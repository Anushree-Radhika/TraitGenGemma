"""Per-epoch loops: training, validation loss and species accuracy."""
import math
from contextlib import nullcontext

import torch
import torch.distributed as dist
from tqdm import tqdm

from dist_utils import is_main
from utils import extract_species, norm_name


def train_one_epoch(model, dataloader, opt, sched, scaler, device, local_rank, ep, eos_id, accum=8):
    model.train()
    loss_sum, count, skipped = 0.0, 0, 0
    pbar = tqdm(dataloader, desc=f"Epoch {ep}") if is_main(local_rank) else dataloader
    opt.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        boundary = (step + 1) % accum == 0 or (step + 1) == len(dataloader)
        sync = model.no_sync() if (not boundary and hasattr(model, "no_sync")) else nullcontext()

        pix, input_ids, attn, labels, species_mask = batch
        with sync:
            loss = model(pix.to(device), input_ids.to(device), attn.to(device),
                         labels.to(device), species_mask.to(device), eos_id)
            # always backward (keeps DDP ranks in lock-step); GradScaler skips the step if grads are inf/nan
            scaler.scale(loss / accum).backward()

        if boundary:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() < before:
                skipped += 1
            if scaler.get_scale() < 1.0:
                raise RuntimeError("GradScaler scale collapsed -> persistent non-finite gradients")
            sched.step()
            opt.zero_grad(set_to_none=True)

        l = loss.item()
        if math.isfinite(l):                 # NaN batches don't poison the running average
            loss_sum += l
            count += 1
        if isinstance(pbar, tqdm):
            pbar.set_postfix(loss=f"{l:.4f}", skipped=skipped)

    return loss_sum / max(count, 1)


@torch.no_grad()
def validate(base_model, loader, device, eos_id, local_rank):
    """Mean weighted loss over the test set, averaged across all GPUs."""
    base_model.eval()
    tot = torch.zeros(2, device=device, dtype=torch.float64)   # [sum of finite batch losses, n batches]
    for pix, input_ids, attn, labels, species_mask in tqdm(loader, desc="Validation", leave=False,
                                                           disable=not is_main(local_rank)):
        l = base_model(pix.to(device), input_ids.to(device), attn.to(device),
                       labels.to(device), species_mask.to(device), eos_id).item()
        if math.isfinite(l):
            tot[0] += l
            tot[1] += 1
    if local_rank != -1:
        dist.all_reduce(tot)
    base_model.train()
    return (tot[0] / tot[1].clamp(min=1)).item()


@torch.no_grad()
def evaluate_accuracy(base_model, loader, device, local_rank, desc, max_new_tokens, stop_after_species):
    """Species accuracy over ALL images in the loader (summed over all GPUs)."""
    base_model.eval()
    tot = torch.zeros(2, device=device, dtype=torch.float64)    # [correct, total]
    for pix, gts in tqdm(loader, desc=desc, leave=False, disable=not is_main(local_rank)):
        gens = base_model.generate(pix.to(device), max_new_tokens, stop_after_species)
        for g, gt in zip(gens, gts):
            if not gt:
                continue
            pred = extract_species(g)
            tot[1] += 1
            if pred and norm_name(pred) == norm_name(gt):
                tot[0] += 1
    if local_rank != -1:
        dist.all_reduce(tot)
    base_model.train()
    return (tot[0] / tot[1].clamp(min=1)).item()
