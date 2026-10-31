"""Training / validation / accuracy loops and the per-process entry point `run`."""
import math
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from data import ImgCapDS, create_collate_fn, load_records, make_acc_loader
from model import GemmaTraitGen
from utils import debug_first_nonfinite, extract_species, norm_name, save_best, save_curves


def train_one_epoch(model, dataloader, opt, sched, scaler, device, local_rank, ep, eos_id, accum=8):
    model.train()
    loss_sum, count, skipped = 0.0, 0, 0
    pbar = tqdm(dataloader, desc=f"Epoch {ep}") if local_rank in [-1, 0] else dataloader
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
        if math.isfinite(l):                 # NaN batches no longer poison the running average
            loss_sum += l
            count += 1
        if local_rank in [-1, 0] and isinstance(pbar, tqdm):
            pbar.set_postfix(loss=f"{l:.4f}", skipped=skipped)

    return loss_sum / max(count, 1)


@torch.no_grad()
def validate(base_model, loader, device, eos_id, local_rank):
    """Mean weighted loss over the test set, averaged across all GPUs."""
    base_model.eval()
    tot = torch.zeros(2, device=device, dtype=torch.float64)   # [sum of finite batch losses, n batches]
    for pix, input_ids, attn, labels, species_mask in tqdm(loader, desc="Validation", leave=False,
                                                           disable=local_rank not in [-1, 0]):
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
    for pix, gts in tqdm(loader, desc=desc, leave=False, disable=local_rank not in [-1, 0]):
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


def run(args):
    """Full training run for one process (one GPU)."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda")

    # rank 0 loads/downloads first, the other ranks wait, then read from the warm HF cache
    if local_rank > 0:
        dist.barrier(device_ids=[local_rank])
    model = GemmaTraitGen(args.lm_id, args.clip_id, device=device)
    if local_rank == 0:
        dist.barrier(device_ids=[local_rank])
    tokenizer = model.tokenizer
    eos_id = tokenizer.eos_token_id

    recs = load_records(args.train_json)
    ds = ImgCapDS(recs, model.preprocess_train, args.img_root)
    sampler = DistributedSampler(ds, shuffle=True) if local_rank != -1 else None
    train_loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                              collate_fn=create_collate_fn(tokenizer, 256), num_workers=2, drop_last=True)

    val_ds = ImgCapDS(load_records(args.test_json), model.preprocess_val, args.img_root)
    val_sampler = DistributedSampler(val_ds, shuffle=False) if local_rank != -1 else None
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
                            collate_fn=create_collate_fn(tokenizer, 256), num_workers=2)

    world = dist.get_world_size() if local_rank != -1 else 1
    rank = max(local_rank, 0)
    # accuracy passes use the plain (non-augmented) transform and cover every unique image
    train_acc_loader = make_acc_loader(recs, model.preprocess_val, args.img_root, args.eval_batch_size, rank, world)
    test_acc_loader = make_acc_loader(load_records(args.test_json), model.preprocess_val, args.img_root,
                                      args.eval_batch_size, rank, world)

    enc_params = [p for n, p in model.visual.named_parameters() if "lora_" in n]
    dec_params = [p for n, p in model.lm.named_parameters() if "lora_" in n]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_enc},
        {"params": dec_params, "lr": args.lr_dec},
        {"params": model.projector_params(), "lr": args.lr_proj},
    ], weight_decay=0.01)

    if args.debug_nan and local_rank in [-1, 0]:
        debug_first_nonfinite(model, next(iter(train_loader)), device, eos_id)

    base_model = model
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 12)
    total_steps = args.epochs * math.ceil(len(train_loader) / args.grad_accum)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)

    best_acc, best_val = -1.0, float("inf")
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "test_acc": []}
    for ep in range(args.epochs):
        if sampler:
            sampler.set_epoch(ep)
        avg_loss = train_one_epoch(model, train_loader, opt, sched, scaler, device, local_rank,
                                   ep + 1, eos_id, args.grad_accum)
        val_loss = validate(base_model, val_loader, device, eos_id, local_rank)
        train_acc = evaluate_accuracy(base_model, train_acc_loader, device, local_rank, "Train accuracy",
                                      args.max_new_tokens, args.stop_after_species)
        test_acc = evaluate_accuracy(base_model, test_acc_loader, device, local_rank, "Test accuracy",
                                     args.max_new_tokens, args.stop_after_species)

        if local_rank in [-1, 0]:
            print(f"Epoch {ep+1} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
                  f"| Val Loss: {val_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
            hist["train_loss"].append(avg_loss)
            hist["val_loss"].append(val_loss)
            hist["train_acc"].append(train_acc * 100)
            hist["test_acc"].append(test_acc * 100)
            save_curves(hist, args.checkpoint_dir)
            # best = highest test accuracy (lower val loss breaks ties)
            if test_acc > best_acc or (test_acc == best_acc and val_loss < best_val):
                best_acc, best_val = test_acc, val_loss
                save_best(base_model, args.checkpoint_dir, ep + 1,
                          {"test_acc": test_acc, "train_acc": train_acc,
                           "val_loss": val_loss, "train_loss": avg_loss})
                print(f"  -> new best (test acc {test_acc*100:.2f}%), saved to "
                      f"{os.path.join(args.checkpoint_dir, 'best')}")

    if local_rank != -1:
        dist.destroy_process_group()
