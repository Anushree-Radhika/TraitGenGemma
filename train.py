"""Training entry point for one process (one GPU): `run(cfg)`."""
import math
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup

from checkpoint import save_best
from data import build_dataloaders
from dist_utils import cleanup_distributed, is_main, rank_and_world, rank_ordered, setup_distributed
from engine import evaluate_accuracy, train_one_epoch, validate
from model import GemmaTraitGen
from utils import debug_first_nonfinite, save_curves


def build_optimizer(model, cfg):
    return torch.optim.AdamW([
        {"params": model.lora_params(model.visual), "lr": cfg.lr_enc},
        {"params": model.lora_params(model.lm), "lr": cfg.lr_dec},
        {"params": model.projector_params(), "lr": cfg.lr_proj},
    ], weight_decay=cfg.weight_decay)


def run(cfg):
    local_rank, device = setup_distributed()
    rank, world = rank_and_world(local_rank)

    model = rank_ordered(lambda: GemmaTraitGen(cfg.lm_id, cfg.clip_id, device=device), local_rank)
    eos_id = model.tokenizer.eos_token_id

    loaders = build_dataloaders(cfg, model.tokenizer, model.preprocess_train, model.preprocess_val,
                                local_rank, rank, world)
    opt = build_optimizer(model, cfg)

    if cfg.debug_nan and is_main(local_rank):
        debug_first_nonfinite(model, next(iter(loaders.train)), device, eos_id)

    base_model = model
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 12)
    total_steps = cfg.epochs * math.ceil(len(loaders.train) / cfg.grad_accum)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)

    best_acc, best_val = -1.0, float("inf")
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "test_acc": []}

    for ep in range(cfg.epochs):
        if loaders.train_sampler:
            loaders.train_sampler.set_epoch(ep)

        avg_loss = train_one_epoch(model, loaders.train, opt, sched, scaler, device, local_rank,
                                   ep + 1, eos_id, cfg.grad_accum)
        val_loss = validate(base_model, loaders.val, device, eos_id, local_rank)
        train_acc = evaluate_accuracy(base_model, loaders.train_acc, device, local_rank, "Train accuracy",
                                      cfg.max_new_tokens, cfg.stop_after_species)
        test_acc = evaluate_accuracy(base_model, loaders.test_acc, device, local_rank, "Test accuracy",
                                     cfg.max_new_tokens, cfg.stop_after_species)

        if is_main(local_rank):
            print(f"Epoch {ep+1} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
                  f"| Val Loss: {val_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
            hist["train_loss"].append(avg_loss)
            hist["val_loss"].append(val_loss)
            hist["train_acc"].append(train_acc * 100)
            hist["test_acc"].append(test_acc * 100)
            save_curves(hist, cfg.checkpoint_dir)

            # best = highest test accuracy (lower val loss breaks ties)
            if test_acc > best_acc or (test_acc == best_acc and val_loss < best_val):
                best_acc, best_val = test_acc, val_loss
                save_best(base_model, cfg.checkpoint_dir, ep + 1,
                          {"test_acc": test_acc, "train_acc": train_acc,
                           "val_loss": val_loss, "train_loss": avg_loss})
                print(f"  -> new best (test acc {test_acc*100:.2f}%), saved to "
                      f"{os.path.join(cfg.checkpoint_dir, 'best')}")

    cleanup_distributed(local_rank)
