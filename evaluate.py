"""Standalone test-set evaluation of the best checkpoint (multi-GPU).

    python evaluate.py
    torchrun --nproc-per-node=2 evaluate.py
"""
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from checkpoint import load_best
from config import CFG, setup_env
from data import ImgCapDS, eval_collate, load_records, unique_by_image
from dist_utils import cleanup_distributed, is_main, launch, rank_and_world, rank_ordered, setup_distributed
from model import GemmaTraitGen
from utils import extract_species, norm_name


def run_eval(cfg):
    local_rank, device = setup_distributed()
    rank, world = rank_and_world(local_rank)

    model = rank_ordered(lambda: GemmaTraitGen(cfg.lm_id, cfg.clip_id, device=device), local_rank)
    load_best(model, os.path.join(cfg.checkpoint_dir, "best"), device)
    model.eval()

    ds = ImgCapDS(unique_by_image(load_records(cfg.test_json)), model.preprocess_val, cfg.img_root)
    loader = DataLoader(Subset(ds, list(range(rank, len(ds), world))), batch_size=cfg.eval_batch_size,
                        shuffle=False, num_workers=2, collate_fn=eval_collate)

    local, no_gt = [], 0
    for pix, caps, sps in tqdm(loader, desc=f"GPU{rank}", disable=not is_main(local_rank)):
        gens = model.generate(pix.to(device), cfg.max_new_tokens, cfg.stop_after_species)
        for g, cap, sp in zip(gens, caps, sps):
            gt = sp or extract_species(cap)
            if not gt:
                no_gt += 1
                continue
            pred = extract_species(g)
            local.append({"gt": gt, "pred": pred,
                          "correct": bool(pred) and norm_name(pred) == norm_name(gt), "generated": g})

    if local_rank != -1:
        gathered = [None] * world
        dist.all_gather_object(gathered, (local, no_gt))
    else:
        gathered = [(local, no_gt)]

    if is_main(local_rank):
        results = [r for part, _ in gathered for r in part]
        correct = sum(r["correct"] for r in results)
        report = {"accuracy": correct / max(len(results), 1), "correct": correct, "total": len(results),
                  "no_species_found": sum(r["pred"] == "" for r in results),
                  "no_ground_truth": sum(n for _, n in gathered), "results": results}
        with open(cfg.predictions_json, "w") as f:
            json.dump(report, f, indent=1)

        print("\n" + "=" * 50)
        print(f"TEST ACCURACY: {report['accuracy']*100:.2f}%   ({correct}/{report['total']})   [{world} GPUs]")
        print(f"No species found: {report['no_species_found']}   |   no ground truth: {report['no_ground_truth']}")
        print("=" * 50)
        for r in results[:5]:
            print(f"GT: {r['gt']!r} | PRED: {r['pred']!r} | {r['generated'][:150]!r}")
        print("Saved predictions to", cfg.predictions_json)

    if local_rank != -1:
        dist.barrier()
    cleanup_distributed(local_rank)


if __name__ == "__main__":
    setup_env(CFG)
    launch(run_eval, CFG, port=CFG.eval_master_port)
