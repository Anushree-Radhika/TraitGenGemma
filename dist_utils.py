"""Distributed helpers: process-group setup, rank-ordered loading and the multi-GPU launcher."""
import os

import torch
import torch.distributed as dist


def setup_distributed():
    """Returns (local_rank, device). local_rank == -1 means single-process."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda")
    return local_rank, device


def cleanup_distributed(local_rank):
    if local_rank != -1 and dist.is_initialized():
        dist.destroy_process_group()


def is_main(local_rank):
    return local_rank in (-1, 0)


def rank_and_world(local_rank):
    world = dist.get_world_size() if local_rank != -1 else 1
    return max(local_rank, 0), world


def rank_ordered(fn, local_rank):
    """Run fn() on rank 0 first (warms the HF cache), then on the other ranks."""
    if local_rank > 0:
        dist.barrier(device_ids=[local_rank])
    out = fn()
    if local_rank == 0:
        dist.barrier(device_ids=[local_rank])
    return out


def _worker(rank, target, world_size, cfg, port):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    target(cfg)


def launch(target, cfg, port=None):
    """Run target(cfg) under torchrun, or spawn one process per GPU, or run in-process."""
    port = port or cfg.master_port
    n_gpu = torch.cuda.device_count()
    world = min(cfg.num_gpus or n_gpu, n_gpu)

    if "LOCAL_RANK" in os.environ:          # started with torchrun -> ranks already exist
        target(cfg)
    elif world > 1:                         # plain python / notebook -> spawn one process per GPU
        assert not torch.cuda.is_initialized(), \
            "CUDA was already used in this session - restart the kernel and run this first."
        torch.multiprocessing.start_processes(_worker, args=(target, world, cfg, port),
                                              nprocs=world, start_method="fork")
    else:
        target(cfg)
