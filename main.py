"""Entry point.

    python main.py                       # spawns one process per GPU (up to CFG.num_gpus)
    torchrun --nproc-per-node=2 main.py  # or let torchrun spawn the ranks
"""
import os
from types import SimpleNamespace

import torch

CFG = SimpleNamespace(
    train_json="/kaggle/input/datasets/neurallogica/dataset/train.json",
    test_json="/kaggle/input/datasets/neurallogica/dataset/test.json",
    img_root="/kaggle/working/CUB_200_2011/images/",
    output_dir="/kaggle/working/gpu_output",
    checkpoint_dir="/kaggle/working/checkpoints",
    lm_id="google/gemma-3-1b-pt",
    clip_id="hf-hub:imageomics/bioclip",
    epochs=10,
    batch_size=4,
    grad_accum=4,
    lr_dec=5e-5,
    lr_enc=1e-5,
    lr_proj=5e-4,
    eval_batch_size=8,
    max_new_tokens=120,
    stop_after_species=True,
    num_gpus=2,
    debug_nan=False,
    master_port="29511",
    nccl_p2p_disable=True,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:  # on Kaggle: read the token from Secrets; elsewhere set HF_TOKEN in the environment
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass


def _worker(rank, world_size, cfg):
    from train import run
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(cfg.master_port))
    run(cfg)


if __name__ == "__main__":
    if CFG.nccl_p2p_disable:
        os.environ["NCCL_P2P_DISABLE"] = "1"

    n_gpu = torch.cuda.device_count()
    world = min(CFG.num_gpus or n_gpu, n_gpu)

    if "LOCAL_RANK" in os.environ:          # started with torchrun -> ranks already spawned
        from train import run
        run(CFG)
    elif world > 1:                         # plain python -> spawn one process per GPU
        assert not torch.cuda.is_initialized(), \
            "CUDA was already used in this session - restart the kernel, then run this first."
        torch.multiprocessing.start_processes(_worker, args=(world, CFG), nprocs=world, start_method="fork")
    else:
        from train import run
        run(CFG)
