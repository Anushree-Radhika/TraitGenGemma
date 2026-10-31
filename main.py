"""Training entry point.

    python main.py                       # spawns one process per GPU (CFG.num_gpus)
    torchrun --nproc-per-node=2 main.py  # or let torchrun spawn the ranks
"""
from config import CFG, setup_env
from dist_utils import launch
from train import run

if __name__ == "__main__":
    setup_env(CFG)
    launch(run, CFG)
