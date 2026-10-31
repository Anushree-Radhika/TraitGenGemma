"""Central configuration and environment setup. Edit values here only."""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # ---- paths
    train_json: str = "train.json"
    test_json: str = "test.json"
    img_root: str = "CUB_200_2011/images/"
    output_dir: str = "gpu_output"
    checkpoint_dir: str = "checkpoints"      # best model -> <checkpoint_dir>/best
    predictions_json: str = "test_predictions.json"

    # ---- models
    lm_id: str = "google/gemma-3-1b-pt"
    clip_id: str = "hf-hub:imageomics/bioclip"

    # ---- optimisation
    epochs: int = 10
    batch_size: int = 4           # per GPU
    grad_accum: int = 4           # effective batch = batch_size * grad_accum * num_gpus
    lr_dec: float = 5e-5
    lr_enc: float = 1e-5
    lr_proj: float = 5e-4
    weight_decay: float = 0.01

    # ---- evaluation / generation
    eval_batch_size: int = 8
    max_new_tokens: int = 120
    stop_after_species: bool = True   # stop once  species of "<name>"  has been generated

    # ---- distributed / debugging
    num_gpus: int = 2                 # None = all visible GPUs
    debug_nan: bool = False
    master_port: str = "29511"
    eval_master_port: str = "29533"
    nccl_p2p_disable: bool = True     # avoids NCCL hangs on Kaggle T4 x2


CFG = Config()


def setup_env(cfg: Config = CFG):
    """Process-level environment: tokenizers, NCCL, and the HF token (Kaggle Secrets)."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if cfg.nccl_p2p_disable:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    try:  # on Kaggle read HF_TOKEN from Secrets; elsewhere export HF_TOKEN yourself
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
