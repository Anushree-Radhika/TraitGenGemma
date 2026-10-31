"""Saving / loading of the trainable parts (encoder LoRA, decoder LoRA, projector)."""
import json
import os

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file


def save_best(model, ckpt_dir, epoch, metrics):
    path = os.path.join(ckpt_dir, "best")
    os.makedirs(path, exist_ok=True)
    model.visual.save_pretrained(os.path.join(path, "enc_lora"))
    model.lm.save_pretrained(os.path.join(path, "dec_lora"))
    torch.save({"vis_norm": model.vis_norm.state_dict(),
                "projector": model.projector.state_dict(),
                "out_scale": model.out_scale.data.cpu()},
               os.path.join(path, "projector.pt"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({"epoch": epoch, **metrics}, f, indent=2)


def _load_adapter(peft_model, folder):
    f = os.path.join(folder, "adapter_model.safetensors")
    if os.path.exists(f):
        sd = load_file(f)
    else:
        sd = torch.load(os.path.join(folder, "adapter_model.bin"), map_location="cpu")
    res = set_peft_model_state_dict(peft_model, sd)
    bad = getattr(res, "unexpected_keys", [])
    assert not bad, f"unexpected adapter keys in {folder}: {bad[:3]}"


def load_best(model, best_dir, device):
    """Load a checkpoint written by save_best into a freshly built GemmaTraitGen."""
    _load_adapter(model.visual, os.path.join(best_dir, "enc_lora"))
    _load_adapter(model.lm, os.path.join(best_dir, "dec_lora"))
    proj = torch.load(os.path.join(best_dir, "projector.pt"), map_location="cpu")
    model.vis_norm.load_state_dict(proj["vis_norm"])
    model.projector.load_state_dict(proj["projector"])
    model.out_scale.data = proj["out_scale"].to(device)
    return model
