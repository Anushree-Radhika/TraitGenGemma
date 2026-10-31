"""Species parsing, generation stopping criterion, NaN debugging and training-curve plots."""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import StoppingCriteria

SPECIES_PAT = re.compile(r'species of "(.*?)"')


def extract_species(text):
    m = SPECIES_PAT.search(text)
    return m.group(1).strip() if m else ""


def norm_name(s):
    return s.strip().strip("'\"").replace("_", " ").lower()


class SpeciesDone(StoppingCriteria):
    """Stop each sequence once  species of "<name>"  has been written."""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tok.batch_decode(input_ids, skip_special_tokens=True)
        return torch.tensor([bool(SPECIES_PAT.search(t)) for t in texts], device=input_ids.device)


@torch.no_grad()
def debug_first_nonfinite(model, batch, device, eos_id):
    """Run one batch with hooks and report the FIRST module that outputs inf/nan."""
    handles = []

    def mk(name):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                raise RuntimeError(f"first non-finite output: {name} (dtype={t.dtype})")
        return hook

    for n, m in model.named_modules():
        handles.append(m.register_forward_hook(mk(n)))
    pix, ids, attn, labels, sm = batch
    try:
        loss = model(pix.to(device), ids.to(device), attn.to(device), labels.to(device), sm.to(device), eos_id)
        print("debug forward OK, loss =", loss.item())
    except RuntimeError as e:
        print("DEBUG:", e)
    finally:
        for h in handles:
            h.remove()


def save_curves(hist, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(hist["train_loss"]) + 1)
    for name, series, ylabel in (
        ("loss.png", {"Train Loss": hist["train_loss"], "Val Loss": hist["val_loss"]}, "Loss"),
        ("accuracy.png", {"Train Acc": hist["train_acc"], "Test Acc": hist["test_acc"]}, "Accuracy (%)"),
    ):
        plt.figure(figsize=(10, 6))
        for label, vals in series.items():
            plt.plot(epochs, vals, label=label, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(ylabel + " per epoch")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, name))
        plt.close()
