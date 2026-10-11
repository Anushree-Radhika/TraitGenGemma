import os
import json
import re
import math
import argparse
from contextlib import nullcontext
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import open_clip
from transformers import (AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup,
                          StoppingCriteria, StoppingCriteriaList)
from peft import LoraConfig, get_peft_model

# =============================================================================
# CONFIG  --  edit here, then just run the cell (no torchrun / argparse needed)
# =============================================================================
from types import SimpleNamespace

CFG = SimpleNamespace(
    train_json="/kaggle/input/datasets/neurallogica/dataset/train.json",
    test_json="/kaggle/input/datasets/neurallogica/dataset/test.json",
    img_root="/kaggle/working/CUB_200_2011/images/",
    output_dir="/kaggle/working/gpu_output",
    checkpoint_dir="/kaggle/working/checkpoints",   # best model (highest test accuracy) is saved to <checkpoint_dir>/best
    lm_id="google/gemma-3-1b-pt",
    clip_id="hf-hub:imageomics/bioclip",
    epochs=5,
    batch_size=4,          # per GPU
    grad_accum=4,          # effective batch = batch_size * grad_accum * num_gpus (4*4*2 = 32)
    lr_dec=5e-5,
    lr_enc=1e-5,
    lr_proj=5e-4,
    eval_batch_size=8,     # batch size for the accuracy passes (train + test, ALL images)
    max_new_tokens=120,
    stop_after_species=True,  # stop each generation once  species of "<name>"  is written (much faster)
    num_gpus=None,         # None = use all visible GPUs (2 on Kaggle T4 x2)
    debug_nan=False,       # True = run one batch with hooks, print first module that gives inf/nan
    master_port="29511",
    nccl_p2p_disable=True, # avoids NCCL hangs on Kaggle T4 x2; set False to try P2P
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Gemma is gated: add your HF token as a Kaggle secret named HF_TOKEN (Add-ons -> Secrets)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

# -----------------------------------------------------------------------------
# 1. Dataset (unchanged logic)
# -----------------------------------------------------------------------------
def load_records(path):
    data = json.load(open(path))
    if isinstance(data, dict):
        if "annotations" in data and "images" in data:
            id2file = {im["id"]: im["file_name"] for im in data["images"]}
            return [(id2file[a["image_id"]], a["caption"], "") for a in data["annotations"]]
        data = next(v for v in data.values() if isinstance(v, list))
    recs = []
    for r in data:
        img = r.get("image") or r.get("imagePath") or r.get("image_path") or r.get("file_name") or r.get("filename")
        cap = r.get("gt") or r.get("caption") or r.get("text") or r.get("description")
        species = ""
        m = re.search(r'species of "(.*?)"', str(cap))
        if m:
            species = m.group(1)
        elif img and "/" in img:
            species = img.split("/")[0].split(".", 1)[-1].replace("_", " ")
        for c in (cap if isinstance(cap, list) else [cap]):
            recs.append((img, str(c), species))
    return recs


class ImgCapDS(Dataset):
    def __init__(self, recs, tf, img_root):
        self.recs, self.tf, self.img_root = recs, tf, img_root

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        img, cap, species = self.recs[i]
        image = Image.open(os.path.join(self.img_root, img)).convert("RGB")
        return self.tf(image), cap, species


def create_collate_fn(tokenizer, max_len):
    def collate(batch):
        pix = torch.stack([b[0] for b in batch])
        caps = [b[1] + tokenizer.eos_token for b in batch]
        species = [b[2] for b in batch]

        enc = tokenizer(caps, padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)

        labels = enc.input_ids.clone()
        labels[enc.attention_mask == 0] = -100

        species_mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, (cap, sp) in enumerate(zip(caps, species)):
            if not sp:
                continue
            s = cap.find(sp)
            if s == -1:
                continue
            e = s + len(sp)
            for j, (o_s, o_e) in enumerate(enc.offset_mapping[i]):
                if o_s < e and o_e > s and enc.attention_mask[i, j] == 1:
                    species_mask[i, j] = True

        return pix, enc.input_ids, enc.attention_mask, labels, species_mask
    return collate


# -----------------------------------------------------------------------------
# 2. Model
# -----------------------------------------------------------------------------
SPECIES_PAT = re.compile(r'species of "(.*?)"')

def extract_species(text):
    m = SPECIES_PAT.search(text)
    return m.group(1).strip() if m else ""

def norm_name(s):
    return s.strip().strip("'\"").replace("_", " ").lower()


class SpeciesDone(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tok.batch_decode(input_ids, skip_special_tokens=True)
        return torch.tensor([bool(SPECIES_PAT.search(t)) for t in texts], device=input_ids.device)


class GemmaTraitGen(nn.Module):
    def __init__(self, lm_id, clip_id, device="cuda", n_img_tokens=49):
        super().__init__()
        self.device = device
        self.n_img_tokens = n_img_tokens

        # tokenizer: right padding (Gemma defaults to left), explicit pad token
        self.tokenizer = AutoTokenizer.from_pretrained(lm_id)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- vision: BioCLIP in fp16 (CLIP is fine in fp16), LoRA on MLPs of last 6 blocks
        clip, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(clip_id)
        visual = clip.visual
        del clip
        visual.proj = None            # native 768-d tokens
        visual.output_tokens = True
        visual = visual.to(device, dtype=torch.float16)
        for p in visual.parameters():
            p.requires_grad = False
        enc_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.2, bias="none",
            target_modules=r".*resblocks\.(6|7|8|9|10|11)\.mlp\.(c_fc|c_proj)",
        )
        self.visual = get_peft_model(visual, enc_cfg)

        # ---- decoder: FP32 weights, eager attention (Gemma 3 overflows in fp16)
        lm = AutoModelForCausalLM.from_pretrained(
            lm_id, torch_dtype=torch.float32, attn_implementation="eager"
        ).to(device)
        hidden = lm.config.hidden_size

        # RMS of real (scaled) text embeddings -> used to scale the image tokens
        embed = lm.get_input_embeddings()
        with torch.no_grad():
            es = getattr(embed, "embed_scale", None)
            es = float(es) if es is not None else math.sqrt(hidden)
            text_rms = float(embed.weight[:20000].float().pow(2).mean().sqrt() * es)

        dec_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.2, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.lm = get_peft_model(lm, dec_cfg)

        # trainable params must be fp32 for GradScaler
        for m in (self.visual, self.lm):
            for n, p in m.named_parameters():
                if "lora_" in n:
                    p.data = p.data.float()
                    p.requires_grad = True

        # ---- projector: LN -> MLP -> LN(no affine) * learnable scale (init = text embedding RMS)
        self.vis_norm = nn.LayerNorm(768)
        self.projector = nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.out_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.out_scale = nn.Parameter(torch.tensor(text_rms))
        self.vis_norm.to(device); self.projector.to(device); self.out_norm.to(device)
        self.out_scale.data = self.out_scale.data.to(device)

    def projector_params(self):
        return (list(self.vis_norm.parameters()) + list(self.projector.parameters()) + [self.out_scale])

    def encode_images(self, pixels):
        with torch.autocast("cuda", dtype=torch.float16):
            _, tokens = self.visual(pixels.to(self.device, dtype=torch.float16))   # (B,196,768)
        tokens = tokens.float()
        B, N, D = tokens.shape
        s = int(math.sqrt(N))
        grid = tokens.transpose(1, 2).reshape(B, D, s, s)
        side = int(math.sqrt(self.n_img_tokens))
        pooled = F.adaptive_avg_pool2d(grid, side).flatten(2).transpose(1, 2)       # (B,49,768)
        with torch.autocast("cuda", enabled=False):
            x = self.projector(self.vis_norm(pooled))
            return self.out_norm(x) * self.out_scale                                 # fp32

    @torch.no_grad()
    def generate(self, pixels, max_new_tokens=120, stop_after_species=True):
        """Greedy-decode a caption for each image: [BOS][image tokens] -> text."""
        B = pixels.size(0)
        embed = self.lm.get_input_embeddings()
        img = self.encode_images(pixels)
        bos = embed(torch.full((B, 1), self.tokenizer.bos_token_id, device=self.device))
        emb = torch.cat([bos, img], dim=1)
        mask = torch.ones(emb.shape[:2], dtype=torch.long, device=self.device)
        kw = {}
        if stop_after_species:
            kw["stopping_criteria"] = StoppingCriteriaList([SpeciesDone(self.tokenizer)])
        with torch.autocast("cuda", enabled=False):
            out = self.lm.generate(inputs_embeds=emb, attention_mask=mask, max_new_tokens=max_new_tokens,
                                   do_sample=False, num_beams=1, top_k=None, top_p=None,
                                   eos_token_id=self.tokenizer.eos_token_id,
                                   pad_token_id=self.tokenizer.pad_token_id, **kw)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def forward(self, pixels, input_ids, attn, labels, species_mask, eos_token_id):
        B, Lt = input_ids.shape
        img = self.encode_images(pixels)                                             # (B,49,H) fp32
        embed = self.lm.get_input_embeddings()

        bos = embed(torch.full((B, 1), self.tokenizer.bos_token_id, device=self.device))
        txt = embed(input_ids)
        emb = torch.cat([bos, img, txt], dim=1)
        mask = torch.cat([torch.ones(B, 1 + img.size(1), device=self.device, dtype=attn.dtype), attn], 1)

        # decoder runs in full fp32 (no autocast)
        with torch.autocast("cuda", enabled=False):
            out = self.lm(inputs_embeds=emb, attention_mask=mask, use_cache=False,
                          logits_to_keep=Lt + 1, return_dict=True)
        # kept positions = [last prefix token, text_0 ... text_{Lt-1}]  ->  logits[:, :-1] predicts text_0..text_{Lt-1}
        logits = out.logits[:, :-1].float()

        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                             reduction="none", ignore_index=-100, label_smoothing=0.1).view_as(labels)
        weights = torch.full_like(ce, 2.0)
        weights[labels == eos_token_id] = 5.0
        weights[species_mask] = 5.0
        valid = labels != -100
        return (ce * weights)[valid].sum() / weights[valid].sum()


# -----------------------------------------------------------------------------
# 3. Debug helper: finds the FIRST module producing inf/nan
# -----------------------------------------------------------------------------
@torch.no_grad()
def debug_first_nonfinite(model, batch, device, eos_id):
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


# -----------------------------------------------------------------------------
# 4. Training
# -----------------------------------------------------------------------------
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


def unique_by_image(recs):
    seen, out = set(), []
    for r in recs:
        if r[0] not in seen:
            seen.add(r[0])
            out.append(r)
    return out


def acc_collate(b):
    # (images, ground-truth species)
    return torch.stack([x[0] for x in b]), [x[2] or extract_species(x[1]) for x in b]


def make_acc_loader(recs, tf, img_root, batch_size, rank, world):
    ds = ImgCapDS(unique_by_image(recs), tf, img_root)
    shard = torch.utils.data.Subset(ds, list(range(rank, len(ds), world)))   # each GPU takes every world-th image
    return DataLoader(shard, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=acc_collate)


@torch.no_grad()
def evaluate_accuracy(base_model, loader, device, local_rank, desc, max_new_tokens, stop_after_species):
    """Species accuracy over ALL images in the loader (summed over both GPUs)."""
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


def save_best(base_model, ckpt_dir, epoch, metrics):
    path = os.path.join(ckpt_dir, "best")
    os.makedirs(path, exist_ok=True)
    base_model.visual.save_pretrained(os.path.join(path, "enc_lora"))
    base_model.lm.save_pretrained(os.path.join(path, "dec_lora"))
    torch.save({"vis_norm": base_model.vis_norm.state_dict(),
                "projector": base_model.projector.state_dict(),
                "out_scale": base_model.out_scale.data.cpu()},
               os.path.join(path, "projector.pt"))
    json.dump({"epoch": epoch, **metrics}, open(os.path.join(path, "meta.json"), "w"), indent=2)


def save_curves(hist, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    epochs = range(1, len(hist["train_loss"]) + 1)
    for name, series, ylabel in (
        ("loss.png", {"Train Loss": hist["train_loss"], "Val Loss": hist["val_loss"]}, "Loss"),
        ("accuracy.png", {"Train Acc": hist["train_acc"], "Test Acc": hist["test_acc"]}, "Accuracy (%)"),
    ):
        plt.figure(figsize=(10, 6))
        for label, vals in series.items():
            plt.plot(epochs, vals, label=label, marker="o")
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(ylabel + " per epoch")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(ckpt_dir, name)); plt.close()


def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda")

    # rank 0 loads/downloads first, the other rank waits, then reads from the warm HF cache
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
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

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
            hist["train_loss"].append(avg_loss); hist["val_loss"].append(val_loss)
            hist["train_acc"].append(train_acc * 100); hist["test_acc"].append(test_acc * 100)
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


def _worker(rank, world_size, cfg):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(cfg.master_port))
    main(cfg)


if __name__ == "__main__":
    if CFG.nccl_p2p_disable:
        os.environ["NCCL_P2P_DISABLE"] = "1"

    n_gpu = torch.cuda.device_count()
    world = min(CFG.num_gpus or n_gpu, n_gpu)

    if "LOCAL_RANK" in os.environ:          # started with torchrun -> it already spawned the ranks
        main(CFG)
    elif world > 1:                         # plain notebook cell / python file -> spawn one process per GPU
        assert not torch.cuda.is_initialized(), \
            "CUDA was already used in this session - restart the kernel, then run this cell first."
        torch.multiprocessing.start_processes(_worker, args=(world, CFG), nprocs=world, start_method="fork")
    else:
        main(CFG)