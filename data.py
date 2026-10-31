"""Records, datasets, collate functions and dataloader construction."""
import json
import os
import re
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from utils import extract_species


# ------------------------------------------------------------------ records
def load_records(path):
    """Returns a list of (image_path, caption, species)."""
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


def unique_by_image(recs):
    seen, out = set(), []
    for r in recs:
        if r[0] not in seen:
            seen.add(r[0])
            out.append(r)
    return out


# ------------------------------------------------------------------ dataset
class ImgCapDS(Dataset):
    def __init__(self, recs, tf, img_root):
        self.recs, self.tf, self.img_root = recs, tf, img_root

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        img, cap, species = self.recs[i]
        image = Image.open(os.path.join(self.img_root, img)).convert("RGB")
        return self.tf(image), cap, species


# ------------------------------------------------------------------ collate
def create_collate_fn(tokenizer, max_len):
    """Training/validation batches: (pixels, input_ids, attention_mask, labels, species_mask)."""
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


def acc_collate(b):
    """Accuracy passes: (pixels, ground-truth species)."""
    return torch.stack([x[0] for x in b]), [x[2] or extract_species(x[1]) for x in b]


def eval_collate(b):
    """Prediction dump: (pixels, captions, species)."""
    return torch.stack([x[0] for x in b]), [x[1] for x in b], [x[2] for x in b]


# ------------------------------------------------------------------ loaders
def shard_loader(recs, tf, img_root, batch_size, rank, world, collate_fn):
    """Unique images, each GPU takes every world-th one."""
    ds = ImgCapDS(unique_by_image(recs), tf, img_root)
    shard = Subset(ds, list(range(rank, len(ds), world)))
    return DataLoader(shard, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)


def build_dataloaders(cfg, tokenizer, preprocess_train, preprocess_val, local_rank, rank, world):
    train_recs = load_records(cfg.train_json)
    test_recs = load_records(cfg.test_json)
    collate = create_collate_fn(tokenizer, 256)

    train_ds = ImgCapDS(train_recs, preprocess_train, cfg.img_root)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if local_rank != -1 else None
    train = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
                       shuffle=(train_sampler is None), collate_fn=collate, num_workers=2, drop_last=True)

    val_ds = ImgCapDS(test_recs, preprocess_val, cfg.img_root)
    val_sampler = DistributedSampler(val_ds, shuffle=False) if local_rank != -1 else None
    val = DataLoader(val_ds, batch_size=cfg.batch_size, sampler=val_sampler, shuffle=False,
                     collate_fn=collate, num_workers=2)

    # accuracy passes use the plain (non-augmented) transform and cover every unique image
    train_acc = shard_loader(train_recs, preprocess_val, cfg.img_root, cfg.eval_batch_size, rank, world, acc_collate)
    test_acc = shard_loader(test_recs, preprocess_val, cfg.img_root, cfg.eval_batch_size, rank, world, acc_collate)

    return SimpleNamespace(train=train, train_sampler=train_sampler, val=val,
                           train_acc=train_acc, test_acc=test_acc)
