# ======== Generate text for images with your fine-tuned weights  (paste in ONE cell; uses both GPUs) ========
# Needs: enc.safetensors, dec.safetensors, projector.pt  +  internet (downloads Gemma 3 1B and BioCLIP once)
import os, re, json, glob, time, random, textwrap
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import open_clip
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
import matplotlib.pyplot as plt

# ------------------------------ settings ------------------------------
ENC_PATH    = "/kaggle/input/datasets/neurallogica/checkpoints/best/enc_lora/adapter_model.safetensors"      # <- where your three files are
DEC_PATH    = "/kaggle/input/datasets/neurallogica/checkpoints/best/dec_lora/adapter_model.safetensors"
PROJ_PATH   = "/kaggle/input/datasets/neurallogica/checkpoints/best/projector.pt"

IMAGE_DIR   = "/kaggle/working/CUB_200_2011/images/"   # folder with one sub-folder per bird class (CUB layout)
TEST_JSON   = "/kaggle/input/datasets/neurallogica/dataset/test.json"   # sample only from these unseen images; "" = whole folder
IMAGE_FILES = [
    "/kaggle/working/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0051_796103.jpg",
    "/kaggle/working/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0010_796097.jpg",
    "/kaggle/working/CUB_200_2011/images/002.Laysan_Albatross/Laysan_Albatross_0039_924.jpg",
    "/kaggle/working/CUB_200_2011/images/002.Laysan_Albatross/Laysan_Albatross_0056_500.jpg",
    "/kaggle/working/CUB_200_2011/images/115.Brewer_Sparrow/Brewer_Sparrow_0068_107422.jpg",
    "/kaggle/working/CUB_200_2011/images/115.Brewer_Sparrow/Brewer_Sparrow_0036_107451.jpg",
    "/kaggle/working/CUB_200_2011/images/176.Prairie_Warbler/Prairie_Warbler_0063_172682.jpg",
    "/kaggle/working/CUB_200_2011/images/176.Prairie_Warbler/Prairie_Warbler_0115_172689.jpg",
]       
N_CLASSES, IMAGES_PER_CLASS = 4, 2                    # 4 random birds x 2 images each = 8 images
SEED        = None                                    # None = new random pick every run; e.g. 42 = repeatable

LM_ID, CLIP_ID = "google/gemma-3-1b-pt", "hf-hub:imageomics/bioclip"
BATCH_SIZE     = 8            # per GPU
MAX_NEW_TOKENS = 200
OUT_JSON       = "/kaggle/working/generated_captions.json"
# ----------------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:   # Gemma is gated: add your HF token as a Kaggle secret named HF_TOKEN
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass


class InferModel(nn.Module):
    """Same architecture as in training: BioCLIP(+LoRA) -> projector -> Gemma 3 1B(+LoRA)."""

    def __init__(self, device, n_img_tokens=49):
        super().__init__()
        self.device, self.n_img_tokens = device, n_img_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(LM_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        clip, _, self.preprocess = open_clip.create_model_and_transforms(CLIP_ID)
        visual = clip.visual
        del clip
        visual.proj, visual.output_tokens = None, True
        visual = visual.to(device, dtype=torch.float16)
        enc_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                             target_modules=r".*resblocks\.(6|7|8|9|10|11)\.mlp\.(c_fc|c_proj)")
        self.visual = get_peft_model(visual, enc_cfg)

        lm = AutoModelForCausalLM.from_pretrained(LM_ID, torch_dtype=torch.float32,
                                                  attn_implementation="eager").to(device)
        hidden = lm.config.hidden_size
        dec_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.lm = get_peft_model(lm, dec_cfg)
        for m in (self.visual, self.lm):
            for n, p in m.named_parameters():
                if "lora_" in n:
                    p.data = p.data.float()

        self.vis_norm = nn.LayerNorm(768).to(device)
        self.projector = nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Linear(hidden, hidden)).to(device)
        self.out_norm = nn.LayerNorm(hidden, elementwise_affine=False).to(device)
        self.out_scale = nn.Parameter(torch.tensor(1.0, device=device))

    def load_weights(self):
        def read(path):
            return load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu")
        for peft_model, path in ((self.visual, ENC_PATH), (self.lm, DEC_PATH)):
            res = set_peft_model_state_dict(peft_model, read(path))
            bad = getattr(res, "unexpected_keys", [])
            assert not bad, f"{path}: unexpected keys {bad[:3]} (architecture does not match the checkpoint)"
        proj = torch.load(PROJ_PATH, map_location="cpu")
        self.vis_norm.load_state_dict(proj["vis_norm"])
        self.projector.load_state_dict(proj["projector"])
        self.out_scale.data = proj["out_scale"].to(self.device)
        self.lm = self.lm.merge_and_unload()     # fold decoder LoRA into the weights -> faster generation
        self.eval()

    def encode_images(self, pixels):
        with torch.autocast("cuda", dtype=torch.float16):
            _, tokens = self.visual(pixels.to(self.device, dtype=torch.float16))
        tokens = tokens.float()
        B, N, D = tokens.shape
        s = int(N ** 0.5)
        grid = tokens.transpose(1, 2).reshape(B, D, s, s)
        pooled = F.adaptive_avg_pool2d(grid, int(self.n_img_tokens ** 0.5)).flatten(2).transpose(1, 2)
        with torch.autocast("cuda", enabled=False):
            return self.out_norm(self.projector(self.vis_norm(pooled))) * self.out_scale

    @torch.no_grad()
    def generate(self, pixels):
        B = pixels.size(0)
        tok = self.tokenizer
        embed = self.lm.get_input_embeddings()
        emb = torch.cat([embed(torch.full((B, 1), tok.bos_token_id, device=self.device)),
                         self.encode_images(pixels)], dim=1)           # [BOS][image tokens] -> text
        mask = torch.ones(emb.shape[:2], dtype=torch.long, device=self.device)
        with torch.autocast("cuda", enabled=False):
            out = self.lm.generate(inputs_embeds=emb, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS,
                                   do_sample=False, num_beams=1, top_k=None, top_p=None,
                                   eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        return tok.batch_decode(out, skip_special_tokens=True)


class ImgDS(Dataset):
    def __init__(self, items, tf):                 # items = [(original_index, path)]
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        idx, path = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), idx


MARKER = "/tmp/rank0_model_loaded"

def _worker(rank, world, files):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    if rank > 0:                                   # let rank 0 download/load first (warm HF cache)
        t0 = time.time()
        while not os.path.exists(MARKER) and time.time() - t0 < 900:
            time.sleep(2)
    model = InferModel(device)
    model.load_weights()
    if rank == 0:
        open(MARKER, "w").close()

    items = [(i, files[i]) for i in range(rank, len(files), world)]    # every world-th image
    loader = DataLoader(ImgDS(items, model.preprocess), batch_size=BATCH_SIZE, num_workers=2)
    out = []
    for pix, idxs in loader:
        for i, text in zip(idxs.tolist(), model.generate(pix.to(device))):
            out.append({"index": i, "image": files[i], "text": text})
        if rank == 0:
            print(f"GPU0: {len(out)}/{len(items)} done", end="\r")
    json.dump(out, open(f"/kaggle/working/_gen_rank{rank}.json", "w"))


# ---- choose images: your hardcoded list, or N_CLASSES random birds x IMAGES_PER_CLASS random images
def _pick_random():
    def _json_images(path):
        data = json.load(open(path))
        if isinstance(data, dict):
            if "images" in data:
                return sorted({im["file_name"] for im in data["images"]})
            data = next(v for v in data.values() if isinstance(v, list))
        keys = ("image", "imagePath", "image_path", "file_name", "filename")
        return sorted({next(r[k] for k in keys if r.get(k)) for r in data})

    all_files = []
    if TEST_JSON and os.path.exists(TEST_JSON):
        all_files = [os.path.join(IMAGE_DIR, r) for r in _json_images(TEST_JSON)]
        print("Sampling from the images listed in test.json (not used for training)")
    if not all_files:
        print("TEST_JSON not found/empty -> sampling from the whole image folder")
        all_files = sorted(f for e in ("jpg", "jpeg", "png", "webp") for f in glob.glob(os.path.join(IMAGE_DIR, "**", f"*.{e}"), recursive=True))
    all_files = [f for f in all_files if os.path.exists(f)]

    by_class = {}
    for f in all_files:
        by_class.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
    by_class = {c: v for c, v in by_class.items() if len(v) >= IMAGES_PER_CLASS}
    assert len(by_class) >= N_CLASSES, "Need images in class sub-folders - check IMAGE_DIR (and TEST_JSON)"

    rng = random.Random(SEED)
    files, gt_class = [], []
    for c in rng.sample(sorted(by_class), N_CLASSES):
        for f in rng.sample(by_class[c], IMAGES_PER_CLASS):
            files.append(f); gt_class.append(c)
    return files, gt_class

if IMAGE_FILES:
    files = list(IMAGE_FILES)
    gt_class = [os.path.basename(os.path.dirname(f)) for f in files]   # class = folder name
else:
    files, gt_class = _pick_random()
print(f"{len(files)} images")

# ---- run: one process per GPU (works with 1 GPU too)
if os.path.exists(MARKER):
    os.remove(MARKER)
world = max(torch.cuda.device_count(), 1)
if world > 1:
    assert not torch.cuda.is_initialized(), "CUDA already used in this kernel - restart the session and run this cell first."
    torch.multiprocessing.start_processes(_worker, args=(world, files), nprocs=world, start_method="fork")
else:
    _worker(0, 1, files)

# ---- merge, compare with the true class, save, show
clean = lambda c: re.sub(r"^\d+\.", "", c).replace("_", " ").strip()
results = sorted((r for k in range(world) for r in json.load(open(f"/kaggle/working/_gen_rank{k}.json"))),
                 key=lambda r: r["index"])
for r in results:
    m = re.search(r'species of "(.*?)"', r["text"])
    r["true_class"] = clean(gt_class[r["index"]])
    r["pred_species"] = m.group(1).strip() if m else ""
    r["match"] = r["pred_species"].lower().replace("_", " ") == r["true_class"].lower()
json.dump(results, open(OUT_JSON, "w"), indent=1)
print(f"\nSaved {len(results)} captions to {OUT_JSON}   |   species correct: {sum(r['match'] for r in results)}/{len(results)}\n")

for r in results:
    print(f"[{os.path.basename(r['image'])}]  TRUE: {r['true_class']}  |  PRED: {r['pred_species'] or '(none)'}  {'OK' if r['match'] else 'X'}")
    print(textwrap.fill(r["text"], 110), "\n")

COLS = IMAGES_PER_CLASS
ROWS = -(-len(results) // COLS)
fig, axes = plt.subplots(ROWS, COLS, figsize=(6 * COLS, 5.5 * ROWS), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")
for i, r in enumerate(results):
    ax = axes[i // COLS][i % COLS]
    ax.imshow(Image.open(r["image"]).convert("RGB"))
    ax.set_title(f"TRUE: {r['true_class']}\n" + textwrap.fill(r["text"], 55)[:300], fontsize=8,
                 color="green" if r["match"] else "red")
plt.tight_layout()
plt.show()
