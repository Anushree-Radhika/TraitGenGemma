# ===================== Test accuracy on test.json (paste in a NEW cell after training) =====================
import os, re, json, importlib.util
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

# ------------------------------ settings ------------------------------
TRAIN_FILE         = "train.py"                                   # the training script (provides the model class)
CKPT_DIR           = "/kaggle/working/checkpoints/best"           # contains enc_lora/, dec_lora/, projector.pt
TEST_JSON          = "/kaggle/input/datasets/neurallogica/dataset/test.json"
IMG_ROOT           = "/kaggle/working/CUB_200_2011/images/"
LM_ID              = "google/gemma-3-1b-pt"
CLIP_ID            = "hf-hub:imageomics/bioclip"
BATCH_SIZE         = 8
MAX_NEW_TOKENS     = 120
STOP_AFTER_SPECIES = True     # stop each sequence once  species of "<name>"  has been written (much faster)
OUT_JSON           = "/kaggle/working/test_predictions.json"
# ----------------------------------------------------------------------

# --- get the model class / dataset helpers (from this kernel if training ran here, else from train.py)
if "GemmaTraitGen" not in globals():
    spec = importlib.util.spec_from_file_location("trainmod", TRAIN_FILE)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    GemmaTraitGen, load_records, ImgCapDS = T.GemmaTraitGen, T.load_records, T.ImgCapDS

device = torch.device("cuda:0")
model = GemmaTraitGen(LM_ID, CLIP_ID, device=device)
tok = model.tokenizer

# --- load the saved best weights
def load_adapter(peft_model, folder):
    f = os.path.join(folder, "adapter_model.safetensors")
    sd = load_file(f) if os.path.exists(f) else torch.load(os.path.join(folder, "adapter_model.bin"), map_location="cpu")
    res = set_peft_model_state_dict(peft_model, sd)
    bad = getattr(res, "unexpected_keys", [])
    assert not bad, f"unexpected adapter keys in {folder}: {bad[:3]}"

load_adapter(model.visual, os.path.join(CKPT_DIR, "enc_lora"))
load_adapter(model.lm, os.path.join(CKPT_DIR, "dec_lora"))
proj = torch.load(os.path.join(CKPT_DIR, "projector.pt"), map_location="cpu")
model.vis_norm.load_state_dict(proj["vis_norm"])
model.projector.load_state_dict(proj["projector"])
model.out_scale.data = proj["out_scale"].to(device)
model.eval()
print("Loaded checkpoint:", CKPT_DIR)
if os.path.exists(os.path.join(CKPT_DIR, "meta.json")):
    print(open(os.path.join(CKPT_DIR, "meta.json")).read())

# --- data: every unique test image
PAT = re.compile(r'species of "(.*?)"')

def extract(text):
    m = PAT.search(text)
    return m.group(1).strip() if m else ""

def norm(s):
    return s.strip().strip("'\"").replace("_", " ").lower()

seen, recs = set(), []
for r in load_records(TEST_JSON):
    if r[0] in seen:
        continue
    seen.add(r[0]); recs.append(r)

ds = ImgCapDS(recs, model.preprocess_val, IMG_ROOT)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                    collate_fn=lambda b: (torch.stack([x[0] for x in b]), [x[1] for x in b], [x[2] for x in b]))
print(f"Test images: {len(ds)}")

# --- generation
class SpeciesDone(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        texts = tok.batch_decode(input_ids, skip_special_tokens=True)
        return torch.tensor([bool(PAT.search(t)) for t in texts], device=input_ids.device)

@torch.no_grad()
def generate(pix):
    B = pix.size(0)
    embed = model.lm.get_input_embeddings()
    img = model.encode_images(pix)                                              # (B,49,H) fp32
    bos = embed(torch.full((B, 1), tok.bos_token_id, device=device))
    emb = torch.cat([bos, img], dim=1)                                          # same prefix as in training
    mask = torch.ones(emb.shape[:2], dtype=torch.long, device=device)
    kw = {"stopping_criteria": StoppingCriteriaList([SpeciesDone()])} if STOP_AFTER_SPECIES else {}
    with torch.autocast("cuda", enabled=False):
        out = model.lm.generate(inputs_embeds=emb, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False, num_beams=1, top_k=None, top_p=None,
                                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, **kw)
    return tok.batch_decode(out, skip_special_tokens=True)

# --- evaluate
correct = total = no_pred = no_gt = 0
results = []
for pix, caps, sps in tqdm(loader, desc="Evaluating"):
    gens = generate(pix.to(device))
    for g, cap, sp in zip(gens, caps, sps):
        gt = sp or extract(cap)
        if not gt:
            no_gt += 1
            continue
        pred = extract(g)
        total += 1
        ok = bool(pred) and norm(pred) == norm(gt)
        correct += ok
        no_pred += (pred == "")
        results.append({"gt": gt, "pred": pred, "correct": ok, "generated": g})

acc = correct / max(total, 1)
print("\n" + "=" * 50)
print(f"TEST ACCURACY: {acc*100:.2f}%   ({correct}/{total})")
print(f"No species found in generated text: {no_pred}   |   test samples without ground truth: {no_gt}")
print("=" * 50)
for r in results[:5]:
    print(f"GT: {r['gt']!r} | PRED: {r['pred']!r} | {r['generated'][:150]!r}")

json.dump({"accuracy": acc, "correct": correct, "total": total, "results": results},
          open(OUT_JSON, "w"), indent=1)
print("Saved predictions to", OUT_JSON)

# Evaluating: 100%|██████████| 750/750 [13:44<00:00,  1.10s/it]

# ==================================================
# TEST ACCURACY: 59.83%   (3586/5994)
# No species found in generated text: 0   |   test samples without ground truth: 0
# ==================================================
# GT: 'Black footed Albatross' | PRED: 'Red breasted Merganser' | 'It is a species of "Red breasted Merganser"'
# GT: 'Black footed Albatross' | PRED: 'Black footed Albatross' | 'It is a species of "Black footed Albatross"'
# GT: 'Black footed Albatross' | PRED: 'Pomarine Jaeger' | 'It is a species of "Pomarine Jaeger"'
# GT: 'Black footed Albatross' | PRED: 'Black footed Albatross' | 'It is a species of "Black footed Albatross"'
# GT: 'Black footed Albatross' | PRED: 'Black footed Albatross' | 'It is a species of "Black footed Albatross"'
# Saved predictions to /kaggle/working/test_predictions.json