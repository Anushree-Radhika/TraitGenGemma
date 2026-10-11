# ============ Test accuracy on test.json, 2 GPUs (paste in a NEW cell after training; no torchrun needed) ============
import os, re, json, importlib.util
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

# ------------------------------ settings ------------------------------
TRAIN_FILE         = "train.py"                                   # training script (provides the model class)
CKPT_DIR           = "/kaggle/working/checkpoints/best"           # contains enc_lora/, dec_lora/, projector.pt
TEST_JSON          = "/kaggle/input/datasets/neurallogica/dataset/test.json"
IMG_ROOT           = "/kaggle/working/CUB_200_2011/images/"
LM_ID              = "google/gemma-3-1b-pt"
CLIP_ID            = "hf-hub:imageomics/bioclip"
BATCH_SIZE         = 8            # per GPU
MAX_NEW_TOKENS     = 120
STOP_AFTER_SPECIES = True         # stop each sequence once  species of "<name>"  is written (much faster)
OUT_JSON           = "/kaggle/working/test_predictions.json"
MASTER_PORT        = "29533"
# ----------------------------------------------------------------------
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# model class / dataset helpers (from this kernel if training ran here, else from train.py)
if "GemmaTraitGen" not in globals():
    _spec = importlib.util.spec_from_file_location("trainmod", TRAIN_FILE)
    _T = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_T)
    GemmaTraitGen, load_records, ImgCapDS = _T.GemmaTraitGen, _T.load_records, _T.ImgCapDS

PAT = re.compile(r'species of "(.*?)"')
extract = lambda text: (PAT.search(text).group(1).strip() if PAT.search(text) else "")
norm = lambda s: s.strip().strip("'\"").replace("_", " ").lower()

def _collate(b):
    return torch.stack([x[0] for x in b]), [x[1] for x in b], [x[2] for x in b]


def _load_adapter(peft_model, folder):
    f = os.path.join(folder, "adapter_model.safetensors")
    sd = load_file(f) if os.path.exists(f) else torch.load(os.path.join(folder, "adapter_model.bin"), map_location="cpu")
    res = set_peft_model_state_dict(peft_model, sd)
    bad = getattr(res, "unexpected_keys", [])
    assert not bad, f"unexpected adapter keys in {folder}: {bad[:3]}"


def _eval_worker(rank, world):
    """Runs once per GPU: evaluates every `world`-th test image, rank 0 merges and saves."""
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=MASTER_PORT)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{rank}")

    if rank > 0:                                   # rank 0 loads first, then the others
        dist.barrier(device_ids=[rank])
    model = GemmaTraitGen(LM_ID, CLIP_ID, device=device)
    if rank == 0:
        dist.barrier(device_ids=[rank])
    tok = model.tokenizer

    _load_adapter(model.visual, os.path.join(CKPT_DIR, "enc_lora"))
    _load_adapter(model.lm, os.path.join(CKPT_DIR, "dec_lora"))
    proj = torch.load(os.path.join(CKPT_DIR, "projector.pt"), map_location="cpu")
    model.vis_norm.load_state_dict(proj["vis_norm"])
    model.projector.load_state_dict(proj["projector"])
    model.out_scale.data = proj["out_scale"].to(device)
    model.eval()

    class SpeciesDone(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            texts = tok.batch_decode(input_ids, skip_special_tokens=True)
            return torch.tensor([bool(PAT.search(t)) for t in texts], device=input_ids.device)

    @torch.no_grad()
    def generate(pix):
        B = pix.size(0)
        embed = model.lm.get_input_embeddings()
        emb = torch.cat([embed(torch.full((B, 1), tok.bos_token_id, device=device)), model.encode_images(pix)], 1)
        mask = torch.ones(emb.shape[:2], dtype=torch.long, device=device)
        kw = {"stopping_criteria": StoppingCriteriaList([SpeciesDone()])} if STOP_AFTER_SPECIES else {}
        with torch.autocast("cuda", enabled=False):
            out = model.lm.generate(inputs_embeds=emb, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS,
                                    do_sample=False, num_beams=1, top_k=None, top_p=None,
                                    eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, **kw)
        return tok.batch_decode(out, skip_special_tokens=True)

    seen, recs = set(), []                         # every unique test image
    for r in load_records(TEST_JSON):
        if r[0] not in seen:
            seen.add(r[0]); recs.append(r)
    ds = ImgCapDS(recs, model.preprocess_val, IMG_ROOT)
    loader = DataLoader(Subset(ds, list(range(rank, len(ds), world))), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2, collate_fn=_collate)

    local, no_gt = [], 0
    for pix, caps, sps in tqdm(loader, desc=f"GPU{rank}", disable=rank != 0):
        for g, cap, sp in zip(generate(pix.to(device)), caps, sps):
            gt = sp or extract(cap)
            if not gt:
                no_gt += 1
                continue
            pred = extract(g)
            local.append({"gt": gt, "pred": pred, "correct": bool(pred) and norm(pred) == norm(gt), "generated": g})

    gathered = [None] * world
    dist.all_gather_object(gathered, (local, no_gt))
    if rank == 0:
        results = [r for part, _ in gathered for r in part]
        correct = sum(r["correct"] for r in results)
        json.dump({"accuracy": correct / max(len(results), 1), "correct": correct, "total": len(results),
                   "no_species_found": sum(r["pred"] == "" for r in results),
                   "no_ground_truth": sum(n for _, n in gathered), "results": results},
                  open(OUT_JSON, "w"), indent=1)
    dist.barrier()
    dist.destroy_process_group()


# ---- launch one process per GPU
world = torch.cuda.device_count()
assert not torch.cuda.is_initialized(), "CUDA already used in this kernel - restart the session and run this cell first."
if world > 1:
    torch.multiprocessing.start_processes(_eval_worker, args=(world,), nprocs=world, start_method="fork")
else:
    print("Only 1 GPU visible - set the accelerator to GPU T4 x2.")
    raise SystemExit

# ---- report
R = json.load(open(OUT_JSON))
print("\n" + "=" * 50)
print(f"TEST ACCURACY: {R['accuracy']*100:.2f}%   ({R['correct']}/{R['total']})   [{world} GPUs]")
print(f"No species found in generated text: {R['no_species_found']}   |   samples without ground truth: {R['no_ground_truth']}")
print("=" * 50)
for r in R["results"][:5]:
    print(f"GT: {r['gt']!r} | PRED: {r['pred']!r} | {r['generated'][:150]!r}")
print("Saved predictions to", OUT_JSON)