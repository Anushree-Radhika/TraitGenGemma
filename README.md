# BioCLIP + Gemma 3 species captioning (CUB-200-2011)

BioCLIP ViT encoder (LoRA) -> MLP projector -> Gemma-3-1B decoder (LoRA). Generates a
caption containing `species of "<name>"`; accuracy is exact match on that name.

## Layout

| File | Responsibility |
|---|---|
| `config.py` | `Config` dataclass (`CFG`) and `setup_env()` (NCCL, tokenizers, HF token) |
| `data.py` | record loading, `ImgCapDS`, collate functions, `build_dataloaders` |
| `model.py` | `GemmaTraitGen` (encoder, decoder, projector, loss, generation) |
| `engine.py` | `train_one_epoch`, `validate`, `evaluate_accuracy` |
| `train.py` | `run(cfg)`: builds everything and runs the epoch loop |
| `evaluate.py` | standalone test-set evaluation of the best checkpoint |
| `checkpoint.py` | `save_best` / `load_best` |
| `dist_utils.py` | process-group setup, rank-ordered loading, `launch` |
| `utils.py` | species parsing, `SpeciesDone`, NaN debugger, curve plots |
| `main.py` | training launcher |

## Usage

```bash
pip install -r requirements.txt
bash download_data.sh          # CUB-200-2011 images
python main.py                 # train (or: torchrun --nproc-per-node=2 main.py)
python evaluate.py             # evaluate <checkpoint_dir>/best on test.json
```

Gemma is gated: provide `HF_TOKEN` as a Kaggle Secret or environment variable.
Never hard-code the token in code or notebooks.
