# Ablation Study: MobileNet → +V3 → +CLIP (Image Captioning)

Supervised image captioning on COCO 2017 with a **pluggable multi-backbone CNN encoder**. The repo is organized as an ablation study (application study): the exact same supervised pipeline is trained once per encoder config, then compared — answering *does adding a second V3 backbone / CLIP actually help?*

## The 3 study configs

| Config | Encoder(s) | CLIP loss | Research question |
|--------|------------|-----------|-------------------|
| **A** `baseline_mobilenet` | MobileNetV3-Small | off | Baseline |
| **B** `mobilenet_v3` | MobileNetV3-Small **+ V3** (MobileNetV3-Large / EfficientNet) | off | Does a 2nd encoder help? |
| **C** `mobilenet_v3_clip` | MobileNet + V3 | on | Does CLIP contrastive loss help further? |

Every config writes its own checkpoints under `checkpoints/<STUDY_TAG>/` so nothing is overwritten.

## Architecture

```
Image (3, 224, 224)
    |
    +-> Backbone A: MobileNetV3-Small (pretrained, late layers fine-tuned) ----+
    |                                                                          |
    +-> Backbone B: MobileNetV3-Large / EfficientNet (optional, +V3) ---------+-> concat
    |                                                                          |   -> Conv1x1 fusion
    +-> CLIP Projection Head [only when USE_CLIP=1, training only]             |   -> BatchNorm
    |                                                                          v
    |                                                            49 spatial tokens x EMBED_SIZE
    |                                                                          |
    +-----------------------------------------------------------------------> Transformer Decoder
                                                    Caption tokens -> Beam Search / Greedy
```

Each backbone branch: `features -> SpatialAttention -> Conv1x1(embed_size) -> BatchNorm`.

**Losses:**
- Captioning cross-entropy + label smoothing — weight 1.0 (always on, supervised)
- CLIP contrastive loss — weight 0.5 (only when `USE_CLIP=1`)

## Requirements

```
torch >= 2.0
torchvision >= 0.15
transformers >= 4.30
pandas, pillow, matplotlib, tqdm, nltk, pycocoevalcap, kagglehub
```

## Setup

```bash
cd "E:\Image Captioning\mobilenet_clip_captioning\mobilenet_clip_captioning"
pip install -r requirements.txt
```

## Quick start

```bash
# 1. Download COCO 2017
python -m src.main coco

# 2. Train ALL study configs and compare (each in its own checkpoint folder)
python -m src.main study

# 3. Show comparison table of finished runs
python -m src.main study-report
```

Results are written to `checkpoints/study_results.json`.

## Train a single config

```bash
# Config A — baseline: MobileNet only, no CLIP
set STUDY_TAG=baseline_mobilenet && set USE_V3=0 && set USE_CLIP=0 && python -m src.main train

# Config B — + V3 encoder branch (feature fusion)
set STUDY_TAG=mobilenet_v3 && set USE_V3=1 && set USE_CLIP=0 && python -m src.main train

# Config C — + V3 + CLIP contrastive loss
set STUDY_TAG=mobilenet_v3_clip && set USE_V3=1 && set USE_CLIP=1 && python -m src.main train
```

Or run from exercise of a single run:

```bash
python -m src.main evaluate            # BLEU/METEOR/CIDEr on COCO val set
python -m src.main predict <image> --beam
python -m src.main train --epochs 0    # train until early stop
```

## Ablation toggles (env vars, defaults in `src/config.py`)

| Var | Default | Description |
|-----|---------|-------------|
| `BACKBONE` | `mobilenet_v3_small` | Primary backbone |
| `V3_BACKBONE` | `mobilenet_v3_large` | Second (V3) backbone; `efficientnet_b0` also supported |
| `USE_V3` | `1` | Enable 2nd encoder branch (feature fusion) |
| `USE_CLIP` | `1` | Enable CLIP contrastive loss + projection head |
| `STUDY_TAG` | (empty) | Checkpoint sub-folder name; empty = default config |

## Project structure

```
mobilenet_clip_captioning/
|-- requirements.txt
|-- src/
|   |-- config.py          # Hyperparameters + ablation toggles (env-driven)
|   |-- model.py           # ImageEncoder (multi-backbone fusion) + TransformerDecoder + CLIP
|   |-- train.py           # Supervised training (AMP, early stopping, BLEU-4 checkpointing)
|   |-- study.py           # Ablation runner + results table
|   |-- dataset.py         # COCO dataset
|   |-- evaluate.py        # BLEU/METEOR/CIDEr
|   |-- predict.py         # Greedy + beam search inference
|   |-- generation.py      # Decode logic
|   |-- vocabulary.py      # CLIP subword tokenizer wrapper
|   |-- main.py            # CLI entry point
|   `-- setup.py           # COCO download
`-- checkpoints/
    |-- <STUDY_TAG>/       # per-config model_best.pth, training_log.txt, resume_state.pth
    `-- study_results.json # ablation comparison table
```

## Study results

| Config | BLEU-1 | BLEU-4 | METEOR | CIDEr |
|--------|--------|--------|--------|-------|
| A: MobileNet (baseline) | -- | -- | -- | -- |
| B: + V3 | -- | -- | -- | -- |
| C: + V3 + CLIP | -- | -- | -- | -- |