# MobileNet + CLIP Image Captioning

Image captioning using MobileNetV3-Small encoder + TransformerDecoder with CLIP contrastive loss, trained on COCO 2017. Uses the CLIP subword BPE tokenizer (openai/clip-vit-base-patch32, 49,408 tokens + a dedicated `[PAD]` token = 49,409).

## Architecture

```
Image (3, 224, 224)
    |
    v
MobileNetV3-Small (pretrained, late layers fine-tuned)
    |
    v
Spatial Attention -> Conv(576 -> 256) -> BatchNorm
    |
    +-> CLIP Projection Head (576 -> 512)  [training only]
    |
    v
49 spatial tokens x 256 dims
    |
    v
Transformer Decoder (2 layers, 4 heads)
    |
    v
Caption tokens
```

**Training losses:**
- Captioning loss (cross-entropy on token prediction) -- weight: 1.0
- CLIP contrastive loss (aligns image + text embeddings) -- weight: 0.5

**Dataset:**
- COCO 2017: 118k images with human-written ground-truth captions
- Optional: BLIP pseudo-captions for CLIP text cache enrichment

## Requirements

```
torch >= 2.0
torchvision >= 0.15
transformers >= 4.30
pandas, pillow, matplotlib, tqdm, nltk, pycocoevalcap, kagglehub
```

## Setup

```bash
cd mobilenet_clip_captioning
pip install -r requirements.txt
```

> **Note:** This project lives at `E:\Image Captioning\mobilenet_clip_captioning\mobilenet_clip_captioning`. Run all commands from that directory (the one containing `src/`), e.g.:
> ```bat
> cd "E:\Image Captioning\mobilenet_clip_captioning\mobilenet_clip_captioning"
> ```

## Quick Start

```bash
# 1. Download COCO 2017 dataset
python -m src.main coco

# 2. Train (default, ~5 min/epoch on GTX 1650)
python -m src.main pipeline

# Optionally generate BLIP pseudo-captions first (~1-2h):
python -m src.main pipeline --blip-captions
```

## Full Commands

```bash
# Download COCO 2017 from Kaggle
python -m src.main coco

# Train only (same as pipeline, no BLIP generation)
python -m src.main train

# Generate BLIP pseudo-captions (optional, for CLIP text cache enrichment)
python -m src.main soft-captions

# Evaluate on validation set
python -m src.main evaluate

# Predict on any image
python -m src.main predict path/to/image.jpg
python -m src.main predict path/to/image.jpg --beam

# Predict with a file dialog (pick the image in a folder window)
python -m src.main predict
```

## Project Structure

```
mobilenet_clip_captioning/
|-- requirements.txt
|-- README.md
|-- src/
|   |-- __init__.py        # Cache dirs, HF/disable warnings
|   |-- config.py          # Hyperparameters, paths, tokenizer config
|   |-- vocabulary.py      # CaptionTokenizer (CLIP tokenizer wrapper)
|   |-- setup.py           # COCO download CLI
|   |-- soft_caption.py    # BLIP caption generation (optional)
|   |-- dataset.py         # COCO dataset + make_collate_fn
|   |-- model.py           # MobileNetEncoder + TransformerDecoder + CLIPContrastiveLoss
|   |-- train.py           # Training with per-epoch image resampling
|   |-- evaluate.py        # BLEU/METEOR/CIDEr evaluation
|   |-- predict.py         # Inference (greedy + beam search)
|   |-- main.py            # CLI entry point
```

## Configuration

All settings in `src/config.py`:

### Model

| Parameter | Default | Description |
|---|---|---|
| `EMBED_SIZE` | 384 | Decoder embedding dimension |
| `HIDDEN_SIZE` | 768 | Transformer feed-forward dimension |
| `NUM_LAYERS` | 4 | Transformer decoder layers |
| `NUM_HEADS` | 6 | Attention heads |
| `BEAM_SIZE` | 5 | Beam search width |

### Training

| Parameter | Default | Description |
|---|---|---|
| `BATCH_SIZE` | 16 | Training batch size |
| `NUM_EPOCHS` | 200 | Max training epochs (early stopping may end sooner) |
| `MAX_BATCHES_PER_EPOCH` | 0 | Batch limit per epoch (0 = unlimited) |
| `TRAIN_IMAGES_PER_EPOCH` | 16000 | Unique images sampled per epoch |
| `ENCODER_LR` | 1e-5 | Encoder learning rate |
| `DECODER_LR` | 1e-4 | Decoder learning rate |
| `WARMUP_EPOCHS` | 2 | Linear warmup epochs |
| `CAPTION_LOSS_WEIGHT` | 1.0 | Captioning loss weight |
| `CLIP_LOSS_WEIGHT` | 0.5 | CLIP contrastive loss weight |
| `LABEL_SMOOTHING` | 0.1 | Label smoothing for caption CE loss |

### Early Stopping

| Parameter | Default | Description |
|---|---|---|
| `EARLY_STOP_PATIENCE` | 5 | Epochs to wait for val loss improvement |
| `EARLY_STOP_MIN_DELTA` | 1e-4 | Minimum change to count as improvement |

### Data & Tokenizer

| Parameter | Default | Description |
|---|---|---|
| `TOKENIZER_MODEL` | openai/clip-vit-base-patch32 | CLIP subword BPE tokenizer (49,408 tokens) |
| `BLIP_MODEL` | Salesforce/blip-image-captioning-base | BLIP model for soft captions (optional) |
| `PSEUDO_CAPTION_BATCH_SIZE` | 16 | Batch size for BLIP inference |

## Checkpoints

Saved to `checkpoints/`:

| File | Description |
|---|---|---|
| `model_best.pth` | Best model by BLEU-4 |
| `model_latest.pth` | Latest epoch |
| `resume_state.pth` | Full trainer state (optimizer, scheduler, scaler) |
| `training_log.txt` | Training log with losses and metrics |
| `pseudo_captions/coco_captions.json` | BLIP captions for COCO (optional) |
