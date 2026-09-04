# MobileNet + CLIP Image Captioning

Semi-supervised image captioning using MobileNetV3 as the encoder backbone, trained with CLIP contrastive loss, and vocabulary built from BLIP-generated soft captions.

## Architecture

```
Image (3, 224, 224)
    │
    ▼
MobileNetV3-Small (pretrained, late layers fine-tuned)
    │
    ▼
Spatial Attention → Conv(576 → 256) → BatchNorm
    │
    ├──→ CLIP Projection Head (576 → 512)  [training only]
    │
    ▼
49 spatial tokens × 256 dims
    │
    ▼
Transformer Decoder (2 layers, 4 heads)
    │
    ▼
Caption tokens
```

**Training losses:**
- Captioning loss (cross-entropy on token prediction) — weight: 1.0
- CLIP contrastive loss (aligns image + text embeddings) — weight: 0.5

**Semi-supervised data:**
- COCO 2017: 118k images with human-written ground-truth captions
- ImageNet 1000 (mini): 34k images from `ifigotin/imagenetmini-1000` with BLIP-generated pseudo-captions

## Tokenizer

Subword tokenization using BPE (Byte-Pair Encoding) via HuggingFace `tokenizers`.

- **Vocab size**: 5000 tokens (configurable via `VOCAB_SIZE`)
- **No OOV**: Any word is decomposed into known subword pieces
  - Example: "cyberpunk" → `"cyber" + "##punk"`
  - Example: "selfie" → `"self" + "##ie"`
- **Special tokens**: `<PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3`
- **Expansion**: New tokens can be added after training, initialized from CLIP embeddings

The BPE tokenizer is trained on all captions (COCO human captions + BLIP pseudo-captions + ImageNet pseudo-captions) and saved to `checkpoints/tokenizer.json`.

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

## Quick Start

```bash
# 1. Download datasets
python -m src.main coco           # COCO 2017 from Kaggle
python -m src.main imagenet       # ImageNet 1000 (mini) from Kaggle

# 2. Run full pipeline (BLIP captions → vocab → training)
python -m src.main pipeline

# Regenerate captions, tokenizer, and checkpoints from scratch:
python -m src.main pipeline --fresh

# If captions already exist, skip regeneration:
python -m src.main pipeline --skip-captions
```

## Full Commands

```bash
# Download COCO 2017 from Kaggle
python -m src.main coco

# Download ImageNet 1000 (mini) from Kaggle (~4GB)
python -m src.main imagenet

# Run full pipeline (captions → vocab → training)
python -m src.main pipeline

# Run full pipeline from scratch (deletes existing artifacts first)
python -m src.main pipeline --fresh

# Run pipeline but skip BLIP caption generation
python -m src.main pipeline --skip-captions

# Generate BLIP soft captions (COCO + ImageNet)
# Optimized: fp16, torch.compile, pre-resize, incremental caching
python -m src.main soft-captions

# Build vocabulary from captions
python -m src.main vocab

# Train with early stopping (stops if val loss stalls for 5 epochs)
python -m src.main train

# Train indefinitely (until Ctrl+C, STOP file, or early stopping)
NUM_EPOCHS=0 python -m src.main train

# Evaluate on validation set
python -m src.main evaluate

# Predict on any image
python -m src.main predict path/to/image.jpg
python -m src.main predict path/to/image.jpg --beam
```

## Vocabulary Expansion

After initial training, you can expand the vocabulary with new tokens:

```bash
python -m src.main expand-vocab --new-data path/to/new_captions.json
```

The `new_captions.json` should be a JSON object mapping image filenames to caption strings:
```json
{
  "image1.jpg": "a cyberpunk cityscape at night",
  "image2.jpg": "a selfie of a cat"
}
```

What happens during expansion:
1. Existing BPE tokenizer is loaded
2. New BPE merges are learned from the new captions
3. New tokens are added to the tokenizer
4. Model embedding layer is resized (old weights preserved)
5. New embedding rows are initialized from CLIP text encoder
6. Updated tokenizer and model are saved

**Note**: After expansion, retrain for a few epochs to fine-tune the new embeddings.

## Project Structure

```
mobilenet_clip_captioning/
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py          # Hyperparameters, paths, special tokens
│   ├── vocabulary.py      # BPE tokenizer (train, encode, decode, expand)
│   ├── setup.py           # Unified dataset setup CLI
│   ├── soft_caption.py    # BLIP caption generation (optimized)
│   ├── dataset.py         # COCO, PseudoCaption, SemiSupervised datasets
│   ├── model.py           # MobileNetEncoder + CLIP head + TransformerDecoder
│   ├── train.py           # Training with subset, early stopping, manual stop
│   ├── evaluate.py        # BLEU/METEOR/CIDEr evaluation
│   ├── predict.py         # Inference (greedy + beam search)
│   └── main.py            # CLI entry point (pipeline, individual steps)
```

## Configuration

All settings in `src/config.py`:

### Model

| Parameter | Default | Description |
|---|---|---|
| `EMBED_SIZE` | 256 | Decoder embedding dimension |
| `HIDDEN_SIZE` | 512 | Transformer feed-forward dimension |
| `NUM_LAYERS` | 2 | Transformer decoder layers |
| `NUM_HEADS` | 4 | Attention heads |
| `BEAM_SIZE` | 5 | Beam search width |

### Training

| Parameter | Default | Description |
|---|---|---|
| `BATCH_SIZE` | 32 | Training batch size |
| `GRAD_ACCUM_STEPS` | 4 | Gradient accumulation (effective batch = 32 × 4 = 128) |
| `NUM_EPOCHS` | 100 | Max training epochs (≤0 = indefinite) |
| `NUM_WORKERS` | 4 | DataLoader workers |
| `MAX_TRAIN_IMAGES` | 20000 | Random image subset per epoch (0 = use all 118K) |
| `ENCODER_LR` | 1e-5 | Encoder learning rate |
| `DECODER_LR` | 1e-4 | Decoder learning rate |
| `WARMUP_EPOCHS` | 2 | Linear warmup epochs |
| `CAPTION_LOSS_WEIGHT` | 1.0 | Captioning loss weight |
| `CLIP_LOSS_WEIGHT` | 0.5 | CLIP contrastive loss weight |

### Validation

| Parameter | Default | Description |
|---|---|---|
| `VAL_LOSS_SUBSET` | 500 | Images for validation loss subset |
| `VAL_LOSS_FULL_EVERY` | 5 | Full validation every N epochs |
| `VAL_BLEU_SUBSET` | 200 | Images for BLEU evaluation subset |
| `VAL_BLEU_EVERY_N_EPOCHS` | 2 | BLEU evaluation every N epochs |

### Early Stopping

| Parameter | Default | Description |
|---|---|---|
| `EARLY_STOP_PATIENCE` | 5 | Epochs to wait for val loss improvement |
| `EARLY_STOP_MIN_DELTA` | 1e-4 | Minimum change to count as improvement |

### Data

| Parameter | Default | Description |
|---|---|---|
| `MIN_WORD_FREQ` | 2 | Minimum token frequency for BPE training |
| `VOCAB_SIZE` | 5000 | BPE vocabulary size |
| `BLIP_MODEL` | Salesforce/blip-image-captioning-base | BLIP model for soft captions |
| `PSEUDO_CAPTION_BATCH_SIZE` | 32 | Batch size for BLIP inference |

## Training Details

- **Per-epoch subset**: With `MAX_TRAIN_IMAGES=20000`, each epoch trains on a random 20K-image subset (~100K caption pairs), reducing epoch time from ~44 min to ~7-8 min. Different subset each epoch — over 100 epochs the model sees every image ~17 times.
- **Gradient accumulation**: `GRAD_ACCUM_STEPS=4` gives effective batch size of 128 without extra VRAM.
- **Manual stop**: Press `Ctrl+C` to finish the current batch and save a checkpoint (press twice to force). Or create `checkpoints\STOP` (`echo x > checkpoints\STOP`) to stop cleanly at end of epoch.
- **CLIP text cache**: Embeddings are cached to disk (keyed by caption hash). Only rebuilt when captions change — shaves ~10 min off restart.

## BLIP Caption Generation

The soft caption step is optimized for speed:

- **Mixed precision (fp16)** — ~2x faster on GPU
- **torch.compile** — fused CUDA kernels for additional speedup
- **Pre-resize to 384x384** — uniform tensors, less padding waste
- **Incremental caching** — skips already-captioned images on re-runs
- **Periodic saves** — progress saved every 50 batches (crash-safe)

## Checkpoints

Saved to `checkpoints/`:

| File | Description |
|---|---|
| `model_best.pth` | Best model by BLEU-4 |
| `model_latest.pth` | Latest epoch |
| `vocab.json` | Vocabulary |
| `tokenizer.json` | BPE tokenizer model |
| `pseudo_captions/coco_captions.json` | BLIP captions for COCO |
| `pseudo_captions/imagenet_captions.json` | Pseudo-captions for ImageNet |
| `clip_text_cache_*.pt` | CLIP text embeddings (auto-cached, auto-invalidated) |
| `STOP` | Create this file to stop training at end of epoch |
| `training_log.txt` | Training log with losses and metrics |
