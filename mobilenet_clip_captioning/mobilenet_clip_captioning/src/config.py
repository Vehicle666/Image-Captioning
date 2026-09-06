import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset paths ──────────────────────────────────────────────
DATASET_DIR = "E:/Image Captioning/coco_dataset/coco2017"
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train2017")
VAL_IMAGES = os.path.join(DATASET_DIR, "val2017")
TRAIN_ANN = os.path.join(DATASET_DIR, "annotations", "captions_train2017.json")
VAL_ANN = os.path.join(DATASET_DIR, "annotations", "captions_val2017.json")

# ── Ablation study (application study) ─────────────────────────
# The same supervised pipeline is re-run with each config to answer:
#   "Does adding another encoder (V3) / CLIP actually help?"
#
#   Config A  baseline_mobilenet : ENCODER_BACKBONE only, no CLIP
#   Config B  +v3                : adds V3_ENCODER_BACKBONE branch (feature fusion)
#   Config C  +v3_clip           : adds CLIP contrastive loss + projection head
#
# All values can be overridden via environment variables so
# `python -m src.main study` can run the 3 configs back-to-back.
def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")

ENCODER_BACKBONE = os.environ.get("BACKBONE", "mobilenet_v3_small")
V3_ENCODER_BACKBONE = os.environ.get("V3_BACKBONE", "mobilenet_v3_large")
USE_V3_ENCODER = _env_bool("USE_V3", True)      # add second encoder branch
USE_CLIP = _env_bool("USE_CLIP", True)          # add CLIP contrastive loss
STUDY_TAG = os.environ.get("STUDY_TAG", "").strip()

# Valid backbones (see src/model.py BACKBONE_REGISTRY):
#   mobilenet_v3_small / mobilenet_v3_large / efficientnet_b0

# ── Model architecture ─────────────────────────────────────────
EMBED_SIZE = 384
HIDDEN_SIZE = 768
NUM_LAYERS = 4
NUM_HEADS = 6
DROPOUT = 0.1
BEAM_SIZE = 5

# CLIP projection dim (openai/clip-vit-base-patch32 = 512)
CLIP_EMBED_DIM = 512

# ── Training ───────────────────────────────────────────────────
NUM_EPOCHS = 600
BATCH_SIZE = 16
MAX_BATCHES_PER_EPOCH = 0     # 0 = unlimited (use TRAIN_IMAGES_PER_EPOCH instead)
TRAIN_IMAGES_PER_EPOCH = 5500   # unique images per epoch (tuned to ~10 min on GTX 1650)
ENCODER_LR = 1e-5
DECODER_LR = 1e-4
WARMUP_EPOCHS = 6
GRAD_CLIP = 5.0
NUM_WORKERS = min(2, os.cpu_count() or 2)  # Windows: giữ <=2 để tránh deadlock

# ── Early stopping ────────────────────────────────────────────
EARLY_STOP_PATIENCE = 24
EARLY_STOP_MIN_DELTA = 1e-4

# ── Loss weights ───────────────────────────────────────────────
CAPTION_LOSS_WEIGHT = 1.0
CLIP_LOSS_WEIGHT = 0.5
LABEL_SMOOTHING = 0.1

# ── Tokenizer ──────────────────────────────────────────────────
TOKENIZER_MODEL = "openai/clip-vit-base-patch32"

# ── CLIP text encoder (for contrastive loss) ───────────────────
CLIP_TEXT_MODEL = "openai/clip-vit-base-patch32"
CLIP_TEMPERATURE = 0.07

# ── Evaluation ─────────────────────────────────────────────────
VAL_EVERY_N_EPOCHS = 15    # run validation loss every N epochs
VAL_BLEU_SUBSET = 200
VAL_BLEU_EVERY_N_EPOCHS = 6
SCHEDULER_PATIENCE = 6     # ReduceLROnPlateau patience (val-based, scaled for ~10 min epochs)
COMPARE_SAMPLE_SIZE = 200

# ── Checkpoints ────────────────────────────────────────────────
# Each ablation config writes into its own sub-folder, so results can be
# compared without overwriting each other.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, "checkpoints")
CHECKPOINT_DIR = os.path.join(_CHECKPOINT_ROOT, STUDY_TAG) if STUDY_TAG else _CHECKPOINT_ROOT
MODEL_BEST_PATH = os.path.join(CHECKPOINT_DIR, "model_best.pth")
MODEL_LATEST_PATH = os.path.join(CHECKPOINT_DIR, "model_latest.pth")
TRAINING_LOG_PATH = os.path.join(CHECKPOINT_DIR, "training_log.txt")
RESUME_PATH = os.path.join(CHECKPOINT_DIR, "resume_state.pth")
CLIP_CACHE_PATH = os.path.join(CHECKPOINT_DIR, "clip_text_cache.pth")

# ── Ablation study results ─────────────────────────────────────
STUDY_RESULTS_PATH = os.path.join(_CHECKPOINT_ROOT, "study_results.json")