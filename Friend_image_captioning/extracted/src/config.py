import os
import torch

os.environ["HF_HOME"] = "D:/AI_Cache/huggingface"
os.environ["HF_HUB_CACHE"] = "D:/AI_Cache/huggingface/hub"
os.environ["TORCH_HOME"] = "D:/AI_Cache/torch"
os.environ["KAGGLE_CACHE_DIR"] = "D:/AI_Cache/kagglehub"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset paths ──────────────────────────────────────────────
DATASET_DIR = "D:/coco_dataset/coco2017"
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train2017")
VAL_IMAGES = os.path.join(DATASET_DIR, "val2017")
TRAIN_ANN = os.path.join(DATASET_DIR, "annotations", "captions_train2017.json")
VAL_ANN = os.path.join(DATASET_DIR, "annotations", "captions_val2017.json")

IMAGENET_DIR = "D:/imagenet"
IMAGENET_TRAIN = os.path.join(IMAGENET_DIR, "train")
IMAGENET_VAL = os.path.join(IMAGENET_DIR, "val")

# ── Model architecture ─────────────────────────────────────────
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1
BEAM_SIZE = 5

# MobileNetV3-Small outputs 576-channel feature maps
MOBILENET_OUT_CHANNELS = 576

# CLIP projection dim (openai/clip-vit-base-patch32 = 512)
CLIP_EMBED_DIM = 512

# ── Training ───────────────────────────────────────────────────
NUM_EPOCHS = 0 
BATCH_SIZE = 32
ENCODER_LR = 1e-5
DECODER_LR = 1e-4
WARMUP_EPOCHS = 2
GRAD_CLIP = 5.0
NUM_WORKERS = min(4, os.cpu_count() or 2)
GRAD_ACCUM_STEPS = 4
MAX_TRAIN_IMAGES = 25000  # random subset per epoch (0 = use all)

# ── Validation ─────────────────────────────────────────────────
VAL_LOSS_SUBSET = 500
VAL_LOSS_FULL_EVERY = 5

# ── Early stopping ────────────────────────────────────────────
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 1e-4

# ── Semi-supervised loss weights ───────────────────────────────
CAPTION_LOSS_WEIGHT = 1.0
CLIP_LOSS_WEIGHT = 0.5

# ── Vocabulary ─────────────────────────────────────────────────
MIN_WORD_FREQ = 2
VOCAB_SIZE = 5000

# ── Special tokens ─────────────────────────────────────────────
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

# ── BLIP / soft captioning ─────────────────────────────────────
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
PSEUDO_CAPTIONS_DIR = "checkpoints/pseudo_captions"
PSEUDO_CAPTION_BATCH_SIZE = 32

# ── CLIP text encoder (for contrastive loss) ───────────────────
CLIP_TEXT_MODEL = "openai/clip-vit-base-patch32"
CLIP_TEMPERATURE = 0.07

# ── Evaluation ─────────────────────────────────────────────────
VAL_BLEU_SUBSET = 200
VAL_BLEU_EVERY_N_EPOCHS = 2
COMPARE_SAMPLE_SIZE = 200

# ── Checkpoints ────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
MODEL_BEST_PATH = os.path.join(CHECKPOINT_DIR, "model_best.pth")
MODEL_LATEST_PATH = os.path.join(CHECKPOINT_DIR, "model_latest.pth")
VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "vocab.json")
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer.json")
TRAINING_LOG_PATH = os.path.join(CHECKPOINT_DIR, "training_log.txt")
STOP_FILE = os.path.join(CHECKPOINT_DIR, "STOP")
