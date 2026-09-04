"""
Unified setup: download datasets, generate soft captions, build vocabulary.

Usage:
    python -m src.setup coco           # Download COCO 2017 from Kaggle
    python -m src.setup imagenet       # Download ImageNet 1000 (mini) from Kaggle
    python -m src.setup soft-captions  # Generate BLIP pseudo-captions
    python -m src.setup vocab          # Build vocabulary from soft captions
    python -m src.setup all            # Run everything in order
"""

import argparse
import json
import os

import kagglehub

from src.config import (
    DATASET_DIR,
    IMAGENET_DIR,
    IMAGENET_TRAIN,
    TRAIN_IMAGES,
    VAL_IMAGES,
    PSEUDO_CAPTIONS_DIR,
    TOKENIZER_PATH,
    MIN_WORD_FREQ,
    VOCAB_SIZE,
)


# ── COCO ───────────────────────────────────────────────────────

def setup_coco():
    if os.path.exists(TRAIN_IMAGES) and os.path.exists(VAL_IMAGES):
        print(f"COCO already present at {DATASET_DIR}")
        return

    dataset_id = "awsaf49/coco-2017-dataset"
    target = "D:/coco_dataset"
    os.makedirs(target, exist_ok=True)
    print(f"Downloading COCO 2017 from Kaggle ({dataset_id})...")
    kagglehub.dataset_download(dataset_id, output_dir=target)
    print(f"Downloaded to: {target}")


# ── ImageNet ───────────────────────────────────────────────────

def setup_imagenet():
    if os.path.exists(IMAGENET_TRAIN) and len(os.listdir(IMAGENET_TRAIN)) > 0:
        print(f"ImageNet already present at {IMAGENET_TRAIN}")
        return

    os.makedirs(IMAGENET_DIR, exist_ok=True)
    dataset_id = "ifigotin/imagenetmini-1000"
    print(f"Downloading ImageNet 1000 (mini) from Kaggle ({dataset_id})...")
    kagglehub.dataset_download(dataset_id, output_dir=IMAGENET_DIR)

    extracted = os.path.join(IMAGENET_DIR, "imagenet-mini", "train")
    if os.path.exists(extracted):
        target = os.path.join(IMAGENET_DIR, "train")
        if not os.path.exists(target):
            os.rename(extracted, target)
        print(f"ImageNet train ready: {target}")

    leftover = os.path.join(IMAGENET_DIR, "imagenet-mini")
    if os.path.exists(leftover):
        import shutil
        shutil.rmtree(leftover, ignore_errors=True)


# ── Soft captions ──────────────────────────────────────────────

def setup_soft_captions():
    from src.soft_caption import generate_coco_captions, generate_imagenet_captions

    os.makedirs(PSEUDO_CAPTIONS_DIR, exist_ok=True)

    coco_path = os.path.join(PSEUDO_CAPTIONS_DIR, "coco_captions.json")
    if os.path.exists(coco_path):
        print(f"COCO soft captions already exist: {coco_path}")
    else:
        print("Generating soft captions for COCO training images...")
        generate_coco_captions(coco_path)

    imagenet_path = os.path.join(PSEUDO_CAPTIONS_DIR, "imagenet_captions.json")
    if os.path.exists(imagenet_path):
        print(f"ImageNet pseudo-captions already exist: {imagenet_path}")
    else:
        if os.path.exists(IMAGENET_TRAIN):
            print("Generating pseudo-captions for ImageNet training images...")
            generate_imagenet_captions(imagenet_path)
        else:
            print(f"ImageNet not found at {IMAGENET_TRAIN}, skipping.")


# ── Vocabulary ─────────────────────────────────────────────────

def setup_vocab():
    from src.vocabulary import CaptionTokenizer

    coco_path = os.path.join(PSEUDO_CAPTIONS_DIR, "coco_captions.json")
    imagenet_path = os.path.join(PSEUDO_CAPTIONS_DIR, "imagenet_captions.json")

    all_captions = []

    if os.path.exists(coco_path):
        with open(coco_path) as f:
            coco_caps = json.load(f)
        all_captions.extend(coco_caps.values())
        print(f"Loaded {len(coco_caps)} COCO soft captions")

    if os.path.exists(imagenet_path):
        with open(imagenet_path) as f:
            in_caps = json.load(f)
        all_captions.extend(in_caps.values())
        print(f"Loaded {len(in_caps)} ImageNet pseudo-captions")

    if not all_captions:
        print("No captions found. Run soft-captions first.")
        return

    tokenizer = CaptionTokenizer.train(
        all_captions, vocab_size=VOCAB_SIZE,
        min_frequency=MIN_WORD_FREQ, save_path=TOKENIZER_PATH,
    )
    print(f"Tokenizer: {tokenizer} saved to {TOKENIZER_PATH}")


# ── Main ───────────────────────────────────────────────────────

STEPS = {
    "coco": setup_coco,
    "imagenet": setup_imagenet,
    "soft-captions": setup_soft_captions,
    "vocab": setup_vocab,
}


def main():
    parser = argparse.ArgumentParser(description="Dataset setup")
    parser.add_argument("step", choices=list(STEPS) + ["all"], help="Which setup step to run")
    args = parser.parse_args()

    if args.step == "all":
        for name, fn in STEPS.items():
            print(f"\n{'=' * 50}")
            print(f"  {name}")
            print(f"{'=' * 50}\n")
            fn()
    else:
        STEPS[args.step]()


if __name__ == "__main__":
    main()
