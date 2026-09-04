"""
Dataset setup: download COCO, generate soft captions.

Usage:
    python -m src.setup coco           # Download COCO 2017 from Kaggle
    python -m src.setup soft-captions  # Generate BLIP pseudo-captions
    python -m src.setup all            # Run everything in order
"""

import argparse
import os

import kagglehub

from src.config import (
    DATASET_DIR,
    TRAIN_IMAGES,
    VAL_IMAGES,
    PSEUDO_CAPTIONS_DIR,
)


# ── COCO ───────────────────────────────────────────────────────

def setup_coco():
    if os.path.exists(TRAIN_IMAGES) and os.path.exists(VAL_IMAGES):
        print(f"COCO already present at {DATASET_DIR}")
        return

    dataset_id = "awsaf49/coco-2017-dataset"
    target = "E:/Image Captioning/coco_dataset"
    os.makedirs(target, exist_ok=True)
    print(f"Downloading COCO 2017 from Kaggle ({dataset_id})...")
    kagglehub.dataset_download(dataset_id, output_dir=target)
    print(f"Downloaded to: {target}")


# ── Soft captions ──────────────────────────────────────────────

def setup_soft_captions():
    from src.soft_caption import generate_coco_captions

    os.makedirs(PSEUDO_CAPTIONS_DIR, exist_ok=True)

    coco_path = os.path.join(PSEUDO_CAPTIONS_DIR, "coco_captions.json")
    if os.path.exists(coco_path):
        print(f"COCO soft captions already exist: {coco_path}")
    else:
        print("Generating soft captions for COCO training images...")
        generate_coco_captions(coco_path)


# ── Main ───────────────────────────────────────────────────────

STEPS = {
    "coco": setup_coco,
    "soft-captions": setup_soft_captions,
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
