"""
Dataset setup: download COCO 2017.

Usage:
    python -m src.setup coco           # Download COCO 2017 from Kaggle
"""

import os

import kagglehub

from src.config import (
    DATASET_DIR,
    TRAIN_IMAGES,
    VAL_IMAGES,
)


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


if __name__ == "__main__":
    setup_coco()