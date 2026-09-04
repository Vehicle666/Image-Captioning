"""
BLIP-based soft caption generation for COCO training images.
Enriches vocabulary with diverse captions beyond human annotations.
Results are cached to JSON so BLIP only runs once per image.
"""

import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.config import (
    BLIP_MODEL,
    PSEUDO_CAPTION_BATCH_SIZE,
    TRAIN_IMAGES,
    TRAIN_ANN,
    device,
)

PRE_RESIZE = (384, 384)


def _load_existing_captions(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_captions(captions, path):
    with open(path, "w") as f:
        json.dump(captions, f, indent=2)


def load_blip():
    print(f"Loading BLIP model ({BLIP_MODEL})...")
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)

    if device.type == "cuda":
        model = model.half()

    try:
        model = torch.compile(model, mode="max-autotune")
        print("  torch.compile enabled (max-autotune)")
    except Exception as e:
        print(f"  torch.compile failed ({e}), using eager mode")

    model.eval()
    return model, processor


def _open_and_resize(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(PRE_RESIZE, Image.BICUBIC)
    return img


@torch.no_grad()
def generate_captions_for_images(
    model, processor, image_paths, batch_size=PSEUDO_CAPTION_BATCH_SIZE,
    existing=None, save_path=None, root_dir=None
):
    if existing is None:
        existing = {}

    remaining = [
        p for p in image_paths
        if (os.path.relpath(p, root_dir).replace("\\", "/") if root_dir else os.path.basename(p)) not in existing
    ]
    if not remaining:
        print("  All images already captioned, skipping.")
        return existing

    print(f"  {len(remaining)}/{len(image_paths)} images left to caption")
    captions = dict(existing)

    for i in tqdm(range(0, len(remaining), batch_size), desc="Captioning"):
        batch_paths = remaining[i : i + batch_size]
        images = []
        valid_paths = []

        for p in batch_paths:
            try:
                images.append(_open_and_resize(p))
                valid_paths.append(p)
            except Exception:
                continue

        if not images:
            continue

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        inputs = {k: v.to(dtype) if v.is_floating_point() else v for k, v in inputs.items()}

        output_ids = model.generate(**inputs, max_length=50, num_beams=3)
        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)

        for path, caption in zip(valid_paths, decoded):
            if root_dir:
                key = os.path.relpath(path, root_dir).replace("\\", "/")
            else:
                key = os.path.basename(path)
            captions[key] = caption.strip()

        if save_path and (i // batch_size + 1) % 50 == 0:
            _save_captions(captions, save_path)

    return captions


def generate_coco_captions(save_path):
    with open(TRAIN_ANN) as f:
        ann_data = json.load(f)

    img_id_to_name = {img["id"]: img["file_name"] for img in ann_data["images"]}
    image_names = list(img_id_to_name.values())
    image_paths = [os.path.join(TRAIN_IMAGES, name) for name in image_names]

    existing = _load_existing_captions(save_path)
    model, processor = load_blip()
    captions = generate_captions_for_images(
        model, processor, image_paths, existing=existing, save_path=save_path, root_dir=TRAIN_IMAGES
    )

    _save_captions(captions, save_path)
    print(f"Saved {len(captions)} COCO soft captions -> {save_path}")


if __name__ == "__main__":
    from src.config import CHECKPOINT_DIR, PSEUDO_CAPTIONS_DIR
    os.makedirs(PSEUDO_CAPTIONS_DIR, exist_ok=True)
    generate_coco_captions(os.path.join(PSEUDO_CAPTIONS_DIR, "coco_captions.json"))
