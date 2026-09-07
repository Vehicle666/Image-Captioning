import argparse
import os
import sys

import torch
from PIL import Image

from src.config import (
    MODEL_BEST_PATH, MODEL_LATEST_PATH,
    device,
)
from src.dataset import val_transform
from src.generation import generate_caption, generate_caption_beam
from src.model import build_model_from_checkpoint
from src.vocabulary import CaptionTokenizer


def load_tokenizer():
    return CaptionTokenizer()


def load_model(tokenizer, checkpoint=None):
    if checkpoint is None:
        checkpoint = MODEL_BEST_PATH if os.path.exists(MODEL_BEST_PATH) else MODEL_LATEST_PATH
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"No checkpoint found in checkpoints/ (tried {MODEL_BEST_PATH}, {MODEL_LATEST_PATH}).\n"
            "Train a model first (study/study-report), or set STUDY_TAG to the config you want."
        )
    return build_model_from_checkpoint(checkpoint, tokenizer, device=device)


def _decode(image_path, model, tokenizer, beam=False):
    image = Image.open(image_path).convert("RGB")
    image = val_transform(image).unsqueeze(0).to(device)
    if beam:
        ids = generate_caption_beam(model, image, tokenizer)
    else:
        ids = generate_caption(model, image, tokenizer)
    return tokenizer.decode(ids)


def pick_image():
    """Open a native file dialog so the user can choose an image."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    root.update()
    path = filedialog.askopenfilename(
        parent=root,
        title="Select an image to caption",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.gif"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path or None


def main():
    parser = argparse.ArgumentParser(description="Generate a caption for an image")
    parser.add_argument("image_path", nargs="?", help="Path to image file (optional; a file dialog opens if omitted)")
    parser.add_argument("--beam", action="store_true", help="Use beam search")
    args = parser.parse_args()

    image_path = args.image_path
    if not image_path:
        print("Opening file dialog... (choose an image)")
        image_path = pick_image()
        if not image_path:
            image_path = input("Enter image path: ").strip().strip('"')
        if not image_path:
            print("No image selected. Usage: python -m src.main predict <image_path> [--beam]")
            sys.exit(1)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    print("Loading model...")
    model = load_model(tokenizer)

    print(f"Caption: {_decode(image_path, model, tokenizer, beam=args.beam)}")


if __name__ == "__main__":
    main()
