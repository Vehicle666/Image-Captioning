import argparse
import os
import sys

import torch
from PIL import Image

from src.config import (
    EMBED_SIZE, HIDDEN_SIZE,
    MODEL_BEST_PATH, MODEL_LATEST_PATH,
    NUM_HEADS, NUM_LAYERS, device,
)
from src.dataset import val_transform
from src.generation import generate_caption, generate_caption_beam
from src.model import CaptioningModel
from src.vocabulary import CaptionTokenizer


def load_tokenizer():
    return CaptionTokenizer()


def load_model(tokenizer, checkpoint=None):
    if checkpoint is None:
        checkpoint = MODEL_BEST_PATH if os.path.exists(MODEL_BEST_PATH) else MODEL_LATEST_PATH
    model = CaptioningModel(
        embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
        vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


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
