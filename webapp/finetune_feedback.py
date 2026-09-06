import json
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mobilenet_clip_captioning", "mobilenet_clip_captioning"))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

WEBAPP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
FEEDBACK_FILE = os.path.join(WEBAPP_DIR, "feedback.json")
UPLOAD_FOLDER = os.path.join(WEBAPP_DIR, "uploads")

from src.config import (
    EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS,
    ENCODER_BACKBONE, V3_ENCODER_BACKBONE, USE_V3_ENCODER, USE_CLIP,
    MODEL_BEST_PATH, MODEL_LATEST_PATH, CHECKPOINT_DIR,
    GRAD_CLIP, LABEL_SMOOTHING,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class FeedbackDataset(Dataset):
    def __init__(self, entries, tokenizer, transform=None):
        self.entries = entries
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image_path = os.path.join(UPLOAD_FOLDER, entry["image_name"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = torch.tensor(self.tokenizer.encode(entry["corrected_caption"]), dtype=torch.long)
        text = entry["corrected_caption"]
        return image, caption, text


def collate_fn(batch, pad_token_id):
    images, captions, texts = zip(*batch)
    images = torch.stack(images)
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_token_id)
    return images, captions, texts


def load_verified_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        all_feedback = json.load(f)
    return [fb for fb in all_feedback if fb.get("verified") is True]


def finetune(epochs=20, lr_multiplier=0.5, batch_size=8):
    from src.config import device
    from src.model import CaptioningModel
    from src.vocabulary import CaptionTokenizer

    verified = load_verified_feedback()
    if not verified:
        print("No verified feedback found. Nothing to fine-tune.")
        return

    print(f"Found {len(verified)} verified feedback entries.")

    for fb in verified:
        img_path = os.path.join(UPLOAD_FOLDER, fb["image_name"])
        if not os.path.exists(img_path):
            print(f"  Warning: Image '{fb['image_name']}' not found, skipping.")
    verified = [fb for fb in verified if os.path.exists(os.path.join(UPLOAD_FOLDER, fb["image_name"]))]

    if not verified:
        print("No valid feedback entries with existing images.")
        return

    tokenizer = CaptionTokenizer()
    print(f"Tokenizer: {tokenizer.__class__.__name__}, vocab size: {len(tokenizer)}")

    checkpoint = MODEL_BEST_PATH if os.path.exists(MODEL_BEST_PATH) else MODEL_LATEST_PATH
    backbones = [ENCODER_BACKBONE]
    if USE_V3_ENCODER:
        backbones.append(V3_ENCODER_BACKBONE)
    model = CaptioningModel(
        embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
        vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.0,
        backbones=tuple(backbones), use_clip_proj=USE_CLIP,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    print(f"Loaded model from {checkpoint}")

    encoder_lr = 1e-5 * lr_multiplier
    decoder_lr = 1e-4 * lr_multiplier
    print(f"Fine-tune LR: encoder={encoder_lr}, decoder={decoder_lr}")

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = list(model.decoder.parameters())

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": encoder_lr},
        {"params": decoder_params, "lr": decoder_lr},
    ])
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id, label_smoothing=LABEL_SMOOTHING
    )
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    ds = FeedbackDataset(verified, tokenizer, transform=train_transform)
    loader = DataLoader(
        ds, batch_size=min(batch_size, len(ds)), shuffle=True,
        num_workers=0, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        pin_memory=True,
    )

    print(f"\nFine-tuning for {epochs} epochs on {len(ds)} samples...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, captions, texts in loader:
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad()
            with autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(images, captions)
                loss = criterion(outputs.reshape(-1, len(tokenizer)), captions[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg = total_loss / max(1, len(loader))
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg:.4f}")

    torch.save(model.state_dict(), MODEL_LATEST_PATH)
    torch.save(model.state_dict(), MODEL_BEST_PATH)
    print(f"\nFine-tune complete. Model saved to {MODEL_BEST_PATH}")
    print("Restart the web app to use the updated model.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune model with verified feedback")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr-mult", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    finetune(epochs=args.epochs, lr_multiplier=args.lr_mult, batch_size=args.batch_size)
