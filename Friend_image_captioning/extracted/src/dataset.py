import json
import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import (
    BATCH_SIZE,
    IMAGENET_TRAIN,
    NUM_WORKERS,
    PAD_TOKEN,
    TRAIN_ANN,
    TRAIN_IMAGES,
    VAL_ANN,
    VAL_IMAGES,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── COCO helpers ───────────────────────────────────────────────

def load_coco_captions(ann_file, images_dir):
    with open(ann_file) as f:
        ann_data = json.load(f)
    img_id_to_name = {img["id"]: img["file_name"] for img in ann_data["images"]}
    rows = []
    for ann in ann_data["annotations"]:
        img_name = img_id_to_name[ann["image_id"]]
        rows.append({"image": img_name, "caption": ann["caption"]})
    return pd.DataFrame(rows), images_dir


def get_coco_captions(split="train"):
    if split == "train":
        return load_coco_captions(TRAIN_ANN, TRAIN_IMAGES)
    return load_coco_captions(VAL_ANN, VAL_IMAGES)


def get_coco_dataloader(df, root_dir, tokenizer, split="train", batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS):
    tfm = train_transform if split == "train" else val_transform
    dataset = COCODataset(df, root_dir, tokenizer, transform=tfm, return_name=(split == "val"))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)


# ── Pseudo-caption loader ─────────────────────────────────────

def load_pseudo_captions(json_path):
    with open(json_path) as f:
        return json.load(f)


# ── Datasets ───────────────────────────────────────────────────

class COCODataset(Dataset):
    def __init__(self, dataframe, root_dir, tokenizer, transform=None, return_name=False):
        self.df = dataframe
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.return_name = return_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(os.path.join(self.root_dir, row["image"])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = torch.tensor(self.tokenizer.encode(row["caption"]), dtype=torch.long)
        text = row["caption"]
        if self.return_name:
            return image, caption, text, row["image"]
        return image, caption, text


class PseudoCaptionDataset(Dataset):
    def __init__(self, images_dir, captions_dict, tokenizer, transform=None, max_samples=0):
        self.tokenizer = tokenizer
        self.transform = transform
        self.samples = []

        for fname, caption in captions_dict.items():
            path = os.path.join(images_dir, fname)
            if os.path.exists(path):
                self.samples.append((path, caption))

        if max_samples > 0 and max_samples < len(self.samples):
            self.samples = random.sample(self.samples, max_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, text = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return image, caption, text


class SemiSupervisedDataset(Dataset):
    def __init__(self, coco_dataset, pseudo_dataset):
        self.coco = coco_dataset
        self.pseudo = pseudo_dataset

    def __len__(self):
        return len(self.coco) + len(self.pseudo)

    def __getitem__(self, index):
        if index < len(self.coco):
            image, caption, text = self.coco[index]
            return image, caption, text, torch.tensor(0, dtype=torch.long)
        idx = index - len(self.coco)
        image, caption, text = self.pseudo[idx]
        return image, caption, text, torch.tensor(1, dtype=torch.long)


# ── Collate ────────────────────────────────────────────────────

def collate_fn(batch):
    if len(batch[0]) == 4:
        images, captions, texts, names = zip(*batch)
        images = torch.stack(images)
        captions = pad_sequence(captions, batch_first=True, padding_value=PAD_TOKEN)
        return images, captions, texts, names
    images, captions, texts = zip(*batch)
    images = torch.stack(images)
    captions = pad_sequence(captions, batch_first=True, padding_value=PAD_TOKEN)
    return images, captions, texts


def semi_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    captions = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=PAD_TOKEN)
    texts = [b[2] for b in batch]
    sources = torch.stack([b[3] for b in batch])
    return images, captions, texts, sources
