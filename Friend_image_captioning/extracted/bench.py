import os, sys, time, torch
os.environ["HF_HOME"] = "D:/AI_Cache/huggingface"
os.environ["TORCH_HOME"] = "D:/AI_Cache/torch"

from src.config import *
from src.vocabulary import CaptionTokenizer
from src.model import CaptioningModel, CLIPContrastiveLoss, CLIPTextCache
from src.dataset import *

tokenizer = CaptionTokenizer.load(TOKENIZER_PATH)
train_df, train_images_dir = get_coco_captions("train")
val_df, val_images_dir = get_coco_captions("val")

import json
imagenet_soft_path = os.path.join(PSEUDO_CAPTIONS_DIR, "imagenet_captions.json")
coco_soft_path = os.path.join(PSEUDO_CAPTIONS_DIR, "coco_captions.json")
has_imagenet = os.path.exists(imagenet_soft_path) and os.path.exists(IMAGENET_TRAIN)
has_coco_soft = os.path.exists(coco_soft_path)

all_captions = train_df["caption"].tolist()
if has_coco_soft:
    soft = load_pseudo_captions(coco_soft_path)
    all_captions.extend(soft.values())
    print(f"COCO soft captions: {len(soft)}")
if has_imagenet:
    in_caps = load_pseudo_captions(imagenet_soft_path)
    all_captions.extend(in_caps.values())
    print(f"ImageNet captions: {len(in_caps)}")

print(f"COCO train pairs: {len(train_df)}")
print(f"COCO train unique images: {train_df['image'].nunique()}")
print(f"COCO val pairs: {len(val_df)}")
print(f"COCO val unique images: {val_df['image'].nunique()}")

if has_imagenet:
    coco_train = COCODataset(train_df, train_images_dir, tokenizer, transform=train_transform)
    in_caps = load_pseudo_captions(imagenet_soft_path)
    pseudo = PseudoCaptionDataset(IMAGENET_TRAIN, in_caps, tokenizer, transform=train_transform)
    train_dataset = SemiSupervisedDataset(coco_train, pseudo)
    print(f"ImageNet pseudo images: {len(pseudo)}")
else:
    train_dataset = COCODataset(train_df, train_images_dir, tokenizer, transform=train_transform)

total = len(train_dataset)
batches = total // BATCH_SIZE
opt_steps = batches // GRAD_ACCUM_STEPS
print(f"\n{'='*60}")
print(f"  TRAINING MATH")
print(f"{'='*60}")
print(f"  Total training samples:  {total:,}")
print(f"  Batch size:              {BATCH_SIZE}")
print(f"  Batches per epoch:       {batches:,}")
print(f"  Grad accum steps:        {GRAD_ACCUM_STEPS}")
print(f"  Effective batch:         {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"  Optimizer steps/epoch:   {opt_steps:,}")
print(f"  Workers:                 {NUM_WORKERS}")
print(f"  Val subset:              {VAL_LOSS_SUBSET} (full every {VAL_LOSS_FULL_EVERY})")
print(f"{'='*60}")

# Benchmark: 1 batch forward+backward
print("\nBenchmarking 1 training batch...")
model = CaptioningModel(
    embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
    vocab_size=len(tokenizer), num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS, dropout=DROPOUT,
).to(device)
clip_loss_fn = CLIPContrastiveLoss().to(device)

clip_cache = CLIPTextCache(device)
unique_captions = list(set(all_captions))
clip_cache.build(unique_captions)
del clip_cache.text_encoder
del clip_loss_fn.text_encoder, clip_loss_fn.tokenizer
torch.cuda.empty_cache()

import torch.nn as nn
caption_criterion = nn.CrossEntropyLoss(ignore_index=0)
scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

collate = semi_collate_fn if has_imagenet else collate_fn
loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, collate_fn=collate, pin_memory=False,
)

model.train()
batch = next(iter(loader))
if len(batch) == 4:
    images, captions, texts, sources = batch
    images, captions, sources = images.to(device), captions.to(device), sources.to(device)
else:
    images, captions, texts = batch
    images, captions = images.to(device), captions.to(device)

# Warmup
for _ in range(3):
    with autocast("cuda", enabled=torch.cuda.is_available()):
        outputs, clip_features = model(images, captions, return_clip_features=True)
        cap_loss = caption_criterion(outputs.reshape(-1, len(tokenizer)), captions[:, 1:].reshape(-1))
        closs = clip_loss_fn(clip_features, captions, text_cache=clip_cache, caption_strings=texts)
        total_loss = cap_loss + 0.5 * closs
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

torch.cuda.synchronize()

# Timed run: 10 batches
N = 10
t0 = time.time()
for i, batch in enumerate(loader):
    if i >= N:
        break
    if len(batch) == 4:
        images, captions, texts, sources = batch
        images, captions, sources = images.to(device), captions.to(device), sources.to(device)
    else:
        images, captions, texts = batch
        images, captions = images.to(device), captions.to(device)

    with autocast("cuda", enabled=torch.cuda.is_available()):
        outputs, clip_features = model(images, captions, return_clip_features=True)
        cap_loss = caption_criterion(outputs.reshape(-1, len(tokenizer)), captions[:, 1:].reshape(-1))
        closs = clip_loss_fn(clip_features, captions, text_cache=clip_cache, caption_strings=texts)
        total_loss = cap_loss + 0.5 * closs
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
torch.cuda.synchronize()
batch_time = (time.time() - t0) / N

print(f"\n  Batch forward+backward: {batch_time*1000:.0f} ms")
print(f"  Estimated train time:   {batch_time * batches / 60:.1f} min (bs={BATCH_SIZE})")

# Benchmark val
print("\nBenchmarking 1 val batch...")
val_ds = COCODataset(val_df, val_images_dir, tokenizer, transform=val_transform, return_name=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
model.eval()
vbatch = next(iter(val_loader))
vimages, vcaptions, _, _ = vbatch
vimages, vcaptions = vimages.to(device), vcaptions.to(device)

for _ in range(2):
    with torch.no_grad():
        out = model(vimages, vcaptions)
        _ = caption_criterion(out.reshape(-1, len(tokenizer)), vcaptions[:, 1:].reshape(-1)).item()
torch.cuda.synchronize()

N2 = 20
t1 = time.time()
for i, (vimages, vcaptions, _, _) in enumerate(val_loader):
    if i >= N2:
        break
    vimages, vcaptions = vimages.to(device), vcaptions.to(device)
    with torch.no_grad():
        out = model(vimages, vcaptions)
        _ = caption_criterion(out.reshape(-1, len(tokenizer)), vcaptions[:, 1:].reshape(-1)).item()
torch.cuda.synchronize()
val_batch_time = (time.time() - t1) / N2
val_batches_full = len(val_ds) // BATCH_SIZE
val_batches_sub = min(VAL_LOSS_SUBSET, len(val_ds)) // BATCH_SIZE

print(f"  Val batch: {val_batch_time*1000:.0f} ms")
print(f"  Full val ({len(val_ds)} imgs, {val_batches_full} batches): {val_batch_time * val_batches_full / 60:.1f} min")
print(f"  Sub val ({VAL_LOSS_SUBSET} imgs, {val_batches_sub} batches): {val_batch_time * val_batches_sub / 60:.1f} min")

# Summary
print(f"\n{'='*60}")
print(f"  ESTIMATED EPOCH TIME COMPARISON")
print(f"{'='*60}")
old_train = batch_time * batches / 60
new_train = batch_time * (batches // GRAD_ACCUM_STEPS) / 60
old_val = val_batch_time * val_batches_full / 60
new_val = val_batch_time * val_batches_sub / 60
print(f"  BEFORE: train {old_train:.1f} + val {old_val:.1f} = {old_train+old_val:.1f} min/epoch")
print(f"  AFTER:  train {new_train:.1f} + val {new_val:.1f} = {new_train+new_val:.1f} min/epoch")
print(f"  Speedup: {(old_train+old_val)/(new_train+new_val):.1f}x")
print(f"{'='*60}")
