import json
import os
import random
import signal
import sys
import time

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from linh_src.config import (
    BATCH_SIZE,
    CAPTION_LOSS_WEIGHT,
    CHECKPOINT_DIR,
    CLIP_LOSS_WEIGHT,
    DROPOUT,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    EMBED_SIZE,
    ENCODER_LR,
    DECODER_LR,
    GRAD_ACCUM_STEPS,
    GRAD_CLIP,
    HIDDEN_SIZE,
    IMAGENET_TRAIN,
    MAX_TRAIN_IMAGES,
    MIN_WORD_FREQ,
    MODEL_BEST_PATH,
    MODEL_LATEST_PATH,
    NUM_EPOCHS,
    NUM_HEADS,
    NUM_LAYERS,
    NUM_WORKERS,
    PSEUDO_CAPTIONS_DIR,
    STOP_FILE,
    TOKENIZER_PATH,
    TRAINING_LOG_PATH,
    VAL_BLEU_EVERY_N_EPOCHS,
    VAL_BLEU_SUBSET,
    VAL_LOSS_FULL_EVERY,
    VAL_LOSS_SUBSET,
    VOCAB_SIZE,
    WARMUP_EPOCHS,
    device,
)
from linh_src.dataset import (
    COCODataset,
    PseudoCaptionDataset,
    SemiSupervisedDataset,
    collate_fn,
    get_coco_captions,
    load_pseudo_captions,
    semi_collate_fn,
    train_transform,
    val_transform,
)
from linh_src.model import CaptioningModel, CLIPContrastiveLoss, CLIPTextCache
from linh_src.vocabulary import CaptionTokenizer

torch.set_num_threads(min(16, os.cpu_count() or 8))

_stop_requested = False


def _signal_handler(sig, frame):
    global _stop_requested
    if _stop_requested:
        print("\nForce quitting...")
        sys.exit(1)
    _stop_requested = True
    print("\n[STOP] Ctrl+C received. Finishing current batch then saving...")


signal.signal(signal.SIGINT, _signal_handler)


def _iter_loader(loader, is_stopped):
    """Iterate a DataLoader, tolerating worker death caused by Ctrl+C.

    On Windows, Ctrl+C is delivered to the whole console process group, so
    DataLoader worker subprocesses die too. The DataLoader then raises
    ``RuntimeError: DataLoader worker ... exited unexpectedly`` instead of
    letting the SIGINT handler finish gracefully. This wrapper converts that
    crash into a clean end-of-iteration whenever a stop has been requested.
    """
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except RuntimeError as e:
            if "exited unexpectedly" in str(e) and is_stopped():
                return
            raise


@torch.no_grad()
def generate_caption(model, image, tokenizer, max_length=50):
    features = model.encoder(image)
    caption = [tokenizer.sos_token_id]
    for _ in range(max_length):
        tgt = torch.tensor([caption]).to(device)
        logits = model.decoder(features, tgt)
        pred = logits[0, -1, :].argmax().item()
        if pred == tokenizer.eos_token_id:
            break
        caption.append(pred)
    return tokenizer.decode(caption[1:]).split()


def _save_checkpoint(model, path, line="", log_file=None):
    torch.save(model.state_dict(), path)
    tag = os.path.basename(path)
    if line:
        print(f"  [saved {tag}] {line}")
    else:
        print(f"  [saved {tag}]")


def train():
    global _stop_requested
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)

    print("=" * 60)
    print("  Semi-Supervised Training: MobileNet + CLIP + BLIP")
    print("=" * 60)
    print(f"\n  To stop gracefully: press Ctrl+C (saves checkpoint)")
    print(f"  Or create file:     echo x > {STOP_FILE}")
    print(f"  Effective batch:    {BATCH_SIZE} x {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    # ── Load COCO captions (ground truth) ──────────────────────
    print("\nLoading COCO training captions...")
    train_df, train_images_dir = get_coco_captions("train")
    print(f"COCO training pairs: {len(train_df)}")

    print("Loading COCO validation captions...")
    val_df, val_images_dir = get_coco_captions("val")
    print(f"COCO validation pairs: {len(val_df)}")
    val_image_list = val_df["image"].unique().tolist()

    # ── Build vocabulary from soft captions + ground truth ──────
    all_captions = train_df["caption"].tolist()

    coco_soft_path = os.path.join(PSEUDO_CAPTIONS_DIR, "coco_captions.json")
    if os.path.exists(coco_soft_path):
        soft_caps = load_pseudo_captions(coco_soft_path)
        all_captions.extend(soft_caps.values())
        print(f"Added {len(soft_caps)} COCO soft captions")

    imagenet_soft_path = os.path.join(PSEUDO_CAPTIONS_DIR, "imagenet_captions.json")
    if os.path.exists(imagenet_soft_path):
        in_caps = load_pseudo_captions(imagenet_soft_path)
        all_captions.extend(in_caps.values())
        print(f"Added {len(in_caps)} ImageNet pseudo-captions")

    tokenizer = CaptionTokenizer.train(
        all_captions, vocab_size=VOCAB_SIZE,
        min_frequency=MIN_WORD_FREQ, save_path=TOKENIZER_PATH,
    )
    print(f"Tokenizer: {tokenizer}")

    # ── Build datasets ─────────────────────────────────────────
    coco_train = COCODataset(train_df, train_images_dir, tokenizer, transform=train_transform)

    pseudo_dataset = None
    if os.path.exists(imagenet_soft_path) and os.path.exists(IMAGENET_TRAIN):
        in_caps = load_pseudo_captions(imagenet_soft_path)
        pseudo_dataset = PseudoCaptionDataset(IMAGENET_TRAIN, in_caps, tokenizer, transform=train_transform)
        print(f"ImageNet pseudo-caption dataset: {len(pseudo_dataset)} images")

    if pseudo_dataset is not None:
        train_dataset = SemiSupervisedDataset(coco_train, pseudo_dataset)
        print(f"Combined training dataset: {len(train_dataset)} images")
    else:
        train_dataset = coco_train
        print("Training on COCO only (no ImageNet pseudo-captions found)")

    if MAX_TRAIN_IMAGES > 0 and len(train_df["image"].unique()) > MAX_TRAIN_IMAGES:
        print(f"Per-epoch subset: {MAX_TRAIN_IMAGES} images (of {len(train_df['image'].unique())} total)")

    pw = NUM_WORKERS > 0
    train_collate = semi_collate_fn if pseudo_dataset else collate_fn
    train_images = list(train_df["image"].unique())

    val_full_dataset = COCODataset(val_df, val_images_dir, tokenizer, transform=val_transform, return_name=True)
    val_loader = DataLoader(
        val_full_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=pw, prefetch_factor=2 if pw else None,
    )

    # ── Model ──────────────────────────────────────────────────
    model = CaptioningModel(
        embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
        vocab_size=len(tokenizer), num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, dropout=DROPOUT,
    ).to(device)

    clip_loss_fn = CLIPContrastiveLoss().to(device)

    # ── Build CLIP text cache ─────────────────────────────────
    clip_cache = CLIPTextCache(device)
    unique_captions = list(set(all_captions))
    clip_cache.build(unique_captions)
    del clip_cache.text_encoder
    del clip_loss_fn.text_encoder, clip_loss_fn.tokenizer

    # ── Resume from checkpoint if available ─────────────────────
    start_epoch = 0
    if os.path.exists(MODEL_LATEST_PATH):
        model.load_state_dict(torch.load(MODEL_LATEST_PATH, map_location=device, weights_only=True))
        # Parse log for last epoch
        if os.path.exists(TRAINING_LOG_PATH):
            with open(TRAINING_LOG_PATH) as f:
                for line in f:
                    if line.startswith("Epoch "):
                        try:
                            epoch_num = int(line.split(":")[0].split()[1])
                            start_epoch = max(start_epoch, epoch_num)
                        except (ValueError, IndexError):
                            pass
        print(f"Resumed from checkpoint (epoch {start_epoch})")
    torch.cuda.empty_cache()

    encoder_params = (
        list(model.encoder.features.parameters())
        + list(model.encoder.attention.parameters())
        + list(model.encoder.conv.parameters())
        + list(model.encoder.bn.parameters())
        + list(model.encoder.clip_proj.parameters())
    )
    decoder_params = list(model.decoder.parameters())

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": ENCODER_LR},
        {"params": decoder_params, "lr": DECODER_LR},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    caption_criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    # ── Logging ────────────────────────────────────────────────
    log_file = open(TRAINING_LOG_PATH, "w")
    log_file.write("Semi-Supervised Training Log\n")
    log_file.write(f"Device: {device}\n")
    epoch_str = "indefinite" if NUM_EPOCHS <= 0 else str(NUM_EPOCHS)
    log_file.write(f"Epochs: {epoch_str} | Batch: {BATCH_SIZE} | GradAccum: {GRAD_ACCUM_STEPS}\n")
    log_file.write(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}\n")
    log_file.write(f"Embed: {EMBED_SIZE} | Hidden: {HIDDEN_SIZE} | Layers: {NUM_LAYERS} | Heads: {NUM_HEADS}\n")
    log_file.write(f"Encoder LR: {ENCODER_LR} | Decoder LR: {DECODER_LR}\n")
    log_file.write(f"Caption loss weight: {CAPTION_LOSS_WEIGHT} | CLIP loss weight: {CLIP_LOSS_WEIGHT}\n")
    log_file.write(f"Vocabulary: {len(tokenizer)} tokens\n")
    log_file.write(f"CLIP text cache: {clip_cache.embeddings.shape[0]} entries\n")
    log_file.write(f"Training images: {len(train_images)} total")
    if MAX_TRAIN_IMAGES > 0:
        log_file.write(f", using {MAX_TRAIN_IMAGES} per epoch")
    log_file.write(f"\n")
    log_file.write(f"Val subset: {VAL_LOSS_SUBSET} (full every {VAL_LOSS_FULL_EVERY} epochs)\n")
    log_file.write(f"Workers: {NUM_WORKERS} | persistent_workers: {pw}\n")
    log_file.write("-" * 60 + "\n\n")
    log_file.flush()

    # ── Training loop ──────────────────────────────────────────
    best_bleu4 = 0
    best_val_loss = float("inf")
    best_val_epoch = 0
    best_bleu_epoch = 0
    patience_counter = 0
    global_start = time.time()
    max_epochs = NUM_EPOCHS if NUM_EPOCHS > 0 else None
    epoch = start_epoch

    while True:
        if max_epochs is not None and epoch >= max_epochs:
            break
        epoch_t0 = time.time()
        model.train()
        epoch_cap_loss = 0
        epoch_clip_loss = 0
        micro_steps = 0

        warmup_factor = min(1.0, (epoch + 1) / WARMUP_EPOCHS) if epoch < WARMUP_EPOCHS else 1.0
        optimizer.param_groups[0]["lr"] = ENCODER_LR * warmup_factor
        optimizer.param_groups[1]["lr"] = DECODER_LR * warmup_factor

        if MAX_TRAIN_IMAGES > 0 and len(train_images) > MAX_TRAIN_IMAGES:
            epoch_images = random.sample(train_images, MAX_TRAIN_IMAGES)
            epoch_df = train_df[train_df["image"].isin(epoch_images)].reset_index(drop=True)
            epoch_coco = COCODataset(epoch_df, train_images_dir, tokenizer, transform=train_transform)
            if pseudo_dataset is not None:
                epoch_dataset = SemiSupervisedDataset(epoch_coco, pseudo_dataset)
            else:
                epoch_dataset = epoch_coco
        else:
            epoch_dataset = train_dataset

        train_loader = DataLoader(
            epoch_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, collate_fn=train_collate,
            pin_memory=True, persistent_workers=False, prefetch_factor=2 if pw else None,
        )

        epoch_label = f"Epoch {epoch+1}" + (f"/{NUM_EPOCHS}" if max_epochs else "")
        progress = tqdm(_iter_loader(train_loader, lambda: _stop_requested), desc=epoch_label, total=len(train_loader))

        for batch in progress:
            if _stop_requested:
                break

            if len(batch) == 4:
                images, captions, texts, sources = batch
                if not isinstance(images, torch.Tensor):
                    images = torch.stack(images)
                if not isinstance(captions, torch.Tensor):
                    captions = torch.stack(captions)
                if not isinstance(sources, torch.Tensor):
                    sources = torch.stack(sources)
                images, captions, sources = images.to(device), captions.to(device), sources.to(device)
            else:
                images, captions, texts = batch
                if not isinstance(images, torch.Tensor):
                    images = torch.stack(images)
                if not isinstance(captions, torch.Tensor):
                    captions = torch.stack(captions)
                images, captions = images.to(device), captions.to(device)

            with autocast("cuda", enabled=torch.cuda.is_available()):
                outputs, clip_features = model(images, captions, return_clip_features=True)

                cap_loss = caption_criterion(
                    outputs.reshape(-1, len(tokenizer)),
                    captions[:, 1:].reshape(-1),
                )
                closs = clip_loss_fn(clip_features, captions, text_cache=clip_cache, caption_strings=texts)
                total_loss = (CAPTION_LOSS_WEIGHT * cap_loss + CLIP_LOSS_WEIGHT * closs) / GRAD_ACCUM_STEPS

            scaler.scale(total_loss).backward()
            micro_steps += 1

            if micro_steps % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_cap_loss += cap_loss.item()
            epoch_clip_loss += closs.item()
            progress.set_postfix(cap=cap_loss.item(), clip=closs.item())

        if _stop_requested:
            torch.save(model.state_dict(), MODEL_LATEST_PATH)
            line = f"Epoch {epoch+1}: stopped early, checkpoint saved to {MODEL_LATEST_PATH}"
            print(line)
            log_file.write(line + "\n")
            log_file.flush()
            break

        num_batches = max(1, len(train_loader))
        avg_cap = epoch_cap_loss / num_batches
        avg_clip = epoch_clip_loss / num_batches

        line = f"Epoch {epoch+1}: CapLoss={avg_cap:.4f} | CLIP={avg_clip:.4f}"

        if not _stop_requested:
            # ── Validation loss ────────────────────────────────
            use_full_val = (epoch + 1) % VAL_LOSS_FULL_EVERY == 0
            if use_full_val:
                val_iter = val_loader
            else:
                n = min(VAL_LOSS_SUBSET, len(val_full_dataset))
                indices = random.sample(range(len(val_full_dataset)), n)
                subset = Subset(val_full_dataset, indices)
                val_iter = DataLoader(
                    subset, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=0, collate_fn=collate_fn,
                )

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, captions, _, _ in _iter_loader(val_iter, lambda: _stop_requested):
                    images, captions = images.to(device), captions.to(device)
                    outputs = model(images, captions)
                    val_loss += caption_criterion(outputs.reshape(-1, len(tokenizer)), captions[:, 1:].reshape(-1)).item()
            avg_val = val_loss / max(1, len(val_iter))

            if not _stop_requested:
                scheduler.step(avg_val)

                # ── Early stopping check ──────────────────────
                if avg_val < best_val_loss - EARLY_STOP_MIN_DELTA:
                    best_val_loss = avg_val
                    best_val_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1

                val_tag = f"(full {len(val_full_dataset)})" if use_full_val else f"(subset {min(VAL_LOSS_SUBSET, len(val_full_dataset))})"
                line += f" | ValLoss={avg_val:.4f} {val_tag}"

                # ── BLEU evaluation ────────────────────────────
                if (epoch + 1) % VAL_BLEU_EVERY_N_EPOCHS == 0 and VAL_BLEU_SUBSET > 0:
                    subset = random.sample(val_image_list, min(VAL_BLEU_SUBSET, len(val_image_list)))
                    subset_df = val_df[val_df["image"].isin(subset)].reset_index(drop=True)
                    subset_loader = DataLoader(
                        COCODataset(subset_df, val_images_dir, tokenizer, transform=val_transform, return_name=True),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn,
                    )

                    hyps, refs = {}, {}
                    with torch.no_grad():
                        for images, captions, texts, names in subset_loader:
                            images = images.to(device)
                            for i, name in enumerate(names):
                                if name not in hyps:
                                    hyps[name] = generate_caption(model, images[i:i+1], tokenizer)
                                ref_idx = captions[i].tolist()
                                ref_text = tokenizer.decode([j for j in ref_idx if j not in (0, 1, 2)])
                                refs.setdefault(name, []).append(ref_text.split())

                    from linh_src.evaluate import compute_caption_metrics
                    metrics = compute_caption_metrics(hyps, refs, bleu_only=True)
                    line += f" | B1={metrics['bleu1']:.4f} B2={metrics['bleu2']:.4f} B3={metrics['bleu3']:.4f} B4={metrics['bleu4']:.4f}"

                    if metrics["bleu4"] > best_bleu4:
                        best_bleu4 = metrics["bleu4"]
                        best_bleu_epoch = epoch + 1
                        torch.save(model.state_dict(), MODEL_BEST_PATH)
                        line += " *best*"

        elapsed = time.time() - epoch_t0
        total_elapsed = time.time() - global_start
        line += f" [{elapsed:.0f}s]"

        print(line)
        log_file.write(line + "\n")
        log_file.flush()
        torch.save(model.state_dict(), MODEL_LATEST_PATH)

        epoch += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            stop_msg = f"\nEarly stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})"
            print(stop_msg)
            log_file.write(stop_msg + "\n")
            break

        if os.path.exists(STOP_FILE):
            stop_msg = f"\nStop file detected at epoch {epoch}"
            print(stop_msg)
            log_file.write(stop_msg + "\n")
            os.remove(STOP_FILE)
            break

    total_time = time.time() - global_start
    summary = f"\nTraining complete ({total_time/60:.1f} min). Best BLEU-4: {best_bleu4:.4f} (epoch {best_bleu_epoch}) | Best val loss: {best_val_loss:.4f} (epoch {best_val_epoch})"
    print(summary)
    log_file.write("=" * 60 + "\n" + summary.strip() + "\n" + "=" * 60 + "\n")
    log_file.close()
    return model, tokenizer


if __name__ == "__main__":
    train()
