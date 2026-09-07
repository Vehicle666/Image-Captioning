import csv
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CAPTION_LOSS_WEIGHT,
    CHECKPOINT_DIR,
    CLIP_CACHE_PATH,
    CLIP_LOSS_WEIGHT,
    DROPOUT,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    EMBED_SIZE,
    ENCODER_BACKBONE,
    ENCODER_LR,
    DECODER_LR,
    GRAD_CLIP,
    HIDDEN_SIZE,
    K_FOLD,
    LABEL_SMOOTHING,
    MAX_BATCHES_PER_EPOCH,
    MODEL_BEST_PATH,
    MODEL_LATEST_PATH,
    NUM_EPOCHS,
    NUM_HEADS,
    NUM_LAYERS,
    NUM_WORKERS,
    RESUME_PATH,
    SCHEDULER_PATIENCE,
    SEED,
    STUDY_TAG,
    TRAINING_CSV_PATH,
    TRAINING_LOG_PATH,
    USE_CLIP,
    USE_V3_ENCODER,
    V3_ENCODER_BACKBONE,
    VAL_BLEU_EVERY_N_EPOCHS,
    VAL_BLEU_SUBSET,
    VAL_EVERY_N_EPOCHS,
    WARMUP_EPOCHS,
    device,
)
from src.dataset import (
    COCODataset,
    make_collate_fn,
    get_coco_captions,
    train_transform,
    val_transform,
)
from src.generation import generate_caption_beam
from src.model import CaptioningModel, CLIPContrastiveLoss, CLIPTextCache
from src.vocabulary import CaptionTokenizer

torch.set_num_threads(min(16, os.cpu_count() or 8))

CSV_HEADER = ["epoch", "fold", "timestamp", "seconds", "lr", "cap_loss", "clip_loss", "val_loss", "b1", "b4", "meteor", "cider", "is_best"]


class LengthBucketedBatchSampler:
    """Yields batches whose items have similar caption lengths (less padding).

    Indices are sorted by token length once, split into buckets, then each epoch
    shuffles bucket order and batch order while keeping items within a bucket
    (similar length) together.
    """

    def __init__(self, lengths, batch_size, bucket_mult=8):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_mult
        order = np.argsort(np.asarray(lengths), kind="stable")
        self.buckets = [
            order[i : i + self.bucket_size].tolist()
            for i in range(0, len(order), self.bucket_size)
        ]
        for b in self.buckets:
            random.shuffle(b)

    def __iter__(self):
        bucket_order = list(range(len(self.buckets)))
        random.shuffle(bucket_order)
        batches = []
        for bi in bucket_order:
            b = self.buckets[bi]
            for i in range(0, len(b) - self.batch_size + 1, self.batch_size):
                batches.append(b[i : i + self.batch_size])
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(max(0, (len(b) - self.batch_size + 1) // self.batch_size) for b in self.buckets)


def get_caption_token_lens(captions, tokenizer, cache_path):
    """Return {caption: token_count}, cached on disk to avoid re-tokenizing."""
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    missing = {c for c in captions if c not in cache}
    if missing:
        for c in tqdm(missing, desc="Tokenizing captions"):
            cache[c] = len(tokenizer.encode(c))
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    return cache


def model_is_finite(model):
    return all(
        torch.isfinite(t).all()
        for t in model.state_dict().values()
        if t.is_floating_point()
    )


def plot_embedding_projection(model, val_loader, clip_cache, epoch, save_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        return

    model.eval()
    all_img = []
    all_txt = []

    with torch.no_grad():
        for images, captions, texts, _ in val_loader:
            images = images.to(device)
            _, clip_feats = model(images, captions.to(device), return_clip_features=True)
            text_feats = clip_cache.lookup(texts)
            all_img.append(clip_feats.cpu().numpy())
            all_txt.append(text_feats.cpu().numpy())

    img_feats = np.concatenate(all_img, axis=0)
    txt_feats = np.concatenate(all_txt, axis=0)

    combined = np.concatenate([img_feats, txt_feats], axis=0)
    if not np.isfinite(combined).all():
        print(f"[warn] Skipping embedding plot (epoch {epoch}): non-finite embeddings")
        plt.close("all")
        return
    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)
    n = len(img_feats)
    img_2d, txt_2d = proj[:n], proj[n:]

    plt.figure(figsize=(10, 8))
    plt.scatter(img_2d[:, 0], img_2d[:, 1], alpha=0.5, s=12, label="Image")
    plt.scatter(txt_2d[:, 0], txt_2d[:, 1], alpha=0.5, s=12, label="Text")
    for i in range(min(30, n)):
        plt.plot([img_2d[i, 0], txt_2d[i, 0]], [img_2d[i, 1], txt_2d[i, 1]], "k-", alpha=0.12, lw=0.5)
    plt.legend()
    plt.title(f"CLIP Embedding Projection (PCA) — Epoch {epoch}")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"embeddings_epoch_{epoch:04d}.png"), dpi=120)
    plt.close()


def make_backbones():
    backbones = [ENCODER_BACKBONE]
    if USE_V3_ENCODER:
        backbones.append(V3_ENCODER_BACKBONE)
    return tuple(backbones)


def build_k_folds(images, k=K_FOLD):
    """Split train images into k deterministic, roughly equal folds.

    Sorted interleaved assignment: image[i] -> fold(i % k). Stable across
    runs/machines because it depends only on filenames, not on RNG state.
    Returns a list of folds (lists of image filenames).
    """
    ordered = sorted(images)
    folds = [[] for _ in range(k)]
    for idx, img in enumerate(ordered):
        folds[idx % k].append(img)
    return folds


def set_reproducible_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_tag():
    parts = [ENCODER_BACKBONE]
    if USE_V3_ENCODER:
        parts.append(f"+{V3_ENCODER_BACKBONE}")
    if USE_CLIP:
        parts.append("+CLIP")
    tag = " ".join(parts)
    return f"[{tag}]" if STUDY_TAG else tag


def train(num_epochs=None):
    effective_epochs = num_epochs or NUM_EPOCHS
    set_reproducible_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    backbones = make_backbones()
    clip_enabled = USE_CLIP

    print("=" * 60)
    print(f"  Supervised Image Captioning - {training_tag()}")
    print(f"  Backbones: {backbones} | CLIP loss: {clip_enabled}")
    print("=" * 60)

    # ── Load COCO captions (ground truth) ──────────────────────
    print("\nLoading COCO training captions...")
    train_df, train_images_dir = get_coco_captions("train")
    print(f"COCO training pairs: {len(train_df)}")

    print("Loading COCO validation captions...")
    val_df, val_images_dir = get_coco_captions("val")
    print(f"COCO validation pairs: {len(val_df)}")
    val_image_list = val_df["image"].unique().tolist()

    # ── Tokenizer ──────────────────────────────────────────────
    tokenizer = CaptionTokenizer()
    print(f"Tokenizer: {tokenizer.__class__.__name__}, vocab size: {len(tokenizer)}")

    # ── Build datasets ─────────────────────────────────────────
    all_train_images = train_df["image"].unique().tolist()
    collate = make_collate_fn(tokenizer.pad_token_id)

    # K-fold fold rotation: split ALL train images into K deterministic folds.
    # Epoch e uses fold (e % K), so every epoch covers a disjoint fixed slice
    # and the whole dataset cycles exactly once per K epochs. No random
    # subsampling -> run-to-run variance is eliminated.
    k_folds = build_k_folds(all_train_images, K_FOLD)
    fold_sizes = [len(f) for f in k_folds]
    print(f"K-fold: {len(k_folds)} folds (sizes {fold_sizes[0]}..{fold_sizes[-1]}, total {sum(fold_sizes)})")

    length_cache_path = os.path.join(CHECKPOINT_DIR, "caption_token_lens.json")
    caption_lens = get_caption_token_lens(train_df["caption"].tolist(), tokenizer, length_cache_path)

    def make_train_loader(fold_index):
        fold_images = k_folds[fold_index % K_FOLD]
        subset_df = train_df[train_df["image"].isin(fold_images)].reset_index(drop=True)
        ds = COCODataset(subset_df, train_images_dir, tokenizer, transform=train_transform)
        lengths = [caption_lens[c] for c in subset_df["caption"].tolist()]
        sampler = LengthBucketedBatchSampler(lengths, BATCH_SIZE)
        return DataLoader(ds, batch_sampler=sampler,
                          num_workers=NUM_WORKERS, collate_fn=collate, pin_memory=True)

    train_loader = make_train_loader(0)
    print(f"Training: {len(train_loader.dataset)} caption pairs in first fold (~{fold_sizes[0]} images)")

    val_loader = DataLoader(
        COCODataset(val_df, val_images_dir, tokenizer, transform=val_transform, return_name=True),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate, pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────
    model = CaptioningModel(
        embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
        vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT,
        backbones=backbones, use_clip_proj=clip_enabled,
    ).to(device)

    clip_loss_fn = None
    clip_cache = None
    if clip_enabled:
        clip_loss_fn = CLIPContrastiveLoss().to(device)

        # ── Build CLIP text cache (persisted to disk) ──────────
        clip_cache = CLIPTextCache(device)
        if os.path.exists(CLIP_CACHE_PATH):
            clip_cache.load(CLIP_CACHE_PATH)
        else:
            unique_captions = list(set(train_df["caption"].tolist() + val_df["caption"].tolist()))
            clip_cache.build(unique_captions)
            clip_cache.save(CLIP_CACHE_PATH)
        missing = clip_cache.missing(train_df["caption"].tolist() + val_df["caption"].tolist())
        if missing:
            clip_cache.add(missing)
            clip_cache.save(CLIP_CACHE_PATH)
        del clip_cache.text_encoder
        del clip_loss_fn.text_encoder, clip_loss_fn.tokenizer
        torch.cuda.empty_cache()

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = list(model.decoder.parameters())

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": ENCODER_LR},
        {"params": decoder_params, "lr": DECODER_LR},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE)
    caption_criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id, label_smoothing=LABEL_SMOOTHING
    )
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    # ── Logging ────────────────────────────────────────────────
    # Append mode: a resume session appends on top of the previous history, so
    # training_log.txt always covers the whole run from epoch 0, not just the
    # latest session. The header is only written for a fresh (empty) file.
    log_file = open(TRAINING_LOG_PATH, "a", encoding="utf-8")
    log_resumed = os.path.exists(TRAINING_LOG_PATH) and os.path.getsize(TRAINING_LOG_PATH) > 0
    if log_resumed:
        log_file.write(f"\n===== Session resumed @ {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    else:
        log_file.write("Training Log\n")
        log_file.write(f"Device: {device}\n")
        log_file.write(f"Config: {training_tag()} (STUDY_TAG={STUDY_TAG or 'default'})\n")
        log_file.write(f"Tokenizer: {tokenizer.__class__.__name__} ({len(tokenizer)} tokens)\n")
        log_file.write(f"Epochs: {effective_epochs} | Batch: {BATCH_SIZE}\n")
        log_file.write(f"Embed: {EMBED_SIZE} | Hidden: {HIDDEN_SIZE} | Layers: {NUM_LAYERS} | Heads: {NUM_HEADS}\n")
        log_file.write(f"Encoder LR: {ENCODER_LR} | Decoder LR: {DECODER_LR}\n")
        log_file.write(f"Caption loss weight: {CAPTION_LOSS_WEIGHT} | CLIP loss weight: {CLIP_LOSS_WEIGHT if clip_enabled else 0}\n")
        if clip_enabled:
            log_file.write(f"CLIP text cache: {clip_cache.embeddings.shape[0]} entries\n")
        log_file.write(f"Training images: {len(train_loader.dataset)} (fold 1/{K_FOLD})\n")
        log_file.write(f"K-fold: {K_FOLD} deterministic folds, epoch e -> fold (e % {K_FOLD}), seed {SEED}\n")
        log_file.write("-" * 60 + "\n\n")
    log_file.flush()

    # Machine-readable per-epoch sidecar: header written once, rows appended
    # every epoch (including resumed sessions) so loss/BLEU curves are trivial
    # to plot from CSV.
    if not (os.path.exists(TRAINING_CSV_PATH) and os.path.getsize(TRAINING_CSV_PATH) > 0):
        with open(TRAINING_CSV_PATH, "w", newline="", encoding="utf-8") as cf:
            csv.writer(cf).writerow(CSV_HEADER)

    # ── Training loop ──────────────────────────────────────────
    best_bleu4 = 0
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 0

    tmp = RESUME_PATH + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    if os.path.exists(RESUME_PATH):
        ckpt = torch.load(RESUME_PATH, map_location=device, weights_only=False)
        bad_tensors = [
            n for n, t in ckpt["model"].items()
            if t.is_floating_point() and not torch.isfinite(t).all()
        ]
        if bad_tensors:
            raise RuntimeError(
                f"{RESUME_PATH} contains {len(bad_tensors)} NaN/Inf tensors "
                f"(e.g. {bad_tensors[:3]}). Refusing to resume from a corrupted "
                f"checkpoint. Restore weights from {MODEL_BEST_PATH} instead."
            )
        ckpt_arch = ckpt.get("arch")
        if ckpt_arch and ckpt_arch != training_tag():
            raise RuntimeError(
                f"Resume checkpoint was trained with architecture '{ckpt_arch}' "
                f"but the current config uses '{training_tag()}'. This would corrupt "
                f"training; set STUDY_TAG/USE_V3/USE_CLIP/BACKBONE to match, or move "
                f"this resume_state.pth away to start fresh."
            )
        model.load_state_dict(ckpt["model"])
        ckpt_groups = len(ckpt["optimizer"]["param_groups"])
        if ckpt_groups == len(optimizer.param_groups):
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            print(
                f"[warn] Checkpoint optimizer has {ckpt_groups} param group(s) "
                f"but current setup has {len(optimizer.param_groups)}; "
                f"starting optimizer/scheduler fresh (model weights kept)."
            )
            log_file.write(
                f"[warn] Optimizer param-group mismatch ({ckpt_groups} vs "
                f"{len(optimizer.param_groups)}); optimizer state reset.\n"
            )
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_bleu4 = ckpt.get("best_bleu4", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"\nResumed from epoch {start_epoch} (best BLEU-4: {best_bleu4:.4f})\n")
        log_file.write(f"Resumed from epoch {start_epoch}\n")

    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, effective_epochs):
            train_loader = make_train_loader(epoch)
            fold_no = (epoch % K_FOLD) + 1

            model.train()
            epoch_cap_loss = 0
            epoch_clip_loss = 0.0

            warmup_factor = min(1.0, (epoch + 1) / WARMUP_EPOCHS) if epoch < WARMUP_EPOCHS else 1.0
            optimizer.param_groups[0]["lr"] = ENCODER_LR * warmup_factor
            optimizer.param_groups[1]["lr"] = DECODER_LR * warmup_factor

            epoch_label = f"{effective_epochs}" if effective_epochs < 90000 else "inf"
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epoch_label} (fold {fold_no}/{K_FOLD})")
            epoch_start = time.time()

            batch_count = 0
            for batch in progress:
                images, captions, texts = batch
                images, captions = images.to(device), captions.to(device)

                optimizer.zero_grad()

                with autocast("cuda", enabled=torch.cuda.is_available()):
                    if clip_enabled:
                        outputs, clip_features = model(images, captions, return_clip_features=True)
                    else:
                        outputs = model(images, captions)
                        clip_features = None

                    cap_loss = caption_criterion(
                        outputs.reshape(-1, len(tokenizer)),
                        captions[:, 1:].reshape(-1),
                    )
                    if clip_enabled:
                        closs = clip_loss_fn(clip_features, captions, text_cache=clip_cache, caption_strings=texts)
                        total_loss = CAPTION_LOSS_WEIGHT * cap_loss + CLIP_LOSS_WEIGHT * closs
                    else:
                        total_loss = CAPTION_LOSS_WEIGHT * cap_loss

                if not torch.isfinite(total_loss):
                    optimizer.zero_grad(set_to_none=True)
                    progress.set_postfix(cap=float("nan"), clip=float("nan"), status="skip")
                    continue

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()

                epoch_cap_loss += cap_loss.item()
                if clip_enabled:
                    epoch_clip_loss += closs.item()
                progress.set_postfix(cap=cap_loss.item(), clip=closs.item() if clip_enabled else 0.0)

                if batch_count % 100 == 0:
                    torch.cuda.empty_cache()

                batch_count += 1
                if MAX_BATCHES_PER_EPOCH > 0 and batch_count >= MAX_BATCHES_PER_EPOCH:
                    break

            num_batches = max(1, batch_count)
            avg_cap = epoch_cap_loss / num_batches
            avg_clip = epoch_clip_loss / num_batches

            if not model_is_finite(model):
                msg = (
                    f"\n[ERROR] Non-finite weights/buffers detected after epoch {epoch+1}. "
                    f"Halting WITHOUT saving so the last good checkpoint is preserved. "
                    f"Resume again to continue from it."
                )
                print(msg)
                log_file.write(msg + "\n")
                log_file.flush()
                break

            model.eval()
            run_val = (epoch + 1) % VAL_EVERY_N_EPOCHS == 0
            if run_val:
                val_loss = 0
                with torch.no_grad():
                    for images, captions, _, _ in tqdm(val_loader, desc="Val"):
                        images, captions = images.to(device), captions.to(device)
                        outputs = model(images, captions)
                        val_loss += caption_criterion(outputs.reshape(-1, len(tokenizer)), captions[:, 1:].reshape(-1)).item()
                avg_val = val_loss / max(1, len(val_loader))
                if clip_enabled:
                    emb_dir = os.path.join(CHECKPOINT_DIR, "embeddings")
                    try:
                        plot_embedding_projection(model, val_loader, clip_cache, epoch + 1, emb_dir)
                    except Exception as e:
                        print(f"[warn] Embedding plot failed (epoch {epoch+1}): {e}")
            else:
                avg_val = float("inf")
            if run_val:
                scheduler.step(avg_val)

            if avg_val < best_val_loss - EARLY_STOP_MIN_DELTA:
                best_val_loss = avg_val
                patience_counter = 0
            elif run_val:
                patience_counter += 1

            elapsed = time.time() - epoch_start
            lr_used = optimizer.param_groups[1]["lr"]

            line = f"Epoch {epoch+1}: fold {fold_no}/{K_FOLD} | CapLoss={avg_cap:.4f}"
            if clip_enabled:
                line += f" | CLIP={avg_clip:.4f}"
            if run_val:
                line += f" | ValLoss={avg_val:.4f}"
            line += f" | lr={lr_used:.2g} | {elapsed:.0f}s"

            b1 = b4 = meteor = cider = ""
            is_best = False
            if (epoch + 1) % VAL_BLEU_EVERY_N_EPOCHS == 0 and VAL_BLEU_SUBSET > 0:
                rng = random.Random(SEED * 1000 + (epoch + 1) // VAL_BLEU_EVERY_N_EPOCHS)
                subset = rng.sample(val_image_list, min(VAL_BLEU_SUBSET, len(val_image_list)))
                subset_df = val_df[val_df["image"].isin(subset)].reset_index(drop=True)
                subset_loader = DataLoader(
                    COCODataset(subset_df, val_images_dir, tokenizer, transform=val_transform, return_name=True),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate,
                )

                hyps, refs = {}, {}
                with torch.no_grad():
                    pbar_bleu = tqdm(subset_loader, desc="BLEU eval", leave=False)
                    for images, captions, texts, names in pbar_bleu:
                        images = images.to(device)
                        for i, name in enumerate(names):
                            if name not in hyps:
                                token_ids = generate_caption_beam(model, images[i:i+1], tokenizer)
                                hyps[name] = tokenizer.decode(token_ids).split()
                            refs.setdefault(name, []).append(texts[i].split())
                        pbar_bleu.set_postfix(hyp=len(hyps))

                from src.evaluate import compute_caption_metrics
                metrics = compute_caption_metrics(hyps, refs)
                b1, b4 = metrics["bleu1"], metrics["bleu4"]
                meteor, cider = metrics["meteor"], metrics["cider"]
                line += f" | B1={b1:.4f} B4={b4:.4f}"
                if meteor is not None:
                    line += f" M={meteor:.4f}"
                if cider is not None:
                    line += f" C={cider:.4f}"

                if metrics["bleu4"] > best_bleu4:
                    best_bleu4 = metrics["bleu4"]
                    torch.save(model.state_dict(), MODEL_BEST_PATH)
                    line += " *best*"
                    is_best = True

            print(line)
            log_file.write(line + "\n")
            log_file.flush()

            with open(TRAINING_CSV_PATH, "a", newline="", encoding="utf-8") as cf:
                csv.writer(cf).writerow([
                    epoch + 1,
                    fold_no,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"{elapsed:.1f}",
                    f"{lr_used:.3g}",
                    f"{avg_cap:.4f}",
                    f"{avg_clip:.4f}",
                    f"{avg_val:.4f}" if run_val else "",
                    f"{b1:.4f}" if isinstance(b1, float) else "",
                    f"{b4:.4f}" if isinstance(b4, float) else "",
                    f"{meteor:.4f}" if isinstance(meteor, float) else "",
                    f"{cider:.4f}" if isinstance(cider, float) else "",
                    1 if is_best else 0,
                ])
            torch.save(model.state_dict(), MODEL_LATEST_PATH)

            tmp = RESUME_PATH + ".tmp"
            torch.save({
                "epoch": epoch,
                "arch": training_tag(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_bleu4": best_bleu4,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
            }, tmp)
            os.replace(tmp, RESUME_PATH)

            if patience_counter >= EARLY_STOP_PATIENCE:
                stop_msg = f"\nEarly stopping at epoch {epoch+1} (patience={EARLY_STOP_PATIENCE})"
                print(stop_msg)
                log_file.write(stop_msg + "\n")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving checkpoint for resume...")
        if not model_is_finite(model):
            print("[ERROR] Weights are non-finite; NOT saving to avoid corrupting checkpoints.")
            log_file.write(f"\nInterrupted at epoch {epoch+1} with non-finite weights (not saved)\n")
            log_file.close()
            return model, tokenizer
        torch.save(model.state_dict(), MODEL_LATEST_PATH)
        tmp = RESUME_PATH + ".tmp"
        torch.save({
            "epoch": epoch,
            "arch": training_tag(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_bleu4": best_bleu4,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        }, tmp)
        os.replace(tmp, RESUME_PATH)
        log_file.write(f"\nInterrupted at epoch {epoch+1}\n")
        log_file.close()
        print("Checkpoint saved. Run again to resume.")
        return model, tokenizer

    summary = f"\nTraining complete. Best BLEU-4: {best_bleu4:.4f}"
    print(summary)
    log_file.write("=" * 60 + "\n" + summary.strip() + "\n" + "=" * 60 + "\n")
    log_file.close()

    if os.path.exists(RESUME_PATH):
        os.remove(RESUME_PATH)

    return model, tokenizer


if __name__ == "__main__":
    train()