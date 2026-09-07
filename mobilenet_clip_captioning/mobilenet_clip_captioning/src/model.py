import math
import os

import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from src.config import (
    CLIP_EMBED_DIM,
    CLIP_TEXT_MODEL,
    CLIP_TEMPERATURE,
)


# ── Backbone registry (all give a 7x7 spatial grid for 224x224 input) ──
# name -> (builder, weights, out_channels, train_from)
#   `train_from` = index in `features` from which parameters stay trainable
#   (earlier blocks are frozen to keep the pretrained backbone stable).
BACKBONE_REGISTRY = {
    "mobilenet_v3_small": (
        models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT, 576, 9,
    ),
    "mobilenet_v3_large": (
        models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT, 960, 10,
    ),
    "efficientnet_b0": (
        models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280, 6,
    ),
}


# ── Positional Encoding ────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


# ── Spatial Attention ──────────────────────────────────────────

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


# ── Single backbone branch ─────────────────────────────────────

class BackboneBranch(nn.Module):
    def __init__(self, backbone_name, embed_size):
        super().__init__()
        builder, weights, out_channels, train_from = BACKBONE_REGISTRY[backbone_name]
        net = builder(weights=weights)

        self.features = net.features
        self.out_channels = out_channels
        self.backbone_name = backbone_name

        for i, mod in enumerate(self.features):
            if i < train_from:
                for p in mod.parameters():
                    p.requires_grad = False

        self.attention = SpatialAttention()
        self.conv = nn.Conv2d(out_channels, embed_size, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_size)

    def forward(self, images):
        features = self.features(images)        # (B, C, 7, 7)
        features = self.attention(features)     # (B, C, 7, 7)
        pooled = torch.mean(features, dim=[2, 3])  # (B, C) for CLIP projection
        features = self.bn(self.conv(features))    # (B, embed_size, 7, 7)
        return features, pooled


# ── Image Encoder (1 or more backbones) ────────────────────────

class ImageEncoder(nn.Module):
    """CNN encoder supporting an ablation-style feature fusion.

    - 1 backbone            : MobileNet-only baseline
    - 2 backbones (+V3)     : each branch 1x1-projects to embed_size, then the
                              branches are concatenated along channels and fused
                              back down to embed_size with a Conv1x1 + BN.
    - clip_proj (optional)  : global-pooled branch features projected to the
                              CLIP embedding space (contrastive loss, training only).
    """

    def __init__(self, embed_size, backbones=("mobilenet_v3_small",),
                 clip_embed_dim=CLIP_EMBED_DIM, use_clip_proj=False):
        super().__init__()
        self.use_clip_proj = use_clip_proj
        self.branches = nn.ModuleList(
            [BackboneBranch(b, embed_size) for b in backbones]
        )

        self.fusion = None
        if len(backbones) > 1:
            self.fusion = nn.Sequential(
                nn.Conv2d(embed_size * len(backbones), embed_size, kernel_size=1),
                nn.BatchNorm2d(embed_size),
            )

        if use_clip_proj:
            total_channels = sum(
                BACKBONE_REGISTRY[b][2] for b in backbones
            )
            self.clip_proj = nn.Linear(total_channels, clip_embed_dim)

    def forward(self, images, return_clip_features=False):
        branch_outs, branch_pools = [], []
        for branch in self.branches:
            out, pooled = branch(images)
            branch_outs.append(out)
            branch_pools.append(pooled)

        if self.fusion is not None:
            features = torch.cat(branch_outs, dim=1)   # (B, k*embed, 7, 7)
            features = self.fusion(features)           # (B, embed, 7, 7)
        else:
            features = branch_outs[0]

        features = features.permute(0, 2, 3, 1)                        # (B, 7, 7, embed)
        features = features.reshape(features.size(0), -1, features.size(-1))  # (B, 49, embed)

        if return_clip_features:
            if not self.use_clip_proj:
                raise ValueError("use_clip_proj=False but CLIP features were requested")
            clip_features = self.clip_proj(torch.cat(branch_pools, dim=1))  # (B, 512)
            return features, clip_features
        return features


# ── CLIP Contrastive Loss ──────────────────────────────────────

class CLIPTextCache:
    def __init__(self, device, batch_size=256):
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_TEXT_MODEL)
        self.text_encoder = CLIPTextModel.from_pretrained(CLIP_TEXT_MODEL).to(device)
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.embeddings = None
        self.str_to_idx = {}

    @torch.no_grad()
    def build(self, captions):
        unique = list(set(captions))
        print(f"  Building CLIP text cache ({len(unique)} unique captions)...")
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        all_embs = []
        for i in tqdm(range(0, len(unique), self.batch_size), desc="  Caching"):
            batch = unique[i : i + self.batch_size]
            token_ids = []
            for cap in batch:
                tokens = self.tokenizer(
                    cap, padding=False, truncation=True, max_length=77
                )["input_ids"]
                token_ids.append(tokens)

            max_len = max(len(t) for t in token_ids)
            padded = [t + [pad_id] * (max_len - len(t)) for t in token_ids]
            inputs = torch.tensor(padded, device=self.device)
            outputs = self.text_encoder(inputs)
            all_embs.append(outputs.pooler_output.cpu())

        self.embeddings = torch.cat(all_embs, dim=0)
        self.str_to_idx = {cap: idx for idx, cap in enumerate(unique)}
        print(f"  CLIP text cache ready ({self.embeddings.shape[0]} entries, {self.embeddings.shape[1]}d)")

    def save(self, path):
        torch.save({"embeddings": self.embeddings, "str_to_idx": self.str_to_idx}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.embeddings = ckpt["embeddings"]
        self.str_to_idx = ckpt["str_to_idx"]
        print(f"  CLIP text cache loaded ({self.embeddings.shape[0]} entries, {self.embeddings.shape[1]}d)")
        return self

    def missing(self, captions):
        return sorted({c for c in captions if c not in self.str_to_idx})

    @torch.no_grad()
    def add(self, captions):
        missing_caps = self.missing(captions)
        if not missing_caps:
            return 0
        print(f"  Adding {len(missing_caps)} missing captions to CLIP text cache...")
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        all_embs = []
        for i in tqdm(range(0, len(missing_caps), self.batch_size), desc="  Caching"):
            batch = missing_caps[i : i + self.batch_size]
            token_ids = []
            for cap in batch:
                tokens = self.tokenizer(
                    cap, padding=False, truncation=True, max_length=77
                )["input_ids"]
                token_ids.append(tokens)

            max_len = max(len(t) for t in token_ids)
            padded = [t + [pad_id] * (max_len - len(t)) for t in token_ids]
            inputs = torch.tensor(padded, device=self.device)
            outputs = self.text_encoder(inputs)
            all_embs.append(outputs.pooler_output.cpu())

        new_embs = torch.cat(all_embs, dim=0)
        start = len(self.embeddings)
        self.embeddings = torch.cat([self.embeddings, new_embs], dim=0)
        for idx, cap in enumerate(missing_caps):
            self.str_to_idx[cap] = start + idx
        print(f"  CLIP text cache now {self.embeddings.shape[0]} entries ({self.embeddings.shape[1]}d)")
        return len(missing_caps)

    def lookup(self, captions):
        indices = torch.tensor(
            [self.str_to_idx[c] for c in captions], dtype=torch.long
        )
        return self.embeddings[indices].to(self.device)


class CLIPContrastiveLoss(nn.Module):
    def __init__(self, temperature=CLIP_TEMPERATURE):
        super().__init__()
        self.temperature = temperature
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_TEXT_MODEL)
        self.text_encoder = CLIPTextModel.from_pretrained(CLIP_TEXT_MODEL)
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_text(self, caption_ids):
        device = next(self.text_encoder.parameters()).device
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        bos_id = self.tokenizer.bos_token_id
        text = []
        for ids in caption_ids:
            tokens = [t for t in ids if t != pad_id]
            if tokens and tokens[0] != bos_id:
                tokens = [bos_id] + tokens
            tokens = tokens[:77]
            text.append(tokens)
        max_len = max(len(t) for t in text)
        padded = [t + [pad_id] * (max_len - len(t)) for t in text]
        text = torch.tensor(padded, device=device)
        outputs = self.text_encoder(text)
        return outputs.pooler_output  # (B, 512)

    def forward(self, image_features, caption_ids, text_cache=None, caption_strings=None):
        if text_cache is not None and caption_strings is not None:
            text_features = text_cache.lookup(caption_strings)
        else:
            text_features = self.encode_text(caption_ids)

        image_features = nn.functional.normalize(image_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)

        logits = (image_features @ text_features.T) / self.temperature
        labels = torch.arange(len(logits), device=logits.device)

        loss_i2t = nn.functional.cross_entropy(logits, labels)
        loss_t2i = nn.functional.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2


# ── Transformer Decoder ────────────────────────────────────────

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, pad_token_id,
                 num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_id)
        self.embedding_norm = nn.LayerNorm(embed_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, nhead=num_heads,
            dim_feedforward=hidden_size, dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)
        # Tie output projection to embedding weights (standard Transformer trick:
        # ~halves the vocab-projection parameter count and improves training).
        self.fc.weight = self.embedding.weight

    def forward(self, encoder_features, captions):
        tgt_len = captions.size(1)
        causal_mask = torch.triu(
            torch.full((tgt_len, tgt_len), float("-inf"), device=captions.device), diagonal=1,
        )
        tgt_padding_mask = captions == self.pad_token_id

        embeddings = self.embedding(captions)
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        embeddings = self.pos_encoder(embeddings)

        output = self.transformer_decoder(
            tgt=embeddings, memory=encoder_features,
            tgt_mask=causal_mask, tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.fc(output)


# ── Full Model ─────────────────────────────────────────────────

class CaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, pad_token_id,
                 num_layers=2, num_heads=4, dropout=0.1,
                 backbones=None, use_clip_proj=False):
        super().__init__()
        if backbones is None:
            backbones = ("mobilenet_v3_small",)
        self.encoder = ImageEncoder(
            embed_size, backbones=backbones, use_clip_proj=use_clip_proj,
        )
        self.decoder = TransformerDecoder(
            embed_size, hidden_size, vocab_size, pad_token_id,
            num_layers, num_heads, dropout,
        )

    def forward(self, images, captions, return_clip_features=False):
        if return_clip_features:
            encoder_features, clip_features = self.encoder(images, return_clip_features=True)
        else:
            encoder_features = self.encoder(images)
            clip_features = None

        outputs = self.decoder(encoder_features, captions[:, :-1])

        if return_clip_features:
            return outputs, clip_features
        return outputs


# ── Building a model that exactly matches a saved checkpoint ───
# The ablation study changes the encoder architecture per config
# (backbones list, CLIP head). Instead of guessing from env vars,
# inspect the checkpoint's state dict so `load` can never be built
# with a mismatched architecture (a recurring source of errors).

def build_model_from_checkpoint(checkpoint, tokenizer, device=None,
                                dropout=0.0, num_layers=None, num_heads=None,
                                backbones=None, use_clip_proj=None,
                                hidden_size=None):
    """Build a CaptioningModel whose architecture matches `checkpoint`.

    Backbones / CLIP head / embed size are detected from the state dict
    keys; num_layers / num_heads / hidden_size come from config (fallback
    to explicit args). Raises FileNotFoundError if the checkpoint is missing
    and RuntimeError if the checkpoint is unusable.
    """
    from src.config import EMBED_SIZE, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, device as _default_device

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Train a model first (python -m src.main study), or point STUDY_TAG "
            "at the config you want to run."
        )

    device = device or _default_device
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    # ── Detect encoder architecture from the checkpoint ─────────
    detected_backbones = []
    i = 0
    while True:
        key = f"encoder.branches.{i}.conv.weight"
        if key not in state:
            break
        out_channels = state[key].shape[1]
        name = next(
            (n for n, (_, _, c, _) in BACKBONE_REGISTRY.items() if c == out_channels),
            None,
        )
        if name is None:
            raise RuntimeError(
                f"Cannot identify backbone for branch {i} ({out_channels} channels) "
                f"in {checkpoint}. Known backbones: {list(BACKBONE_REGISTRY)}"
            )
        detected_backbones.append(name)
        i += 1

    if i > 0:
        backbones = tuple(detected_backbones)
        embed_size = state["encoder.branches.0.conv.weight"].shape[0]
    else:
        backbones = backbones or ("mobilenet_v3_small",)
        embed_size = EMBED_SIZE

    if use_clip_proj is None:
        use_clip_proj = "encoder.clip_proj.weight" in state

    model = CaptioningModel(
        embed_size=embed_size,
        hidden_size=hidden_size or HIDDEN_SIZE,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        num_layers=num_layers or NUM_LAYERS,
        num_heads=num_heads or NUM_HEADS,
        dropout=dropout,
        backbones=backbones,
        use_clip_proj=use_clip_proj,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model