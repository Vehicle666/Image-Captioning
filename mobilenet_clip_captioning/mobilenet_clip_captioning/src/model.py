import math

import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from src.config import (
    CLIP_EMBED_DIM,
    CLIP_TEXT_MODEL,
    CLIP_TEMPERATURE,
    MOBILENET_OUT_CHANNELS,
)


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


# ── MobileNet Encoder ─────────────────────────────────────────

class MobileNetEncoder(nn.Module):
    def __init__(self, embed_size, clip_embed_dim=CLIP_EMBED_DIM):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        # Keep all layers except the classifier
        self.features = mobilenet.features

        # Freeze early layers (0-8), fine-tune late layers (9-12)
        for i, param in enumerate(self.features.parameters()):
            if i < 9:
                param.requires_grad = False

        self.attention = SpatialAttention()
        self.conv = nn.Conv2d(MOBILENET_OUT_CHANNELS, embed_size, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_size)

        # CLIP projection head (training only)
        self.clip_proj = nn.Linear(MOBILENET_OUT_CHANNELS, clip_embed_dim)

    def forward(self, images, return_clip_features=False):
        features = self.features(images)           # (B, 576, 7, 7)
        features = self.attention(features)         # (B, 576, 7, 7)

        clip_features = None
        if return_clip_features:
            pooled = torch.mean(features, dim=[2, 3])  # (B, 576)
            clip_features = self.clip_proj(pooled)      # (B, 512)

        features = self.conv(features)              # (B, embed_size, 7, 7)
        features = self.bn(features)                # (B, embed_size, 7, 7)
        features = features.permute(0, 2, 3, 1)    # (B, 7, 7, embed_size)
        features = features.reshape(features.size(0), -1, features.size(-1))  # (B, 49, embed_size)

        if return_clip_features:
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
                 num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.encoder = MobileNetEncoder(embed_size)
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
