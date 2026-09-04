"""MobileNetV3 + CLIP Transformer Captioning Model.

Uses MobileNetV3-Small encoder with spatial attention,
CLIP contrastive loss during training, and a Transformer decoder.
Checkpoint: mobilenet_clip_captioning/checkpoints/model_best.pth
"""

import os
import sys

import torch
from PIL import Image

from models.base import BaseModel, CaptionResult

# Add the training project to sys.path so we can import its modules
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mobilenet_clip_captioning", "mobilenet_clip_captioning"))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class MobileNetCLIPModel(BaseModel):
    name = "MobileNetV3 + CLIP (WIP)"
    description = "MobileNetV3-Small encoder + Transformer decoder (576ch, 384d, 4 layers)"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None

    def load(self):
        from src.config import device, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, MODEL_BEST_PATH, MODEL_LATEST_PATH
        from src.model import CaptioningModel
        from src.vocabulary import CaptionTokenizer

        self._device = device
        self._tokenizer = CaptionTokenizer()

        checkpoint = MODEL_BEST_PATH if os.path.exists(MODEL_BEST_PATH) else MODEL_LATEST_PATH
        self._model = CaptioningModel(
            embed_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            vocab_size=len(self._tokenizer),
            pad_token_id=self._tokenizer.pad_token_id,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            dropout=0.0,
        ).to(device)
        self._model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        self._model.eval()

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            self._tokenizer = None
        if self._device is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self):
        return self._model is not None

    def get_caption(self, image_path: str) -> CaptionResult:
        from src.dataset import val_transform
        from src.generation import generate_caption_beam

        image = Image.open(image_path).convert("RGB")
        image = val_transform(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            ids = generate_caption_beam(self._model, image, self._tokenizer)

        caption = self._tokenizer.decode(ids)
        return CaptionResult(caption=caption, model_name=self.name)
