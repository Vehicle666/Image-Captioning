"""MobileNetV3 (+V3 +CLIP) Supervised Captioning Model.

Ablation-study encoder: 1 backbone (MobileNetV3-Small) by default,
optionally fused with a second V3 backbone and CLIP contrastive loss
(see src/config.py: BACKBONE / V3_BACKBONE / USE_V3 / USE_CLIP).
Checkpoint: mobilenet_clip_captioning/checkpoints/<STUDY_TAG>/model_best.pth
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
    name = "MobileNetV3 (+V3 +CLIP)"
    description = "Supervised CNN encoder(s) + Transformer decoder (COCO). Toggle backbones via config."

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None

    def load(self):
        from src.config import (
            device, MODEL_BEST_PATH, MODEL_LATEST_PATH,
        )
        from src.model import build_model_from_checkpoint
        from src.vocabulary import CaptionTokenizer

        import src.config as cfg

        self._device = device
        self._tokenizer = CaptionTokenizer()

        checkpoint = MODEL_BEST_PATH if os.path.exists(MODEL_BEST_PATH) else MODEL_LATEST_PATH
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"No trained model found for this config. "
                f"Looking in {cfg.CHECKPOINT_DIR} ({MODEL_BEST_PATH} / {MODEL_LATEST_PATH}).\n"
                "Train first via `python -m src.main study` (or `train`), then restart the web app."
            )
        self._model = build_model_from_checkpoint(
            checkpoint, self._tokenizer, device=device,
        )

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

        self._ensure_loaded()
        image = Image.open(image_path).convert("RGB")
        image = val_transform(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            ids = generate_caption_beam(self._model, image, self._tokenizer)

        caption = self._tokenizer.decode(ids)
        return CaptionResult(caption=caption, model_name=self.name)