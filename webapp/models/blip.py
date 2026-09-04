"""Salesforce BLIP Image Captioning Model.

Pre-trained BLIP model from Hugging Face Transformers.
Requires: pip install transformers
"""

import os
import sys

import torch
from PIL import Image

from models.base import BaseModel, CaptionResult


class BLIPModel(BaseModel):
    name = "BLIP (Salesforce)"
    description = "Salesforce/blip-image-captioning-base (pre-trained, no fine-tuning)"

    def __init__(self):
        self._processor = None
        self._model = None
        self._device = None

    def load(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "Salesforce/blip-image-captioning-base"
        self._processor = BlipProcessor.from_pretrained(model_id)
        self._model = BlipForConditionalGeneration.from_pretrained(model_id).to(self._device)

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._device and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self):
        return self._model is not None

    def get_caption(self, image_path: str) -> CaptionResult:
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)

        with torch.no_grad():
            output = self._model.generate(**inputs, max_length=64, num_beams=5)

        caption = self._processor.decode(output[0], skip_special_tokens=True)
        return CaptionResult(caption=caption, model_name=self.name)
