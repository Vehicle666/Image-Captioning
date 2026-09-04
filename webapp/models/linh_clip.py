"""MobileNetV3 + CLIP Captioning Model (Linh).

Deployment package from friend 'Linh' (mobilenet_clip_deploy.zip).
Uses an isolated `linh_src` package so it does not clash with the
`src` package of the original MobileNetV3 + CLIP model.
Checkpoint: Friend_image_captioning/extracted/checkpoints/model_best.pth
"""

import os
import sys

import torch
from PIL import Image

from models.base import BaseModel, CaptionResult

# Package root that contains the `linh_src` package
_LINH_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Friend_image_captioning", "extracted")
)
if _LINH_ROOT not in sys.path:
    sys.path.insert(0, _LINH_ROOT)


class LinhClipModel(BaseModel):
    name = "MobileNetV3 + CLIP (Linh)"
    description = "Model của Linh - MobileNetV3-Small encoder + Transformer decoder (256d, 512h, 2 layers)"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None

    def load(self):
        from linh_src.config import EMBED_SIZE, HIDDEN_SIZE, MODEL_BEST_PATH, MODEL_LATEST_PATH, NUM_HEADS, NUM_LAYERS, TOKENIZER_PATH, device
        from linh_src.model import CaptioningModel
        from linh_src.vocabulary import CaptionTokenizer

        print("[linh_clip] loading...", flush=True)
        self._device = device
        self._tokenizer = CaptionTokenizer.load(TOKENIZER_PATH)

        checkpoint = MODEL_BEST_PATH if os.path.exists(MODEL_BEST_PATH) else MODEL_LATEST_PATH
        print(f"[linh_clip] checkpoint: {checkpoint}", flush=True)
        self._model = CaptioningModel(
            embed_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            vocab_size=len(self._tokenizer),
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            dropout=0.0,
        ).to(device)
        self._model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        self._model.eval()
        print(f"[linh_clip] loaded, model={self._model is not None}", flush=True)

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
        from linh_src.config import BEAM_SIZE, device
        from linh_src.dataset import val_transform

        image = Image.open(image_path).convert("RGB")
        image = val_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = self._model.encoder(image)

            start_token = self._tokenizer.sos_token_id
            end_token = self._tokenizer.eos_token_id

            sequences = [[start_token]]
            scores = [0.0]

            for _ in range(50):
                all_candidates = []
                for seq, score in zip(sequences, scores):
                    if seq[-1] == end_token:
                        all_candidates.append((seq, score))
                        continue
                    tgt = torch.tensor([seq]).to(device)
                    logits = self._model.decoder(features, tgt)
                    probs = torch.softmax(logits[0, -1, :], dim=0)
                    topk_probs, topk_indices = torch.topk(probs, BEAM_SIZE)
                    for i in range(BEAM_SIZE):
                        all_candidates.append(
                            (seq + [topk_indices[i].item()], score + torch.log(topk_probs[i]).item())
                        )

                ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
                sequences = [seq for seq, _ in ordered[:BEAM_SIZE]]
                scores = [s for _, s in ordered[:BEAM_SIZE]]

                if all(seq[-1] == end_token for seq in sequences):
                    break

            best = sequences[0][1:]
            if best and best[-1] == end_token:
                best = best[:-1]

        caption = self._tokenizer.decode(best)
        return CaptionResult(caption=caption, model_name=self.name)
