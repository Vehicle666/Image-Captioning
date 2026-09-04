import argparse
import os
import sys

import torch
from PIL import Image

from linh_src.config import BEAM_SIZE, EMBED_SIZE, HIDDEN_SIZE, MODEL_BEST_PATH, MODEL_LATEST_PATH, NUM_HEADS, NUM_LAYERS, TOKENIZER_PATH, device
from linh_src.dataset import val_transform
from linh_src.model import CaptioningModel
from linh_src.vocabulary import CaptionTokenizer


def load_tokenizer(path=TOKENIZER_PATH):
    return CaptionTokenizer.load(path)


def load_model(tokenizer, checkpoint=None):
    if checkpoint is None:
        checkpoint = MODEL_BEST_PATH if os.path.exists(MODEL_BEST_PATH) else MODEL_LATEST_PATH
    model = CaptioningModel(
        embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
        vocab_size=len(tokenizer), num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def generate_caption(model, image_path, tokenizer, max_length=50):
    image = Image.open(image_path).convert("RGB")
    image = val_transform(image).unsqueeze(0).to(device)

    features = model.encoder(image)
    caption = [tokenizer.sos_token_id]

    for _ in range(max_length):
        tgt = torch.tensor([caption]).to(device)
        logits = model.decoder(features, tgt)
        pred = logits[0, -1, :].argmax().item()
        if pred == tokenizer.eos_token_id:
            break
        caption.append(pred)

    return tokenizer.decode(caption[1:])


@torch.no_grad()
def generate_caption_beam(model, image_path, tokenizer, beam_size=BEAM_SIZE, max_length=50):
    image = Image.open(image_path).convert("RGB")
    image = val_transform(image).unsqueeze(0).to(device)

    features = model.encoder(image)
    start_token = tokenizer.sos_token_id
    end_token = tokenizer.eos_token_id

    sequences = [[start_token]]
    scores = [0.0]

    for _ in range(max_length):
        all_candidates = []
        for seq, score in zip(sequences, scores):
            if seq[-1] == end_token:
                all_candidates.append((seq, score))
                continue

            tgt = torch.tensor([seq]).to(device)
            logits = model.decoder(features, tgt)
            probs = torch.softmax(logits[0, -1, :], dim=0)
            topk_probs, topk_indices = torch.topk(probs, beam_size)

            for i in range(beam_size):
                all_candidates.append(
                    (seq + [topk_indices[i].item()], score + torch.log(topk_probs[i]).item())
                )

        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = [seq for seq, _ in ordered[:beam_size]]
        scores = [s for _, s in ordered[:beam_size]]

        if all(seq[-1] == end_token for seq in sequences):
            break

    best = sequences[0][1:]
    if best and best[-1] == end_token:
        best = best[:-1]
    return tokenizer.decode(best)


def main():
    parser = argparse.ArgumentParser(description="Generate a caption for an image")
    parser.add_argument("image_path", nargs="?", help="Path to image file")
    parser.add_argument("--beam", action="store_true", help="Use beam search")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    model = load_model(tokenizer)

    if not args.image_path:
        print("Usage: python -m src.predict <image_path> [--beam]")
        sys.exit(1)

    if args.beam:
        print(f"Caption: {generate_caption_beam(model, args.image_path, tokenizer)}")
    else:
        print(f"Caption: {generate_caption(model, args.image_path, tokenizer)}")


if __name__ == "__main__":
    main()
