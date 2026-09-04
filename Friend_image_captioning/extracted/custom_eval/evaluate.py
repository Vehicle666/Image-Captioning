"""Evaluate the trained captioning model on your own set of images.

Usage (from the project root):

    # Generate captions only (no reference captions needed)
    python -m custom_eval.evaluate --images path/to/your/images

    # Use beam search instead of greedy decoding
    python -m custom_eval.evaluate --images path/to/your/images --beam

    # Compute BLEU / METEOR / CIDEr against reference captions
    python -m custom_eval.evaluate --images path/to/your/images --references path/to/refs.json

Folder layout created here:

    custom_eval/
    ├── evaluate.py              # this script
    ├── references.example.json  # template for reference captions
    ├── images/                  # drop your images here
    └── output/                  # predictions.json + results.txt are written here
"""

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from src.config import (
    BEAM_SIZE,
    EMBED_SIZE,
    HIDDEN_SIZE,
    MODEL_BEST_PATH,
    MODEL_LATEST_PATH,
    NUM_HEADS,
    NUM_LAYERS,
    TOKENIZER_PATH,
    device,
)
from src.dataset import val_transform
from src.model import CaptioningModel
from src.vocabulary import CaptionTokenizer

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "predictions.json")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.txt")


def compute_metrics(hypotheses_dict, references_dict):
    """BLEU-1..4 (nltk) + CIDEr (pycocoevalcap). METEOR is skipped on Windows
    because pycocoevalcap's METEOR hangs on Java pipe I/O there."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    hyps = [hypotheses_dict[n] for n in hypotheses_dict if n in references_dict]
    refs = [references_dict[n] for n in hypotheses_dict if n in references_dict]
    smoothing = SmoothingFunction().method1

    bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(refs, hyps, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    meteor = None
    cider = None
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {n: [" ".join(r) for r in references_dict[n]] for n in hypotheses_dict if n in references_dict}
        res = {n: [" ".join(hypotheses_dict[n])] for n in hypotheses_dict if n in references_dict}
        cider = Cider().compute_score(gts, res)[0]
    except Exception:
        pass

    return {"bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4,
            "meteor": meteor, "cider": cider, "meteor_skipped": os.name == "nt"}


def load_model(tokenizer):
    checkpoint = MODEL_BEST_PATH if os.path.exists(MODEL_BEST_PATH) else MODEL_LATEST_PATH
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint}. Train the model first."
        )
    model = CaptioningModel(
        embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
        vocab_size=len(tokenizer), num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def list_images(images_dir):
    files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not files:
        raise FileNotFoundError(
            f"No images found in {images_dir}. Supported formats: {', '.join(IMAGE_EXTENSIONS)}"
        )
    return files


@torch.no_grad()
def generate_caption(model, image_tensor, tokenizer, max_length=50, beam_size=1):
    features = model.encoder(image_tensor)

    if beam_size <= 1:
        caption = [tokenizer.sos_token_id]
        for _ in range(max_length):
            tgt = torch.tensor([caption]).to(device)
            logits = model.decoder(features, tgt)
            pred = logits[0, -1, :].argmax().item()
            if pred == tokenizer.eos_token_id:
                break
            caption.append(pred)
        return tokenizer.decode(caption[1:])

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
    parser = argparse.ArgumentParser(description="Evaluate on your own images")
    parser.add_argument("--images", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "images"),
        help="Folder containing your images")
    parser.add_argument("--references", type=str, default=None,
        help="JSON file mapping image filenames to reference caption lists. "
             "If omitted, only predictions are generated (no metrics).")
    parser.add_argument("--beam", type=int, nargs="?", const=BEAM_SIZE, default=1,
        help="Beam size for decoding (default: 1 = greedy, e.g. --beam for size 5)")
    parser.add_argument("--max-length", type=int, default=50,
        help="Maximum caption length in tokens")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
        help="Where to write predictions.json and results.txt")
    args = parser.parse_args()

    images_dir = args.images
    if not os.path.isdir(images_dir):
        parser.error(f"Images folder not found: {images_dir}")
    images = list_images(images_dir)

    references = None
    if args.references:
        with open(args.references) as f:
            references = json.load(f)
        missing = [name for name in images if name not in references]
        if missing:
            print(f"Warning: {len(missing)} images have no reference captions:")
            for name in missing[:10]:
                print(f"  {name}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
            print("These will still be captioned but excluded from metrics.")

    tokenizer = CaptionTokenizer.load(TOKENIZER_PATH)
    model = load_model(tokenizer)
    print(f"Device: {device}")
    print(f"Model checkpoint: {MODEL_BEST_PATH if os.path.exists(MODEL_BEST_PATH) else MODEL_LATEST_PATH}")
    print(f"Images: {len(images)} in {images_dir}")
    if args.beam > 1:
        print(f"Decoding: beam search (beam={args.beam})")
    else:
        print("Decoding: greedy")

    hypotheses = {}
    for name in tqdm(images, desc="Captioning"):
        path = os.path.join(images_dir, name)
        image = Image.open(path).convert("RGB")
        image = val_transform(image).unsqueeze(0).to(device)
        hypotheses[name] = generate_caption(
            model, image, tokenizer, max_length=args.max_length, beam_size=args.beam,
        )

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(hypotheses, f, indent=2, ensure_ascii=False)

    report_lines = [
        "=" * 50,
        "  Custom Evaluation Report",
        "=" * 50,
        f"  Images: {len(images)}",
        f"  Decoding: beam={args.beam}",
        f"  Predictions: {os.path.join(args.output, 'predictions.json')}",
    ]

    metrics = None
    if references:
        hyps = {n: h.split() for n, h in hypotheses.items() if n in references}
        refs = {n: [r.split() for r in ref_list] for n, ref_list in references.items() if n in hypotheses}
        metrics = compute_metrics(hyps, refs)
        report_lines += [
            "",
            f"  Evaluated on {len(hyps)} images with references",
            f"  BLEU-1:  {metrics['bleu1']:.4f}",
            f"  BLEU-2:  {metrics['bleu2']:.4f}",
            f"  BLEU-3:  {metrics['bleu3']:.4f}",
            f"  BLEU-4:  {metrics['bleu4']:.4f}",
        ]
        if metrics["meteor"] is not None:
            report_lines.append(f"  METEOR:  {metrics['meteor']:.4f}")
        elif metrics["meteor_skipped"]:
            report_lines.append("  METEOR:  skipped (unstable on Windows)")
        else:
            report_lines.append("  METEOR:  unavailable")
        if metrics["cider"] is not None:
            report_lines.append(f"  CIDEr:   {metrics['cider']:.4f}")

    report_lines += [
        "",
        "  Sample predictions:",
    ]
    for name in images[:5]:
        report_lines.append(f"    {name}")
        report_lines.append(f"      Generated: {hypotheses[name]}")
        if references and name in references:
            for r in references[name][:2]:
                report_lines.append(f"      Reference: {r}")

    report = "\n".join(report_lines)
    print(report)

    with open(os.path.join(args.output, "results.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nResults saved to {os.path.join(args.output, 'results.txt')}")

    return metrics


if __name__ == "__main__":
    main()
