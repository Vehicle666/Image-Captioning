import os

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    EMBED_SIZE,
    ENCODER_BACKBONE,
    HIDDEN_SIZE,
    MODEL_BEST_PATH,
    MODEL_LATEST_PATH,
    NUM_HEADS,
    NUM_LAYERS,
    USE_CLIP,
    USE_V3_ENCODER,
    V3_ENCODER_BACKBONE,
    VAL_IMAGES,
    device,
)
from src.dataset import COCODataset, make_collate_fn, get_coco_captions, val_transform
from src.generation import generate_caption, generate_caption_beam
from src.model import CaptioningModel
from src.vocabulary import CaptionTokenizer

torch.set_num_threads(min(16, os.cpu_count() or 8))


def compute_caption_metrics(hypotheses_dict, references_dict):
    smoothing = SmoothingFunction().method1

    refs = [references_dict[n] for n in hypotheses_dict if n in references_dict]
    hyps = [hypotheses_dict[n] for n in hypotheses_dict if n in references_dict]

    bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(refs, hyps, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    cider = None
    try:
        from pycocoevalcap.cider.cider import Cider

        gts = {}
        res = {}
        for name in hypotheses_dict:
            if name not in references_dict:
                continue
            gts[name] = [" ".join(r) for r in references_dict[name]]
            res[name] = [" ".join(hypotheses_dict[name])]

        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)
        cider = cider_score
    except Exception:
        pass

    return {
        "bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4,
        "meteor": None, "cider": cider,
    }


def make_backbones():
    backbones = [ENCODER_BACKBONE]
    if USE_V3_ENCODER:
        backbones.append(V3_ENCODER_BACKBONE)
    return tuple(backbones)


def load_model_for_eval(tokenizer):
    checkpoint = MODEL_BEST_PATH if os.path.exists(MODEL_BEST_PATH) else MODEL_LATEST_PATH
    model = CaptioningModel(
        embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
        vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.0,
        backbones=make_backbones(), use_clip_proj=USE_CLIP,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def _generate_ids(model, image, tokenizer, beam=False):
    if beam:
        return generate_caption_beam(model, image, tokenizer)
    return generate_caption(model, image, tokenizer)


def evaluate(beam=True):
    tokenizer = CaptionTokenizer()
    model = load_model_for_eval(tokenizer)

    val_df, val_images_dir = get_coco_captions("val")
    val_loader = torch.utils.data.DataLoader(
        COCODataset(val_df, val_images_dir, tokenizer, transform=val_transform, return_name=True),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )

    hyps, refs = {}, {}
    for images, captions, texts, names in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        for i, name in enumerate(names):
            if name not in hyps:
                token_ids = _generate_ids(model, images[i:i+1], tokenizer, beam=beam)
                hyps[name] = tokenizer.decode(token_ids).split()
            refs.setdefault(name, []).append(texts[i].split())

    metrics = compute_caption_metrics(hyps, refs)

    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    print(f"  BLEU-1:  {metrics['bleu1']:.4f}")
    print(f"  BLEU-2:  {metrics['bleu2']:.4f}")
    print(f"  BLEU-3:  {metrics['bleu3']:.4f}")
    print(f"  BLEU-4:  {metrics['bleu4']:.4f}")
    if metrics["meteor"] is not None:
        print(f"  METEOR:  {metrics['meteor']:.4f}")
    print(f"  CIDEr:   {metrics['cider']:.4f}")
    print("=" * 50)

    import random
    keys = random.sample(list(hyps.keys()), min(5, len(hyps)))
    print("\nSample predictions:")
    for k in keys:
        print(f"  {k}")
        print(f"    Generated: {' '.join(hyps[k])}")
        for r in refs[k][:2]:
            print(f"    Reference: {' '.join(r)}")

    return metrics


def main():
    evaluate()


if __name__ == "__main__":
    main()
