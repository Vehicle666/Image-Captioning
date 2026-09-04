import os

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

from linh_src.config import (
    BATCH_SIZE,
    EMBED_SIZE,
    HIDDEN_SIZE,
    MODEL_BEST_PATH,
    MODEL_LATEST_PATH,
    NUM_HEADS,
    NUM_LAYERS,
    TOKENIZER_PATH,
    VAL_IMAGES,
    device,
)
from linh_src.dataset import COCODataset, collate_fn, get_coco_captions, val_transform
from linh_src.model import CaptioningModel
from linh_src.vocabulary import CaptionTokenizer

torch.set_num_threads(min(16, os.cpu_count() or 8))

METEOR_CIDER_TIMEOUT = 30


def _compute_meteor_cider(hypotheses_dict, references_dict):
    import shutil

    from pycocoevalcap.cider.cider import Cider

    gts = {}
    res = {}
    for name in hypotheses_dict:
        if name not in references_dict:
            continue
        gts[name] = [" ".join(r) for r in references_dict[name]]
        res[name] = [" ".join(hypotheses_dict[name])]

    meteor, cider = None, None

    try:
        cider = Cider().compute_score(gts, res)[0]
    except Exception:
        pass

    if shutil.which("java") is not None:
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            meteor = Meteor().compute_score(gts, res)[0]
        except Exception:
            pass

    return meteor, cider


def compute_caption_metrics(hypotheses_dict, references_dict, bleu_only=False):
    smoothing = SmoothingFunction().method1

    refs = [references_dict[n] for n in hypotheses_dict if n in references_dict]
    hyps = [hypotheses_dict[n] for n in hypotheses_dict if n in references_dict]

    bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(refs, hyps, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    meteor, cider = None, None
    if not bleu_only:
        try:
            meteor, cider = _compute_meteor_cider(hypotheses_dict, references_dict)
        except Exception:
            pass

    return {
        "bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4,
        "meteor": meteor, "cider": cider,
    }


def load_model_for_eval(tokenizer):
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
def generate_caption(model, image, tokenizer, max_length=50):
    features = model.encoder(image)
    caption = [tokenizer.sos_token_id]
    for _ in range(max_length):
        tgt = torch.tensor([caption]).to(device)
        logits = model.decoder(features, tgt)
        pred = logits[0, -1, :].argmax().item()
        if pred == tokenizer.eos_token_id:
            break
        caption.append(pred)
    return tokenizer.decode(caption[1:]).split()


def evaluate():
    tokenizer = CaptionTokenizer.load(TOKENIZER_PATH)
    model = load_model_for_eval(tokenizer)

    val_df, val_images_dir = get_coco_captions("val")
    val_loader = torch.utils.data.DataLoader(
        COCODataset(val_df, val_images_dir, tokenizer, transform=val_transform, return_name=True),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    hyps, refs = {}, {}
    for images, captions, texts, names in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        for i, name in enumerate(names):
            if name not in hyps:
                hyps[name] = generate_caption(model, images[i:i+1], tokenizer)
            ref_idx = captions[i].tolist()
            ref_text = tokenizer.decode([j for j in ref_idx if j not in (0, 1, 2)])
            refs.setdefault(name, []).append(ref_text.split())

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
    if metrics["cider"] is not None:
        print(f"  CIDEr:   {metrics['cider']:.4f}")
    print("=" * 50)

    import random
    keys = random.sample(list(hyps.keys()), min(5, len(hyps)))
    print("\nSample predictions:")
    for k in keys:
        print(f"  {k}")
        print(f"    Generated: {hyps[k]}")
        for r in refs[k][:2]:
            print(f"    Reference: {' '.join(r)}")

    return metrics


def main():
    evaluate()


if __name__ == "__main__":
    main()
