import argparse
import json
import os
import random
import sys

from linh_src.config import MODEL_BEST_PATH, MODEL_LATEST_PATH, TOKENIZER_PATH, TRAINING_LOG_PATH, VAL_IMAGES, device


def run_pipeline(skip_captions=False, fresh=False):
    print(f"Device: {device}\n")
    print("=" * 60)
    print("  Full Pipeline: BLIP Captions -> Vocab -> Training")
    print("=" * 60)

    coco_caps_path = os.path.join("checkpoints/pseudo_captions", "coco_captions.json")
    imagenet_caps_path = os.path.join("checkpoints/pseudo_captions", "imagenet_captions.json")

    # ── Fresh start: delete existing artifacts ─────────────────
    if fresh:
        print("\n[ fresh ] Removing existing captions, tokenizer, and checkpoints...")
        for p in [coco_caps_path, imagenet_caps_path, TOKENIZER_PATH,
                  MODEL_BEST_PATH, MODEL_LATEST_PATH, TRAINING_LOG_PATH]:
            if os.path.exists(p):
                os.remove(p)
                print(f"  removed {p}")

    # ── Step 1: Generate soft captions ─────────────────────────
    if not skip_captions:
        from linh_src.soft_caption import generate_coco_captions, generate_imagenet_captions

        os.makedirs("checkpoints/pseudo_captions", exist_ok=True)

        if not os.path.exists(coco_caps_path):
            print("\n[1/3] Generating BLIP captions for COCO...")
            generate_coco_captions(coco_caps_path)
        else:
            print(f"\n[1/3] COCO captions already exist: {coco_caps_path}")

        if not os.path.exists(imagenet_caps_path):
            print("\n[2/3] Generating BLIP captions for ImageNet...")
            generate_imagenet_captions(imagenet_caps_path)
        else:
            print(f"\n[2/3] ImageNet captions already exist: {imagenet_caps_path}")
    else:
        print("\n[1/2] Skipping captions (--skip-captions)")

    # ── Step 2: Train ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Starting Training")
    print("=" * 60 + "\n")

    from linh_src.train import train
    train()

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


def run_expand_vocab(new_data_path):
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    from linh_src.config import CLIP_TEXT_MODEL, EMBED_SIZE, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS
    from linh_src.model import CaptioningModel
    from linh_src.vocabulary import CaptionTokenizer

    if not os.path.exists(TOKENIZER_PATH):
        print(f"No tokenizer found at {TOKENIZER_PATH}. Run training first.")
        return

    tokenizer = CaptionTokenizer.load(TOKENIZER_PATH)
    old_vocab_size = len(tokenizer)

    with open(new_data_path) as f:
        new_data = json.load(f)

    new_captions = list(new_data.values())
    new_tokens = set()
    for caption in new_captions:
        words = caption.lower().split()
        for word in words:
            word_tokens = tokenizer.tokenizer.encode(word).tokens
            for t in word_tokens:
                if t not in tokenizer.tokenizer.get_vocab():
                    new_tokens.add(t)

    if not new_tokens:
        print("No new tokens found in the provided data.")
        return

    print(f"Old vocab size: {old_vocab_size}")
    print(f"Found {len(new_tokens)} new tokens to add")

    tokenizer.add_tokens(list(new_tokens))
    print(f"New vocab size: {len(tokenizer)}")
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer saved to {TOKENIZER_PATH}")

    if os.path.exists(MODEL_BEST_PATH):
        print("\nExpanding model embeddings...")
        clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_TEXT_MODEL)
        clip_text_encoder = CLIPTextModel.from_pretrained(CLIP_TEXT_MODEL)

        model = CaptioningModel(
            embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
            vocab_size=old_vocab_size, num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS, dropout=0.0,
        ).to(device)
        model.load_state_dict(torch.load(MODEL_BEST_PATH, map_location=device, weights_only=True))

        model.expand_vocabulary(list(new_tokens), clip_tokenizer, clip_text_encoder)
        torch.save(model.state_dict(), MODEL_BEST_PATH)
        print(f"Model saved to {MODEL_BEST_PATH}")
        print("Retrain for a few epochs to fine-tune the new embeddings.")
    else:
        print(f"\nNo model found at {MODEL_BEST_PATH}. Skipping model expansion.")


def print_menu():
    print(f"Device: {device}\n")

    print("=" * 60)
    print("  MobileNet + CLIP Image Captioning Pipeline")
    print("=" * 60)

    print("\nStep 1: Setup datasets")
    print("  python -m src.main coco          # Download COCO 2017")
    print("  python -m src.main imagenet       # Download ImageNet 1000 (mini)")

    print("\nStep 2: Run full pipeline (captions + training)")
    print("  python -m src.main pipeline              # BLIP captions -> vocab -> train")
    print("  python -m src.main pipeline --fresh      # Regenerate everything from scratch")
    print("  python -m src.main pipeline --skip-captions  # Skip BLIP caption generation")

    print("\nStep 3: Individual steps")
    print("  python -m src.main soft-captions  # Generate BLIP captions only")
    print("  python -m src.main vocab          # Build tokenizer only")
    print("  python -m src.main train          # Train only")

    print("\nStep 4: Evaluate & predict")
    print("  python -m src.main evaluate")
    print("  python -m src.main predict <path> --beam")

    print("\nStep 5: Expand vocabulary")
    print("  python -m src.main expand-vocab --new-data <path.json>")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        from linh_src.predict import main as pred_main
        sys.argv = sys.argv[1:]
        pred_main()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        from linh_src.evaluate import main as eval_main
        eval_main()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="MobileNet + CLIP Captioning")
    parser.add_argument(
        "command",
        nargs="?",
        default="menu",
        choices=["menu", "pipeline", "soft-captions", "vocab", "train",
                 "coco", "imagenet", "expand-vocab"],
        help="Which step to run",
    )
    parser.add_argument("--skip-captions", action="store_true",
                        help="Skip caption generation in pipeline mode")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing captions, tokenizer, and checkpoints before starting a fresh pipeline")
    parser.add_argument("--new-data", type=str,
                        help="Path to new captions JSON for expand-vocab")
    args = parser.parse_args()

    if args.command == "menu":
        print_menu()

    elif args.command == "pipeline":
        run_pipeline(skip_captions=args.skip_captions, fresh=args.fresh)

    elif args.command == "soft-captions":
        from linh_src.setup import setup_soft_captions
        setup_soft_captions()

    elif args.command == "vocab":
        from linh_src.setup import setup_vocab
        setup_vocab()

    elif args.command == "train":
        from linh_src.train import train
        train()

    elif args.command == "coco":
        from linh_src.setup import setup_coco
        setup_coco()

    elif args.command == "imagenet":
        from linh_src.setup import setup_imagenet
        setup_imagenet()

    elif args.command == "expand-vocab":
        if not args.new_data:
            print("Usage: python -m src.main expand-vocab --new-data path/to/captions.json")
            sys.exit(1)
        run_expand_vocab(args.new_data)

    # Quick demo if no command and model exists
    if args.command == "menu" and os.path.exists(MODEL_BEST_PATH):
        from linh_src.predict import load_tokenizer, load_model, generate_caption

        tokenizer = load_tokenizer()
        model = load_model(tokenizer)
        val_dir = VAL_IMAGES
        if os.path.exists(val_dir):
            images = [f for f in os.listdir(val_dir) if f.endswith(".jpg")]
            if images:
                sample = os.path.join(val_dir, random.choice(images))
                print(f"\nSample: {sample}")
                print(f"Caption: {generate_caption(model, sample, tokenizer)}")
