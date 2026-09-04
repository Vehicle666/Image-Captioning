import argparse
import os
import random
import sys

from src.config import VAL_IMAGES, device


def run_pipeline(blip_captions=False, num_epochs=None):
    print(f"Device: {device}\n")
    print("=" * 60)
    print("  MobileNet + CLIP Captioning Pipeline")
    print("=" * 60)

    if blip_captions:
        from src.soft_caption import generate_coco_captions

        os.makedirs(os.path.join("checkpoints", "pseudo_captions"), exist_ok=True)
        coco_caps_path = os.path.join("checkpoints", "pseudo_captions", "coco_captions.json")

        if not os.path.exists(coco_caps_path):
            print("\nGenerating BLIP captions for COCO (~1-2 hours)...")
            generate_coco_captions(coco_caps_path)
        else:
            print("\nBLIP captions already exist, skipping generation.")

    # ── Train ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Starting Training")
    print("=" * 60 + "\n")

    from src.train import train
    train(num_epochs=num_epochs)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


def print_menu():
    print(f"Device: {device}\n")

    print("=" * 60)
    print("  MobileNet + CLIP Image Captioning (COCO only)")
    print("=" * 60)

    print("\nStep 1: Setup dataset")
    print("  python -m src.main coco                # Download COCO 2017")

    print("\nStep 2: Run training pipeline")
    print("  python -m src.main pipeline                          # Train only (default)")
    print("  python -m src.main pipeline --blip-captions            # Generate BLIP captions first")
    print("  python -m src.main pipeline --epochs 0                 # Train indefinitely (until early stop)")

    print("\nStep 3: Individual steps")
    print("  python -m src.main soft-captions    # Generate BLIP captions only")
    print("  python -m src.main train            # Train only")
    print("  python -m src.main train --epochs 0 # Train indefinitely")

    print("\nStep 4: Evaluate & predict")
    print("  python -m src.main evaluate")
    print("  python -m src.main predict <path> --beam")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        from src.predict import main as pred_main
        sys.argv = sys.argv[1:]
        pred_main()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        from src.evaluate import main as eval_main
        eval_main()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="MobileNet + CLIP Captioning (COCO)")
    parser.add_argument(
        "command",
        nargs="?",
        default="menu",
        choices=["menu", "pipeline", "soft-captions", "train", "coco"],
        help="Which step to run",
    )
    parser.add_argument("--blip-captions", action="store_true",
                        help="Also generate BLIP pseudo-captions before training")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override NUM_EPOCHS (0 = unlimited, until early stop)")
    args = parser.parse_args()

    epochs = None if args.epochs is None else (99999 if args.epochs == 0 else args.epochs)

    if args.command == "menu":
        print_menu()

    elif args.command == "pipeline":
        run_pipeline(blip_captions=args.blip_captions, num_epochs=epochs)

    elif args.command == "soft-captions":
        from src.setup import setup_soft_captions
        setup_soft_captions()

    elif args.command == "train":
        from src.train import train
        train(num_epochs=epochs)

    elif args.command == "coco":
        from src.setup import setup_coco
        setup_coco()

    # Quick demo if no command and model exists
    if args.command == "menu" and (os.path.exists("checkpoints/model_best.pth") or os.path.exists("checkpoints/model_latest.pth")):
        from src.predict import load_tokenizer, load_model, generate_caption
        from src.dataset import val_transform

        tokenizer = load_tokenizer()
        model = load_model(tokenizer)
        val_dir = VAL_IMAGES
        if os.path.exists(val_dir):
            images = [f for f in os.listdir(val_dir) if f.endswith(".jpg")]
            if images:
                sample = os.path.join(val_dir, random.choice(images))
                print(f"\nSample: {sample}")
                print(f"Caption: {generate_caption(model, sample, tokenizer)}")
