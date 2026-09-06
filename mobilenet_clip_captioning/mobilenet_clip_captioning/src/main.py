import argparse
import os
import random
import sys

from src.config import VAL_IMAGES, device


def run_pipeline(num_epochs=None):
    print(f"Device: {device}\n")
    print("=" * 60)
    print("  Supervised Captioning Pipeline (COCO)")
    print("=" * 60)

    # ── Train ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Starting Training")
    print("=" * 60 + "\n")

    from src.train import train
    train(num_epochs=num_epochs)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


def run_study():
    """Ablation study: MobileNet baseline → +V3 encoder → +CLIP."""
    from src.study import run_all_configs
    run_all_configs()


def print_menu():
    print(f"Device: {device}\n")

    print("=" * 60)
    print("  Supervised Image Captioning (COCO)")
    print("=" * 60)

    print("\nAblation study (application study): compare encoder variants")
    print("  python -m src.main study             # Train ALL configs & compare")
    print("  python -m src.main study-report      # Print comparison of finished runs")

    print("\nSingle-config training (env overrides):")
    print("  python -m src.main train                                     # default (current config)")
    print("  python -m src.main train --epochs 0                          # train until early stop")
    print("  set STUDY_TAG=baseline_mobilenet && set USE_V3=0 && set USE_CLIP=0 ^\n"
          "      && python -m src.main train        # Config A: MobileNet only")
    print("  set STUDY_TAG=mobilenet_v3 && set USE_V3=1 && set USE_CLIP=0 ^\n"
          "      && python -m src.main train        # Config B: + V3 encoder")
    print("  set STUDY_TAG=mobilenet_v3_clip && set USE_V3=1 && set USE_CLIP=1 ^\n"
          "      && python -m src.main train        # Config C: + V3 + CLIP")

    print("\nSetup / evaluate / predict:")
    print("  python -m src.main coco                # Download COCO 2017")
    print("  python -m src.main evaluate            # Evaluate on COCO val set")
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

    if len(sys.argv) > 1 and sys.argv[1] == "study-report":
        from src.study import print_results
        print_results()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Supervised Captioning (COCO)")
    parser.add_argument(
        "command",
        nargs="?",
        default="menu",
        choices=["menu", "study", "train", "coco"],
        help="Which step to run",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override NUM_EPOCHS (0 = unlimited, until early stop)")
    args = parser.parse_args()

    epochs = None if args.epochs is None else (99999 if args.epochs == 0 else args.epochs)

    if args.command == "menu":
        print_menu()

    elif args.command == "study":
        run_study()

    elif args.command == "train":
        from src.train import train
        train(num_epochs=epochs)

    elif args.command == "coco":
        from src.setup import setup_coco
        setup_coco()

    # Quick demo if no command and model exists
    if args.command == "menu" and (os.path.exists("checkpoints/model_best.pth") or os.path.exists("checkpoints/model_latest.pth")):
        from src.predict import load_tokenizer, load_model, _decode

        tokenizer = load_tokenizer()
        model = load_model(tokenizer)
        val_dir = VAL_IMAGES
        if os.path.exists(val_dir):
            images = [f for f in os.listdir(val_dir) if f.endswith(".jpg")]
            if images:
                sample = os.path.join(val_dir, random.choice(images))
                print(f"\nSample: {sample}")
                print(f"Caption: {_decode(sample, model, tokenizer)}")