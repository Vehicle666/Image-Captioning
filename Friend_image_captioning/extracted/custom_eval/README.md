# Custom Evaluation

Run the trained model on **your own images** and get BLEU / METEOR / CIDEr scores when
you provide reference captions.

## Layout

```
custom_eval/
├── evaluate.py              # the evaluation script
├── references.example.json  # template showing the reference format
├── images/                  # ← drop your images here (jpg, jpeg, png, bmp, webp)
└── output/                  # predictions.json + results.txt are written here
```

## Usage

Run from the project root (`mobilenet_clip_captioning/`):

### 1. Captions only (no references needed)

```bash
python -m custom_eval.evaluate
```

Uses the default `custom_eval/images/` folder. Or point at another folder:

```bash
python -m custom_eval.evaluate --images path/to/my/images
```

### 2. Beam search instead of greedy decoding

```bash
python -m custom_eval.evaluate --beam
python -m custom_eval.evaluate --beam 7        # custom beam width
```

### 3. With reference captions (get BLEU-1..4, METEOR, CIDEr)

Create a JSON file mapping each image filename to a list of human reference captions:

```json
{
  "photo_1.jpg": ["a cat sitting on a sofa", "a fluffy cat resting on the couch"],
  "photo_2.jpg": ["a dog running in the park"]
}
```

Then:

```bash
python -m custom_eval.evaluate --references path/to/references.json
```

Images without references are still captioned but are excluded from the metrics.

### All options

| Flag | Default | Description |
|---|---|---|
| `--images` | `custom_eval/images/` | Folder of images to caption |
| `--references` | none | JSON file: `{filename: [caption, ...]}` |
| `--beam` | 1 (greedy) | Beam search width, `--beam` alone = 5 |
| `--max-length` | 50 | Max generated caption length (tokens) |
| `--output` | `custom_eval/output/` | Where `predictions.json` and `results.txt` are saved |

## Outputs

- `output/predictions.json` — every image filename → generated caption
- `output/results.txt` — a human-readable report with metrics (if references were given)

## Notes

- Uses `checkpoints/model_best.pth` (or `model_latest.pth` if no best model).
- METEOR is skipped on Windows (pycocoevalcap's METEOR hangs on Java pipe I/O there);
  it runs on Linux/macOS if Java is available.
