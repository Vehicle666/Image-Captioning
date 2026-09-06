"""Ablation study (application study) runner.

Trains the same supervised pipeline with incrementally richer encoders and
compares the results, answering:

    Config A  baseline_mobilenet : MobileNetV3-Small             (baseline)
    Config B  +v3                : MobileNet + V3 encoder branch (feature fusion)
    Config C  +v3_clip           : above + CLIP contrastive loss

Each config runs in its own subprocess (isolated env) and writes its own
checkpoint folder, so runs never overwrite each other.

Usage:
    python -m src.main study            # train all 3 configs then print table
    python -m src.main study-report     # print table from finished runs only
"""

import json
import os
import re
import subprocess
import sys

from src.config import PROJECT_ROOT, STUDY_RESULTS_PATH

CONFIGS = [
    {
        "tag": "baseline_mobilenet",
        "env": {"BACKBONE": "mobilenet_v3_small", "USE_V3": "0", "USE_CLIP": "0"},
        "desc": "MobileNetV3-Small only (baseline)",
    },
    {
        "tag": "mobilenet_v3",
        "env": {"BACKBONE": "mobilenet_v3_small", "USE_V3": "1", "USE_CLIP": "0"},
        "desc": "MobileNet + V3 encoder branch (+fusion)",
    },
    {
        "tag": "mobilenet_v3_clip",
        "env": {"BACKBONE": "mobilenet_v3_small", "USE_V3": "1", "USE_CLIP": "1"},
        "desc": "MobileNet + V3 + CLIP contrastive loss",
    },
]

_BLEU_LINE_RE = re.compile(
    r"B1=(?P<b1>[0-9.]+)\s+B4=(?P<b4>[0-9.]+)(?:\s+M=(?P<meteor>[0-9.]+))?\s+C=(?P<cider>[0-9.]+)"
)
_COMPLETE_RE = re.compile(r"Training complete\. Best BLEU-4:\s*(?P<b4>[0-9.]+)")


def _read_log(tag):
    path = os.path.join(PROJECT_ROOT, "checkpoints", tag, "training_log.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def summary_for(tag):
    """Return a metrics summary dict for one finished config, or None."""
    log = _read_log(tag)
    if log is None:
        return None

    result = {"completed": False, "epochs": 0, "bleu1": None, "bleu4": None,
              "meteor": None, "cider": None}

    best_epoch = None
    best_line = None
    for line in log.splitlines():
        m = _BLEU_LINE_RE.search(line)
        if m:
            best_epoch = line
            best_line = m
    if best_line:
        result["bleu1"] = float(best_line.group("b1"))
        result["bleu4"] = float(best_line.group("b4"))
        if best_line.group("meteor"):
            result["meteor"] = float(best_line.group("meteor"))
        result["cider"] = float(best_line.group("cider"))
        epoch_match = re.match(r"Epoch (\d+)", best_epoch)
        if epoch_match:
            result["epochs"] = int(epoch_match.group(1))

    if _COMPLETE_RE.search(log):
        result["completed"] = True
        result["best_bleu4"] = float(_COMPLETE_RE.search(log).group("b4"))

    return result


def _load_results():
    if os.path.exists(STUDY_RESULTS_PATH):
        with open(STUDY_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_results(results):
    os.makedirs(os.path.dirname(STUDY_RESULTS_PATH), exist_ok=True)
    with open(STUDY_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_config(cfg, num_epochs=None):
    tag = cfg["tag"]
    print("\n" + "=" * 70)
    print(f"  [{tag}] {cfg['desc']}")
    print("=" * 70)

    env = dict(os.environ)
    env.update({"STUDY_TAG": tag})
    env.update(cfg["env"])

    cmd = [sys.executable, "-m", "src.main", "train"]
    if num_epochs:
        cmd += ["--epochs", str(num_epochs)]

    subprocess.check_call(cmd, cwd=PROJECT_ROOT, env=env)
    log = _read_log(tag)
    return tag, log


def run_all_configs(num_epochs=None):
    results = _load_results()

    for cfg in CONFIGS:
        tag, log = run_config(cfg, num_epochs=num_epochs)
        results[tag] = summary_for(tag)

    _save_results(results)
    print_results(results)
    return results


def print_results(results=None):
    if results is None:
        results = _load_results()

    print("\n" + "=" * 70)
    print("  Ablation Study (application study) - results")
    print("=" * 70)
    header = f"{'Config':<22}{'State':<10}{'BLEU-1':>8}{'BLEU-4':>9}{'METEOR':>9}{'CIDEr':>9}{'BestEp':>8}"
    print(header)
    print("-" * len(header))

    for cfg in CONFIGS:
        tag = cfg["tag"]
        s = results.get(tag)
        if s is None:
            print(f"{tag:<22}{'-':<10}")
            continue
        state = "done" if s.get("completed") else ("partial" if s.get("bleu4") else "running")
        b1 = f"{s['bleu1']:.4f}" if s.get("bleu1") is not None else "-"
        b4 = f"{s['bleu4']:.4f}" if s.get("bleu4") is not None else "-"
        mt = f"{s['meteor']:.4f}" if s.get("meteor") is not None else "-"
        cd = f"{s['cider']:.4f}" if s.get("cider") is not None else "-"
        ep = str(s.get("epochs") or "-")
        print(f"{tag:<22}{state:<10}{b1:>8}{b4:>9}{mt:>9}{cd:>9}{ep:>8}")

    print("=" * 70)
    print("Result file:", STUDY_RESULTS_PATH)
    print('Run "python -m src.main study" to (re)train every config.')
    return results


if __name__ == "__main__":
    run_all_configs()