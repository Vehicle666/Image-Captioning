import sys, json, os
BASE = "E:/Image Captioning/mobilenet_clip_captioning/mobilenet_clip_captioning"
sys.path.insert(0, BASE)
os.chdir(BASE)

from src.vocabulary import Vocabulary
from src.dataset import get_coco_captions
from src.config import MIN_WORD_FREQ, VOCAB_PATH, PSEUDO_CAPTIONS_DIR

# COCO ground truth
train_df, _ = get_coco_captions("train")
all_caps = train_df["caption"].tolist()
print(f"COCO GT: {len(all_caps)} captions")

# COCO BLIP
coco_blip = json.load(open(os.path.join(PSEUDO_CAPTIONS_DIR, "coco_captions.json")))
all_caps.extend(coco_blip.values())
print(f"+ COCO BLIP: {len(coco_blip)}")

# ImageNet BLIP
imgnet = json.load(open(os.path.join(PSEUDO_CAPTIONS_DIR, "imagenet_captions.json")))
all_caps.extend(imgnet.values())
print(f"+ ImageNet BLIP: {len(imgnet)}")

# Build and save
vocab = Vocabulary()
vocab.build_vocab(all_caps, min_freq=MIN_WORD_FREQ)
vocab.save(VOCAB_PATH)
print(f"Saved vocab: {len(vocab)} words -> {VOCAB_PATH}")
