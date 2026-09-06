# AI Captioning Project Flowchart

## Ablation study (application study)

Three supervised configs, same pipeline, same COCO data — only the encoder changes:

```
Config A  baseline_mobilenet  : MobileNetV3-Small only
Config B  mobilenet_v3        : MobileNet + V3 encoder branch (feature fusion)
Config C  mobilenet_v3_clip   : MobileNet + V3 + CLIP contrastive loss
```

```mermaid
flowchart TB
    subgraph Study["🧪 Ablation Study (study.py)"]
        direction LR
        A["Config A<br/>baseline_mobilenet<br/>MobileNet only"] --> B["Config B<br/>mobilenet_v3<br/>MobileNet + V3"]
        B --> C["Config C<br/>mobilenet_v3_clip<br/>+ CLIP loss"]
        C --> COMPARE["Compare BLEU/CIDEr<br/>checkpoints/study_results.json"]
    end

    subgraph Data["📦 Data (dataset.py)"]
        COCO["COCO 2017<br/>(train2017 + val2017)"] --> ANN["COCO Annotations<br/>captions_*.json"]
        ANN --> DLD["DataLoader<br/>- RandomResizedCrop(224)<br/>- Collate (pad to PAD)<br/>- LengthBucketedBatchSampler"]
    end

    subgraph Model["🧠 Model (model.py)"]
        direction TB
        BR_A["Backbone A: MobileNetV3-Small<br/>features + SpatialAttention<br/>Conv1x1 -> embed_size"] --> FUSE["Feature Fusion<br/>concat -> Conv1x1 -> BN<br/>(only when USE_V3=1)"]
        BR_B["Backbone B (optional, +V3):<br/>MobileNetV3-Large / EfficientNet<br/>features + SpatialAttention<br/>Conv1x1 -> embed_size"] --> FUSE
        FUSE --> TOKENS["49 spatial tokens<br/>(B, 49, embed_size)"]
        TOKENS --> DEC["TransformerDecoder<br/>- cross-attention over tokens<br/>- causal + padding mask"]
        EMB["Embedding + LayerNorm<br/>+ Dropout + PosEncoding"] --> DEC
        DEC --> OUTPUT["Linear(embed -> vocab)<br/>tied to embedding"]
        BR_A --> POOL["GlobalPool concat<br/>(only CLIP branch)"]
        BR_B --> POOL
        POOL --> CLIP_HEAD["CLIP Projection Head<br/>-> 512d (training only,<br/>when USE_CLIP=1)"]
    end

    subgraph Train["⚙️ Train Loop (train.py)"]
        direction TB
        STEP["Training Step"] --> FW["Forward (autocast)"]
        FW --> CAP["Caption CrossEntropy<br/>weight 1.0 (always)"]
        FW --> CLC["CLIP Contrastive Loss<br/>weight 0.5 (if USE_CLIP)"]
        CAP --> TOT["Total Loss"]
        CLC --> TOT
        TOT --> BW["Backward + GradClip(5.0)"]
        BW --> OPT["Adam<br/>Encoder 1e-5 / Decoder 1e-4"]
        VAL["Validation (every N epochs)<br/>caption CE on val split"] --> SCHED["ReduceLROnPlateau"]
        VAL --> BLEU["BLEU-4 eval on subset<br/>-> save model_best.pth"]
    end

    subgraph Infer["🔮 Inference (predict.py / generation.py)"]
        IMG["Input Image"] --> ENC["Encoder(s) — same arch, eval"]
        ENC --> TOK_I["49 spatial tokens"]
        TOK_I --> DEC_I["Autoregressive decode"]
        DEC_I --> GREEDY["Greedy argmax"]
        DEC_I --> BEAM["Beam Search (beam=5)"]
    end

    subgraph Eval["📊 Evaluation (evaluate.py)"]
        VSET["COCO Val set"] --> GEN["Generate captions"]
        GEN --> METRICS["BLEU-1/2/3/4 (NLTK)<br/>CIDEr (pycocoevalcap)"]
    end

    Data --> Train
    Model --> Train
    Train --> Study
    Study --> Eval
    Study --> Infer
```

## CLI commands

```
python -m src.main coco            # download COCO 2017
python -m src.main study           # train all 3 configs, then compare
python -m src.main study-report    # print comparison of finished runs
python -m src.main train           # train default (current) config
python -m src.main evaluate        # BLEU/CIDEr on COCO val
python -m src.main predict <img> --beam
```

Env overrides to test a single config:

```
STUDY_TAG=baseline_mobilenet  USE_V3=0  USE_CLIP=0  -> Config A
STUDY_TAG=mobilenet_v3        USE_V3=1  USE_CLIP=0  -> Config B
STUDY_TAG=mobilenet_v3_clip   USE_V3=1  USE_CLIP=1  -> Config C
```

## Loss formula

```
total_loss = 1.0 × CrossEntropy(decoder_logits, caption_tokens)     [always]
           + 0.5 × CLIPContrastiveLoss(image_feats, text_feats)     [USE_CLIP=1]

CLIPContrastiveLoss = [CE(img @ text.T / τ) + CE(text @ img.T / τ)] / 2,  τ = 0.07
```