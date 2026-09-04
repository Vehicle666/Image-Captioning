# AI Captioning Project Flowchart

```mermaid
flowchart TB
    subgraph Data["📦 Data Preparation"]
        COCO["COCO 2017 Dataset<br/>(118K train + 5K val images)"] --> ANN["COCO Annotations<br/>(captions_train2017.json)"]
        COCO --> BLIP_GEN["BLIP Pseudo-Caption Generation<br/>(soft_caption.py)<br/>Model: Salesforce/blip-image-captioning-base"]
        ANN --> VOCAB["Vocabulary Builder<br/>(vocabulary.py)<br/>min_freq=3 → 18,672 tokens"]
        BLIP_GEN --> VOCAB
        VOCAB --> VOCAB_JSON["checkpoints/vocab.json"]
    end

    subgraph Loader["📥 Data Loading (dataset.py)"]
        COCO_DATASET["COCODataset<br/>- Loads image + caption pairs<br/>- Applies augmentations<br/>- Numericalizes captions"] --> COLLATE["Collate Function<br/>- Pads sequences with &lt;PAD&gt;<br/>- Stacks images"]
        TRANS["Train Transforms:<br/>RandomResizedCrop(224)<br/>HorizontalFlip, ColorJitter<br/>Normalize<br/><br/>Val Transforms:<br/>Resize(256), CenterCrop(224)<br/>Normalize"]
        COCO_DATASET --> TRANS
        COLLATE --> DATALOADER["DataLoader<br/>(batch_size=16)"]
    end

    subgraph Model["🧠 Model Architecture (model.py)"]
        direction TB
        ENC["Encoder: MobileNetV3-Small<br/>- Pretrained, layers 0-8 frozen<br/>- Layers 9-12 fine-tuned<br/>- Output: 576 channels"] --> ATT["SpatialAttention<br/>(7x7 conv, sigmoid gate)"]
        ATT --> PROJ["Conv2D(576→256, 1×1)<br/>+ BatchNorm2d"]
        PROJ --> TOKENS["49 spatial tokens<br/>(B, 49, 256)"]
        
        TOKENS --> DEC["TransformerDecoder<br/>- 2 layers, 4 heads<br/>- FF=512, dropout=0.1<br/>- Causal + padding mask"]
        EMB["Embedding(18672→256)<br/>+ LayerNorm + Dropout<br/>+ PositionalEncoding"] --> DEC
        DEC --> OUTPUT["Linear(256→18672)<br/>→ vocab logits"]
        
        ENC --> CLIP_HEAD["CLIP Projection Head<br/>AdaptiveAvgPool<br/>Linear(576→512)"]
        CLIP_HEAD --> CLIP_FEAT["CLIP Image Features<br/>(512-dim)"]
    end

    subgraph TrainLoop["⚙️ Training Loop (train.py)"]
        direction TB
        TRAIN_STEP["Training Step"] --> FORWARD["Forward Pass (autocast)<br/>Image → Encoder → Decoder"]
        FORWARD --> CAP_LOSS["Caption Cross-Entropy Loss<br/>Weight: 1.0<br/>(decoder logits vs ground truth)"]
        FORWARD --> CLIP_LOSS["CLIP Contrastive Loss<br/>Weight: 0.5<br/>(image feat vs cached text feat)"]
        CAP_LOSS --> TOTAL["Total Loss = cap_loss + 0.5 × clip_loss"]
        CLIP_LOSS --> TOTAL
        TOTAL --> BACKWARD["Backward (GradScaler)<br/>Gradient Clipping (5.0)"]
        BACKWARD --> OPT["Optimizer: Adam<br/>Encoder LR: 1e-5<br/>Decoder LR: 1e-4<br/>ReduceLROnPlateau scheduler"]
        
        VAL_STEP["Validation Step<br/>(caption CE loss only)"] --> VAL_LOSS["Validation Loss"]
        VAL_LOSS --> CHECKPOINT["Checkpointing<br/>- model_latest.pth (every epoch)<br/>- model_best.pth (BLEU-4 ↑)"]
        VAL_LOSS --> EARLY_STOP["Early Stopping<br/>patience=5, min_delta=1e-4"]
        
        CLIP_CACHE["CLIPTextCache<br/>Pre-encode 619K unique captions<br/>via frozen CLIP text model"] -.-> CLIP_LOSS
    end

    subgraph Infer["🔮 Inference (predict.py)"]
        IMG["Input Image"] --> ENC_INF["MobileNetEncoder<br/>(same architecture, eval mode)"]
        ENC_INF --> TOKENS_INF["49 spatial tokens"]
        TOKENS_INF --> DEC_INF["Autoregressive Decoding"]
        DEC_INF --> GREEDY["Greedy:<br/>argmax at each step"]
        DEC_INF --> BEAM["Beam Search (beam=5):<br/>top-K sequences"]
        GREEDY --> CAPTION["Generated Caption"]
        BEAM --> CAPTION
    end

    subgraph Eval["📊 Evaluation (evaluate.py)"]
        VAL_SET["COCO Validation Set<br/>(5K images, 5 refs each)"] --> GEN_CAP["Generate Captions<br/>(greedy)"]
        GEN_CAP --> METRICS["Metrics:<br/>BLEU-1/2/3/4 (NLTK)<br/>METEOR (pycocoevalcap)<br/>CIDEr (pycocoevalcap)"]
    end

    Data --> Loader
    Loader --> TrainLoop
    Model --> TrainLoop
    TrainLoop --> Checkpoints["checkpoints/<br/>model_latest.pth<br/>model_best.pth"]
    Checkpoints --> Infer
    Checkpoints --> Eval
```

## Pipeline Overview

```
main.py CLI:
  pipeline          → soft-captions → train
  pipeline --skip-captions → train only
  soft-captions     → BLIP pseudo-captions only
  vocab             → build vocabulary only
  train             → train model only
  evaluate          → eval on COCO val set
  predict <image>   → generate caption for image
```

## Loss Formula

```
total_loss = 1.0 × CrossEntropy(decoder_logits, caption_tokens)
           + 0.5 × CLIPContrastiveLoss(image_features, text_features)

CLIPContrastiveLoss = [CE(img @ text.T / τ, labels) + CE(text @ img.T / τ, labels)] / 2
                    where τ = 0.07
```
