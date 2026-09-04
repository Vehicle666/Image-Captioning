# Image Captioning

Hệ thống tạo chú thích hình ảnh (image captioning) sử dụng kiến trúc lai **MobileNetV3-Small** (CNN encoder) + **Transformer Decoder**, huấn luyện với **CLIP contrastive loss** để căn chỉnh ngữ nghĩa. Hệ thống hỗ trợ bán giám sát với pseudo-captions từ **BLIP**, và tích hợp web app cho inference tương tác với pipeline human-in-the-loop feedback.

## Kiến trúc tổng quát

```
Image (3, 224, 224)
    |
MobileNetV3-Small (pretrained, fine-tune layers 9-12)
    |
Spatial Attention -> Conv -> BatchNorm
    |
+-> CLIP Projection Head [training only]
    |
N spatial tokens x embedding_dim
    |
Transformer Decoder (cross-attention to image features)
    |
Caption tokens -> Beam Search / Greedy Decoding
```

**Training losses:**
- Captioning loss (cross-entropy + label smoothing) -- weight: 1.0
- CLIP contrastive loss (aligns image + text embeddings) -- weight: 0.5

## Cấu trúc thư mục

```
Image Captioning/
├── webapp/                                    # Flask web application (inference + feedback)
│   ├── app.py                                 # Flask routes (/, /caption, /feedback, /verify, /finetune)
│   ├── model_service.py                       # Model service
│   ├── gemini_verify.py                       # Gemini API verify corrections
│   ├── finetune_feedback.py                   # Online fine-tuning từ human feedback
│   ├── models/                                # Plugin-based model registry
│   │   ├── base.py                            # BaseModel abstract + CaptionResult
│   │   ├── registry.py                        # Auto-discovery model registry
│   │   ├── blip.py                            # BLIP (Salesforce/blip-image-captioning-base)
│   │   ├── mobilenet_clip.py                  # MobileNetV3+CLIP (384d, 4 layers)
│   │   └── linh_clip.py                       # MobileNetV3+CLIP (256d, 2 layers)
│   ├── templates/index.html                   # Web UI (dark theme, drag-and-drop)
│   └── requirements.txt
├── mobilenet_clip_captioning/                 # Main training codebase (evolved version)
│   └── mobilenet_clip_captioning/
│       ├── src/                               # Source code
│       │   ├── config.py                      # Hyperparameters (384d/768h/4 layers/6 heads)
│       │   ├── model.py                       # MobileNetEncoder + TransformerDecoder + CLIPContrastiveLoss
│       │   ├── vocabulary.py                  # CLIP tokenizer wrapper (49,409 tokens)
│       │   ├── dataset.py                     # COCO dataset
│       │   ├── train.py                       # Training loop (AMP, gradient accumulation, early stopping)
│       │   ├── evaluate.py                    # BLEU/METEOR/CIDEr evaluation
│       │   ├── predict.py                     # CLI inference (greedy + beam search)
│       │   ├── generation.py                  # Greedy/beam decode logic
│       │   ├── soft_caption.py                # BLIP pseudo-caption generation
│       │   ├── setup.py                       # Dataset download from Kaggle
│       │   └── main.py                        # CLI entry point
│       ├── checkpoints/                       # Model weights, logs, embeddings visualization
│       └── requirements.txt
├── Friend_image_captioning/                   # Model variant của collaborator (Linh)
│   └── extracted/
│       ├── src/ & linh_src/                   # Source code (256d/512h/2 layers/4 heads)
│       ├── custom_eval/                       # Custom evaluation tool
│       ├── checkpoints/                       # model_best.pth, vocab.json, tokenizer.json
│       ├── GUIDE_MOBILENET_CLIP.md            # Hướng dẫn chi tiết kiến trúc
│       └── requirements.txt
├── _rebuild_vocab.py                          # Utility: rebuild vocab từ COCO + ImageNet BLIP captions
└── FLOWCHART.md                               # Sơ đồ kiến trúc Mermaid
```

## Yêu cầu hệ thống

- Python 3.10+
- CUDA GPU (tested on GTX 1650, CUDA 12.6)
- ~4GB VRAM

## Cài đặt

```bash
# Clone repo
git clone https://github.com/Vehicle666/Image-Captioning.git
cd Image-Captioning

# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Cài dependencies
pip install -r mobilenet_clip_captioning/mobilenet_clip_captioning/requirements.txt
pip install -r webapp/requirements.txt
```

## Chạy Web Application

```bash
cd webapp
python app.py
```

Web chạy tại: **http://127.0.0.1:5000**

Public URL (ngrok): **https://snowfall-idealize-uncombed.ngrok-free.dev**

## Các mô hình có sẵn trên Web

| Model | Kiến trúc | Checkpoint |
|-------|-----------|------------|
| BLIP | Salesforce/blip-image-captioning-base | Download từ HuggingFace |
| MobileNetV3 + CLIP (Main) | 384d, 768h, 4 layers, 6 heads | `mobilenet_clip_captioning/checkpoints/` |
| MobileNetV3 + CLIP (Linh) | 256d, 512h, 2 layers, 4 heads | `Friend_image_captioning/extracted/checkpoints/` |

### Web API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/` | GET | Giao diện web |
| `/models` | GET | Danh sách models |
| `/models/switch` | POST | Chuyển model |
| `/caption` | POST | Upload ảnh, nhận caption |
| `/feedback` | POST | Gửi correction cho caption |
| `/feedback` | GET | Xem tất cả feedback |
| `/verify` | POST | Xác minh feedback qua Gemini |
| `/finetune` | POST | Trigger fine-tuning |
| `/finetune/status` | GET | Kiểm tra tiến trình fine-tune |

### Human-in-the-Loop Pipeline

```
User upload ảnh -> AI generate caption -> User sửa caption sai
    -> Gemini API verify correction -> Lưu vào feedback.json
    -> Trigger fine-tuning -> Model tự reload với weights mới
```

## Huấn luyện mô hình (CLI)

```bash
cd mobilenet_clip_captioning/mobilenet_clip_captioning

# 1. Download COCO 2017 từ Kaggle
python -m src.main coco

# 2. Tạo BLIP pseudo-captions (tùy chọn)
python -m src.main soft-captions

# 3. Huấn luyện
python -m src.main train

# 4. Pipeline đầy đủ (BLIP captions -> vocab -> train)
python -m src.main pipeline

# 5. Đánh giá trên validation set
python -m src.main evaluate

# 6. Predict trên ảnh
python -m src.main predict path/to/image.jpg
python -m src.main predict path/to/image.jpg --beam
```

### Training Configuration

| Param | Giá trị | Mô tả |
|-------|---------|-------|
| `EMBED_SIZE` | 384 | Decoder embedding dimension |
| `HIDDEN_SIZE` | 768 | Transformer feed-forward dimension |
| `NUM_LAYERS` | 4 | Transformer decoder layers |
| `NUM_HEADS` | 6 | Attention heads |
| `BATCH_SIZE` | 16 | Training batch size |
| `NUM_EPOCHS` | 600 | Max epochs (early stopping) |
| `TRAIN_IMAGES_PER_EPOCH` | 5500 | Ảnh sampling mỗi epoch |
| `LABEL_SMOOTHING` | 0.1 | Label smoothing |
| `CAPTION_LOSS_WEIGHT` | 1.0 | Cross-entropy weight |
| `CLIP_LOSS_WEIGHT` | 0.5 | CLIP contrastive loss weight |
| `EARLY_STOP_PATIENCE` | 24 | Epochs chờ improvement |

### Kết quả huấn luyện

| Model | BLEU-1 | BLEU-4 | METEOR | CIDEr |
|-------|--------|--------|--------|-------|
| Linh (256d/2 layers) | 0.7356 | **0.2812** | -- | -- |
| Main (384d/4 layers) | -- | 0.1910 | -- | -- |

## Đánh giá tùy chỉnh (Custom Eval)

```bash
cd Friend_image_captioning/extracted
python custom_eval/evaluate.py
```

Đặt ảnh vào `custom_eval/images/`, kết quả xuất ra `custom_eval/output/`.

## Models sử dụng

| Model | Vai trò | Loại |
|-------|---------|------|
| **MobileNetV3-Small** | Image encoder | CNN (pretrained ImageNet, fine-tune layers 9-12) |
| **CLIP** (`openai/clip-vit-base-patch32`) | Contrastive loss + tokenizer | Transformer (frozen) |
| **BLIP** (`Salesforce/blip-image-captioning-base`) | Pseudo-caption generation | Encoder-decoder (offline) |
| **Google Gemini** (`gemini-3.6-flash`) | Feedback verification | External API |

## License

Educational project.
