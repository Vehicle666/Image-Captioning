# Ablation Study: MobileNet → +V3 → +CLIP (Image Captioning)

Hệ thống tạo chú thích hình ảnh **supervised** trên COCO 2017 với encoder CNN có thể ghép nhiều backbone, và được dùng như một **application study (ablation study)**: train cùng một pipeline với từng cấu hình encoder, so sánh xem *thêm mỗi thành phần có thực sự tốt hơn hay không*.

## 3 cấu hình ablation

| Config | Backbone(s) | CLIP loss | Câu hỏi |
|--------|-------------|-----------|---------|
| **A** `baseline_mobilenet` | MobileNetV3-Small | ✗ | Baseline |
| **B** `mobilenet_v3` | MobileNetV3-Small + **MobileNetV3-Large** (`+V3`) | ✗ | Thêm encoder thứ 2 có tốt hơn? |
| **C** `mobilenet_v3_clip` | MobileNet + V3 | ✓ | Thêm CLIP contrastive loss có tốt hơn nữa? |

Encoder fusion: mỗi backbone → SpatialAttention → Conv1x1 → tổng hợp bằng channel concat + Conv1x1 fusion về `EMBED_SIZE`. CLIP projection head (chỉ khi `USE_CLIP=1`) dùng global-pool đặc trưng 2 branch.

## Kiến trúc chung

```
Image (3, 224, 224)
    |
    +-> MobileNetV3-Small (pretrained, fine-tune late layers) ----+
    |                                                              |
    +-> V3 LARGE: MobileNetV3-Large/EfficientNet (optional) ------+---> concat -> Conv1x1 fusion
    |                                                              |
    +-> CLIP Projection Head [only when USE_CLIP=1, training only]
    |
49 spatial tokens x EMBED_SIZE
    |
Transformer Decoder (cross-attention to image features)
    |
Caption tokens -> Beam Search / Greedy Decoding
```

**Losses:**
- Captioning cross-entropy + label smoothing -- weight: 1.0 (luôn bật, supervised)
- CLIP contrastive loss -- weight: 0.5 (chỉ khi `USE_CLIP=1`)

## Cấu trúc thư mục

```
Image Captioning/
├── webapp/                                    # Flask web app (inference + feedback)
│   ├── app.py                                 # Routes: /, /caption, /feedback, /verify, /finetune
│   ├── models/mobilenet_clip.py               # Model plugin (đọc cấu hình src/config.py)
│   └── templates/index.html                   # Web UI (dark theme, drag-and-drop)
├── mobilenet_clip_captioning/mobilenet_clip_captioning/
│   ├── src/
│   │   ├── config.py                          # Hyperparameters + ablation toggles (env)
│   │   ├── model.py                           # ImageEncoder (multi-backbone) + TransformerDecoder + CLIP
│   │   ├── train.py                           # Training loop (supervised, AMP, early stopping)
│   │   ├── study.py                           # Chạy & so sánh 3 cấu hình ablation
│   │   ├── evaluate.py / predict.py           # BLEU/METEOR/CIDEr + CLI inference
│   │   ├── vocabulary.py                      # CLIP tokenizer wrapper (49,409 tokens)
│   │   └── main.py                            # CLI entry point
│   └── checkpoints/<STUDY_TAG>/               # Mỗi config một thư mục riêng
└── FLOWCHART.md                               # Sơ đồ kiến trúc Mermaid
```

## Cài đặt & chạy web app

```bash
git clone https://github.com/Vehicle666/Image-Captioning.git
cd Image-Captioning
python -m venv .venv
.venv\Scripts\activate
pip install -r mobilenet_clip_captioning/mobilenet_clip_captioning/requirements.txt
pip install -r webapp/requirements.txt

cd webapp
python app.py            # http://127.0.0.1:5000
```

## Chạy ablation study

```bash
cd mobilenet_clip_captioning/mobilenet_clip_captioning

# 1. Download COCO 2017 từ Kaggle
python -m src.main coco

# 2. Train cả 3 cấu hình và so sánh (mỗi config checkpoint riêng)
python -m src.main study

# 3. Chỉ xem bảng kết quả của các run đã xong
python -m src.main study-report
```

Bảng kết quả được lưu ở `checkpoints/study_results.json`.

### Train đơn lẻ một cấu hình

```bash
# Config A: MobileNet-only baseline
set STUDY_TAG=baseline_mobilenet && set USE_V3=0 && set USE_CLIP=0 && python -m src.main train

# Config B: +V3 encoder
set STUDY_TAG=mobilenet_v3 && set USE_V3=1 && set USE_CLIP=0 && python -m src.main train

# Config C: +V3 + CLIP
set STUDY_TAG=mobilenet_v3_clip && set USE_V3=1 && set USE_CLIP=1 && python -m src.main train
```

### Train config A bằng venv (máy này)

```powershell
# Kích hoạt venv (nếu chưa)
& "E:\Image Captioning\.venv\Scripts\Activate.ps1"

# Vào thư mục project captioning
cd "E:\Image Captioning\mobilenet_clip_captioning\mobilenet_clip_captioning"

# Chạy config A: baseline_mobilenet (600 epochs)
$env:STUDY_TAG="baseline_mobilenet"; $env:USE_V3="0"; $env:USE_CLIP="0"; python -m src.main train
```

Nếu chạy từ **cmd.exe** thay vì PowerShell:

```cmd
cd E:\Image Captioning\mobilenet_clip_captioning\mobilenet_clip_captioning
set STUDY_TAG=baseline_mobilenet && set USE_V3=0 && set USE_CLIP=0 && python -m src.main train
```

> Nếu bị ngắt giữa chừng, chạy lại **cùng lệnh** để resume từ `resume_state.pth`.
> Log theo dõi ở `checkpoints/baseline_mobilenet/training_log.txt`.

### Chia việc train với bạn (B & C trên máy khác)

Máy này train **Config A**. Bạn của bạn train **Config B & C** với đúng thông số dưới đây (giống hệt `src/study.py`):

| Config | `STUDY_TAG` | `BACKBONE` | `V3_BACKBONE` | `USE_V3` | `USE_CLIP` | Lệnh (cmd) |
|--------|-------------|------------|-----------------|----------|-----------|------------|
| **B** | `mobilenet_v3` | `mobilenet_v3_small` | `mobilenet_v3_large` | `1` | `0` | `set STUDY_TAG=mobilenet_v3 && set USE_V3=1 && set USE_CLIP=0 && python -m src.main train` |
| **C** | `mobilenet_v3_clip` | `mobilenet_v3_small` | `mobilenet_v3_large` | `1` | `1` | `set STUDY_TAG=mobilenet_v3_clip && set USE_V3=1 && set USE_CLIP=1 && python -m src.main train` |

Lưu ý cho bạn train B/C:
- Dataset COCO 2017 phải nằm đúng đường dẫn trong `src/config.py` (`DATASET_DIR`) hoặc sửa cho khớp máy của bạn.
- Sau khi train xong, gửi lại thư mục `checkpoints/mobilenet_v3/` và `checkpoints/mobilenet_v3_clip/` (hoặc cả 2 tệp `model_best.pth`/`model_latest.pth` + file `training_log.txt`) để gộp vào bảng ablation.
- Kết quả của cả 3 config sẽ được gom vào `checkpoints/study_results.json` khi gộp dữ liệu.

**Ablation toggles** (đọc từ biến môi trường, có default trong `src/config.py`):

| Env | Default | Mô tả |
|-----|---------|-------|
| `BACKBONE` | `mobilenet_v3_small` | Backbone chính |
| `V3_BACKBONE` | `mobilenet_v3_large` | Backbone thêm (có thể đổi `efficientnet_b0`) |
| `USE_V3` | `1` | Bật encoder branch thứ 2 (feature fusion) |
| `USE_CLIP` | `1` | Bật CLIP contrastive loss + projection head |
| `STUDY_TAG` | (trống) | Tên thư mục checkpoint; để trống = default config |

**K-fold fold rotation** (không còn lấy mẫu ngẫu nhiên mỗi epoch):
- `K_FOLD` (default `20`): toàn bộ train2017 được chia MỘT LẦN thành `K_FOLD` fold cố định (sắp xếp theo tên file, xen kẽ). Epoch `e` train trên fold `(e % K_FOLD)`. Sau `K_FOLD` epoch, toàn bộ dataset được thấy đúng một lần — loại bỏ variance do `random.sample` mỗi epoch gây ra.
- `SEED` (default `42`): gieo seed cho random/numpy/torch và subset BLEU val, đảm bảo kết quả tái lập được giữa các config trong ablation.
- Thay đổi trực tiếp trong `src/config.py` (không phải biến môi trường).

## Web API

| Endpoint | Phương thức | Mô tả |
|----------|-------------|-------|
| `/` | GET | Giao diện web |
| `/models` | GET | Danh sách model |
| `/models/switch` | POST | Chuyển model |
| `/caption` | POST | Upload ảnh → caption |
| `/feedback` | POST/GET | Gửi/xem correction |
| `/verify` | POST | Xác minh correction qua Gemini |
| `/finetune` | POST | Fine-tune từ feedback |
| `/finetune/status` | GET | Trạng thái fine-tune |

## Kết quả ablation (cập nhật sau khi chạy)

| Config | BLEU-1 | BLEU-4 | METEOR | CIDEr |
|--------|--------|--------|--------|-------|
| A: MobileNet (baseline) | -- | -- | -- | -- |
| B: + V3 | -- | -- | -- | -- |
| C: + V3 + CLIP | -- | -- | -- | -- |

## License

Educational project.