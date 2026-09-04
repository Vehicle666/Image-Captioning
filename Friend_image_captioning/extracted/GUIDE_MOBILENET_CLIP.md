# Hướng dẫn chi tiết: MobileNet + CLIP Image Captioning

> File này giải thích **toàn bộ** model `mobilenet_clip_captioning` trong thư mục này — model là gì, kiến trúc CNN hay Transformer, và trả lời các câu hỏi về **từng phần** của con AI này. Mọi giải thích đều dựa trên code thực tế trong `src/`.

---

## Mục lục

- [1. Model này là gì?](#1-model-này-là-gì)
- [2. Tổng quan kiến trúc (sơ đồ)](#2-tổng-quan-kiến-trúc-sơ-đồ)
- [3. CNN hay Transformer? — Câu trả lời ngắn](#3-cnn-hay-transformer--câu-trả-lời-ngắn)
- [4. Phần 1: MobileNetV3 Encoder (CNN)](#4-phần-1-mobilenetv3-encoder-cnn)
- [5. Phần 2: Spatial Attention](#5-phần-2-spatial-attention)
- [6. Phần 3: Conv 1x1 + BatchNorm (chiếu sang không gian embedding)](#6-phần-3-conv-1x1--batchnorm-chiếu-sang-không-gian-embedding)
- [7. Phần 4: CLIP Projection Head + CLIP Contrastive Loss](#7-phần-4-clip-projection-head--clip-contrastive-loss)
- [8. Phần 5: Transformer Decoder (phần "Transformer" thật sự)](#8-phần-5-transformer-decoder-phần-transformer-thật-sự)
- [9. Phần 6: Positional Encoding](#9-phần-6-positional-encoding)
- [10. Phần 7: BPE Tokenizer + Embedding](#10-phần-7-bpe-tokenizer--embedding)
- [11. Hai hàm loss cùng lúc — training diễn ra thế nào](#11-hai-hàm-loss-cùng-lúc--training-diễn-ra-thế-nào)
- [12. Dữ liệu: COCO + ImageNet + "soft captions" từ BLIP](#12-dữ-liệu-coco--imagenet--soft-captions-từ-blip)
- [13. Semi-supervised learning là gì ở đây](#13-semi-supervised-learning-là-gì-ở-đây)
- [14. Inference: greedy vs beam search](#14-inference-greedy-vs-beam-search)
- [15. Vocabulary Expansion (thêm từ mới sau khi train)](#15-vocabulary-expansion-thêm-từ-mới-sau-khi-train)
- [16. Hỏi & Đáp chi tiết từng phần](#16-hỏi--đáp-chi-tiết-từng-phần)
- [17. Kết quả thực tế từ log huấn luyện](#17-kết-quả-thực-tế-từ-log-huấn-luyện)
- [18. Các câu lệnh chạy](#18-các-câu-lệnh-chạy)

---

## 1. Model này là gì?

Đây là một **hệ thống Image Captioning** (tự động viết mô tả cho ảnh) được lắp ghép từ **ba mô hình nổi tiếng**:

| Mô hình | Vai trò | Bản chất |
|---|---|---|
| **MobileNetV3-Small** | Đọc ảnh → trích đặc trưng | **CNN** (Convolutional Neural Network) |
| **CLIP (text encoder)** | Mã hóa văn bản → dùng làm loss phụ | Transformer (bên trong CLIP) |
| **Transformer Decoder** | Sinh ra câu mô tả từng từ một | **Transformer** |

Tên đầy đủ: *Semi-supervised image captioning using MobileNetV3 encoder backbone, trained with CLIP contrastive loss, vocabulary built from BLIP-generated soft captions.*

Tóm lại:
- **"mobilenet"** = encoder CNN để "nhìn" ảnh.
- **"clip"** = dùng CLIP để so sánh ảnh và câu mô tả, ép model hiểu đúng ý nghĩa.
- **"captioning"** = mục tiêu cuối cùng là viết câu mô tả.
- **BLIP** = mô hình khác được dùng *một lần* để tạo nhãn giả cho dữ liệu không có chú thích.

---

## 2. Tổng quan kiến trúc (sơ đồ)

Nhìn từ `README.md` và `src/model.py`:

```
Ảnh đầu vào (3, 224, 224)
        │
        ▼
┌────────────────────────────────────────────┐
│ MobileNetV3-Small (pretrained)             │   ← CNN
│  - giữ 12 khối features                     │
│  - đóng băng layer 0-8, chỉ tinh chỉnh 9-12 │
└────────────────────────────────────────────┘
        │  (B, 576, 7, 7)
        ▼
┌────────────────────────────────────────────┐
│ SpatialAttention (trọng số theo vị trí)     │
└────────────────────────────────────────────┘
        │  (B, 576, 7, 7)
        ▼
┌────────────────────────────────────────────┐
│ Conv 1x1 (576 → 256) + BatchNorm           │   ← chiếu xuống 256 chiều
└────────────────────────────────────────────┘
        │
        ├───────────────┐
        │  (B, 49, 256) │          (B, 576) ──► CLIP Proj (576→512) [chỉ khi train]
        ▼               │                                  │
┌──────────────────┐   │                                  ▼
│ Transformer      │   │                      CLIP Contrastive Loss
│ Decoder          │   │                    (so ảnh ↔ câu văn bản)
│ 2 layers, 4 heads│   │
└──────────────────┘   │
        │              │
        ▼              │
  Token dự đoán ◄──────┘  ← nối câu để học cả hai loss
```

**Đọc từ dưới lên:** ảnh → CNN lấy đặc trưng → cắt thành 49 "token không gian" → Transformer Decoder sinh từng chữ.

---

## 3. CNN hay Transformer? — Câu trả lời ngắn

**Cả hai.** Model này là **kiến trúc hybrid (lai)**:

- **Phần xử lý ẢNH = CNN** (MobileNetV3). Đây là điểm khác biệt với các mô hình mới hơn như ViT — thay vì chia ảnh thành patch rồi đưa vào Transformer, model này dùng CNN để trích đặc trưng.
- **Phần sinh VĂN BẢN = Transformer** (TransformerDecoder). Đây là nơi "tư duy" ngôn ngữ xảy ra.
- **Attention trong encoder** là **Spatial Attention** (không phải self-attention của Transformer) — dùng để tăng/giảm độ quan trọng của từng vùng ảnh.

Câu trả lời cho phần "cái gì là gì":

| Thành phần | CNN? | Transformer? |
|---|---|---|
| MobileNetV3 | ✅ | ❌ |
| SpatialAttention | ✅ (dùng Conv2d) | ❌ |
| CLIP Projection Head | ❌ | ❌ (chỉ là Linear layer) |
| CLIP text encoder (trong loss) | ❌ | ✅ (nhưng chỉ chạy để tạo label) |
| TransformerDecoder | ❌ | ✅ |
| PositionalEncoding | ❌ | ✅ (của Transformer) |
| BPE Tokenizer | ❌ | ✅ (theo chuẩn Transformer/LLM) |

---

## 4. Phần 1: MobileNetV3 Encoder (CNN)

File: `src/model.py`, class `MobileNetEncoder`.

### 4.1 MobileNetV3-Small là gì?
MobileNetV3-Small là một **mạng CNN nhỏ gọn** dành cho thiết bị di động, pretrained trên **ImageNet** (phân loại 1000 lớp ảnh). Nó đã "biết" nhận diện các vật thể phổ quát (mèo, chó, xe, người...) nên ta tái sử dụng nó làm "mắt".

Trong code:
```python
mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
self.features = mobilenet.features   # bỏ classifier, chỉ giữ phần trích đặc trưng
```

### 4.2 Transfer Learning & Freeze (đóng băng) layer
```python
for i, param in enumerate(self.features.parameters()):
    if i < 9:
        param.requires_grad = False
```
- MobileNetV3-Small có 12 khối features.
- **Layer 0–8: đóng băng** (không học nữa) — các layer đầu học các đặc trưng cơ bản (cạnh, màu, texture) nên giữ nguyên.
- **Layer 9–12: vẫn học** (fine-tune) — các layer cuối học đặc trưng cao cấp, cần điều chỉnh cho nhiệm vụ captioning.

> Vì sao đóng băng? Tiết kiệm VRAM + tránh phá hỏng kiến thức đã học + encoder chỉ học với LR nhỏ (1e-5).

### 4.3 Đầu ra
Ảnh `(B, 3, 224, 224)` đi qua `features` → **`(B, 576, 7, 7)`**:
- `B` = batch size
- `576` = số channel (kênh đặc trưng)
- `7×7` = lưới không gian. Ảnh 224×224 co dần thành 7×7 qua các lớp pooling/stride. Mỗi ô 7×7 "nhìn" một vùng ảnh ~32×32 pixel.

---

## 5. Phần 2: Spatial Attention

File: `src/model.py`, class `SpatialAttention`.

### 5.1 Đây là attention gì?
Đây **không phải** attention kiểu Transformer (Q/K/V). Đây là **CBAM-style spatial attention** — dùng CNN để học trọng số theo vị trí.

```python
avg_out = torch.mean(x, dim=1, keepdim=True)        # trung bình theo channel → (B,1,7,7)
max_out, _ = torch.max(x, dim=1, keepdim=True)      # max theo channel → (B,1,7,7)
attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
return x * attention
```

### 5.2 Ý nghĩa
Với mỗi ô trong lưới 7×7, model tự quyết định **vùng nào của ảnh quan trọng**:
- Với ảnh "con chó trên bãi cỏ", vùng có con chó được nhân với trọng số cao (~1), vùng cỏ bị nhân với trọng số thấp (~0).
- Cách tính: trung bình và max theo các channel rồi đẩy qua Conv 1×1 (kernel 7, padding 3) + sigmoid → điểm 0..1 cho từng ô.

> Tóm tắt: "Đây là chỗ attention CNN — hướng sự chú ý tới vùng quan trọng của ảnh, khác với self-attention của Transformer."

---

## 6. Phần 3: Conv 1x1 + BatchNorm (chiếu sang không gian embedding)

File: `src/model.py`:

```python
self.conv = nn.Conv2d(MOBILENET_OUT_CHANNELS, embed_size, kernel_size=1)  # 576 → 256
self.bn = nn.BatchNorm2d(embed_size)
```

- Conv **1×1** không thấy không gian, chỉ đổi số channel (576 → 256) — giống một fully-connected layer áp lên từng vị trí.
- Sau đó BatchNorm chuẩn hóa.
- Cuối cùng:
```python
features = features.permute(0, 2, 3, 1)          # (B, 7, 7, 256)
features = features.reshape(B, -1, 256)          # (B, 49, 256)
```

Mỗi ảnh giờ trở thành **49 "token" không gian**, mỗi token là vector 256 chiều. Điều này giống hệt cách ViT chia ảnh thành patch — chỉ khác là ViT dùng patch thẳng từ pixel, còn ở đây dùng đặc trưng CNN.

**49 token này chính là `memory` (bộ nhớ) mà Transformer Decoder sẽ "nhìn" vào.**

---

## 7. Phần 4: CLIP Projection Head + CLIP Contrastive Loss

File: `src/model.py`, classes `CLIPTextCache`, `CLIPContrastiveLoss`.

### 7.1 CLIP là gì?
CLIP (Contrastive Language–Image Pre-training) là model của OpenAI được train để **kéo ảnh và câu mô tả đúng lại gần nhau** trong không gian vector, đẩy các cặp sai ra xa.

Ở đây ta dùng **`openai/clip-vit-base-patch32`** (text encoder, 512 chiều).

### 7.2 Projection Head
```python
self.clip_proj = nn.Linear(MOBILENET_OUT_CHANNELS, clip_embed_dim)  # 576 → 512
```
- Lấy trung bình đặc trưng `(B, 576, 7, 7)` theo không gian → `(B, 576)`.
- Đẩy qua `clip_proj` → `(B, 512)` — **cùng chiều với embedding văn bản của CLIP**.

### 7.3 Loss so sánh (Contrastive Loss)
```python
logits = (image_features @ text_features.T) / temperature   # ma trận B×B
labels = torch.arange(len(logits))
loss_i2t = cross_entropy(logits, labels)      # ảnh → đúng caption
loss_t2i = cross_entropy(logits.T, labels)    # caption → đúng ảnh
return (loss_i2t + loss_t2i) / 2
```
- Với batch 32: ma trận 32×32. Ô chéo (i,i) là cặp (ảnh_i, caption_i) — **phải là cặp đúng**.
- Loss ép model học: "ảnh này phải khớp với câu này chứ không phải câu khác trong batch".
- `temperature = 0.07`: chia để làm sắc nét phân bố xác suất.

### 7.4 CLIPTextCache — điểm tối ưu thông minh
```python
h = hashlib.md5("\n".join(sorted(unique_captions)).encode()).hexdigest()[:12]
```
Toàn bộ caption được mã hóa bằng CLIP text encoder **một lần** rồi cache ra đĩa (`clip_text_cache_*.pt`). Mỗi epoch không cần chạy lại — tiết kiệm ~10 phút mỗi lần khởi động lại. Cache tự vô hiệu khi captions đổi (vì hash thay đổi).

> CLIP text encoder **luôn bị đóng băng** (requires_grad=False) — nó chỉ là "cái thước" để đo, không được huấn luyện.

---

## 8. Phần 5: Transformer Decoder (phần "Transformer" thật sự)

File: `src/model.py`, classes `PositionalEncoding`, `TransformerDecoder`.

### 8.1 Cấu hình
```python
decoder_layer = nn.TransformerDecoderLayer(
    d_model=256, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True)
self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
self.fc = nn.Linear(256, vocab_size)
```
- **2 layer**, **4 attention head**, embedding 256 chiều, feed-forward 512.
- Đây là **decoder-only phần ngôn ngữ** (chỉ sinh văn bản, không có encoder stack riêng — dùng đặc trưng CNN làm memory).

### 8.2 Hai loại attention trong TransformerDecoderLayer
`nn.TransformerDecoderLayer` có 2 khối attention:
1. **Self-Attention trên caption**: mỗi từ đang sinh nhìn các từ đã sinh trước đó (bị mask để không nhìn tương lai).
2. **Cross-Attention**: các từ văn bản **hỏi** 49 token ảnh (`memory`) — đây là cầu nối "ảnh → chữ". Khi sinh từ "dog", model tìm xem vùng ảnh nào liên quan (vùng con chó) thông qua cross-attention.

### 8.3 Causal mask (mask nhân quả)
```python
causal_mask = torch.triu(torch.full((tgt_len, tgt_len), float("-inf"), device=...), diagonal=1)
```
Khi dự đoán từ thứ `t`, model chỉ được nhìn từ 0..t-1. Phần tam giác trên bị đặt `-inf` → softmax cho xác suất 0. Đảm bảo model không "nhìn lén" từ tương lai.

### 8.4 Padding mask
```python
tgt_padding_mask = captions == PAD_TOKEN
```
Các vị trí `<PAD>` (đệm cho caption ngắn dài bằng nhau) bị loại khỏi attention.

### 8.5 Dòng dữ liệu (forward)
```python
def forward(self, encoder_features, captions):
    embeddings = self.embedding(captions)          # token → vector 256
    embeddings = self.embedding_norm(embeddings)   # LayerNorm
    embeddings = self.embedding_dropout(embeddings)
    embeddings = self.pos_encoder(embeddings)      # + positional encoding
    output = self.transformer_decoder(
        tgt=embeddings, memory=encoder_features,
        tgt_mask=causal_mask, tgt_key_padding_mask=tgt_padding_mask)
    return self.fc(output)                         # → logits trên toàn bộ vocab
```

### 8.6 Shift right (dịch phải)
Trong `CaptioningModel.forward`:
```python
outputs = self.decoder(encoder_features, captions[:, :-1])
```
- Input decoder = caption **bỏ từ cuối**.
- Label = caption **bỏ từ đầu** (`captions[:, 1:]` trong train.py).
- Mục tiêu: dự đoán từ tiếp theo. Input `<SOS> a cat sits` → dự đoán `a cat sits <EOS>`.

---

## 9. Phần 6: Positional Encoding

File: `src/model.py`:

```python
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```
- Đây là **sinusoidal positional encoding** gốc từ paper "Attention Is All You Need".
- Vì attention xử lý các token song song, không biết thứ tự → cộng thêm thông tin vị trí vào embedding.
- Tần số khác nhau cho từng chiều: chiều chẵn dùng sin, chiều lẻ dùng cos.

---

## 10. Phần 7: BPE Tokenizer + Embedding

File: `src/vocabulary.py`.

### 10.1 BPE (Byte-Pair Encoding)
Dùng thư viện HuggingFace `tokenizers`:
```python
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=5000, min_frequency=2, continuing_subword_prefix="##")
```
- Học cách chia từ thành các mảnh (subword) phổ biến nhất từ dữ liệu.
- `vocab_size=5000`: 5000 token (mảnh) đủ biểu diễn hầu hết từ tiếng Anh.
- `min_frequency=2`: mảnh phải xuất hiện ≥2 lần mới được giữ.
- `##` đánh dấu mảnh nằm giữa/cuối từ (vd `"cyberpunk"` → `"cyber" + "##punk"`).
- **Không có từ nào bị "không biết"** (no OOV): từ lạ luôn tách được thành các mảnh quen.

### 10.2 Special tokens
```
<PAD>=0   <SOS>=1   <EOS>=2   <UNK>=3
```
- `<SOS>` mở đầu câu, `<EOS>` kết thúc câu, `<PAD>` đệm, `<UNK>` từ không nhận diện được.
- Post-processor tự động thêm `<SOS>` đầu và `<EOS>` cuối:
```python
single=f"<SOS>:0 $A:0 <EOS>:0"
```

### 10.3 Embedding layer
```python
self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_TOKEN)
```
- Ánh xạ id token → vector 256 chiều, **được học trong lúc train** (khác Word2Vec tĩnh).
- `padding_idx=0`: gradient của token `<PAD>` bị bỏ qua.

---

## 11. Hai hàm loss cùng lúc — training diễn ra thế nào

File: `src/train.py`.

### 11.1 Hai loss
```python
cap_loss = caption_criterion(outputs, captions[:, 1:])   # CrossEntropy, ignore_index=0
closs = clip_loss_fn(clip_features, captions, ...)        # CLIP contrastive
total_loss = (1.0 * cap_loss + 0.5 * closs) / GRAD_ACCUM_STEPS
```
1. **Caption loss** (trọng số 1.0): dự đoán từ tiếp theo chính xác đến đâu. Đây là nhiệm vụ chính.
2. **CLIP loss** (trọng số 0.5): ảnh và câu mô tả của nó có "nghĩa" gần nhau không. Đây là nhiệm vụ phụ giúp model hiểu ngữ nghĩa, không chỉ thuộc lòng câu.

### 11.2 AMP (mixed precision)
```python
with autocast("cuda", enabled=torch.cuda.is_available()):
scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
```
Forward chạy ở fp16 (nhanh hơn ~2x), optimizer ở fp32. Giảm VRAM + tăng tốc.

### 11.3 Gradient accumulation
```python
total_loss = (...) / GRAD_ACCUM_STEPS
if micro_steps % GRAD_ACCUM_STEPS == 0:
    scaler.step(optimizer); optimizer.zero_grad()
```
`GRAD_ACCUM_STEPS=4`: cộng dồn gradient 4 batch rồi mới cập nhật → **effective batch = 32 × 4 = 128** mà không cần thêm VRAM.

### 11.4 Hai learning rate
```python
optimizer = torch.optim.Adam([
    {"params": encoder_params, "lr": 1e-5},   # CNN đã pretrain, học chậm
    {"params": decoder_params, "lr": 1e-4},   # Transformer mới, học nhanh hơn
])
```
Encoder quý giá → LR nhỏ. Decoder mới toanh → LR lớn hơn.

### 11.5 Warmup + giảm LR
```python
warmup_factor = min(1.0, (epoch+1) / WARMUP_EPOCHS)   # 2 epoch đầu tăng dần
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
```
- 2 epoch đầu LR tăng dần từ 0 → ổn định.
- Sau đó nếu val loss không giảm 2 epoch → giảm LR đi một nửa.

### 11.6 Subset mỗi epoch (20.000 ảnh/epoch)
```python
epoch_images = random.sample(train_images, MAX_TRAIN_IMAGES)  # 20000
```
118K ảnh COCO quá lâu → mỗi epoch lấy ngẫu nhiên 20K ảnh (~7-8 phút/epoch thay vì ~44 phút). Mỗi epoch khác nhau, nên sau ~6 epoch model thấy đủ toàn bộ, sau 100 epoch mỗi ảnh thấy ~17 lần.

### 11.7 Early stopping
- `patience=25`: nếu val loss không cải thiện 25 epoch → dừng.
- `MIN_DELTA=1e-4`: cải thiện phải ≥ 1e-4 mới tính là "tốt hơn".

### 11.8 Grad clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # 5.0
```
Chặn gradient quá lớn gây loss "nổ" (exploding gradient).

---

## 12. Dữ liệu: COCO + ImageNet + "soft captions" từ BLIP

### 12.1 Ba nguồn dữ liệu
| Nguồn | Số lượng | Nhãn từ đâu |
|---|---|---|
| **COCO 2017 train** | ~118K ảnh | Người viết (5 caption/ảnh) — **ground truth** |
| **COCO 2017 val** | ~25K ảnh | Người viết (dùng để đánh giá) |
| **ImageNet mini-1000** | ~34K ảnh | **BLIP sinh ra** (pseudo-caption) |

### 12.2 Soft captions (nhãn mềm) từ BLIP
File: `src/soft_caption.py`.
- ImageNet chỉ có nhãn *lớp* ("golden retriever"), không có *câu mô tả*. Để model học captioning, ta dùng **BLIP** (`Salesforce/blip-image-captioning-base`) chạy trên từng ảnh để sinh câu.
- Còn COCO cũng được chạy thêm BLIP để làm **giàu vốn từ** (soft captions có từ vựng đa dạng hơn caption người viết).

```python
output_ids = model.generate(**inputs, max_length=50, num_beams=3)
decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
```
- BLIP sinh câu với beam search (3 beam), tối đa 50 token.
- Kết quả lưu vào `checkpoints/pseudo_captions/coco_captions.json` và `imagenet_captions.json`.
- Tối ưu: fp16, `torch.compile`, resize trước 384×384, cache tăng dần, lưu mỗi 50 batch (an toàn khi crash).

### 12.3 Các class dataset
- `COCODataset` (`dataset.py:78`): đọc ảnh + caption người viết.
- `PseudoCaptionDataset` (`dataset.py:101`): ảnh ImageNet + caption từ JSON.
- `SemiSupervisedDataset` (`dataset.py:127`): gộp hai cái trên, thêm cờ `source` (0 = COCO, 1 = pseudo).

### 12.4 Transform ảnh
```python
# Train: ngẫu nhiên hóa (chống overfit)
RandomResizedCrop(224), RandomHorizontalFlip(), ColorJitter(...), RandomGrayscale(...)
# Val: cố định
Resize(256) → CenterCrop(224)
```
Cả hai chuẩn hóa theo ImageNet mean/std `[0.485,0.456,0.406] / [0.229,0.224,0.225]` — vì MobileNet pretrained trên ImageNet.

---

## 13. Semi-supervised learning là gì ở đây

**Semi-supervised** = nửa có nhãn, nửa không:
- COCO: có nhãn người viết (supervised).
- ImageNet: **không có** caption → ta tự tạo nhãn giả bằng BLIP, rồi coi như nhãn thật để train (self-training / pseudo-labeling).

Lợi ích:
- Thêm ~34K ảnh với nhãn giả → model thấy nhiều ảnh và từ vựng hơn, tránh overfit lên COCO.
- Rủi ro: nếu BLIP sinh sai, model học sai — vì vậy loss CLIP giúp kiểm soát chất lượng ngữ nghĩa.

---

## 14. Inference: greedy vs beam search

File: `src/predict.py`.

### 14.1 Greedy (`generate_caption`)
```python
pred = logits[0, -1, :].argmax().item()   # chọn từ có xác suất cao nhất
```
- Mỗi bước chọn từ tốt nhất ngay lập tức. Nhanh nhưng có thể mắc kẹt cục bộ (chọn sai từ đầu kéo theo cả câu sai).

### 14.2 Beam search (`generate_caption_beam`)
```python
topk_probs, topk_indices = torch.topk(probs, beam_size)   # BEAM_SIZE=5
```
- Giữ **5 chuỗi tốt nhất cùng lúc**, mỗi bước mở rộng 5 lựa chọn → 25 ứng viên → chọn 5 tốt nhất.
- Điểm = tổng log-xác suất.
- Cuối cùng chọn chuỗi có điểm cao nhất.
- Chậm hơn nhưng câu mạch lạc hơn hẳn.

### 14.3 Tại sao val đánh giá model phải `model.eval()` + dropout=0
- Khi nạp model để predict/đánh giá, code set `dropout=0.0` và `model.eval()` — tắt Dropout và BatchNorm dùng thống kê cố định, kết quả ổn định.

---

## 15. Vocabulary Expansion (thêm từ mới sau khi train)

File: `src/main.py` (`run_expand_vocab`), `src/model.py` (`expand_vocabulary`).

Quy trình khi bạn thêm từ mới ("cyberpunk", "selfie"...):
1. Đọc tokenizer hiện tại.
2. Dò từ nào không có trong vocab.
3. Học BPE merge mới trên caption mới → thêm token vào tokenizer.
4. `resize_embedding`: tạo Embedding và `fc` mới lớn hơn, **giữ nguyên trọng số cũ** (không phá model).
5. `init_new_embeddings_from_clip`: các hàng embedding mới được khởi tạo từ **CLIP text encoder** — vì CLIP biết nghĩa từ → embedding mới đã có ý nghĩa ngay lập tức (thay vì random).
6. Lưu lại và retrain vài epoch.

```python
def init_new_embeddings_from_clip(self, new_token_strings, ...):
    outputs = clip_text_encoder(tokens)
    emb = outputs.pooler_output   # 512 chiều từ CLIP → copy vào hàng mới
```
> Lưu ý kỹ thuật: embedding model là 256 chiều, CLIP trả 512 → code dùng `outputs.pooler_output` rồi gán trực tiếp vào `weight.data[old_size:new_size]`. (Kích thước khớp ở đây vì cả hai là 512? Thực tế 256 vs 512 — điểm cần kiểm tra khi chạy thật.)

---

## 16. Hỏi & Đáp chi tiết từng phần

### Hỏi về model tổng thể

**H: Model này thuộc loại nào?**
Đ: Hybrid CNN + Transformer. CNN nhìn ảnh, Transformer sinh chữ.

**H: Vì sao không dùng luôn ViT/CLIP visual encoder làm encoder?**
Đ: MobileNet nhẹ hơn nhiều, chạy tốt trên máy thường; dùng lại model pretrained ImageNet sẵn có. Đây là lựa chọn thiết kế để cân bằng hiệu năng/tài nguyên.

**H: CLIP đóng vai trò gì trong model?**
Đ: CLIP không nằm trong đường "dự đoán" mà chỉ xuất hiện khi train — làm *thước đo ngữ nghĩa* qua contrastive loss và làm *nguồn khởi tạo* embedding khi mở rộng từ vựng.

### Hỏi về MobileNetEncoder

**H: Tại sao freeze layer 0-8?** (model.py:62)
Đ: Layer đầu học đặc trưng chung (cạnh, màu...) đã rất tốt từ ImageNet, giữ nguyên để không phá và tiết kiệm tài nguyên. Chỉ 4 khối cuối fine-tune.

**H: 49 token từ đâu ra?** (model.py:85)
Đ: Đặc trưng `(B, 576, 7, 7)` được "duỗi phẳng" không gian 7×7 → 49 vị trí, mỗi vị trí là vector 256 chiều sau Conv 1×1.

**H: Tại sao 224×224 mà ra 7×7?**
Đ: Do chuỗi stride/pooling của MobileNet giảm 32 lần: 224/32 = 7.

### Hỏi về SpatialAttention

**H: SpatialAttention có phải self-attention không?** (model.py:38)
Đ: Không. Nó nhân trọng số theo vùng ảnh dùng Conv + sigmoid, không có Q/K/V. "Attention" ở đây theo nghĩa *chọn vùng quan trọng*, không phải *quan hệ giữa các token*.

**H: kernel_size=7 có nghĩa gì?**
Đ: Conv 2D nhận 2 channel (avg + max) → 1 channel với cửa sổ 7×7. Lưới đặc trưng cũng 7×7 nên cửa sổ này bao phủ cả ảnh, giúp mỗi ô biết bối cảnh xung quanh.

### Hỏi về CLIP loss

**H: CLIP loss trọng số 0.5 nghĩa là gì?**
Đ: Mỗi bước, loss tổng = caption_loss × 1.0 + clip_loss × 0.5. Model ưu tiên học dự đoán từ đúng, CLIP chỉ "hỗ trợ" định hướng ngữ nghĩa.

**H: CLIP text encoder có được train không?** (model.py:176)
Đ: Không, `requires_grad=False`. Nó đóng băng, chỉ dùng để sinh embedding chuẩn cho caption.

**H: Vì sao cần temperature 0.07?** (config.py:77)
Đ: Đẩy logits nhân ra xa nhau → softmax quyết đoán hơn → loss phân biệt cặp đúng/sai rõ ràng hơn.

**H: CLIPTextCache để làm gì?** (model.py:94)
Đ: Tránh chạy CLIP text encoder (Transformer nặng) mỗi batch. Mã hóa 1 lần, lưu đĩa, tra theo hash của caption. Nhanh + không đổi khi dữ liệu giống nhau.

### Hỏi về Transformer Decoder

**H: Vì sao chỉ 2 layer decoder?**
Đ: Caption ngắn (≤50 từ), không cần mô hình khổng lồ. 2 layer đủ học cú pháp; nhiều layer hơn tốn VRAM/giờ train.

**H: Causal mask dùng khi nào?** (model.py:231)
Đ: Khi train (dự đoán song song toàn câu) bắt buộc phải có. Khi inference từng từ một, mask vẫn áp dụng nhưng không ảnh hưởng vì chỉ có 1 chuỗi.

**H: Cross-attention làm gì?**
Đ: Trong `TransformerDecoderLayer`, khối attention thứ 2 lấy `memory` = 49 token ảnh. Khi sinh từ, model tìm vùng ảnh liên quan. Đây là nơi "nhìn" ảnh trong lúc viết chữ.

**H: Shift-right để làm gì?** (model.py:305)
Đ: Dạy model "dự đoán từ tiếp theo": input là caption thiếu từ cuối, label là caption thiếu từ đầu. Chuẩn mô hình ngôn ngữ autoregressive.

### Hỏi về Tokenizer & Embedding

**H: Vì sao vocab chỉ 5000?**
Đ: BPE tận dụng tối đa — 5000 mảnh con phủ được gần như mọi từ tiếng Anh. Nhỏ = Embedding layer và `fc` nhỏ = ít tham số, train nhanh.

**H: `ignore_index=0` nghĩa là gì?** (train.py:246)
Đ: CrossEntropy bỏ qua token `<PAD>` (id 0) khi tính loss — các ô padding không bị phạt.

**H: Từ mới làm sao model biết nghĩa?** (model.py:263)
Đ: Khởi tạo embedding mới từ CLIP text encoder → có sẵn ngữ nghĩa, rồi fine-tune vài epoch.

### Hỏi về dữ liệu & training

**H: 20K ảnh/epoch có đủ không?**
Đ: Đủ vì mỗi epoch subset khác nhau. Qua nhiều epoch model thấy toàn bộ 118K ảnh, lại được ngẫu nhiên hóa (data augmentation) mỗi lần.

**H: ImageNet có nhãn không?**
Đ: Không có caption — đây chính là lý do phải dùng BLIP sinh pseudo-caption (semi-supervised).

**H: Vì sao COCO soft captions cũng cần BLIP?**
Đ: Để đa dạng hóa từ vựng và ngữ pháp (caption người viết khá "khuôn mẫu"), đồng thời dùng để train tokenizer.

**H: Effective batch 128 nghĩa là gì?**
Đ: Cập nhật 1 lần sau mỗi 4 batch (4×32=128 ảnh). Batch lớn → gradient ổn định, ít noise.

### Hỏi về inference

**H: Khi nào dùng beam, khi nào greedy?**
Đ: Beam (chất lượng cao, chậm) cho demo/đánh giá. Greedy cho tốc độ hoặc khi cần sinh nhanh hàng loạt.

**H: Vì sao greedy đôi khi ra câu dở?**
Đ: Chọn từng từ tốt nhất ngay tại chỗ có thể dẫn vào ngõ cụt ("greedy bias"). Beam thử nhiều đường, ít bị kẹt hơn.

**H: Có thể giới hạn caption dài không?**
Đ: Có, `max_length=50` trong predict.py — quá 50 từ không sinh nữa (dù chưa ra `<EOS>`).

---

## 17. Kết quả thực tế từ log huấn luyện

Từ `checkpoints/training_log.txt` (348 epochs, ~438 phút):

| Chỉ số | Giá trị tốt nhất |
|---|---|
| Caption Loss (train) | ~2.12 (giảm dần đều) |
| CLIP Loss (train) | ~0.38 |
| Val Loss | ~2.18 |
| BLEU-1 | 0.7356 |
| BLEU-2 | 0.5329 |
| BLEU-3 | 0.3895 |
| **BLEU-4** | **0.2812** (tốt nhất, epoch 312) |

- **BLEU-4 ~0.28** với captioning là mức hợp lý cho model nhỏ (MobileNet + 2-layer decoder); BLIP/COCO SOTA thường ~0.35–0.40 với model lớn hơn nhiều.
- CLIP loss giảm chậm (0.40→0.38) — phù hợp vì trọng số thấp và text encoder đóng băng.
- Model được dừng bởi **early stopping** (val loss không cải thiện 25 epoch).

---

## 18. Các câu lệnh chạy

```bash
# Tải dữ liệu (Kaggle)
python -m src.main coco
python -m src.main imagenet

# Toàn bộ pipeline: BLIP caption → tokenizer → train
python -m src.main pipeline
python -m src.main pipeline --fresh           # chạy lại từ đầu
python -m src.main pipeline --skip-captions   # bỏ qua bước sinh caption

# Từng bước
python -m src.main soft-captions   # sinh BLIP captions
python -m src.main vocab           # dựng tokenizer
python -m src.main train           # huấn luyện

# Đánh giá & dự đoán
python -m src.main evaluate
python -m src.main predict path/to/image.jpg
python -m src.main predict path/to/image.jpg --beam

# Thêm từ vựng mới
python -m src.main expand-vocab --new-data path/to/captions.json

# Dừng sạch trong lúc train
#  1) Ctrl+C một lần (kết thúc batch, lưu checkpoint) — 2 lần = hủy
#  2) echo x > checkpoints\STOP  (dừng cuối epoch)
```

---

## Phụ lục: map code → khái niệm

| Khái niệm | Nằm ở đâu trong code |
|---|---|
| CNN encoder | `MobileNetEncoder` — model.py:53 |
| Transfer learning / freeze | model.py:61-64 |
| Spatial attention (CBAM) | `SpatialAttention` — model.py:38 |
| Ảnh thành 49 token | model.py:84-85 |
| CLIP projection | model.py:71, 77-80 |
| CLIP contrastive loss | `CLIPContrastiveLoss` — model.py:164 |
| CLIP text cache | `CLIPTextCache` — model.py:94 |
| Transformer decoder | `TransformerDecoder` — model.py:214 |
| Causal mask | model.py:231-233 |
| Positional encoding | model.py:21-33 |
| BPE tokenizer | `CaptionTokenizer` — vocabulary.py:10 |
| Shift-right | model.py:305 |
| Mở rộng vocab từ CLIP | model.py:263-287 |
| Hai loss + grad accum | train.py:332-350 |
| Subset 20K/epoch | train.py:291-293 |
| Early stopping | train.py:441 |
| BLIP sinh caption | soft_caption.py:65 |
| Greedy decode | predict.py:32 |
| Beam search | predict.py:51 |
| BLEU evaluation | evaluate.py:58 |

---

*File được viết dựa trên đọc toàn bộ code trong `mobilenet_clip_captioning/`. Nếu có phần nào bạn muốn đào sâu hơn (ví dụ đi sâu vào một hàm cụ thể từng dòng), cứ hỏi.*
