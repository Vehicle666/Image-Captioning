# Deploy lên Oracle Cloud Free Tier (24/7, miễn phí)

Host web Image Captioning trên VPS ARM free vĩnh viễn của Oracle Cloud — không cần máy bạn bật.

## 1. Đăng ký Oracle Cloud (10 phút)

1. Vào **https://signup.cloud.oracle.com**
2. Đăng ký bằng email, nhập **thẻ tín dụng/ghi nợ** để xác minh (chỉ giữ 1$ tạm, **không bị trừ** nếu chỉ dùng Free Tier)
3. Chọn đúng region có sẵn Free ARM (thường: dùng region gần bạn, nếu hết tài nguyên thì thử region khác)
4. Vào Console → **Compute → Instances → Create instance**
5. Điền:
   - **Image**: Ubuntu 24.04 (canonical)
   - **Shape**: `VM.Standard.A1.Flex` (Ampere ARM)
   - **OCPUs**: 2–4 / **Memory**: 12–24 GB (trong hạn mức free)
   - **Add SSH keys**: dán public key (xem bước dưới)
6. Tạo → chờ instance **Running** → copy **Public IP**

### Tạo SSH key nếu chưa có (trên máy Windows của bạn)

```powershell
ssh-keygen -t rsa -b 4096 -f "$HOME\.ssh\cloudflare_vps"
```

Public key nằm ở `$HOME\.ssh\cloudflare_vps.pub` — dán nội dung file này vào Oracle khi tạo instance.

## 2. Mở cổng 5000 trong firewall Oracle

Console → **Networking → Virtual Cloud Networks → Security Lists → Default Security List** → **Add Ingress Rules**:
- Source: `0.0.0.0/0`
- Destination Port: `5000` (TCP)

## 3. SSH vào VPS (từ Windows)

```powershell
ssh -i "$HOME\.ssh\cloudflare_vps" ubuntu@<PUBLIC_IP>
```

## 4. Cài đặt web (chạy 1 lần)

```bash
curl -fsSL https://raw.githubusercontent.com/Vehicle666/Image-Captioning/main/webapp/deploy_vps/setup.sh -o setup.sh
sudo bash setup.sh
```

Script tự: cài package → clone repo → tải checkpoint C (Git LFS) → cài torch CPU → cache CLIP tokenizer.

## 5. Cài dịch vụ tự chạy 24/7

```bash
REPO=/opt/image-captioning
sudo cp $REPO/webapp/deploy_vps/image-captioning.service /etc/systemd/system/
sudo sed -i 's/^User=.*/User=ubuntu/' /etc/systemd/system/image-captioning.service
sudo systemctl daemon-reload
sudo systemctl enable --now image-captioning
```

## 6. Kiểm tra

- Trạng thái: `sudo systemctl status image-captioning`
- Log: `sudo journalctl -u image-captioning -f`
- Mở browser: **http://<PUBLIC_IP>:5000**

## Lưu ý

- Model C (~128MB) tải qua Git LFS lúc setup; lần đầu load model mất ~10–30s (CPU).
- Mỗi caption trên CPU ARM mất vài giây — bình thường.
- Cập nhật code mới: `cd /opt/image-captioning && sudo git pull` rồi `sudo systemctl restart image-captioning`.
- Hết hạn Free tier lấy lại được: giữ instance running, không lên CPU/ướt lốt quota;
  free tier ARM luôn miễn phí khi ở trong hạn mức (4 OCPU, 24GB RAM tổng).