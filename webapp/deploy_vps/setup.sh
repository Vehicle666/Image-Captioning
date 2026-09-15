#!/usr/bin/env bash
# Oracle Cloud Free Tier (ARM) setup for the Image Captioning web app.
# Run as: sudo bash setup.sh
set -euo pipefail

APP_DIR="/opt/image-captioning"
REPO_URL="https://github.com/Vehicle666/Image-Captioning.git"
CFG_USER="ubuntu"

echo "==> Installing system packages..."
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-venv python3-pip python3-dev git git-lfs curl

echo "==> Cloning repository..."
rm -rf "$APP_DIR"
git clone "$REPO_URL" "$APP_DIR"
cd "$APP_DIR"

echo "==> Downloading model checkpoints (Git LFS)..."
git lfs install
git lfs pull --include="*checkpoints/mobilenet_v3_clip/*"

echo "==> Creating virtualenv..."
python3 -m venv "$APP_DIR/.venv"
source "$APP_DIR/.venv/bin/activate"

echo "==> Installing Python dependencies (CPU torch)..."
pip install --upgrade pip
pip install -r "$APP_DIR/webapp/requirements-vps.txt"

echo "==> Caching CLIP tokenizer..."
export HF_HOME="$APP_DIR/.cache/huggingface"
export HF_HUB_CACHE="$APP_DIR/.cache/huggingface/hub"
python - <<'PY'
from transformers import CLIPTokenizer
CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP tokenizer cached.")
PY

echo "==> Giving write access to $CFG_USER..."
chown -R "$CFG_USER:$CFG_USER" "$APP_DIR"
chmod -R u+rwX "$APP_DIR"

echo "==> Done. Next steps (as $CFG_USER):"
echo "    sudo cp $APP_DIR/webapp/deploy_vps/image-captioning.service /etc/systemd/system/"
echo "    sudo systemctl daemon-reload"
echo "    sudo systemctl enable --now image-captioning"