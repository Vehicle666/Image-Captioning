import json
import os
import sys
import uuid
from datetime import datetime

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

# Ensure models package is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.registry import registry

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.json")
ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp", "gif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_feedback(data):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/models", methods=["GET"])
def list_models():
    return jsonify(registry.list_models())


@app.route("/models/switch", methods=["POST"])
def switch_model():
    data = request.get_json(force=True)
    name = data.get("name")
    if not name:
        return jsonify({"error": "Missing 'name' field"}), 400
    try:
        model = registry.switch_to(name)
        return jsonify({"ok": True, "active": model.name})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/caption", methods=["POST"])
def caption():
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "No image uploaded"}), 400
    if not allowed(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    model = registry.get_active()
    if model is None:
        return jsonify({"error": "No model loaded. Select a model first."}), 400

    # Unique filename so feedback images are never overwritten by later uploads
    filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        result = model.get_caption(path)
        return jsonify({
            "caption": result.caption,
            "model": result.model_name,
            "confidence": result.confidence,
            "image_name": filename,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(force=True)
    image_name = data.get("image_name")
    original_caption = data.get("original_caption")
    corrected_caption = data.get("corrected_caption")

    if not image_name or not original_caption or not corrected_caption:
        return jsonify({"error": "Missing fields: image_name, original_caption, corrected_caption"}), 400

    feedback_entry = {
        "id": str(uuid.uuid4())[:8],
        "image_name": image_name,
        "original_caption": original_caption,
        "corrected_caption": corrected_caption,
        "model": data.get("model"),
        "verified": None,
        "verify_reason": None,
        "timestamp": datetime.now().isoformat(),
    }

    all_feedback = _load_feedback()
    all_feedback.append(feedback_entry)
    _save_feedback(all_feedback)

    return jsonify({"ok": True, "id": feedback_entry["id"], "message": "Feedback saved. Use /verify to verify."})


@app.route("/verify", methods=["POST"])
def verify():
    data = request.get_json(force=True)
    feedback_id = data.get("id")

    if not feedback_id:
        return jsonify({"error": "Missing 'id' field"}), 400

    all_feedback = _load_feedback()
    entry = None
    for fb in all_feedback:
        if fb["id"] == feedback_id:
            entry = fb
            break

    if entry is None:
        return jsonify({"error": f"Feedback '{feedback_id}' not found"}), 404

    image_path = os.path.join(UPLOAD_FOLDER, entry["image_name"])
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image '{entry['image_name']}' not found on server"}), 404

    from gemini_verify import verify_caption
    try:
        result = verify_caption(image_path, entry["original_caption"], entry["corrected_caption"])
    except Exception as e:
        return jsonify({"error": f"Gemini API error: {str(e)}"}), 500

    if "error" in result:
        return jsonify(result), 500

    entry["verified"] = result.get("verified", False)
    entry["verify_reason"] = result.get("reason", "")
    _save_feedback(all_feedback)

    return jsonify({
        "ok": True,
        "id": feedback_id,
        "verified": entry["verified"],
        "reason": entry["verify_reason"],
    })


@app.route("/feedback", methods=["GET"])
def list_feedback():
    all_feedback = _load_feedback()
    verified = [fb for fb in all_feedback if fb.get("verified") is True]
    return jsonify({
        "total": len(all_feedback),
        "verified": len(verified),
        "items": all_feedback,
    })


FINETUNE_LOG = os.path.join(BASE_DIR, "finetune.log")
finetune_state = {"status": "idle", "message": "", "log_tail": []}


def _read_log_tail(max_lines=8):
    if not os.path.exists(FINETUNE_LOG):
        return []
    try:
        with open(FINETUNE_LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f.readlines() if l.strip()]
        return lines[-max_lines:]
    except OSError:
        return []


@app.route("/finetune/status", methods=["GET"])
def finetune_status():
    state = dict(finetune_state)
    state["log_tail"] = _read_log_tail()
    return jsonify(state)


@app.route("/finetune", methods=["POST"])
def finetune():
    import subprocess
    import threading

    finetune_script = os.path.join(BASE_DIR, "finetune_feedback.py")
    if not os.path.exists(finetune_script):
        return jsonify({"error": "finetune_feedback.py not found"}), 500

    if finetune_state["status"] == "running":
        return jsonify({"error": "Fine-tune đang chạy, vui lòng đợi."}), 409

    data = request.get_json(force=True) if request.is_json else {}
    epochs = data.get("epochs", 20)
    lr_mult = data.get("lr_mult", 0.5)

    def run_finetune():
        finetune_state["status"] = "running"
        finetune_state["message"] = f"Đang fine-tune ({epochs} epochs)..."
        try:
            with open(FINETUNE_LOG, "w", encoding="utf-8") as log:
                proc = subprocess.run(
                    [sys.executable, finetune_script,
                     "--epochs", str(epochs), "--lr-mult", str(lr_mult)],
                    cwd=BASE_DIR, stdout=log, stderr=subprocess.STDOUT,
                )
            tail = _read_log_tail(3)
            if proc.returncode == 0:
                msg = "Fine-tune hoàn tất. Đang nạp lại model..."
                try:
                    active = registry.get_active_name()
                    registry.reload(active)
                    msg += " Xong! Model đã dùng trọng số mới."
                except Exception as e:
                    msg += f" Không nạp lại được ({e}). Hãy restart web app."
                finetune_state["status"] = "done"
                finetune_state["message"] = msg
            else:
                finetune_state["status"] = "error"
                finetune_state["message"] = "Fine-tune lỗi: " + (tail[-1] if tail else f"exit code {proc.returncode}")
        except Exception as e:
            finetune_state["status"] = "error"
            finetune_state["message"] = str(e)

    thread = threading.Thread(target=run_finetune, daemon=True)
    thread.start()

    return jsonify({"ok": True, "message": f"Đã bắt đầu fine-tune ({epochs} epochs)."})


def init():
    """Discover models and activate the fastest local one."""
    found = registry.discover()
    print(f"[webapp] Discovered {len(found)} model(s): {found}")
    preferred = next((m for m in ["MobileNetV3 + CLIP (WIP)", "BLIP (Salesforce)"] if m in found), found[0] if found else None)
    if preferred:
        registry.switch_to(preferred)
        print(f"[webapp] Active model: {registry.get_active_name()}")


if __name__ == "__main__":
    import socket

    def get_local_ips():
        ips = []
        try:
            hostname = socket.gethostname()
            for addr in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ip = addr[4][0]
                if ip not in ips:
                    ips.append(ip)
        except Exception:
            pass
        if "127.0.0.1" not in ips:
            ips.append("127.0.0.1")
        return ips

    init()
    ips = get_local_ips()
    print("=" * 50)
    print("  Image Caption(Linh) Web App")
    print("=" * 50)
    for ip in ips:
        print(f"  Local:   http://{ip}:5000")
    print(f"  Public:  http://127.0.0.1:5000")
    print("=" * 50)
    print("  Other devices on your network can use the Local URLs above.")
    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=5000)
