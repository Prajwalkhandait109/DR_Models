"""Flask backend for RetinaScan — DR screening web app."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict

from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from gradcam import generate_gradcam
from inference import predict
from model_loader import available_models, has_weights
from utils.preprocessing import MAX_BYTES, validate_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("retinascan")

BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
GRADCAM_DIR = os.path.join(BACKEND_DIR, "outputs", "gradcam")
PREDICTIONS_DIR = os.path.join(BACKEND_DIR, "outputs", "predictions")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "frontend", "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "frontend", "static"),
    static_url_path="/static",
)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES + 1024  # leave a sliver for multipart headers


# --------------------------- routes ---------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.route("/models")
def models_page():
    return send_from_directory(app.template_folder, "models.html")


@app.route("/api/models")
def api_models():
    return jsonify({"models": available_models()})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return _err("No 'image' field in form data.", 400)
    file = request.files["image"]
    model_key = request.form.get("model", "").strip()
    if not model_key:
        return _err("Missing 'model' field.", 400)

    raw = file.read()
    ok, why = validate_upload(file.filename or "", len(raw))
    if not ok:
        return _err(why, 400)

    image_id = uuid.uuid4().hex
    ext = (file.filename or "img.png").rsplit(".", 1)[-1].lower()
    safe_name = secure_filename(f"{image_id}.{ext}")
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(save_path, "wb") as f:
        f.write(raw)

    log.info("Predict request | model=%s | image=%s | bytes=%d", model_key, safe_name, len(raw))

    try:
        result: Dict[str, Any] = predict(model_key, save_path)
    except RuntimeError as exc:
        return _err(str(exc), 503)
    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failed.")
        return _err(f"Inference failed: {exc}", 500)

    result["image_id"] = image_id
    result["image_filename"] = safe_name
    result["image_url"] = f"/uploads/{safe_name}"
    return jsonify(result)


@app.route("/api/gradcam", methods=["POST"])
def api_gradcam():
    payload = request.get_json(silent=True) or {}
    model_key = (payload.get("model") or "").strip()
    image_filename = (payload.get("image_filename") or "").strip()
    class_idx = payload.get("class_idx")

    if not model_key or not image_filename or class_idx is None:
        return _err("Missing one of: model, image_filename, class_idx.", 400)

    safe = secure_filename(image_filename)
    image_path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(image_path):
        return _err("Original image not found. Please re-upload.", 404)

    log.info("GradCAM request | model=%s | image=%s | class=%s", model_key, safe, class_idx)

    try:
        out_path = generate_gradcam(model_key, image_path, int(class_idx))
    except RuntimeError as exc:
        return _err(str(exc), 503)
    except Exception as exc:  # noqa: BLE001
        log.exception("GradCAM failed.")
        return _err(f"GradCAM failed: {exc}", 500)

    fname = os.path.basename(out_path)
    return jsonify({"gradcam_url": f"/gradcam/{fname}"})


@app.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/gradcam/<path:filename>")
def serve_gradcam(filename: str):
    return send_from_directory(GRADCAM_DIR, filename)


# --------------------------- helpers --------------------------------------

@app.errorhandler(413)
def too_large(_):
    return _err("Upload too large. Maximum file size is 10 MB.", 413)


@app.errorhandler(404)
def not_found(_):
    return _err("Not found.", 404)


def _err(message: str, code: int):
    return jsonify({"error": message}), code


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    log.info("Starting RetinaScan on http://127.0.0.1:%d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
