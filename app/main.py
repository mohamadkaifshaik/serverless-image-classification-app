"""Flask app exposing the classifier. Runs under gunicorn in the container (see Dockerfile)."""
import base64
import binascii
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from app import model

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 10))
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app, origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")])


@app.get("/")
def index():
    return jsonify({"service": "image-classifier", "endpoints": ["/health", "/predict"]})


@app.get("/health")
def health():
    try:
        labels = model.get_labels()
    except Exception as exc:  # model missing / corrupt -> report unhealthy so Cloud Run restarts us
        return jsonify({"status": "error", "detail": str(exc)}), 503
    return jsonify({"status": "ok", "classes": labels})


def _read_image_from_request() -> bytes:
    # 1) multipart/form-data with a "file" field (what the frontend sends)
    if "file" in request.files:
        data = request.files["file"].read()
        if not data:
            raise ValueError("uploaded file is empty")
        return data

    # 2) JSON body {"image": "<base64>"} - handy for curl / other services
    payload = request.get_json(silent=True) or {}
    b64 = payload.get("image")
    if b64:
        if b64.strip().startswith("data:") and "," in b64:  # strip data-url prefix
            b64 = b64.split(",", 1)[1]
        try:
            return base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image field is not valid base64") from exc

    raise ValueError("send an image as multipart field 'file' or JSON {'image': '<base64>'}")


@app.post("/predict")
def predict():
    try:
        image_bytes = _read_image_from_request()
        top_k = int(request.args.get("top_k", 3))
        result = model.predict(image_bytes, top_k=max(1, min(top_k, 10)))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"file too large, max {MAX_UPLOAD_MB} MB"}), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "not found"}), 404


if __name__ == "__main__":
    # local dev only; Cloud Run uses gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
