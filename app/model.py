"""Model loading + preprocessing + inference. Kept separate from the Flask layer so it's easy to test."""
import io
import json
import os
import pathlib
import threading
import time

import numpy as np
from PIL import Image, ImageOps

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = pathlib.Path(os.environ.get("MODEL_PATH", ROOT / "model" / "flowers_mobilenetv2.keras"))
LABELS_PATH = pathlib.Path(os.environ.get("LABELS_PATH", ROOT / "model" / "labels.json"))

_lock = threading.Lock()
_model = None
_labels = None
_img_size = None


def _load():
    """Load the keras model once. tensorflow is imported lazily so importing this module stays cheap."""
    global _model, _labels, _img_size
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        import tensorflow as tf  # lazy on purpose

        t0 = time.time()
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        labels = json.loads(LABELS_PATH.read_text())
        h, w = model.input_shape[1], model.input_shape[2]
        if len(labels) != model.output_shape[-1]:
            raise RuntimeError(
                f"labels.json has {len(labels)} entries but the model outputs {model.output_shape[-1]} classes"
            )
        # warm-up so the first real request does not pay for graph tracing
        model.predict(np.zeros((1, h, w, 3), dtype=np.float32), verbose=0)
        _model, _labels, _img_size = model, labels, (w, h)
        print(f"model loaded from {MODEL_PATH} in {time.time() - t0:.1f}s, classes={labels}")


def get_labels():
    _load()
    return list(_labels)


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Decode bytes -> RGB -> resize to the model input -> float32 batch of 1.

    Pixel scaling to [-1, 1] happens inside the model (Rescaling layer), so we hand over raw 0-255 floats.
    """
    _load()
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)  # respect phone orientation metadata
        img = img.convert("RGB")
    except Exception as exc:  # PIL raises a bunch of different things for garbage input
        raise ValueError("could not decode image") from exc
    img = ImageOps.fit(img, _img_size, method=Image.Resampling.BILINEAR)  # center crop instead of squashing
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict(image_bytes: bytes, top_k: int = 3) -> dict:
    _load()
    batch = preprocess(image_bytes)
    t0 = time.time()
    probs = _model.predict(batch, verbose=0)[0]
    elapsed_ms = int((time.time() - t0) * 1000)

    order = np.argsort(probs)[::-1][:top_k]
    preds = [{"label": _labels[i], "confidence": round(float(probs[i]), 4)} for i in order]
    return {
        "label": preds[0]["label"],
        "confidence": preds[0]["confidence"],
        "predictions": preds,
        "inference_ms": elapsed_ms,
    }
