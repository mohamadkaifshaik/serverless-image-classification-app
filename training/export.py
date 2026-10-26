"""Re-save an already trained .keras file without optimizer state (see export_for_inference in train.py).

    python training/export.py model/flowers_mobilenetv2.keras
"""
import pathlib
import sys

import tensorflow as tf

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "model/flowers_mobilenetv2.keras")
before = src.stat().st_size
model = tf.keras.models.load_model(src, compile=False)
model.save(src)
print(f"{src}: {before / 1e6:.1f} MB -> {src.stat().st_size / 1e6:.1f} MB")
