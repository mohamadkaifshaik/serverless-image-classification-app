"""
Train a MobileNetV2 flower classifier with transfer learning.

Two phases:
  1. freeze the ImageNet base, train only the new classification head
  2. unfreeze the top N layers of the base and fine-tune everything with a small LR

Outputs (relative to repo root):
  model/flowers_mobilenetv2.keras
  model/labels.json
  training/logs/history.json

Usage:
  python training/train.py                  # full run
  python training/train.py --quick          # 1 epoch per phase, for smoke tests
"""
import argparse
import json
import os
import pathlib
import time

import tensorflow as tf

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
LOG_DIR = ROOT / "training" / "logs"

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

IMG_SIZE = int(os.environ.get("IMG_SIZE", 224))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--head-epochs", type=int, default=8)
    p.add_argument("--finetune-epochs", type=int, default=8)
    p.add_argument("--finetune-layers", type=int, default=50,
                   help="how many layers from the top of MobileNetV2 to unfreeze")
    p.add_argument("--quick", action="store_true", help="1 epoch per phase, tiny subset")
    p.add_argument("--data-dir", default=None, help="existing flower_photos dir (skips download)")
    return p.parse_args()


def load_datasets(data_dir, quick=False):
    if data_dir is None:
        # keras 2 returns the archive path and extracts next to it, keras 3 returns
        # <name>_extracted/ and the real folder sits one level below. handle both.
        got = pathlib.Path(tf.keras.utils.get_file("flower_photos.tgz", origin=DATASET_URL, extract=True))
        candidates = [
            got / "flower_photos",
            got.with_suffix(""),
            got.parent / "flower_photos_extracted" / "flower_photos",
            got.parent / "flower_photos",
        ]
        data_dir = next(c for c in candidates if (c / "daisy").is_dir())
    data_dir = pathlib.Path(data_dir)
    print(f"dataset dir: {data_dir}")

    common = dict(
        validation_split=0.2,
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, subset="validation", **common)
    class_names = train_ds.class_names
    print("classes:", class_names)

    if quick:
        train_ds = train_ds.take(8)
        val_ds = val_ds.take(4)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names


def build_model(num_classes):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = data_augmentation(inputs)
    # MobileNetV2 expects pixels in [-1, 1]
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(x)
    # training=False keeps the BatchNorm layers in inference mode even when we fine-tune later
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    model = tf.keras.Model(inputs, outputs, name="flowers_mobilenetv2")
    return model, base


def export_for_inference(model, path):
    """Save the model without optimizer state.

    A compiled keras 3 model writes the Adam slots into model.weights.h5 (~3x the size, 27MB
    instead of 9MB for MobileNetV2) and include_optimizer=False is ignored for the .keras format.
    Reloading with compile=False and saving again gives us just the weights.
    """
    tmp = path.with_suffix(".tmp.keras")
    model.save(tmp)
    clean = tf.keras.models.load_model(tmp, compile=False)
    clean.save(path)
    tmp.unlink()


def merge_history(h1, h2):
    out = {}
    for k in h1.history:
        out[k] = [float(v) for v in h1.history[k]] + [float(v) for v in h2.history.get(k, [])]
    return out


def main():
    args = parse_args()
    head_epochs = 1 if args.quick else args.head_epochs
    ft_epochs = 1 if args.quick else args.finetune_epochs

    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = load_datasets(args.data_dir, args.quick)
    model, base = build_model(len(class_names))

    # ---- phase 1: train the head ------------------------------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    print(f"\n== phase 1: head only, {head_epochs} epochs ==")
    t0 = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs)
    print(f"phase 1 done in {time.time() - t0:.0f}s, val_acc={h1.history['val_accuracy'][-1]:.4f}")

    # ---- phase 2: fine-tune top of the base --------------------------------------
    base.trainable = True
    for layer in base.layers[:-args.finetune_layers]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        )
    ]
    print(f"\n== phase 2: fine-tune top {args.finetune_layers} layers, {ft_epochs} epochs ==")
    t0 = time.time()
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs + ft_epochs,
        initial_epoch=head_epochs,
        callbacks=callbacks,
    )
    print(f"phase 2 done in {time.time() - t0:.0f}s")

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nfinal validation accuracy: {val_acc:.4f} (loss {val_loss:.4f})")

    # ---- export -------------------------------------------------------------------
    model_path = MODEL_DIR / "flowers_mobilenetv2.keras"
    export_for_inference(model, model_path)
    (MODEL_DIR / "labels.json").write_text(json.dumps(class_names, indent=2))
    history = merge_history(h1, h2)
    history["final_val_accuracy"] = float(val_acc)
    history["final_val_loss"] = float(val_loss)
    history["img_size"] = IMG_SIZE
    history["batch_size"] = BATCH_SIZE
    history["head_epochs"] = head_epochs
    history["finetune_epochs"] = ft_epochs
    (LOG_DIR / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved model -> {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
