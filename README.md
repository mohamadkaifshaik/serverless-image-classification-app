# Serverless Image Classification App

End-to-end image classifier: a MobileNetV2 model fine-tuned on the TensorFlow flowers dataset,
served from a Flask API in a Docker container on **GCP Cloud Run**, with a static front-end on
**Firebase Hosting** for uploading a photo and getting a prediction back in real time.

```
browser (Firebase Hosting)  --POST /predict-->  Cloud Run (Flask + gunicorn + TensorFlow)
```

**Stack:** TensorFlow / Keras 3, Flask, gunicorn, Docker, Cloud Run, Artifact Registry,
Cloud Build, Firebase Hosting. Front-end is plain HTML/CSS/JS, no framework or build step.

## Results

| Phase                              | Val accuracy |
|------------------------------------|--------------|
| Head only (frozen ImageNet base)   | 88.8%        |
| + fine-tune top 50 layers          | **91.3%**    |

5 classes (daisy, dandelion, roses, sunflowers, tulips), 3,670 images, 80/20 split, 224x224,
batch 32, 8 + 8 epochs with early stopping. Full per-epoch numbers are in
`training/logs/history.json`. Exported model is 9.7 MB; inference is ~130-170 ms per image on
one CPU inside the container, cold start (gunicorn + model load) is ~6 s.

## Layout

```
app/          Flask API (model.py = load/preprocess/predict, main.py = routes)
training/     train.py, export.py, training Dockerfile, logs/history.json
model/        exported .keras model + labels.json
frontend/     Firebase Hosting site (public/, firebase.json)
tests/        pytest suite for the API
scripts/      deploy_cloud_run.sh, run_local_docker.sh
Dockerfile    the Cloud Run image
cloudbuild.yaml   build -> push -> deploy pipeline for Cloud Build
```

## 1. Train

```bash
pip install -r requirements.txt
python training/train.py            # downloads tf_flowers (~220 MB) on first run
python training/train.py --quick    # 1 epoch per phase, just to see it work
```

Writes `model/flowers_mobilenetv2.keras`, `model/labels.json` and `training/logs/history.json`.

No TensorFlow wheel for your Python (I'm on 3.14 locally, TF stops at 3.12/3.13)? Use the
training container instead, it's the same script:

```bash
docker build -t flowers-train -f training/Dockerfile .
docker run --rm -v "$PWD/model:/app/model" -v "$PWD/training:/app/training" flowers-train
```

Training on a 16-core CPU takes about 25 minutes. A GPU gets it down to a couple of minutes.

## 2. Run the API locally

```bash
./scripts/run_local_docker.sh         # builds + runs the exact Cloud Run image on :8080
# or without docker:
python -m app.main
```

```bash
curl -s -F "file=@sunflower.jpg" http://localhost:8080/predict
```

```json
{
  "label": "sunflowers",
  "confidence": 0.9994,
  "predictions": [
    {"label": "sunflowers", "confidence": 0.9994},
    {"label": "daisy", "confidence": 0.0003},
    {"label": "dandelion", "confidence": 0.0002}
  ],
  "inference_ms": 131
}
```

| Method | Path       | Notes                                                                 |
|--------|------------|-----------------------------------------------------------------------|
| GET    | `/health`  | 200 + class list when the model is loaded, 503 otherwise              |
| POST   | `/predict` | multipart field `file`, or JSON `{"image": "<base64 or data-url>"}`. `?top_k=N` (1-10) |

Bad input comes back as JSON with a 400 (undecodable image, missing field, bad base64) or
413 (over `MAX_UPLOAD_MB`, default 10).

## 3. Deploy the API to Cloud Run

```bash
export PROJECT_ID=your-gcp-project
export REGION=us-central1
./scripts/deploy_cloud_run.sh
```

The script enables the APIs, creates an Artifact Registry repo if it's missing, builds the
image with Cloud Build and deploys to Cloud Run (`--allow-unauthenticated`, 1 vCPU, 1 GiB,
concurrency 8, scale to zero). `cloudbuild.yaml` does the same thing as a Cloud Build
trigger if you want it on every push.

## 4. Deploy the front-end to Firebase Hosting

```bash
cd frontend
firebase login
firebase use --add                 # same GCP project as the Cloud Run service
firebase deploy --only hosting
```

`firebase.json` rewrites `/api/**` to the Cloud Run service, so in production the page calls
`/api/predict` on its own origin and there is no CORS to deal with. When opened from
localhost the front-end talks to `http://localhost:8080` instead (see `public/config.js`).
If your service isn't called `image-classifier` in `us-central1`, change the rewrite.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

10 tests: health, multipart and base64 predictions, `top_k`, error handling, and the
preprocessing (resize/crop, grayscale and RGBA inputs, garbage bytes). They use the Flask
test client against the committed model and skip if `model/` is empty.

## Notes / things I ran into

- Keras 3 `get_file(extract=True)` returns a `*_extracted` directory, not the archive path.
  First run trained on a single class called `flower_photos`. See `load_datasets`.
- A compiled Keras 3 model saves the Adam slots into `model.weights.h5` and
  `include_optimizer=False` is a no-op for the `.keras` format, so the export was 27 MB.
  Reloading with `compile=False` and saving again brings it to 9.7 MB (`training/export.py`).
- gunicorn runs with `--preload` and 1 worker so the model is loaded once at boot and the
  first request isn't a 6 second cold start on top of Cloud Run's own.
