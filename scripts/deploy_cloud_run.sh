#!/usr/bin/env bash
# Build the image with Cloud Build and deploy it to Cloud Run.
#   PROJECT_ID=my-project REGION=us-central1 ./scripts/deploy_cloud_run.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-image-classifier}"
SERVICE="${SERVICE:-image-classifier}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "PROJECT_ID is not set and no default gcloud project configured" >&2
  exit 1
fi

if [[ ! -f model/flowers_mobilenetv2.keras ]]; then
  echo "model/flowers_mobilenetv2.keras not found - run training/train.py first" >&2
  exit 1
fi

echo ">> project=$PROJECT_ID region=$REGION image=$IMAGE"

gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
  --project "$PROJECT_ID"

if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" --repository-format docker --location "$REGION" \
    --project "$PROJECT_ID" --description "image classifier containers"
fi

gcloud builds submit --tag "$IMAGE" --project "$PROJECT_ID" .

gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 8 \
  --min-instances 0 \
  --max-instances 5 \
  --port 8080

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT_ID" --format "value(status.url)")
echo
echo "deployed: $URL"
echo "try:      curl -s -F file=@some_flower.jpg $URL/predict"
echo "next:     put the URL into frontend/public/config.js and run firebase deploy from ./frontend"
