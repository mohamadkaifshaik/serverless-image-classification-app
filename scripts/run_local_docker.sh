#!/usr/bin/env bash
# Build and run the API container locally on :8080, same as it would run on Cloud Run.
set -euo pipefail
docker build -t image-classifier .
docker run --rm -p 8080:8080 -e PORT=8080 image-classifier
