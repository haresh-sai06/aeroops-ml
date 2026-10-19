#!/usr/bin/env bash
# Build + push to Artifact Registry and deploy to Cloud Run. Requires: gcloud auth login, a GCP project.
set -euo pipefail
PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"; REGION="${REGION:-asia-south1}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/aeroops/aeroops-api:$(git rev-parse --short HEAD)"
gcloud artifacts repositories create aeroops --repository-format=docker --location="$REGION" 2>/dev/null || true
gcloud builds submit --tag "$IMAGE" .
gcloud run deploy aeroops-api --image "$IMAGE" --region "$REGION" --port 8080 --cpu 2 --memory 2Gi \
  --concurrency 32 --min-instances 0 --max-instances 5 --allow-unauthenticated
