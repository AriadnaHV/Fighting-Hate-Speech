#!/bin/bash
# =======================================================================
# SinOdio/SansHaine — Cloud Run Deployment Script
# These commands are run these commands one by one from the 
# sinodio_api/ directory on local machine (where gcloud CLI is installed)
# =======================================================================

# -- 0. Variables  ------------------------------------------------------
PROJECT_ID="project-5c89dcac-34cb-453d-bd7" # GCP project ID
REGION="europe-west4"                       # same region as your buckets
SERVICE_NAME="sinodio-api"
IMAGE_NAME=IMAGE_NAME="europe-west4-docker.pkg.dev/${PROJECT_ID}/sinodio-api/sinodio-api"

# -- 1. Authenticate with GCP -------------------------------------------
gcloud auth login
gcloud config set project ${PROJECT_ID}

# -- 2. Enable required APIs (only needed once) -------------------------
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# -- 3. Build the Docker image locally and push to GCR ------------------
# This takes ~5-10 minutes the first time (downloading base image + packages)
gcloud builds submit --tag ${IMAGE_NAME} .

# -- 4. Deploy to Cloud Run ---------------------------------------------
# --memory 2Gi   : XLM-RoBERTa needs ~1.5GB RAM on CPU
# --timeout 300  : allow 5 min for cold start (model download from GCS)
# --no-allow-unauthenticated : requires auth token — remove flag to make public
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 4 \
    --min-instances 0 \
    --max-instances 3 \
    --set-env-vars GOOGLE_CLOUD_PROJECT=${PROJECT_ID} \
    --allow-unauthenticated

# -- 5. Get the deployed URL -------------------------------------------
gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format "value(status.url)"

# -- 6. Test the deployed API ------------------------------------------
# Replace SERVICE_URL with the URL from step 5
SERVICE_URL="https://sinodio-api-xxxx-ew.a.run.app"   # <-- update this

# Health check
curl ${SERVICE_URL}/health

# Prediction test — hate speech example
curl -X POST "${SERVICE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "Ojalá que todos los inmigrantes se vayan de aquí"}'

# Prediction test — non-hate speech example  
curl -X POST "${SERVICE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "Hoy ha sido un día tranquilo en el parque"}'

# -- 7. View interactive API docs --------------------------------------
# Open in browser:  ${SERVICE_URL}/docs
echo "API docs: ${SERVICE_URL}/docs"
