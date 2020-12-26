#!/bin/bash
# This scripts performs cloud training for a PyTorch model.
echo "Running cloud ML model"

# IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry
IMAGE_REPO_NAME=benchmark

# IMAGE_TAG: an easily identifiable tag for your docker image
IMAGE_TAG=conv

PROJECT_ID=convbenchmark

BUCKET_ID=convbenchmark

# IMAGE_URI: the complete URI location for Cloud Container Registry
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
REGION=us-central1

# Build the docker image
docker build -f Dockerfile -t ${IMAGE_URI} ./

# Deploy the docker image to Cloud Container Registry
docker push ${IMAGE_URI}

# Submit your training job
echo "Submitting the training job"

# These variables are passed to the docker image
JOB_DIR=gs://${BUCKET_ID}/
# Note: these files have already been copied over when the image was built

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=custom_cpu_container_job_$(date +%Y%m%d_%H%M%S)

gcloud beta ai-platform jobs submit prediction ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --config config.yaml \
    -- \
    --job-dir ${JOB_DIR} \
    --gpu false \

gcloud ai-platform jobs stream-logs ${JOB_NAME}

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=custom_gpu_container_job_$(date +%Y%m%d_%H%M%S)

gcloud beta ai-platform jobs submit prediction ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier BASIC_GPU \
    -- \
    --job-dir ${JOB_DIR} \
    --gpu true \

# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}

# Verify the model was exported
echo "Verify the model was exported:"
gsutil ls ${JOB_DIR}/

