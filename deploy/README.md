# Cloud Deployment Guide

This project can be deployed for free (or nearly free) on either AWS or GCP.
Pick one — you don't need both for the internship application, but knowing
both is a plus.

## Option A: GCP Cloud Run (recommended — simplest)

Cloud Run runs containers and scales to zero when idle, so it's effectively
free for a low-traffic demo.

```bash
# 1. Install gcloud CLI and authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Build and push the container to Google Artifact Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ai-support-assistant

# 3. Deploy to Cloud Run
gcloud run deploy ai-support-assistant \
  --image gcr.io/YOUR_PROJECT_ID/ai-support-assistant \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key_here \
  --memory 2Gi

# 4. Cloud Run prints a public URL — visit {url}/docs
```

## Option B: AWS (EC2, simplest path for a first deployment)

```bash
# 1. Launch a small EC2 instance (t3.medium or larger — the RAG/embedding
#    step needs a bit of RAM), Ubuntu 22.04, open port 8000 in the security group

# 2. SSH in, install Docker
sudo apt update && sudo apt install -y docker.io
sudo systemctl start docker

# 3. Copy the project to the instance (scp or git clone), then:
sudo docker build -t ai-support-assistant .
sudo docker run -d -p 8000:8000 \
  -e GEMINI_API_KEY=your_key_here \
  ai-support-assistant

# 4. Visit http://<EC2_PUBLIC_IP>:8000/docs
```

For a more "production" AWS setup once you're comfortable with EC2:
push the image to **Elastic Container Registry (ECR)** and run it on
**ECS Fargate** (serverless containers, no server management) or
**AWS Lambda** with a container image (good for the classify-only endpoint,
since Lambda has cold-start/time limits that don't suit long RAG calls well).

## Environment variables needed in any environment

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Used by rag_pipeline.py / agent.py to call the Gemini LLM (free-tier key from Google AI Studio) |

## What to say about this in an interview

"I containerized the FastAPI service with Docker and deployed it to Cloud
Run, which auto-scales based on traffic and scales to zero when idle to
control cost. I kept model loading in the app's startup lifecycle rather
than per-request to avoid reloading the model on every call."
