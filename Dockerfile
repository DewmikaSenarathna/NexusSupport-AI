# --- WHY DOCKER ---
# Containerizing means "it works on my machine" becomes "it works everywhere":
# the same image runs identically on your laptop, a teammate's machine, and
# AWS/GCP. This is standard practice for shipping ML services to production.

FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss / torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the classical model + generate data at build time so the image is
# ready to serve immediately on start (skip this in real production and
# instead load a pre-trained model artifact from cloud storage / a model
# registry — training inside the image is fine for a demo, not for scale).
RUN python data/generate_sample_data.py && \
    cd src && python classical_ml.py

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
