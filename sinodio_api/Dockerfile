# ─────────────────────────────────────────────────────────
# SinOdio/SansHaine — Hate Speech Detection API
# Dockerfile for Cloud Run deployment
# ─────────────────────────────────────────────────────────

# Python 3.11 slim — smaller than full image, no GPU needed for inference
FROM python:3.11-slim

# Prevent .pyc files and enable unbuffered stdout (important for Cloud Run logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# torch CPU-only build is ~200MB instead of ~2GB — sufficient for inference
COPY requirements_api.txt .
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY main.py .

# Cloud Run injects PORT environment variable; default to 8080
ENV PORT=8080
EXPOSE 8080

# Start with uvicorn — single worker is fine for Cloud Run (scales via instances)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
