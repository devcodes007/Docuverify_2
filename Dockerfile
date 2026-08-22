# DocuVerify v2 backend image.
# Multi-stage not used here: torch/transformers wheels dominate image size
# regardless of build stage, so a single stage keeps this simple without a
# meaningful size trade-off.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: build tools for any C-extension wheels without a prebuilt
# manylinux release, curl for the healthcheck below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY training ./training
COPY evaluation ./evaluation
COPY data/raw ./data/raw
COPY scripts ./scripts

# models/ and data/processed/ are populated at runtime by /ingest and
# training/train.py; mount them as a volume in docker-compose so they
# persist across container restarts instead of baking them into the image.
RUN mkdir -p data/processed models

EXPOSE 8000

# Render (and most PaaS Docker runtimes) inject $PORT and require the
# container to bind to it; default to 8000 for local `docker compose up`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
