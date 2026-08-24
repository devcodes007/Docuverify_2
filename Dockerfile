# DocuVerify v2 backend image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: curl for the healthcheck below. Heavy training dependencies
# are intentionally excluded from the production image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

COPY app ./app
COPY data/raw ./data/raw

# models/ and data/processed/ are populated by explicit ingestion/training
# steps. Production should not rebuild dense indexes on every startup.
RUN mkdir -p data/processed models

EXPOSE 8000

# Render (and most PaaS Docker runtimes) inject $PORT and require the
# container to bind to it; default to 8000 for local `docker compose up`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
