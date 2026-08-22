"""
Central configuration.

Every tunable in this file is backed by an environment variable (see
.env.example). Nothing here is hardcoded for a specific deployment; the
same code runs against a laptop dev setup or a container in production by
changing environment variables only.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    app_name: str = "DocuVerify v2"
    environment: str = "development"  # development | production
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:8080,http://127.0.0.1:8080"
    auto_ingest_on_startup: bool = True

    # --- LLM ---
    llm_provider: str = "ollama"  # "ollama" | "openai_compatible" | "mock"
    llm_model: str = "llama3.1"
    llm_api_base: str = "http://localhost:11434"
    openai_api_key: str | None = None  # never hardcoded; loaded from env only
    hf_token: str | None = None

    # --- Embeddings / vector store ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    vector_db_path: str = "./data/processed/chroma"
    vector_db_collection: str = "docuverify_chunks"

    # --- BM25 ---
    bm25_index_path: str = "./data/processed/bm25_index.pkl"

    # --- Retrieval ---
    top_k: int = 8
    hybrid_alpha: float = 0.6  # weight on dense score vs bm25 score
    max_retrieval_attempts: int = 3
    evidence_sufficiency_threshold: float = 0.55

    # --- Chunking ---
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 50

    # --- Groundedness classifier ---
    groundedness_model: str = "./models/groundedness-classifier"
    groundedness_base_model: str = "microsoft/deberta-v3-small"
    groundedness_max_length: int = 512

    # --- Ingestion ---
    raw_data_dir: str = "./data/raw"
    allowed_ingest_domains: list[str] = ["fastapi.tiangolo.com"]

    # --- Misc ---
    request_timeout_seconds: int = 30

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS_ORIGINS is a comma-separated string (not a JSON list) so it's
        trivial to set as a single Render/Vercel environment variable
        (e.g. "https://docuverify.vercel.app,https://docuverify-git-main.vercel.app").
        Always includes localhost dev origins so local frontend development
        keeps working regardless of what's set for production."""
        configured = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        dev_origins = ["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:5173"]
        return sorted(set(configured) | set(dev_origins)) if self.environment != "production" else configured


@lru_cache
def get_settings() -> Settings:
    """Settings are cached as a singleton; call get_settings() everywhere
    instead of instantiating Settings() directly so env is read once."""
    return Settings()
