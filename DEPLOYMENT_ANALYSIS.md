# DocuVerify v2 Deployment Analysis

## Current Architecture

DocuVerify v2 is a FastAPI backend plus a static HTML frontend.

- Frontend: `frontend/index.html`, vanilla HTML/CSS/JavaScript, deployed by Vercel from `vercel.json`.
- Backend entry point: `app/main.py`, served with `uvicorn app.main:app`.
- Configuration: `app/config.py`, using environment variables through `pydantic-settings`.
- Dependency wiring: `app/dependencies.py`, using `@lru_cache` singletons for expensive objects.
- Ingestion: `app/ingestion/*`, exposed through `POST /ingest`, `POST /ingest/pdf`, and `scripts/ingest_docs.py`.
- Current vector database: local ChromaDB in `app/retrieval/dense.py`.
- Current dense embedding model: `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions.
- BM25: `app/retrieval/bm25.py`, in-memory `rank_bm25.BM25Okapi` with chunk metadata.
- Hybrid retrieval: `app/retrieval/hybrid.py`, transparent BM25 + dense score normalization and weighted merge.
- Query routing and agentic retrieval: `app/agents/query_router.py`, `app/agents/retrieval_agent.py`, `app/agents/query_rewriter.py`.
- Evidence sufficiency: `app/agents/evidence_evaluator.py`.
- LLM provider: `app/generation/llm.py`, supporting `ollama`, `openai_compatible`, and `mock`.
- Groundedness classifier: `app/verification/groundedness.py`, preferring a fine-tuned transformer at `GROUNDEDNESS_MODEL`, falling back to a heuristic classifier if no model artifact exists.
- Docker: `Dockerfile`, currently installs the monolithic `requirements.txt`.
- Render: `render.yaml`, currently configured for Docker and the `standard` plan because the free 512 MB tier is too tight.

The query path is:

```text
FastAPI /query
-> QueryRouter
-> RetrievalAgent
-> BM25 + Chroma dense retrieval
-> HybridRetriever
-> EvidenceEvaluator
-> LLMProvider
-> GroundednessClassifier
-> retry/refusal policy
```

## Current Memory-Heavy Components

The following are estimates based on package/model characteristics and observed behavior. Exact measurements are recorded separately in `MEMORY_OPTIMIZATION.md` after profiling.

| Component | Current Location | Estimated Memory Impact | Notes |
| --- | --- | ---: | --- |
| PyTorch runtime | `torch`, required by sentence-transformers and transformer classifier | 250-500 MB+ | Often dominates resident memory before model weights. |
| SentenceTransformer embedder | `SentenceTransformerEmbedder` | 80-150 MB model + PyTorch overhead | Loaded during startup auto-ingest or first dense query. |
| ChromaDB | `ChromaVectorStore` | 50-150 MB+ depending on index/corpus | Local HNSW/SQLite client plus documents and vectors. |
| Groundedness transformer | `TransformerGroundednessClassifier` | 150-400 MB+ with PyTorch | Currently falls back if no model artifact exists, but full production model would add memory. |
| BM25 index | `BM25Index` | corpus-dependent | Keeps chunk text and tokenized BM25 state in process memory. |
| Raw document/chunk duplication | BM25 pickle, Chroma documents, in-memory chunks | corpus-dependent | Same content can exist in BM25 RAM, Chroma storage, and payloads. |
| Training/dev dependencies | `datasets`, `accelerate`, `pandas`, `pytest` | install size and import risk | Not needed in production API runtime. |

## What Can Move To External Services

- Vector storage and dense vector search can move from ChromaDB to Qdrant Cloud.
- Dense embedding generation can move to Qdrant Cloud Inference when `QDRANT_CLOUD_INFERENCE=true`.
- LLM generation should use an external OpenAI-compatible API instead of local Ollama on Render.
- Large trained model artifacts should live outside Git, for example Hugging Face Hub or external artifact storage.

## What Must Remain In FastAPI

- API orchestration and routing.
- Query classification.
- BM25 lexical retrieval.
- Hybrid score merge and explainable score fields.
- Evidence sufficiency evaluation.
- Query reformulation/retry.
- LLM prompt construction and external API call.
- Groundedness classification interface and inference path.
- Refusal behavior.
- Frontend CORS/API contract.

## Qdrant Cloud Does Not Solve Everything

Qdrant replaces ChromaDB storage/search memory, but not automatically the embedding model memory. To make Render Free realistic, production should use Qdrant Cloud Inference for both document and query embeddings where available. That allows the backend to avoid loading `sentence-transformers` and PyTorch for embeddings.

The groundedness classifier remains a separate memory concern. It should be lazy-loaded, loaded once, optionally quantized for CPU inference, and kept as a fine-tuned classifier rather than replaced with an LLM prompt.

## Proposed Production Architecture

```text
Vercel static frontend
        |
        v
Render Free FastAPI backend
        |
        |-- BM25Index loaded from prepared artifact
        |-- Qdrant Cloud vector search
        |-- Qdrant Cloud Inference for dense embeddings
        |-- External OpenAI-compatible LLM API
        |-- Lazy groundedness classifier
        v
Verified QueryResponse
```

Production should use explicit ingestion, not startup ingestion:

```text
python scripts/ingest_docs.py
```

That command should build BM25 locally and upsert vectors/payloads into Qdrant Cloud. The API should then start with `AUTO_INGEST_ON_STARTUP=false` and only load the BM25 artifact plus lightweight Qdrant client.

## Proposed Configuration

Add or update environment variables:

```text
VECTOR_STORE_BACKEND=qdrant
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=docuverify_chunks
QDRANT_CLOUD_INFERENCE=true
QDRANT_INFERENCE_MODEL=sentence-transformers/all-MiniLM-L6-v2

LLM_PROVIDER=openai_compatible
LLM_MODEL=
LLM_API_BASE=
OPENAI_API_KEY=

AUTO_INGEST_ON_STARTUP=false
```

Local development can keep Chroma by setting:

```text
VECTOR_STORE_BACKEND=chroma
```

## Production Dependency Strategy

Split the single `requirements.txt` into:

- `requirements-prod.txt`: FastAPI, Qdrant client, BM25, parsing, external LLM client, minimal inference dependencies.
- `requirements-dev.txt`: tests and local Chroma support.
- `requirements-train.txt`: training-only dependencies such as `datasets`, `accelerate`, and heavyweight training packages.

If production uses Qdrant Cloud Inference and no local groundedness artifact is deployed, production can avoid `sentence-transformers`, `chromadb`, and training dependencies. If the transformer groundedness model is deployed, production still needs `torch` and `transformers`, and fitting 512 MB is not guaranteed.

## Key Implementation Decisions

1. Keep the current retrieval interfaces and add `QdrantVectorStore` behind the existing `VectorStore` protocol.
2. Use `VECTOR_STORE_BACKEND` to select `chroma` or `qdrant`.
3. Preserve `ChromaVectorStore` for local/offline development.
4. Use Qdrant payloads to preserve `document_id`, `chunk_id`, `content`, `title`, `section`, `heading_path`, `source_url`, and `content_type`.
5. Make ingestion explicit in production.
6. Change health checks so `/health` does not eagerly load the groundedness model.
7. Add a memory profiling script and record actual measurements in `MEMORY_OPTIMIZATION.md`.

## Current Risks

- Render Free may still fail if the fine-tuned groundedness transformer is loaded in-process.
- Qdrant Cloud Inference availability/model dimensions must be verified in the user's Qdrant Cloud cluster.
- Without Qdrant credentials, local tests must mock Qdrant and avoid live external calls.
- BM25 still stores chunks in memory; this is acceptable for the bundled corpus but must be measured for larger PDFs/docs.
