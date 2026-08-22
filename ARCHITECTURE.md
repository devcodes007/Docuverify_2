# DocuVerify v2 — Architecture

## 1. Purpose

DocuVerify v2 answers questions over a documentation corpus (default: FastAPI's
own docs) using retrieval-augmented generation, but treats retrieval and
verification as *decisions* rather than a fixed pipeline:

1. A query router decides **how** to retrieve (simple hybrid search, per-concept
   retrieval for comparisons, or decomposed multi-hop retrieval).
2. An evidence evaluator decides **whether retrieval succeeded**, and if not,
   reformulates the query and retries (bounded at `MAX_RETRIEVAL_ATTEMPTS`).
3. After generation, a **fine-tuned transformer classifier** (not another LLM
   prompt) independently judges whether the answer is `SUPPORTED`,
   `CONTRADICTED`, or `UNSUPPORTED` by the retrieved evidence, and the system
   retries or refuses based on that verdict.

This separates three concerns that a naive `query -> vector search -> LLM`
pipeline conflates: *retrieval strategy selection*, *evidence sufficiency*,
and *output verification*. Each is implemented as its own module with a
narrow interface so any piece (vector store, LLM provider, embedding model,
classifier) can be swapped without touching the others.

## 2. Data flow

```
Docs (HTML/Markdown)
  -> ingestion.loader        (fetch/read raw docs)
  -> ingestion.cleaner       (strip nav/boilerplate, keep headings+code)
  -> ingestion.chunker       (structure-aware chunking, heading_path aware)
  -> ingestion.metadata      (attach document_id, source_url, section, ...)
  -> retrieval.dense.embed   (sentence-transformers -> Chroma)
  -> retrieval.bm25.index    (rank_bm25 over the same chunks)

Query
  -> agents.query_router          classify: SIMPLE_LOOKUP | COMPARISON | MULTI_HOP
  -> agents.retrieval_agent       loop (<=3 attempts):
       plan_retrieval -> retrieve (bm25/dense/hybrid) -> evidence_evaluator
       -> if insufficient: query_rewriter -> retry
  -> generation.llm               generate answer strictly from evidence
  -> verification.groundedness    fine-tuned classifier -> SUPPORTED/CONTRADICTED/UNSUPPORTED
       -> CONTRADICTED or UNSUPPORTED: one more retrieval+generation attempt
       -> still not SUPPORTED: explicit refusal, no hallucinated answer
  -> agents.orchestrator          assembles AgentTrace + QueryResponse
```

## 3. Components

### Ingestion (`app/ingestion`)
- `loader.py` — pluggable `DocumentSource` interface. Ships a
  `LocalMarkdownSource` (reads `data/raw/*.md`/`*.html`) and a
  `WebDocSource` (fetches a list of URLs). Swapping documentation sets means
  writing a new `DocumentSource`, not touching the rest of the pipeline.
- `cleaner.py` — strips nav/sidebar/footer boilerplate from HTML via
  BeautifulSoup, normalizes whitespace, preserves fenced code blocks verbatim.
- `chunker.py` — walks a heading tree (`h1 > h2 > h3 > paragraph/code`) and
  emits chunks that never split a code block from its preceding explanation.
  Configurable `chunk_size`/`chunk_overlap`, measured in tokens (approximated
  by whitespace split, replaceable with a real tokenizer).
- `metadata.py` — attaches `document_id, source_url, title, section,
  subsection, heading_path, chunk_id, content_type`.

### Retrieval (`app/retrieval`)
- `bm25.py` — thin wrapper around `rank_bm25.BM25Okapi`, tokenized with a
  simple code-aware tokenizer (keeps `snake_case`/`CamelCase`/dotted
  identifiers intact so `HTTPException`, `Depends` etc. are retrievable).
- `dense.py` — `VectorStore` protocol + `ChromaVectorStore` implementation.
  Embedding model name comes from `EMBEDDING_MODEL` env var. The interface
  (`upsert`, `query`) is the only thing the rest of the app depends on, so
  swapping Chroma for Qdrant/Pinecone/Weaviate means writing one new class.
- `hybrid.py` — `hybrid_score = alpha * normalize(dense) + (1-alpha) *
  normalize(bm25)`, `alpha` from `HYBRID_ALPHA`. Normalizes each score list
  to [0,1] before combining so the two scales are comparable.
- `reranker.py` — optional cross-encoder-style rerank hook (no-op by default,
  pluggable so a real cross-encoder can be dropped in later).

### Agents (`app/agents`)
- `query_router.py` — rule-based classifier first (keyword patterns for
  "difference between", "compare", "vs", "and how does ... interact with"),
  falls back to an LLM classification prompt when rules are ambiguous. This
  is intentionally not "just an LLM call" — rules are cheap, deterministic,
  and testable; the LLM is the fallback for genuinely ambiguous phrasing.
- `retrieval_agent.py` — implements the bounded retry loop described above.
- `query_rewriter.py` — reformulates using the *previous* evidence gap
  (missing_information from the evaluator), not a generic rewrite.
- `evidence_evaluator.py` — deterministic signals (top score, score spread,
  chunk count, keyword/entity coverage of the query against retrieved text)
  combined into a confidence score; returns the structured
  `sufficient/confidence/missing_information/reason` object described in the
  spec. LLM-based evaluation is an optional strategy behind the same
  interface (`EvidenceEvaluator.evaluate`) — deterministic is the default
  because it's cheap, fast, and doesn't add another point of LLM failure.
- `orchestrator.py` — wires router -> retrieval agent -> generation ->
  groundedness -> retry/refusal, and builds the `AgentTrace`.

### Generation (`app/generation`)
- `llm.py` — `LLMProvider` protocol with `OllamaProvider` (local, default)
  and `OpenAICompatibleProvider` (optional, for any OpenAI-compatible API,
  key from env, never hardcoded) behind one interface selected by
  `LLM_PROVIDER`.
- `prompts.py` — the answer-generation prompt, which explicitly instructs
  the model not to use outside knowledge and to cite chunk ids.

### Verification (`app/verification`)
- `groundedness.py` — loads the fine-tuned classifier
  (`GROUNDEDNESS_MODEL` path) as a singleton; `GroundednessClassifier.predict(
  question, context, answer) -> {label, confidence}`.
- `claim_checker.py` — optional claim-level verification: splits the answer
  into sentences/claims (simple sentence segmentation), and re-runs the
  groundedness classifier per claim against the retrieved context.
- `retry_policy.py` — encodes the "CONTRADICTED -> retry once,
  UNSUPPORTED -> retry once, still bad -> refuse" state machine as a small,
  independently testable function so the orchestrator stays thin.

### Models (`app/models`)
- `schemas.py` — all Pydantic request/response/trace models.

### API (`app/api`)
FastAPI routers: `routes_health` (`/health`, `/info`), `routes_ingestion`
(`/ingest`, `/documents`), `routes_query` (`/query`, `/query/stream`,
`/retrieve`, `/evaluate`), `routes_debug` (`/metrics`, trace inspection).

## 4. Deep learning component: groundedness classifier

A `deberta-v3-small`-class sequence classifier (configurable, default
`microsoft/deberta-v3-small` — small enough for CPU inference, strong enough
for NLI-style tasks) fine-tuned on
`[CLS] question [SEP] context [SEP] answer [SEP]` -> 3-way label. Training
pipeline is in `training/`: `generate_dataset.py` builds
SUPPORTED/CONTRADICTED/UNSUPPORTED examples from ingested chunks,
`prepare_dataset.py` builds leakage-safe train/val/test splits (split by
source document, not randomly), `train.py` fine-tunes with Hugging Face
`Trainer`, `evaluate.py` reports per-class precision/recall/F1 + confusion
matrix. See README "Known limitations" for the sandbox's model-download
constraint.

## 5. Deployment

- Backend (FastAPI + ML deps) → Render/Railway/Fly.io style container host,
  because it needs to hold PyTorch + Transformers + a vector index in
  memory/disk, which does not fit Vercel's serverless function model
  (function size limits, no persistent disk, cold-start cost of loading a
  transformer per invocation).
- Frontend (static HTML/JS, calls the backend over HTTPS) → Vercel, via
  `vercel.json` with a rewrite to the backend's public URL.
- `docker-compose.yml` runs backend + a volume-mounted Chroma store for local
  dev; `Dockerfile` is the production backend image.

## 6. What is NOT included, on purpose

No Kafka/Kubernetes/Redis/Celery/microservices — a single FastAPI process
with an in-process retry loop and a local vector store is sufficient for this
workload and easier to reason about and deploy.
