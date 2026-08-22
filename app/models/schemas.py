"""All Pydantic models shared across the app: ingestion chunks, API
request/response bodies, and the agent trace object."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    PROSE = "prose"
    CODE = "code"
    MIXED = "mixed"


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    source_url: str
    title: str
    section: str = ""
    subsection: str = ""
    heading_path: list[str] = Field(default_factory=list)
    content_type: ContentType = ContentType.PROSE
    text: str

    def display_heading(self) -> str:
        return " > ".join(self.heading_path) if self.heading_path else self.title


# ---------------------------------------------------------------------------
# Query classification / retrieval
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    SIMPLE_LOOKUP = "SIMPLE_LOOKUP"
    COMPARISON = "COMPARISON"
    MULTI_HOP = "MULTI_HOP"


class RetrievalStrategy(str, Enum):
    BM25 = "BM25"
    DENSE = "DENSE"
    HYBRID = "HYBRID"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    bm25_score: float | None = None
    dense_score: float | None = None


class EvidenceEvaluation(BaseModel):
    sufficient: bool
    confidence: float
    missing_information: list[str] = Field(default_factory=list)
    reason: str


class RetrievalAttempt(BaseModel):
    attempt: int
    query: str
    strategy: RetrievalStrategy
    documents_retrieved: int
    evidence: EvidenceEvaluation


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------

class GroundednessLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"


class GroundednessResult(BaseModel):
    label: GroundednessLabel
    confidence: float
    per_class_scores: dict[str, float] = Field(default_factory=dict)


class ClaimVerification(BaseModel):
    claim: str
    status: GroundednessLabel
    evidence: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent trace
# ---------------------------------------------------------------------------

class TraceStep(BaseModel):
    step: str
    detail: dict[str, Any] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    query: str
    classification: QueryType | None = None
    subqueries: list[str] = Field(default_factory=list)
    attempts: list[RetrievalAttempt] = Field(default_factory=list)
    steps: list[TraceStep] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API request/response bodies
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)
    debug: bool = False


class SourceRef(BaseModel):
    chunk_id: str
    title: str
    section: str
    source_url: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    query_type: QueryType
    retrieval_strategy: RetrievalStrategy
    groundedness: GroundednessResult
    sources: list[SourceRef]
    retrieval_attempts: int
    latency_ms: int
    claims: list[ClaimVerification] | None = None
    trace: AgentTrace | None = None
    refused: bool = False


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = Field(default=8, ge=1, le=50)


class RetrieveResponse(BaseModel):
    results: list[RetrievedChunk]


class EvaluateRequest(BaseModel):
    question: str
    context: str
    answer: str


class EvaluateResponse(BaseModel):
    groundedness: GroundednessResult


class IngestRequest(BaseModel):
    source: str = Field(
        default="local",
        description="'local' reads data/raw, or a specific loader name registered in ingestion.loader",
    )
    urls: list[str] | None = None


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_created: int


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
    bm25_index_ready: bool
    groundedness_model_ready: bool


class InfoResponse(BaseModel):
    app_name: str
    environment: str
    llm_provider: str
    embedding_model: str
    max_retrieval_attempts: int
    hybrid_alpha: float
