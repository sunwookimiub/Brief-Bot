from pydantic import BaseModel, Field
from typing import Optional


class IngestRequest(BaseModel):
    gcs_prefix: str = Field(..., description="GCS path prefix, e.g. 'manuals/euv/'")
    doc_version: str = Field("latest", description="Version tag for rollback support")
    async_mode: bool = Field(False, description="Run ingestion in background")


class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_created: int
    doc_version: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language or technical ID query")
    prompt_version: str = Field("v1", description="Which prompt template to load")
    filters: Optional[dict] = Field(None, description="Metadata filters, e.g. {'doc_version': 'v2'}")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    search_strategy: str = Field(..., description="'semantic', 'keyword', or 'hybrid'")
    prompt_version: str
    latency_ms: Optional[float] = None
