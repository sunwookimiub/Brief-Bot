"""
Documentation Oracle - Hybrid Agentic RAG System
FastAPI entry point
"""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import uvicorn

from app.models import IngestRequest, QueryRequest, IngestResponse, QueryResponse
from app.ingestion.pipeline import IngestionPipeline
from app.agents.rag_agent import RAGAgent
from app.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()
ingestion_pipeline: IngestionPipeline | None = None
rag_agent: RAGAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy resources once on startup."""
    global ingestion_pipeline, rag_agent
    logger.info("Initializing RAG system components...")
    ingestion_pipeline = IngestionPipeline(settings)
    rag_agent = RAGAgent(settings)
    await rag_agent.initialize()
    logger.info("RAG system ready.")
    yield
    logger.info("Shutting down RAG system.")


app = FastAPI(
    title="Documentation Oracle",
    description="Hybrid Agentic RAG for technical PDFs",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """
    Kubernetes readiness/liveness probe endpoint.
    Returns 200 only when vector DB connection is live.
    """
    if rag_agent is None or not await rag_agent.is_healthy():
        raise HTTPException(status_code=503, detail="Vector DB not ready")
    return {"status": "healthy", "vector_db": "connected"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Phase 1: Reads PDFs from GCS, chunks them, embeds and stores in vector DB.
    Supports background ingestion for large document sets.
    """
    if ingestion_pipeline is None:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not ready")

    if request.async_mode:
        background_tasks.add_task(
            ingestion_pipeline.run,
            gcs_prefix=request.gcs_prefix,
            doc_version=request.doc_version,
        )
        return IngestResponse(
            status="accepted",
            message=f"Ingestion started for prefix: {request.gcs_prefix}",
            chunks_created=0,
        )

    result = await ingestion_pipeline.run(
        gcs_prefix=request.gcs_prefix,
        doc_version=request.doc_version,
    )
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Phase 2: Agentic RAG loop. The agent chooses between:
    - vector_search: semantic similarity for conceptual queries
    - keyword_search: BM25 exact match for part numbers, function names, IDs
    """
    if rag_agent is None:
        raise HTTPException(status_code=503, detail="RAG agent not ready")

    start_time = time.time()
    result = await rag_agent.query(
        question=request.question,
        prompt_version=request.prompt_version,
        filters=request.filters,
    )
    result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
    return QueryResponse(**result)


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming endpoint — important for Nginx/Gunicorn config discussion.
    Gunicorn + Uvicorn workers handle concurrent streams without blocking.
    """
    if rag_agent is None:
        raise HTTPException(status_code=503, detail="RAG agent not ready")

    async def event_generator():
        async for chunk in rag_agent.query_stream(
            question=request.question,
            prompt_version=request.prompt_version,
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)
