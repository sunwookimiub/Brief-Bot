# Documentation Oracle — Hybrid Agentic RAG System

A production-grade RAG system for technical PDFs (manuals, white papers, specs).

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT REQUEST                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │    Nginx    │  proxy_buffering off → SSE streams
                    │   Ingress   │  proxy_read_timeout 300s → slow LLMs
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │  Gunicorn + Uvicorn     │  2 workers, non-blocking event loop
              │       FastAPI           │
              │  /ingest  /query        │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │      RAG Agent          │  ReAct loop (Thought→Action→Obs)
              │   (LangChain AgentExec) │
              └──────┬──────────┬───────┘
                     │          │
          ┌──────────▼──┐  ┌────▼──────────┐
          │ vector_search│  │keyword_search  │
          │  (ChromaDB)  │  │  (BM25 index)  │
          │   semantic   │  │  exact-match   │
          └──────────────┘  └───────────────┘
                     │          │
              ┌──────▼──────────▼──────┐
              │    Prompt Manager       │
              │  GCS versioned YAMLs   │  v1, v2, ... → logged per response
              └────────────────────────┘
```

## Project Structure

```
rag-system/
│
├── .env                          ← your secrets (copy from .env.example)
├── .env.example                  ← template showing all required vars
├── requirements.txt              ← all Python dependencies
├── Dockerfile                    ← multi-stage container build
│
├── app/
│   ├── __init__.py
│   ├── main.py                   ← FastAPI app, /ingest /query /health endpoints
│   ├── models.py                 ← Pydantic request/response shapes
│   ├── config.py                 ← all settings loaded from .env
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   └── rag_agent.py          ← LangChain ReAct agent, picks search tool
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── pipeline.py           ← GCS → PDF → chunks → ChromaDB + BM25
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   └── search_tools.py       ← vector_search and keyword_search tools
│   │
│   └── prompts/
│       ├── __init__.py
│       └── manager.py            ← loads prompt YAMLs from GCS by version
│
├── prompts/
│   └── rag_agent_v1.yaml         ← default prompt template (upload to GCS)
│
└── k8s/
    └── manifests.yaml            ← Deployment, Service, Ingress, PVC for GKE
```

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env: set GCS_BUCKET, ANTHROPIC_API_KEY, VOYAGE_API_KEY etc.

# 2. Install deps
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# 4. Ingest documents
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{"gcs_prefix": "manuals/test/", "doc_version": "v1"}'

# 5. Query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "prompt_version": "v1"}'
```

## Environment Variables

| Variable | Description | Example |
|---|---|---|
| `GCS_BUCKET` | Your GCS bucket name | `rag-docs-yourname` |
| `GOOGLE_CLOUD_PROJECT` | Your GCP project ID | `rag-system-2026` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `VOYAGE_API_KEY` | Voyage AI API key | `pa-...` |
| `EMBEDDING_MODEL` | Voyage model to use | `voyage-3` |
| `LLM_MODEL` | Claude model to use | `claude-sonnet-4-6` |
| `VECTOR_DB_TYPE` | Vector store | `chroma` |
| `CHROMA_PERSIST_DIR` | Local ChromaDB path | `./chroma_store` |
| `CHUNK_SIZE` | Tokens per chunk | `512` |
| `CHUNK_OVERLAP` | Overlap between chunks | `64` |
| `TOP_K_SEMANTIC` | Semantic results to fetch | `5` |
| `TOP_K_KEYWORD` | Keyword results to fetch | `5` |

## Endpoints

### `GET /health`
Kubernetes readiness/liveness probe. Returns 200 only when ChromaDB is connected.
```json
{"status": "healthy", "vector_db": "connected"}
```

### `POST /ingest`
Reads PDFs from GCS, chunks them, embeds with Voyage AI, stores in ChromaDB and builds a BM25 keyword index.
```json
// Request
{"gcs_prefix": "manuals/euv/", "doc_version": "v2.1", "async_mode": false}

// Response
{"status": "success", "message": "Ingested 12 pages → 89 chunks", "chunks_created": 89, "doc_version": "v2.1"}
```

### `POST /query`
Agentic RAG loop. The agent chooses between semantic and keyword search based on the question.
```json
// Request
{"question": "What is the error recovery for ASML_EUV_001?", "prompt_version": "v1"}

// Response
{
  "answer": "According to page 42 of euv_manual.pdf...",
  "sources": [{"tool": "keyword_search", "reference": "File: euv_manual.pdf, Page: 42"}],
  "search_strategy": "keyword",
  "prompt_version": "v1",
  "latency_ms": 1823.4
}
```

### `POST /query/stream`
Same as `/query` but streams the response as Server-Sent Events.

## Key Design Decisions

### Hybrid Search (Semantic + BM25)
Technical docs contain exact identifiers (`ASML_EUV_001`, `get_laser_power()`, `0x4F2`)
that pure vector search misses. The agent chooses the right tool:
- **vector_search** → conceptual questions ("how does the EUV lens work")
- **keyword_search** → exact IDs ("find all mentions of ASML_EUV_001")
- **hybrid** → ambiguous queries (agent calls both)

### Voyage AI Embeddings
Anthropic's recommended embedding partner. `voyage-3` is optimized for retrieval
and pairs well with Claude for technical content. No OpenAI dependency needed.

### Prompt Versioning
Prompts live in GCS as YAML files, not in code. Every API response logs `prompt_version`.
This means:
- Zero-downtime prompt updates (no redeploy needed)
- Full audit trail for which prompt generated which answer
- Easy A/B testing between prompt versions

### GCS Object Versioning
Enable on your bucket:
```bash
gsutil versioning set on gs://your-docs-bucket
```
When a manual is updated, the old version is preserved. Pass `doc_version` in `/ingest`
to tag chunks — then filter by version in `/query`.

## GKE Deployment

```bash
# Build and push image
docker build -t gcr.io/YOUR_PROJECT/rag-api:latest .
docker push gcr.io/YOUR_PROJECT/rag-api:latest

# Create namespace + deploy
kubectl apply -f k8s/manifests.yaml

# Check rollout
kubectl rollout status deployment/rag-api -n rag-system

# Watch pods
kubectl get pods -n rag-system -w
```

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `DefaultCredentialsError` | `GOOGLE_APPLICATION_CREDENTIALS` not set | Set env var to path of `rag-sa-key.json` |
| `ModuleNotFoundError: langchain_voyageai` | Missing package | `pip install langchain-voyageai` |
| `No PDFs found at gs://...` | Wrong prefix | Check trailing slash, verify with `gsutil ls` |
| `NotEnoughElementsException` | Fewer docs than TOP_K | Set `TOP_K_SEMANTIC=3` in `.env` |
| `403 Forbidden` on GCS | Service account missing role | Grant `roles/storage.admin` |

## Interview Talking Points

| Topic | What to Say |
|---|---|
| **Hybrid RAG** | "Pure vector search fails on exact technical IDs. Adding BM25 as a second tool and letting the agent pick the right one covers both conceptual questions and exact-match lookups." |
| **Voyage AI** | "Anthropic specifically recommends Voyage AI as the embedding complement to Claude. voyage-3 is optimized for retrieval — better recall on technical content than general-purpose embedding models." |
| **Prompt versioning** | "Prompts are YAML files in GCS. The API logs which version was used per request. I can update a prompt without redeploying, and I can reproduce any past response by looking up the prompt version in logs." |
| **Multi-stage Dockerfile** | "Stage 1 installs build deps and compiles packages. Stage 2 copies only the installed libraries — final image has no gcc or build tools, reducing attack surface and image size by ~60%." |
| **Readiness probe** | "The /health endpoint checks the ChromaDB connection. K8s won't route traffic to a pod until it passes — so during rolling deploys, users never hit a pod whose vector DB isn't warm yet." |
| **Gunicorn + Uvicorn** | "Gunicorn spawns N worker processes. Each worker runs an async Uvicorn event loop. LLM streaming calls don't block other requests because they're async — critical when multiple users query simultaneously." |
| **Nginx SSE streaming** | "I set proxy_buffering off and proxy_read_timeout 300s. Without this, Nginx buffers the entire LLM response before forwarding it — you'd lose the streaming UX entirely." |
