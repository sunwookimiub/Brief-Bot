# Documentation Oracle — Hybrid Agentic RAG System

A production-grade RAG system for technical PDFs (manuals, white papers, specs).

```
┌──────────────────────────────────────────────────────────────────────┐
│           CLIENT REQUEST (HTTP or MCP protocol)                      │
└──────────────┬─────────────────────────────┬────────────────────────┘
               │ HTTP                         │ MCP (stdio)
        ┌──────▼──────┐               ┌───────▼────────┐
        │    Nginx    │               │   MCP Server   │  ← Claude Desktop,
        │   Ingress   │               │ app/mcp/server │     Cursor, VS Code
        └──────┬──────┘               └───────┬────────┘
               │                              │
    ┌──────────▼────────────────────────────── ▼──────────┐
    │           Gunicorn + Uvicorn / FastAPI               │
    │              /ingest   /query   /health              │
    └─────────────────────────┬────────────────────────────┘
                              │
             ┌────────────────▼─────────────────┐
             │           RAG Agent               │  ReAct loop
             │      (LangChain AgentExec)        │  Thought→Action→Obs
             └──────┬──────────┬──────────┬──────┘
                    │          │          │
         ┌──────────▼──┐ ┌─────▼──────┐ ┌▼─────────────────────┐
         │vector_search│ │keyword_    │ │  MCP Client Bridge    │
         │ (ChromaDB)  │ │search(BM25)│ │  (app/mcp/client_     │
         │  semantic   │ │exact-match │ │   bridge.py)          │
         └─────────────┘ └────────────┘ └──┬────────┬───────────┘
                                           │        │
                                   ┌───────▼──┐ ┌───▼───────┐
                                   │Confluence│ │   JIRA    │  (any MCP
                                   │MCP Server│ │MCP Server │   server)
                                   └──────────┘ └───────────┘
                    │
         ┌──────────▼──────────┐
         │    Prompt Manager   │  GCS versioned YAMLs
         │  (app/prompts/)     │  v1, v2 ... logged per response
         └─────────────────────┘
```

## Project Structure

```
rag-system/
├── app/
│   ├── main.py              # FastAPI app + lifespan
│   ├── models.py            # Pydantic request/response models
│   ├── config.py            # Settings (env-driven)
│   ├── agents/
│   │   └── rag_agent.py     # LangChain ReAct agent (+ MCP tool injection)
│   ├── tools/
│   │   └── search_tools.py  # vector_search + keyword_search tools
│   ├── ingestion/
│   │   └── pipeline.py      # GCS → PDF → chunk → embed → store
│   ├── mcp/
│   │   ├── server.py        # MCP Server: exposes RAG as tools to Claude Desktop etc.
│   │   └── client_bridge.py # MCP Client: pulls in external tools (Confluence, JIRA...)
│   └── prompts/
│       └── manager.py       # GCS prompt version loader
├── k8s/
│   └── manifests.yaml       # Deployment + Service + Ingress + PVC
├── prompts/
│   └── rag_agent_v1.yaml    # Default prompt (upload to GCS)
├── mcp_servers.yaml         # Configure which external MCP servers to connect
├── Dockerfile               # Multi-stage build
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env: set GCS_BUCKET, ANTHROPIC_API_KEY, etc.

# 2. Install deps
pip install -r requirements.txt

# 3. Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# 4. Ingest documents
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{"gcs_prefix": "manuals/euv/", "doc_version": "v2.1"}'

# 5. Query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the error recovery for ASML_EUV_001?", "prompt_version": "v1"}'
```

## MCP Integration

### MCP Server — Use the RAG system from Claude Desktop / Cursor

The `app/mcp/server.py` exposes the RAG system as 5 MCP tools. Any MCP-compatible client can call them directly — no HTTP required.

**Claude Desktop config** (`~/.config/claude/config.json`):
```json
{
  "mcpServers": {
    "documentation-oracle": {
      "command": "python",
      "args": ["-m", "app.mcp.server"],
      "env": { "GCS_BUCKET": "your-bucket", "ANTHROPIC_API_KEY": "..." }
    }
  }
}
```

Once added, Claude Desktop gets these tools automatically:
- `ingest_documents` — index new PDFs from GCS
- `query_documents` — full agentic RAG query
- `vector_search` — direct semantic search
- `keyword_search` — direct BM25 exact-match
- `list_indexed_docs` — inspect what's indexed

### MCP Client — Bring live external data into the agent

Edit `mcp_servers.yaml` to connect external MCP servers. The RAG agent will call them alongside its local search tools in the same ReAct loop.

**Example query that uses both:**
> "What does the manual say about error ASML_EUV_001, and is there an open JIRA ticket for it?"

The agent will: `keyword_search` the manual → `jira__search_issues` for the ticket → synthesize both into one answer.

Enable servers in `mcp_servers.yaml`:
```yaml
servers:
  jira:
    enabled: true      # ← flip this
    command: npx
    args: ["-y", "@modelcontextprotocol/server-jira"]
    env:
      JIRA_URL: "https://your-company.atlassian.net"
      JIRA_API_TOKEN: "..."
```

## Key Design Decisions

### Hybrid Search (Semantic + BM25)
Technical docs contain exact identifiers (`ASML_EUV_001`, `get_laser_power()`,
`0x4F2`) that pure vector search misses. The agent chooses the right tool:
- **vector_search** → conceptual questions ("how does the EUV lens work")
- **keyword_search** → exact IDs ("find all mentions of ASML_EUV_001")
- **both** → ambiguous queries

### Prompt Versioning
Prompts live in GCS as YAML files, not in code. Every API response logs
`prompt_version`. This means:
- Zero-downtime prompt updates (no redeploy needed)
- Full audit trail for which prompt generated which answer
- Easy A/B testing between prompt versions

### GCS Object Versioning
Enable on your bucket:
```bash
gsutil versioning set on gs://your-docs-bucket
```
When a manual is updated, the old version is preserved. Pass `doc_version`
in `/ingest` to tag chunks — then filter by version in `/query`.

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

## Interview Talking Points

| Topic | What to Say |
|-------|-------------|
| **MCP Server** | "I exposed the RAG system as an MCP server so Claude Desktop and Cursor can call it directly as tools — no API client needed. It's the same tool protocol Anthropic uses internally." |
| **MCP Client** | "The agent can call external MCP servers — JIRA, Confluence, GitHub — as tools in the same ReAct loop as the vector search. So it can answer 'what does the spec say AND is there an open bug for this' in one query." |
| **Multi-stage Dockerfile** | "Stage 1 installs build deps and compiles packages. Stage 2 copies only the installed libraries — final image has no gcc or build tools, reducing attack surface and image size by ~60%." |
| **Readiness probe** | "The /health endpoint checks the ChromaDB connection. K8s won't route traffic to a pod until it passes — so during rolling deploys, users never hit a pod whose vector DB isn't warm yet." |
| **Gunicorn + Uvicorn** | "Gunicorn spawns N worker processes. Each worker runs an async Uvicorn event loop. LLM streaming calls don't block other requests because they're async — critical when you have multiple users querying simultaneously." |
| **Nginx SSE streaming** | "I set proxy_buffering off and proxy_read_timeout 300s. Without this, Nginx buffers the entire LLM response before forwarding it — you'd lose the streaming UX entirely." |
| **Hybrid RAG** | "Pure vector search fails on exact technical IDs. Adding BM25 as a second tool and letting the agent pick the right one covers both conceptual questions and exact-match lookups." |
| **Prompt versioning** | "Prompts are YAML files in GCS. The API logs which version was used per request. I can update a prompt without redeploying, and I can reproduce any past response by looking up the prompt version in logs." |
