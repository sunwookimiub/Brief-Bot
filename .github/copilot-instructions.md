<!-- Copilot / AI agent instructions for the rag-system-2026 repo -->
# Repo-specific guidance for AI coding agents

This file captures the minimal, actionable knowledge an AI coding assistant needs to be productive in this repository.

1) Big picture
- **Purpose:** a production-ready hybrid RAG system for technical PDFs: ingest PDFs from GCS, create dual indexes (semantic + BM25), and answer queries via an agentic ReAct loop.
- **Main components:**
  - API entry: [main.py](main.py) — FastAPI app with `/ingest`, `/query`, `/query/stream`, and `/health` endpoints.
  - Ingestion: `IngestionPipeline` in [pipeline.py](pipeline.py) — downloads PDFs from GCS, chunks, embeds, stores in vector DB and persists BM25 indices (`{chroma_persist_dir}/bm25_{doc_version}.pkl`).
  - Agent: `RAGAgent` in [rag_agent.py](rag_agent.py) — LangChain ReAct agent that composes `vector_search`, `keyword_search`, and optional MCP tools.
  - Tools: `VectorSearchTool` and `KeywordSearchTool` in [search_tools.py](search_tools.py).
  - Prompts: `PromptVersionManager` in [manager.py](manager.py) — loads prompts from GCS path `gs://{bucket}/{gcs_prompts_prefix}rag_agent_{version}.yaml` and falls back to internal defaults.
  - Config: `Settings` in [config.py](config.py) — environment-driven; check `.env` and `requirements.txt` for runtime/dependency expectations.

2) Key runtime & developer workflows (what agents should know)
- Run locally: `uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload` (see README.md Quick Start).
- Ingest: call `POST /ingest` with `gcs_prefix` and optional `doc_version`. Use `async_mode: true` for background ingestion.
- Query (sync): `POST /query` → returns `answer`, `sources`, `search_strategy`, and `prompt_version`.
- Streaming (SSE): `POST /query/stream` — the code yields SSE events; Nginx/Gunicorn proxy settings must allow streaming (README explains `proxy_buffering off` and timeouts).
- Health probe: `GET /health` depends on vector DB accessibility (used by K8s readiness/liveness).
- Prompt updates: prompts are stored in GCS (not hardcoded); use `PromptVersionManager.upload_prompt()` or upload a `rag_agent_{version}.yaml` to GCS. Prompt version is passed via `QueryRequest.prompt_version`.

3) Project-specific conventions & patterns
- Dual-index strategy: semantic (Chroma/FAISS) + BM25. BM25 files are persisted under `chroma_persist_dir` as `bm25_{doc_version}.pkl` (see [pipeline.py](pipeline.py)).
- Tool routing heuristic: `rag_agent.py` uses `TECHNICAL_ID_PATTERN` (regex) to hint `keyword_search` for alphanumeric IDs and function names. Do not remove this heuristic without updating tests and prompt guidance.
- Prompts are authoritative and versioned in GCS — avoid editing fallback text in code unless adding an explicit offline fallback; follow the prompt-versioning pattern.
- Executor caching: `RAGAgent` caches AgentExecutors per `(prompt_version, filters)`. Be careful when changing prompt loading behavior — invalidation is required.

4) Integration points & external dependencies
- GCS: `google-cloud-storage` used for documents and prompts. `Settings` contains `gcs_bucket` and `gcs_prompts_prefix` (see [config.py](config.py)).
- Vector DB: supports Chroma (default) or FAISS — controlled by `vector_db_type` in `Settings`.
- LLM providers: anthropic or openai; `Settings.llm_provider` chooses `ChatAnthropic` vs `ChatOpenAI` in `RAGAgent._build_llm()`.
- MCP: optional external tools are loaded from `mcp_servers.yaml` via the MCP client bridge. MCP tools are appended to the agent tool list and use `server__tool` naming (see `rag_agent.py`).

5) Concrete examples for code changes
- Add a new search tool: update [search_tools.py](search_tools.py), expose it via `RAGAgent._get_executor()` by appending to `local_tools` or `mcp_tools`, and update `react_template` guidance.
- Change prompt shape: update prompt YAML in GCS `prompts/rag_agent_{version}.yaml`. The code expects keys `system` and `agent_prefix`.
- Change chunking: adjust `chunk_size`/`chunk_overlap` in [config.py](config.py) and verify downstream consumers (BM25 tokenization and vector embeddings).

6) Short troubleshooting notes (discoverable from code)
- No BM25 index found → `KeywordSearchTool` returns a clear message: run `/ingest` first.
- If `/health` returns 503, check embeddings initialization in `RAGAgent.initialize()` and that the `chroma_persist_dir` is accessible.
- Long-running streaming responses require proxy settings that allow SSE (README documents Nginx/Gunicorn hints).

7) Where to look first (file map)
- App entry & routing: [main.py](main.py)
- Agent core: [rag_agent.py](rag_agent.py)
- Tools: [search_tools.py](search_tools.py)
- Ingest pipeline: [pipeline.py](pipeline.py)
- Prompt manager: [manager.py](manager.py)
- Config & models: [config.py](config.py), [models.py](models.py)
- Deployment hints: [README.md](README.md) and `manifests.yaml` for k8s

If anything here is unclear or you want additional examples (e.g., tests to update when changing prompt shape, or a sample prompt YAML), tell me which section to expand and I will iterate.
