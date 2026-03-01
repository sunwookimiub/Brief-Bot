"""
test_api.py
-----------
Smoke tests for the RAG API.
Run against local or GKE by setting BASE_URL.

Usage:
    # Local
    python test_api.py

    # GKE (after port-forward or with external IP)
    BASE_URL=http://34.123.45.67 python test_api.py
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

BASE_URL = os.getenv("BASE_URL", "http://localhost:8080")

# ── Colours for terminal output ───────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def passed(msg): print(f"  {GREEN}✓ {msg}{RESET}")
def failed(msg): print(f"  {RED}✗ {msg}{RESET}")
def info(msg):   print(f"  {YELLOW}→ {msg}{RESET}")


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as resp:
        return json.loads(resp.read())


# ── Tests ─────────────────────────────────────────────────────

def test_health():
    print(f"\n{BOLD}[1] Health Check{RESET}")
    resp = get("/health")
    assert resp["status"] == "healthy", f"Expected healthy, got: {resp}"
    passed(f"Status: {resp['status']} | Vector DB: {resp['vector_db']}")


def test_ingest(gcs_prefix: str, doc_version: str = "v1"):
    print(f"\n{BOLD}[2] Ingest — {gcs_prefix}{RESET}")
    start = time.time()
    resp = post("/ingest", {"gcs_prefix": gcs_prefix, "doc_version": doc_version})
    elapsed = round(time.time() - start, 2)

    if resp["status"] == "success":
        passed(f"{resp['message']} ({elapsed}s)")
    elif resp["status"] == "no_documents":
        failed(f"No PDFs found at prefix: {gcs_prefix}")
    else:
        failed(f"Unexpected status: {resp}")

    return resp


def test_query(question: str, expected_strategy: str = None, label: str = ""):
    print(f"\n{BOLD}[Query] {label or question[:60]}{RESET}")
    info(f"Question: {question}")

    start = time.time()
    resp = post("/query", {"question": question, "prompt_version": "v1"})
    elapsed = round(time.time() - start, 2)

    # Strategy check
    strategy = resp.get("search_strategy", "unknown")
    if expected_strategy and strategy != expected_strategy:
        failed(f"Expected strategy '{expected_strategy}', got '{strategy}'")
    else:
        passed(f"Search strategy: {strategy}")

    # Answer present
    answer = resp.get("answer", "")
    if answer and len(answer) > 20:
        passed(f"Got answer ({len(answer)} chars, {elapsed}s)")
        # Print first 200 chars of answer
        preview = answer[:200].replace("\n", " ")
        info(f"Preview: {preview}...")
    else:
        failed(f"Answer too short or missing: '{answer}'")

    # Sources present
    sources = resp.get("sources", [])
    if sources:
        passed(f"Sources returned: {len(sources)}")
        for s in sources[:3]:
            info(f"  {s['tool']} → {s['reference']}")
    else:
        failed("No sources returned")

    # Latency
    latency = resp.get("latency_ms", elapsed * 1000)
    if latency < 30000:
        passed(f"Latency: {latency}ms")
    else:
        failed(f"Latency too high: {latency}ms")

    return resp


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{BOLD}{'='*50}")
    print(f"RAG API Test Suite")
    print(f"Target: {BASE_URL}")
    print(f"{'='*50}{RESET}")

    errors = []

    # 1. Health
    try:
        test_health()
    except Exception as e:
        failed(f"Health check failed: {e}")
        errors.append("health")
        print(f"\n{RED}Server not reachable at {BASE_URL}. Is it running?{RESET}")
        sys.exit(1)

    # 2. Ingest — update this prefix to match your GCS bucket
    GCS_PREFIX = os.getenv("GCS_PREFIX", "manuals/test/")
    try:
        test_ingest(GCS_PREFIX, doc_version="v1")
    except Exception as e:
        failed(f"Ingest failed: {e}")
        errors.append("ingest")

    # 3. Semantic query — conceptual question, should use vector_search
    try:
        test_query(
            question="What is this document about? Give a summary.",
            expected_strategy=None,   # could be semantic or hybrid
            label="Semantic — summary question",
        )
    except Exception as e:
        failed(f"Semantic query failed: {e}")
        errors.append("semantic_query")

    # 4. Add your own domain-specific queries below
    # Replace these examples with terms from your actual PDFs

    # Example: keyword query (exact identifier)
    # try:
    #     test_query(
    #         question="What does ASML_EUV_001 refer to?",
    #         expected_strategy="keyword",
    #         label="Keyword — exact part number",
    #     )
    # except Exception as e:
    #     failed(f"Keyword query failed: {e}")
    #     errors.append("keyword_query")

    # Example: hybrid query
    # try:
    #     test_query(
    #         question="What is the error recovery procedure for error code 0x4F2?",
    #         expected_strategy="hybrid",
    #         label="Hybrid — concept + exact code",
    #     )
    # except Exception as e:
    #     failed(f"Hybrid query failed: {e}")
    #     errors.append("hybrid_query")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{BOLD}{'='*50}")
    if errors:
        print(f"{RED}FAILED — {len(errors)} error(s): {', '.join(errors)}{RESET}")
        sys.exit(1)
    else:
        print(f"{GREEN}ALL TESTS PASSED{RESET}")
    print(f"{'='*50}{RESET}\n")
