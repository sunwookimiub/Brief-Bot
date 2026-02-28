"""
Retrieval Tools
---------------
Two tools the LangChain agent can invoke:
  1. vector_search   — semantic similarity via embeddings
  2. keyword_search  — BM25 exact-match (great for part numbers, function names)
"""
import logging
import pickle
from typing import Optional

from langchain.tools import BaseTool
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from pydantic import Field

from app.config import Settings

logger = logging.getLogger(__name__)


class VectorSearchTool(BaseTool):
    """
    Semantic search tool.
    Best for: conceptual questions, 'how does X work', summarization tasks.
    """
    name: str = "vector_search"
    description: str = (
        "Use this for conceptual or descriptive questions. "
        "Input: the user's question as plain text. "
        "Do NOT use for queries containing specific part numbers, IDs, or function names."
    )
    settings: Settings = Field(exclude=True)
    embeddings: OpenAIEmbeddings = Field(exclude=True)
    filters: Optional[dict] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        s = self.settings
        if s.vector_db_type == "chroma":
            db = Chroma(
                persist_directory=s.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name="rag_docs",
            )
        else:
            db = FAISS.load_local(s.faiss_index_path, self.embeddings)

        results = db.similarity_search_with_relevance_scores(
            query,
            k=s.top_k_semantic,
            filter=self.filters,
        )
        return self._format_results(results, "semantic")

    async def _arun(self, query: str) -> str:
        return self._run(query)

    def _format_results(self, results, strategy: str) -> str:
        if not results:
            return "No relevant documents found."
        parts = [f"[{strategy.upper()} SEARCH RESULTS]"]
        for i, (doc, score) in enumerate(results, 1):
            parts.append(
                f"\n--- Source {i} (score: {score:.3f}) ---\n"
                f"File: {doc.metadata.get('filename', 'unknown')}, "
                f"Page: {doc.metadata.get('page', '?')}, "
                f"Version: {doc.metadata.get('doc_version', '?')}\n"
                f"{doc.page_content}"
            )
        return "\n".join(parts)


class KeywordSearchTool(BaseTool):
    """
    BM25 keyword search tool.
    Best for: exact technical identifiers, part numbers, function names.
    Example queries: 'ASML_EUV_001', 'get_laser_power()', 'error code 0x4F2'
    """
    name: str = "keyword_search"
    description: str = (
        "Use this for queries containing EXACT terms: part numbers, error codes, "
        "function names, model IDs, or any specific alphanumeric identifier. "
        "Input: the specific term or identifier to search for."
    )
    settings: Settings = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        s = self.settings
        # Load the most recent BM25 index (production: load by version tag)
        import glob
        index_files = glob.glob(f"{s.chroma_persist_dir}/bm25_*.pkl")
        if not index_files:
            return "Keyword index not found. Please run /ingest first."

        # Use latest index
        latest_index = max(index_files)
        with open(latest_index, "rb") as f:
            data = pickle.load(f)

        bm25 = data["bm25"]
        chunks = data["chunks"]

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = top_indices[:s.top_k_keyword]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = chunks[idx]
                results.append((doc, scores[idx]))

        return self._format_results(results)

    async def _arun(self, query: str) -> str:
        return self._run(query)

    def _format_results(self, results: list) -> str:
        if not results:
            return "No exact keyword matches found."
        parts = ["[KEYWORD SEARCH RESULTS]"]
        for i, (doc, score) in enumerate(results, 1):
            parts.append(
                f"\n--- Match {i} (BM25 score: {score:.3f}) ---\n"
                f"File: {doc.metadata.get('filename', 'unknown')}, "
                f"Page: {doc.metadata.get('page', '?')}\n"
                f"{doc.page_content}"
            )
        return "\n".join(parts)
