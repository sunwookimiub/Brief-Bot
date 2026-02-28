"""
Ingestion Pipeline
-----------------
GCS → PDF extraction → semantic chunking → dual-index storage
(vector DB for semantic search + BM25 index for keyword search)
"""
import io
import logging
from pathlib import Path
from typing import AsyncIterator

from google.cloud import storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
import pickle
import tempfile

from app.config import Settings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            # These separators work well for technical PDFs with sections/tables
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._gcs_client = storage.Client(project=settings.google_cloud_project)

    async def run(self, gcs_prefix: str, doc_version: str = "latest") -> dict:
        """Full ingestion: GCS → chunks → dual index."""
        logger.info(f"Starting ingestion: prefix={gcs_prefix}, version={doc_version}")

        # 1. Download PDFs from GCS
        documents = await self._load_pdfs_from_gcs(gcs_prefix, doc_version)
        if not documents:
            return {"status": "no_documents", "message": "No PDFs found", "chunks_created": 0}

        # 2. Chunk
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages")

        # 3. Attach version metadata to every chunk (enables filtered retrieval + rollback)
        for chunk in chunks:
            chunk.metadata["doc_version"] = doc_version

        # 4. Store in vector DB (semantic search)
        await self._store_vector(chunks)

        # 5. Build BM25 index (keyword search)
        await self._build_bm25_index(chunks, doc_version)

        return {
            "status": "success",
            "message": f"Ingested {len(documents)} pages → {len(chunks)} chunks",
            "chunks_created": len(chunks),
            "doc_version": doc_version,
        }

    async def _load_pdfs_from_gcs(self, prefix: str, doc_version: str) -> list:
        """Download PDFs from GCS and parse them with PyPDFLoader."""
        bucket = self._gcs_client.bucket(self.settings.gcs_bucket)
        blobs = list(bucket.list_blobs(prefix=prefix))
        pdf_blobs = [b for b in blobs if b.name.endswith(".pdf")]

        if not pdf_blobs:
            logger.warning(f"No PDFs found at gs://{self.settings.gcs_bucket}/{prefix}")
            return []

        all_docs = []
        for blob in pdf_blobs:
            logger.info(f"Loading: {blob.name}")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                blob.download_to_file(tmp)
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Enrich metadata with GCS source info
            for page in pages:
                page.metadata.update({
                    "source_gcs": f"gs://{self.settings.gcs_bucket}/{blob.name}",
                    "filename": Path(blob.name).name,
                    "gcs_generation": blob.generation,   # GCS versioning: enables rollback
                    "doc_version": doc_version,
                })
            all_docs.extend(pages)

        return all_docs

    async def _store_vector(self, chunks: list):
        """Upsert chunks into ChromaDB or FAISS."""
        s = self.settings
        if s.vector_db_type == "chroma":
            Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=s.chroma_persist_dir,
                collection_name="rag_docs",
            )
            logger.info(f"Stored {len(chunks)} chunks in ChromaDB @ {s.chroma_persist_dir}")
        else:
            db = FAISS.from_documents(chunks, self.embeddings)
            db.save_local(s.faiss_index_path)
            logger.info(f"Stored {len(chunks)} chunks in FAISS @ {s.faiss_index_path}")

    async def _build_bm25_index(self, chunks: list, doc_version: str):
        """Build and persist a BM25 index for keyword/exact-match search."""
        tokenized = [chunk.page_content.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized)

        index_path = f"{self.settings.chroma_persist_dir}/bm25_{doc_version}.pkl"
        with open(index_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunks": chunks}, f)
        logger.info(f"BM25 index saved to {index_path}")
