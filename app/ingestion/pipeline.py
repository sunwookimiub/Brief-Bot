"""
Ingestion Pipeline
-----------------
GCS → PDF extraction → semantic chunking → dual-index storage
(vector DB for semantic search + BM25 index for keyword search)

Re-ingestion behaviour: SKIP
If a filename + doc_version combination is already present in ChromaDB,
that file is skipped entirely. To force re-ingestion, change the doc_version
tag (e.g. v1 → v2) or delete the chroma_store directory.
"""
import logging
import pickle
import tempfile
from pathlib import Path

from google.cloud import storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_voyageai import VoyageAIEmbeddings
from rank_bm25 import BM25Okapi

from app.config import Settings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = VoyageAIEmbeddings(
            voyage_api_key=settings.voyage_api_key,
            model=settings.embedding_model,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._gcs_client = storage.Client(project=settings.google_cloud_project)

    async def run(self, gcs_prefix: str, doc_version: str = "latest") -> dict:
        """Full ingestion: GCS → chunks → dual index, skipping already-indexed files."""
        logger.info(f"Starting ingestion: prefix={gcs_prefix}, version={doc_version}")

        # 1. Find out what's already indexed so we can skip those files
        already_indexed = self._get_indexed_files(doc_version)
        if already_indexed:
            logger.info(f"Already indexed ({doc_version}): {already_indexed}")

        # 2. Download PDFs from GCS, skipping already-indexed ones
        documents, skipped, new_files = await self._load_pdfs_from_gcs(
            gcs_prefix, doc_version, skip_filenames=already_indexed
        )

        if skipped:
            logger.info(f"Skipped {len(skipped)} already-indexed file(s): {skipped}")

        if not documents:
            return {
                "status": "skipped",
                "message": f"All files already indexed at version {doc_version}. Skipped: {skipped}",
                "chunks_created": 0,
                "skipped_files": skipped,
                "doc_version": doc_version,
            }

        # 3. Chunk
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages ({new_files})")

        # 4. Attach version metadata to every chunk
        for chunk in chunks:
            chunk.metadata["doc_version"] = doc_version

        # 5. Store in vector DB
        await self._store_vector(chunks)

        # 6. Rebuild BM25 index (merge new chunks with existing ones for this version)
        await self._build_bm25_index(chunks, doc_version)

        return {
            "status": "success",
            "message": f"Ingested {len(new_files)} new file(s) → {len(chunks)} chunks. Skipped: {skipped}",
            "chunks_created": len(chunks),
            "new_files": new_files,
            "skipped_files": skipped,
            "doc_version": doc_version,
        }

    def _get_indexed_files(self, doc_version: str) -> set[str]:
        """
        Query ChromaDB metadata to find filenames already indexed at this version.
        Returns a set of filenames e.g. {'STaR.pdf', 'manual_v2.pdf'}
        """
        try:
            db = Chroma(
                persist_directory=self.settings.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name="rag_docs",
            )
            results = db._collection.get(
                where={"doc_version": doc_version},
                include=["metadatas"],
            )
            return {
                m["filename"]
                for m in results["metadatas"]
                if "filename" in m
            }
        except Exception:
            # Collection doesn't exist yet on first run — that's fine
            return set()

    async def _load_pdfs_from_gcs(
        self,
        prefix: str,
        doc_version: str,
        skip_filenames: set[str],
    ) -> tuple[list, list, list]:
        """
        Download and parse PDFs from GCS.
        Returns: (documents, skipped_filenames, new_filenames)
        """
        bucket = self._gcs_client.bucket(self.settings.gcs_bucket)
        blobs = list(bucket.list_blobs(prefix=prefix))
        pdf_blobs = [b for b in blobs if b.name.endswith(".pdf")]

        if not pdf_blobs:
            logger.warning(f"No PDFs found at gs://{self.settings.gcs_bucket}/{prefix}")
            return [], [], []

        all_docs = []
        skipped = []
        new_files = []

        for blob in pdf_blobs:
            filename = Path(blob.name).name

            # Skip if already indexed at this version
            if filename in skip_filenames:
                logger.info(f"Skipping (already indexed): {filename} @ {doc_version}")
                skipped.append(filename)
                continue

            logger.info(f"Loading new file: {filename}")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                blob.download_to_file(tmp)
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            for page in pages:
                page.metadata.update({
                    "source_gcs": f"gs://{self.settings.gcs_bucket}/{blob.name}",
                    "filename": filename,
                    "gcs_generation": blob.generation,
                    "doc_version": doc_version,
                })
            all_docs.extend(pages)
            new_files.append(filename)

        return all_docs, skipped, new_files

    async def _store_vector(self, chunks: list):
        """Append new chunks into ChromaDB or FAISS."""
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
        """
        Build BM25 index, merging new chunks with any existing ones for this version.
        This ensures the keyword index stays in sync with ChromaDB.
        """
        index_path = f"{self.settings.chroma_persist_dir}/bm25_{doc_version}.pkl"

        # Load existing chunks for this version if index already exists
        existing_chunks = []
        try:
            with open(index_path, "rb") as f:
                existing_data = pickle.load(f)
                existing_chunks = existing_data.get("chunks", [])
                logger.info(f"Loaded {len(existing_chunks)} existing BM25 chunks for {doc_version}")
        except FileNotFoundError:
            pass  # First time — no existing index

        # Merge and rebuild
        all_chunks = existing_chunks + chunks
        tokenized = [c.page_content.lower().split() for c in all_chunks]
        bm25 = BM25Okapi(tokenized)

        with open(index_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)
        logger.info(f"BM25 index saved: {len(all_chunks)} total chunks @ {index_path}")
