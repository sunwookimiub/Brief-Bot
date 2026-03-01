from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # GCP
    gcs_bucket: str = "rag-docs-sunwookim"
    gcs_prompts_prefix: str = "prompts/"
    google_cloud_project: str = "rag-system-2026"

    # LLM
    anthropic_api_key: Optional[str] = None
    voyage_api_key: Optional[str] = None
    llm_provider: str = "anthropic"           # "anthropic" | "openai"
    llm_model: str = "claude-sonnet-4-6"

    # Embeddings
    embedding_model: str = "voyage-3"

    # Vector DB  (ChromaDB local path or "faiss")
    vector_db_type: str = "chroma"            # "chroma" | "faiss"
    chroma_persist_dir: str = "./chroma_store"
    faiss_index_path: str = "./faiss_index"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k_semantic: int = 5
    top_k_keyword: int = 5

    class Config:
        env_file = ".env"
        case_sensitive = False
