"""
Prompt Version Manager
----------------------
Loads prompt templates from GCS with version tagging.
Never hardcode prompts — this enables A/B testing and audit trails.
"""
import logging
from functools import lru_cache
from typing import Optional

import yaml
from google.cloud import storage

from app.config import Settings

logger = logging.getLogger(__name__)

# Fallback prompts if GCS is unavailable (development / offline)
FALLBACK_PROMPTS = {
    "v1": {
        "system": (
            "You are a precise technical documentation assistant. "
            "Answer questions using ONLY the provided context from the documents. "
            "If the context doesn't contain enough information, say so explicitly. "
            "For technical specifications, always cite the source document and page number."
        ),
        "agent_prefix": (
            "You have access to two search tools:\n"
            "- vector_search: for conceptual and general questions\n"
            "- keyword_search: for specific part numbers, IDs, function names, error codes\n\n"
            "Choose the appropriate tool based on the query type. "
            "You may call both tools if the query is ambiguous.\n\n"
            "Always cite your sources with filename and page number.\n\n"
            "Question: {question}"
        ),
        "version": "v1",
        "description": "Default RAG agent prompt",
    },
    "v2": {
        "system": (
            "You are an expert technical documentation analyst specializing in complex engineering manuals. "
            "Provide structured, precise answers. Always include: "
            "1) Direct answer, 2) Technical context, 3) Source citation."
        ),
        "agent_prefix": (
            "Analyze this technical query carefully.\n"
            "Available tools: vector_search (semantic) and keyword_search (exact match).\n"
            "Strategy: If query contains alphanumeric IDs or exact names → keyword_search first. "
            "Otherwise → vector_search first.\n\n"
            "Question: {question}"
        ),
        "version": "v2",
        "description": "Structured output prompt with routing guidance",
    },
}


class PromptVersionManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._gcs_client: Optional[storage.Client] = None

    def _get_gcs_client(self) -> storage.Client:
        if self._gcs_client is None:
            self._gcs_client = storage.Client(project=self.settings.google_cloud_project)
        return self._gcs_client

    def load(self, version: str = "v1") -> dict:
        """
        Load prompt by version. Tries GCS first, falls back to local defaults.
        GCS path: gs://{bucket}/prompts/rag_agent_{version}.yaml
        """
        try:
            return self._load_from_gcs(version)
        except Exception as e:
            logger.warning(f"GCS prompt load failed ({e}), using fallback for {version}")
            return self._load_fallback(version)

    def _load_from_gcs(self, version: str) -> dict:
        client = self._get_gcs_client()
        bucket = client.bucket(self.settings.gcs_bucket)
        blob_name = f"{self.settings.gcs_prompts_prefix}rag_agent_{version}.yaml"
        blob = bucket.blob(blob_name)

        content = blob.download_as_text()
        prompt_data = yaml.safe_load(content)
        prompt_data["version"] = version
        logger.info(f"Loaded prompt {version} from GCS: {blob_name}")
        return prompt_data

    def _load_fallback(self, version: str) -> dict:
        if version in FALLBACK_PROMPTS:
            return FALLBACK_PROMPTS[version]
        logger.warning(f"Prompt version '{version}' not found, defaulting to v1")
        return FALLBACK_PROMPTS["v1"]

    def upload_prompt(self, version: str, prompt_data: dict) -> str:
        """Upload a new prompt version to GCS. Returns the GCS path."""
        client = self._get_gcs_client()
        bucket = client.bucket(self.settings.gcs_bucket)
        blob_name = f"{self.settings.gcs_prompts_prefix}rag_agent_{version}.yaml"
        blob = bucket.blob(blob_name)

        content = yaml.dump(prompt_data, default_flow_style=False)
        blob.upload_from_string(content, content_type="text/yaml")
        gcs_path = f"gs://{self.settings.gcs_bucket}/{blob_name}"
        logger.info(f"Uploaded prompt {version} to {gcs_path}")
        return gcs_path
