"""Settings shared by *indexing* and *search*.

All values are sourced from environment variables (or a ``.env`` file loaded at
import time).  They control the OpenAI-compatible embedding client used
throughout the project.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings(BaseSettings):
    """Embedding client configuration.

    Fields
    ------
    openai_base_url
        Base URL of an OpenAI-compatible API server.
    embedding_model_name
        Name of the embedding model to request from the server.
    embedding_batch_size
        Maximum number of texts sent per embeddings request.
    openai_api_key
        API key carried in the `Authorization` header (set to a dummy value if
        the backend doesn't require authentication).
    """

    openai_base_url: str = Field("http://localhost:8000/v1", env="OPENAI_BASE_URL")
    embedding_model_name: str = Field(
        "text-embedding-3-small", env="EMBEDDING_MODEL_NAME"
    )
    embedding_batch_size: int = Field(128, env="EMBEDDING_BATCH_SIZE")
    openai_api_key: str = Field("not-needed", env="OPENAI_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
