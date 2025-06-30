"""Configuration for interactive search client.

Reads values from environment variables or a .env file (shared with indexing).
Only parameters required by *search_cli.py* are included.
"""

from __future__ import annotations

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

load_dotenv(override=True)


class Settings(BaseSettings):
    """Runtime knobs for the interactive *search_cli* utility.

    Fields
    ------
    index_name
        Elasticsearch index to query.
    es_host
        Elasticsearch HTTP endpoint.
    top_k
        Number of final documents displayed to the user.
    """

    index_name: str = Field("papers", env="INDEX_NAME")
    es_host: str = Field("http://localhost:9200", env="ES_HOST")

    top_k: int = Field(5, env="TOP_K")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
