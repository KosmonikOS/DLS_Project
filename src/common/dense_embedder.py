"""Shared helper for dense embeddings.

The :class:`DenseEmbedder` wraps `langchain_openai.OpenAIEmbeddings` and is used
by

* *indexing* – to create `text_embedding` vectors before documents are stored
  in Elasticsearch
* *search*   – to embed the user's query for k-NN retrieval

Configuration is provided via :pyfile:`src.common.settings` (values come from
environment variables / ``.env``).  No global instance is created; each caller
instantiates its own `DenseEmbedder` and re-uses it.
"""

from __future__ import annotations


from typing import Sequence
from langchain_openai import OpenAIEmbeddings
from src.common.settings import settings


class DenseEmbedder:
    """Lightweight wrapper around `OpenAIEmbeddings` with cached client."""

    def __init__(self) -> None:
        self._client = OpenAIEmbeddings(
            model=settings.embedding_model_name,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            chunk_size=settings.embedding_batch_size,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._client.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
