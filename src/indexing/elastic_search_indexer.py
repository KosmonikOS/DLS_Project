"""elastic_search_indexer.py
A lightweight wrapper around the official Elasticsearch Python client that
simplifies BM25-based indexing.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Iterable

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError
from itertools import batched

from src.indexing.entities import IndexedDocument

logger = logging.getLogger(__name__)


class ElasticsearchConnectionError(RuntimeError):
    """Raised when the client fails to connect to Elasticsearch."""


class ElasticSearchIndexer:
    """High-level helper for BM25 indexing of plain-text documents."""

    def __init__(self, hosts: list[str] | str = "http://localhost:9200") -> None:
        """Instantiate the indexer and verify connectivity.

        Args:
            hosts: Single host or list of hosts where Elasticsearch is available.

        Raises:
            ElasticsearchConnectionError: If the cluster is unreachable.
        """
        self._client = Elasticsearch(hosts)
        if not self._client.ping():
            raise ElasticsearchConnectionError(
                f"Unable to connect to Elasticsearch at {hosts}"
            )

    def create_index(self, index_name: str, force_delete: bool = False) -> None:
        """Create a text index configured for BM25 similarity.

        Args:
            index_name: Name of the index to create.
            force_delete: Delete an existing index of the same name before creation.
        """
        if force_delete:
            try:
                self._client.indices.delete(index=index_name)
                logger.info("Deleted existing index '%s'.", index_name)
            except NotFoundError:
                pass

        if self._client.indices.exists(index=index_name):
            logger.info("Index '%s' already exists; skipping creation.", index_name)
            return

        body: dict[str, Any] = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "_source": {"excludes": ["text"]},
                "properties": {
                    "text": {"type": "text", "analyzer": "standard"},
                    "title": {"type": "text"},
                    "author": {"type": "text"},
                    "year": {"type": "keyword"},
                    "url": {"type": "keyword"},
                },
            },
        }
        self._client.indices.create(index=index_name, body=body)
        logger.info("Created index '%s'.", index_name)

    def index_documents(
        self, index_name: str, docs: Iterable[IndexedDocument], batch_size: int = 500
    ) -> None:
        """Bulk-index :pydata:Document objects.

        Each document is an arbitrary mapping.  At minimum it must contain a
        text field which will be used for BM25 search.

        Args:
            index_name: Target index.
            docs: Iterable of document mappings.
            batch_size: Documents per bulk request.
        """
        for chunk in batched(docs, batch_size):
            actions = [
                {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": str(uuid.uuid4()),
                    **doc,
                }
                for doc in chunk
            ]
            helpers.bulk(self._client, actions)
            logger.info("Indexed %d documents into '%s'.", len(chunk), index_name)
