"""Thin wrapper around the official Elasticsearch client.

Provides two high-level operations used by the pipeline:

* :py:meth:`create_index` – create a text+vector mapping with optional force-delete.
* :py:meth:`index_documents` – stream bulk requests in configurable batches.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Iterable, Optional, Dict

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError

from itertools import batched

from src.indexing.entities import IndexedDocument

import hashlib
import time

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
        self._client = Elasticsearch(hosts, request_timeout=10)

        # Elasticsearch container may still be starting – retry a few times
        for attempt in range(6):  # ~30 s total
            try:
                if self._client.ping():
                    break
            except Exception:
                pass

            if attempt == 5:
                raise ElasticsearchConnectionError(
                    f"Unable to connect to Elasticsearch at {hosts}"
                )

            logger.info("ES ping failed (attempt %d/6); retrying in 5s…", attempt + 1)
            time.sleep(5)

    def create_index(
        self,
        index_name: str,
        force_delete: bool = False,
        embedding_dim: Optional[int] = None,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ) -> None:
        """Create a text index configured for BM25 similarity.

        Args:
            index_name: Name of the index to create.
            force_delete: Delete an existing index of the same name before creation.
            embedding_dim: Dimension of the dense vector for embeddings.
            bm25_k1: BM25 k1 parameter.
            bm25_b: BM25 b parameter.
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

        properties: dict[str, Any] = {
            "text": {"type": "text", "analyzer": "standard"},
            "title": {"type": "text"},
            "author": {"type": "text"},
            "year": {"type": "keyword"},
            "url": {"type": "keyword"},
            "doi": {"type": "keyword"},
            "pagerank": {"type": "float"},
        }
        if embedding_dim:
            properties["text_embedding"] = {
                "type": "dense_vector",
                "dims": embedding_dim,
            }

        body: dict[str, Any] = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "similarity": {
                    "default": {
                        "type": "BM25",
                        "k1": bm25_k1,
                        "b": bm25_b,
                    }
                },
            },
            "mappings": {
                "_source": {"excludes": ["text"]},
                "properties": properties,
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

    @staticmethod
    def _create_synth_id(title: str | None, year: str | int | None, author: list[str] | str | None) -> str:
        """Generate a stable synthetic identifier for a paper without DOI."""
        title_part = (title or "").strip()[:50]
        year_part = str(year or "")
        first_author = ""
        if isinstance(author, list) and author:
            first_author = author[0]
        elif isinstance(author, str):
            first_author = author.split(",")[0]
        first_author = first_author[:30]
        base = f"{title_part}_{year_part}_{first_author}".lower().replace(" ", "_")
        return f"SYNTH_{hashlib.md5(base.encode()).hexdigest()[:12]}"

    def build_internal_id_map(
        self,
        index_name: str,
        *,
        include_fields: list[str] | None = None,
    ) -> Dict[str, str]:
        """Return a mapping of *internal_id -> Elasticsearch _id*.

        *internal_id* is the DOI if present, otherwise a synthetic ID derived
        from title/year/author so that every document participates in graph
        enrichment jobs.
        """

        src_fields = ["doi", "title", "year", "author"]
        if include_fields:
            src_fields.extend(include_fields)

        id_map: Dict[str, str] = {}
        for hit in helpers.scan(self._client, index=index_name, _source=src_fields):
            src = hit["_source"]
            doc_doi = src.get("doi")
            if doc_doi:
                internal_id = doc_doi
            else:
                internal_id = self._create_synth_id(src.get("title"), src.get("year"), src.get("author"))

            id_map[internal_id] = hit["_id"]

        return id_map

    def bulk_update_field(
        self,
        index_name: str,
        id_value_map: Dict[str, float],
        field: str = "pagerank",
    ) -> None:
        """Update *field* for each ES doc where `_id` is a key in id_value_map."""

        actions = [
            {
                "_op_type": "update",
                "_index": index_name,
                "_id": es_id,
                "doc": {field: value},
            }
            for es_id, value in id_value_map.items()
        ]

        if actions:
            helpers.bulk(self._client, actions, refresh=True)
            logger.info("Updated %d documents (%s field).", len(actions), field)