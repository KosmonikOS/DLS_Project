"""PDF ingestion → embedding → Elasticsearch indexing pipeline.

Workflow
---------
1. Read the BibTeX file specified by ``BIB_FILE``.
2. Download every referenced PDF concurrently with asynchronous HTTP requests.
3. Extract plain text from each PDF and embed it with
   :class:`src.common.dense_embedder.DenseEmbedder`.
4. Create the target ES index – mapping includes a `dense_vector` field sized
   to the embedding dimension.
5. Bulk-index ``text``, metadata, and ``text_embedding``.

Environment variables are consumed via :pyfile:`src.indexing.settings`.
"""

from __future__ import annotations

import logging
import asyncio

from itertools import batched
from tqdm import tqdm

import httpx
from httpx import Limits

from src.indexing.elastic_search_indexer import ElasticSearchIndexer
from src.indexing.entities import IndexedDocument, BibEntry
from src.indexing.bib_parser import extract_bib_entries
from src.indexing.settings import settings
from src.common.dense_embedder import DenseEmbedder
from src.indexing.parse import fetch_and_parse

logger = logging.getLogger(__name__)


def _process_batch(
    entries: list[BibEntry], embedder: DenseEmbedder
) -> list[IndexedDocument]:
    """Convert one batch of BibTeX entries to indexable documents.

    Args:
        entries: A list of BibEntry mappings to be converted. Each entry must
            contain at least the keys url, title, year and author.
        embedder: DenseEmbedder instance used for embedding text.

    Returns:
        A list of :class:IndexedDocument dictionaries that passed conversion
        successfully (failed PDFs are silently skipped).
    """

    urls = []
    for e in entries:
        url = e["url"]
        if not url.endswith(".pdf"):
            url = url.rstrip("/") + ".pdf"
        urls.append(url)

    # Download and parse PDFs concurrently
    texts = asyncio.run(fetch_and_parse(urls))

    docs: list[IndexedDocument] = []
    raw_texts: list[str] = []
    for entry, text in zip(entries, texts):
        if not text:
            continue
        # Keep metadata fields separate and embed only the main document text.
        title_part: str = entry.get("title") or ""
        author_list = entry.get("author") or []

        raw_texts.append(text)  # embeddings are created from PDF body only
        docs.append(
            {
                "title": title_part or None,
                "year": entry.get("year"),
                "url": entry.get("url"),
                "author": author_list or None,
                "doi": entry.get("doi"),
                "text": text,  # BM25 body
            }
        )

    if docs:
        embeddings = embedder.embed_documents(raw_texts)
        for doc, emb in zip(docs, embeddings):
            doc["text_embedding"] = emb

    return docs


async def _process_batch_async(
    entries: list[BibEntry],
    embedder: DenseEmbedder,
    client: httpx.AsyncClient,
) -> list[IndexedDocument]:
    """Convert one batch of BibTeX entries to indexable documents.

    Args:
        entries: A list of BibEntry mappings to be converted. Each entry must
            contain at least the keys url, title, year and author.
        embedder: DenseEmbedder instance used for embedding text.
        client: httpx.AsyncClient instance used for downloading PDFs.

    Returns:
        A list of :class:IndexedDocument dictionaries that passed conversion
        successfully (failed PDFs are silently skipped).
    """

    urls = []
    for e in entries:
        url = e["url"]
        if not url.endswith(".pdf"):
            url = url.rstrip("/") + ".pdf"
        urls.append(url)

    # Download and parse PDFs concurrently (shared HTTP client)
    texts = await fetch_and_parse(urls, client=client)

    docs: list[IndexedDocument] = []
    raw_texts: list[str] = []
    for entry, text in zip(entries, texts):
        if not text:
            continue
        # Keep metadata fields separate and embed only the main document text.
        title_part: str = entry.get("title") or ""
        author_list = entry.get("author") or []

        raw_texts.append(text)  # embeddings are created from PDF body only
        docs.append(
            {
                "title": title_part,
                "year": entry.get("year"),
                "url": entry.get("url"),
                "author": author_list,
                "doi": entry.get("doi"),
                "text": text,  # BM25 body
            }
        )

    if docs:
        embeddings = embedder.embed_documents(raw_texts)
        for doc, emb in zip(docs, embeddings):
            doc["text_embedding"] = emb

    return docs


async def _ingest_bib_async() -> None:
    """Async implementation of the ingestion pipeline (single event loop)."""

    bib_file = settings.bib_file
    index_name = settings.index_name
    batch_size = settings.batch_size
    force_delete_index = settings.force_delete_index
    es_host = settings.es_host
    max_entries = settings.max_entries

    entries = extract_bib_entries(bib_file)
    if max_entries is not None and max_entries > 0:
        entries = entries[:max_entries]
    if not entries:
        logger.warning("No entries with a URL found in %s", bib_file)
        return

    indexer = ElasticSearchIndexer(es_host)
    index_created = False

    for entry in entries:
        if not entry["url"].endswith(".pdf"):
            entry["url"] = entry["url"].rstrip("/") + ".pdf"
    embedder = DenseEmbedder()

    # Shared HTTP client across all batches – HTTP/2 + keep-alive
    async with httpx.AsyncClient(
        http2=True,
        limits=Limits(
            max_connections=settings.acl_concurrency,
            max_keepalive_connections=settings.acl_concurrency * 2,
        ),
    ) as client:
        with tqdm(
            total=len(entries), desc="Ingesting PDF batches", unit="doc"
        ) as progress:
            for batch_entries in batched(entries, batch_size):
                batch_list = list(batch_entries)
                docs = await _process_batch_async(batch_list, embedder, client)
                if not docs:
                    progress.update(len(batch_list))
                    continue

                if not index_created:
                    dim = len(docs[0]["text_embedding"])
                    indexer.create_index(
                        index_name,
                        force_delete=force_delete_index,
                        embedding_dim=dim,
                        bm25_k1=settings.bm25_k1,
                        bm25_b=settings.bm25_b,
                    )
                    index_created = True

                indexer.index_documents(index_name, docs, batch_size=len(docs))
                progress.update(len(batch_list))


if __name__ == "__main__":
    asyncio.run(_ingest_bib_async())
