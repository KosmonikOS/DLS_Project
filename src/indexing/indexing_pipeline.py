"""PDF ingestion → text extraction → Elasticsearch indexing pipeline.

Workflow
---------
1. Read the BibTeX file specified by ``BIB_FILE``.
2. Download every referenced PDF concurrently with asynchronous HTTP requests.
3. Create the target Elasticsearch index configured for BM25 similarity.
4. Bulk-index full-text content (``text``) together with bibliographic metadata.

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
from src.indexing.parse import fetch_and_parse

logger = logging.getLogger(__name__)


async def _process_batch_async(
    entries: list[BibEntry],
    client: httpx.AsyncClient,
) -> list[IndexedDocument]:
    """Convert one batch of BibTeX entries to indexable documents.

    Args:
        entries: List of BibTeX entry mappings to convert. Each entry must
            provide at least the keys ``url``, ``title``, ``year`` and ``author``.
        client: Asynchronous HTTP client used for downloading PDFs.

    Returns:
        A list of :class:`IndexedDocument` dictionaries for successfully parsed
        PDFs. Entries whose PDF failed to download or parse are skipped.
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
    for entry, text in zip(entries, texts):
        if not text:
            continue
        # Keep metadata fields separate and embed only the main document text.
        title_part: str = entry.get("title") or ""
        author_list = entry.get("author") or []

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
                docs = await _process_batch_async(batch_list, client)
                if not docs:
                    progress.update(len(batch_list))
                    continue

                if not index_created:
                    indexer.create_index(
                        index_name,
                        force_delete=force_delete_index,
                        bm25_k1=settings.bm25_k1,
                        bm25_b=settings.bm25_b,
                    )
                    index_created = True

                indexer.index_documents(index_name, docs, batch_size=len(docs))
                progress.update(len(batch_list))


if __name__ == "__main__":
    asyncio.run(_ingest_bib_async())
