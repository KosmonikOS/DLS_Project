"""bib_ingest.py
End-to-end pipeline for ingesting scholarly PDFs listed in a BibTeX file into
an Elasticsearch index using BM25 similarity.

The flow is:
    1. Read a `.bib` file and extract entries containing a URL.
    2. Split entries into batches.
    3. For each batch, asynchronously download the PDFs and parse their text.
    4. Index the extracted text into Elasticsearch via `BM25Indexer`.

All heavy blocking operations (PDF parsing and Elasticsearch I/O) are delegated
to background threads to keep the event loop responsive.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import aiohttp
from tqdm import tqdm
from itertools import batched

from src.indexing.elastic_search_indexer import ElasticSearchIndexer
from src.indexing.entities import IndexedDocument, BibEntry
from src.indexing.bib_parser import extract_bib_entries
from src.indexing.pdf_text import fetch_pdf, pdf_to_text

logger = logging.getLogger(__name__)


async def _process_entry(
    session: aiohttp.ClientSession,
    entry: BibEntry,
    semaphore: asyncio.Semaphore,
) -> tuple[BibEntry, bytes | None]:
    """Download a PDF for *entry* and return the raw bytes (or None)."""
    pdf_bytes = await fetch_pdf(session, entry["url"], semaphore)
    return entry, pdf_bytes


async def _process_batch(
    entries: list[BibEntry], concurrency: int
) -> list[IndexedDocument]:
    """Download and parse a batch of PDFs concurrently."""
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [_process_entry(session, e, semaphore) for e in entries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    docs: list[IndexedDocument] = []
    for res in results:
        if isinstance(res, Exception):
            logger.warning("Unhandled exception during batch processing: %s", res)
            continue
        entry, data = res
        if not data:
            continue
        text = pdf_to_text(data)
        if not text:
            continue
        docs.append(
            {
                "title": entry.get("title"),
                "year": entry.get("year"),
                "url": entry.get("url"),
                "author": entry.get("author"),
                "text": text,
            }
        )

    return docs


async def ingest_bib_async(
    bib_file: Path,
    index_name: str = "papers",
    batch_size: int = 100,
    concurrency: int = os.cpu_count() or 4,
    force_delete_index: bool = False,
    es_hosts: list[str] | str = "http://localhost:9200",
    max_entries: int | None = None,
) -> None:
    """Asynchronously ingest PDFs referenced in *bib_file* into Elasticsearch.

    Args:
        bib_file: Path to the BibTeX file.
        index_name: Name of the target Elasticsearch index.
        batch_size: Number of BibTeX entries processed per batch.
        concurrency: Maximum concurrent PDF downloads.
        force_delete_index: Delete the index before ingestion if it exists.
        es_hosts: Elasticsearch host(s).
        max_entries: Maximum number of entries to process.
    """

    entries = extract_bib_entries(bib_file)
    if max_entries is not None and max_entries > 0:
        entries = entries[:max_entries]
    if not entries:
        logger.warning("No entries with a URL found in %s", bib_file)
        return

    indexer = ElasticSearchIndexer(es_hosts)
    indexer.create_index(index_name, force_delete=force_delete_index)

    for entry in entries:
        if not entry["url"].endswith(".pdf"):
            entry["url"] = entry["url"].rstrip("/") + ".pdf"

    with tqdm(total=len(entries), desc="Ingesting PDF batches", unit="doc") as progress:
        for batch_entries in batched(entries, batch_size):
            docs = await _process_batch(list(batch_entries), concurrency)
            if docs:
                indexer.index_documents(index_name, docs, batch_size=len(docs))
            progress.update(len(batch_entries))


def ingest_bib(
    bib_file: Path,
    index_name: str = "papers",
    batch_size: int = 100,
    concurrency: int = os.cpu_count() or 4,
    force_delete_index: bool = False,
    es_hosts: list[str] | str = "http://localhost:9200",
    max_entries: int | None = None,
) -> None:
    """Run :pyfunc:`ingest_bib_async` inside a new event loop."""

    asyncio.run(
        ingest_bib_async(
            bib_file=bib_file,
            index_name=index_name,
            batch_size=batch_size,
            concurrency=concurrency,
            force_delete_index=force_delete_index,
            es_hosts=es_hosts,
            max_entries=max_entries,
        )
    )
