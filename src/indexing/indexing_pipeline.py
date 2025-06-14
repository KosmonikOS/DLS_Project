"""indexing_pipeline.py
End-to-end pipeline for ingesting scholarly PDFs listed in a BibTeX file into
an Elasticsearch index using BM25 similarity.

The flow is:
    1. Read a .bib file and extract entries containing a URL.
    2. Split entries into batches.
    3. For each batch, asynchronously download the PDFs and parse their text.
    4. Index the extracted text into Elasticsearch via BM25Indexer.

All heavy blocking operations (PDF parsing and Elasticsearch I/O) are delegated
to background threads to keep the event loop responsive.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from itertools import batched
from tqdm import tqdm

from src.indexing.elastic_search_indexer import ElasticSearchIndexer
from src.indexing.entities import IndexedDocument, BibEntry
from src.indexing.bib_parser import extract_bib_entries
from src.indexing.docling_parallel import convert_in_parallel, close_pool

logger = logging.getLogger(__name__)


def _process_batch(entries: list[BibEntry], workers: int) -> list[IndexedDocument]:
    """Convert one batch of BibTeX entries to indexable documents.

    Args:
        entries: A list of BibEntry mappings to be converted. Each entry must
            contain at least the keys url, title, year and author.
        workers: Number of worker processes used by Docling during PDF → Markdown
            conversion.

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

    texts = convert_in_parallel(urls, workers)

    docs: list[IndexedDocument] = []
    for entry, text in zip(entries, texts):
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


def ingest_bib(
    bib_file: Path,
    index_name: str = "papers",
    batch_size: int = 100,
    concurrency: int = os.cpu_count() or 4,
    force_delete_index: bool = False,
    es_hosts: list[str] | str = "http://localhost:9200",
    max_entries: int | None = None,
) -> None:
    """Ingest PDFs referenced in *bib_file* into an Elasticsearch index.

    Args:
        bib_file: Path to the input .bib file.
        index_name: Name of the target Elasticsearch index (default: "papers").
        batch_size: Number of BibTeX entries processed per batch (default: 100).
        concurrency: Worker processes used for Docling conversion (default: os.cpu_count()).
        force_delete_index: If True, delete the target index before ingestion.
        es_hosts: A single host string or a list of hosts where Elasticsearch is running.
        max_entries: Optional hard limit on the number of BibTeX entries to ingest.
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
            docs = _process_batch(list(batch_entries), concurrency)
            if docs:
                indexer.index_documents(index_name, docs, batch_size=len(docs))
            progress.update(len(batch_entries))

    # tear down worker pool
    close_pool()
