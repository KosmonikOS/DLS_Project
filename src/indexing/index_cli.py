"""indexing.py
Command-line entry point for PDF ingestion from BibTeX into Elasticsearch.

This module only handles CLI parsing and delegates all heavy lifting to
:pyfunc:`bib_ingest.ingest_bib_async`.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.indexing.indexing_pipeline import ingest_bib_async


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs referenced in a BibTeX file into Elasticsearch.",
    )
    parser.add_argument(
        "--bib-file", type=Path, required=True, help="Path to .bib file"
    )
    parser.add_argument(
        "--index-name", type=str, default="papers", help="Elasticsearch index name"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Entries per processing batch"
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Concurrent downloads within a batch"
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="Maximum number of BibTeX records to ingest (0 for no limit)",
    )
    parser.add_argument(
        "--force-delete-index", action="store_true", help="Delete index if it exists"
    )
    parser.add_argument(
        "--es-hosts",
        nargs="+",
        default=["http://localhost:9200"],
        help="One or more Elasticsearch hosts",
    )
    return parser.parse_args()


def main() -> None:  # noqa: D401
    """Parse CLI options and launch the ingestion coroutine."""

    args = _parse_args()

    asyncio.run(
        ingest_bib_async(
            bib_file=args.bib_file,
            index_name=args.index_name,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            max_entries=args.max_entries if args.max_entries > 0 else None,
            force_delete_index=args.force_delete_index,
            es_hosts=args.es_hosts,
        )
    )


if __name__ == "__main__":
    main()
