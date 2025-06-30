"""entities.py
Shared type definitions used across the ingestion pipeline.
"""

from __future__ import annotations

from typing import TypedDict, Union


class IndexedDocument(TypedDict, total=False):
    """Canonical schema for documents indexed in Elasticsearch."""

    text: str
    title: str
    author: list[str]
    url: str
    year: Union[int, str]
    doi: str
    pagerank: float


class BibEntry(TypedDict):
    """Essential metadata extracted from a BibTeX entry prior to PDF download."""

    url: str
    title: str
    author: list[str]
    year: Union[int, str]
    doi: str | None
