"""bib_parser.py
Utility helpers for converting LaTeX-encoded metadata to plain Unicode and
loading BibTeX files into validated BibEntry structures.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import re
import codecs
import pandas as pd
from pybtex.database import parse_file

from src.indexing.entities import BibEntry

_LATEX_PATTERN = re.compile(r"[{}]")
_MATH_PATTERN = re.compile(r"(\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|\$.*?\$)", re.DOTALL)


def latex_to_unicode(text: str | None) -> str | None:
    """Convert LaTeX-escaped *text* to Unicode.

    Falls back to simply removing curly braces if *latexcodec* is unavailable
    (the dependency is optional).
    """
    if text is None:
        return None
    # drop inline / display math – keeps titles readable like "MDC^3:" -> "MDC:".
    text = _MATH_PATTERN.sub("", text)

    try:
        return codecs.decode(text, "latex")
    except Exception:
        return _LATEX_PATTERN.sub("", text)


def extract_bib_entries(path: Path) -> List[BibEntry]:
    """Parse *path* and return a list of validated BibEntry dicts."""

    bib_data = parse_file(str(path), bib_format="bibtex")

    rows: list[dict[str, object]] = []
    for entry in bib_data.entries.values():
        authors = [str(p) for p in entry.persons.get("author", [])]
        authors = [latex_to_unicode(a) for a in authors]
        rows.append(
            {
                "url": entry.fields.get("url"),
                "title": latex_to_unicode(entry.fields.get("title")),
                "year": entry.fields.get("year"),
                "author": authors if authors else None,
                "language": entry.fields.get("language"),
            }
        )

    df = pd.DataFrame(rows)
    df = df[(df["language"] == "eng") | (df["language"].isna())]
    df = df.dropna(subset=["url", "title", "year", "author"])

    return df[["url", "title", "year", "author"]].to_dict(orient="records")
