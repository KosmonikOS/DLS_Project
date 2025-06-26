"""compute_pagerank.py
Batch job that enriches an existing Elasticsearch index with PageRank scores.

Steps
-----
1. Fetch all documents (with a DOI) from the configured index.
2. Retrieve outbound references for every DOI via the Crossref API (async).
3. Build the citation graph (optionally adding virtual nodes for external papers).
4. Run PageRank.
5. Bulk-update each indexed document with its PageRank score (field ``pagerank``).

Run as a one-off script *after* the main indexing pipeline has finished:

    python -m src.indexing.compute_pagerank
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import aiohttp

import networkx as nx

from src.indexing.settings import settings
from src.indexing.elastic_search_indexer import ElasticSearchIndexer

from tqdm import tqdm

logger = logging.getLogger(__name__)


async def _fetch_single_reference_list(
    session: aiohttp.ClientSession, doi: str
) -> Tuple[str, List[str]]:
    """Return (doi, references[]) or (doi, []) on error."""
    url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": "Academic Research Tool (mailto:your-email@example.com)"}

    try:
        async with session.get(url, headers=headers, timeout=30) as resp:
            resp.raise_for_status()
            data = await resp.json()
            refs = data.get("message", {}).get("reference", [])
            ref_dois = [r["DOI"] for r in refs if "DOI" in r]
            return doi, ref_dois
    except Exception as exc:  # noqa: BLE001,E722
        logger.warning("Crossref lookup failed for %s: %s", doi, exc)
        return doi, []


async def fetch_references(
    dois: List[str], concurrency: int = 20
) -> Dict[str, List[str]]:
    """Fetch reference lists for *dois* concurrently using Crossref with progress bar."""
    results: Dict[str, List[str]] = {}

    with tqdm(total=len(dois), desc="Crossref", unit="doi") as pbar:
        for i in range(0, len(dois), concurrency):
            batch = dois[i : i + concurrency]
            async with aiohttp.ClientSession() as session:
                tasks = [_fetch_single_reference_list(session, d) for d in batch]
                batch_res = await asyncio.gather(*tasks, return_exceptions=False)
                results.update(dict(batch_res))
            pbar.update(len(batch))
    return results


def pagerank_with_virtual_nodes(
    references_dict: Dict[str, List[str]],
    alpha: float = 0.85,
) -> Dict[str, float]:
    """Run PageRank. Only original papers are returned in the result."""

    # Collect external papers referenced > *min_external_citations* times
    external_counter: Dict[str, int] = defaultdict(int)
    for cited_list in references_dict.values():
        for cited in cited_list:
            if cited not in references_dict:
                external_counter[cited] += 1

    virtual_nodes = {doi for doi, cnt in external_counter.items()}

    original_nodes = set(references_dict.keys())
    all_nodes = original_nodes | virtual_nodes

    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)

    for src, tgt_list in references_dict.items():
        for tgt in tgt_list:
            if tgt in all_nodes:
                G.add_edge(src, tgt)

    pagerank_full = nx.pagerank(G, alpha=alpha, max_iter=1000, tol=1e-9)
    return {doi: score for doi, score in pagerank_full.items() if doi in original_nodes}


def main() -> None:
    index_name = settings.index_name
    indexer = ElasticSearchIndexer(settings.es_host)

    logger.info("Scanning index '%s' for documents …", index_name)
    internal_to_esid = indexer.build_internal_id_map(index_name)
    if not internal_to_esid:
        logger.error("No documents found in index '%s'.", index_name)
        return

    # Separate DOIs (needed for Crossref) and synthetic IDs
    dois = [iid for iid in internal_to_esid if not iid.startswith("SYNTH_")]

    logger.info("Fetching Crossref references for %d DOIs …", len(dois))
    references = asyncio.run(
        fetch_references(dois, concurrency=settings.crossref_concurrency)
    )

    # Ensure synthetic IDs are present in graph even if they have no outgoing refs or references
    for internal_id in internal_to_esid:
        if internal_id not in references:
            references[internal_id] = []

    logger.info("Running PageRank on citation graph …")
    pagerank_scores = pagerank_with_virtual_nodes(
        references,
        alpha=settings.pagerank_alpha,
    )

    id_score_map = {
        internal_to_esid[iid]: score
        for iid, score in pagerank_scores.items()
        if iid in internal_to_esid
    }

    logger.info("Updating Elasticsearch documents with PageRank scores …")
    indexer.bulk_update_field(index_name, id_score_map, field="pagerank")
    logger.info(
        "PageRank enrichment completed: %d documents updated.", len(id_score_map)
    )


if __name__ == "__main__":
    main()
