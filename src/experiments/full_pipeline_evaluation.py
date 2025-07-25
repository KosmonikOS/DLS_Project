"""full_pipeline_evaluation.py
End-to-end evaluation of the search stack against the small benchmark that
ships with the repository.

This script is a cleaned-up, *scriptified* variant of the former
``full_pipeline_experiments.ipynb`` notebook.  It performs the following steps:

1.  Build a 1-indexed mapping ``position -> citation_key`` from the
    ``data/benchmark.bib`` file.  Citation keys follow ACL Anthology style,
    e.g. ``C10-5001`` or ``2024.emnlp-main.391``.
2.  Connect to Elasticsearch using the parameters from
    :pyfile:`src.search.settings`.
3.  Create a *UUID → citation* mapping for all indexed documents.  The
    mapping is generated by extracting the citation key from the document URL.
4.  For every benchmark query the hybrid search pipeline is executed:

        – lexical   : BM25 on the ``text`` field
        – authority : PageRank score (if present)

    The three result lists are fused via *Reciprocal Rank Fusion* (RRF).
5.  Standard IR metrics (MRR, Precision@{1,5}, Recall@5, MAP, NDCG@5) are
    computed and aggregated.

Run the module directly:

    python -m src.experiments.full_pipeline_evaluation

Environment variables (or a ``.env`` file) are respected through
``src.search.settings``.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from src.search.settings import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark definition (queries + relevance judgements)
# ---------------------------------------------------------------------------

QUERY_ARTICLE_DICT: dict[str, list[int]] = {
    "kernel methods support vector machines NLP": [3, 8, 2, 10, 18],
    "machine translation history": [21, 4, 20, 13, 14],
    "What is BERT in NLP": [6, 3, 2, 15, 17],
    "transformers in NLP": [6, 3, 2, 15, 17],
    "typical errors associated with SMT output": [9, 1, 19, 20, 13],
    "kernel engineering natural language applications tutorial": [3, 13, 1, 20, 22],
    "neural lattice search domain": [14, 19, 10, 15, 5],
    "Neural machine translation evaluation methods": [14, 22, 23, 2, 9],
    "generation of visual tables": [18, 19, 20, 3, 10],
    "Context-Vector Analysis": [10, 18, 3, 17, 1],
    "How does multimodal input affect lexical retrieval?": [29, 31, 32, 33, 18],
    "Key initiating events in narrative endings": [24, 34, 35, 8, 26],
    "Semantic Representation": [26, 36, 37, 38, 27],
    "Challenges in creating LLD resources": [28, 40, 39, 41, 13],
    "Large language models Semantic Web vocabulary": [41, 39, 40, 30, 22],
    "What is diachronic lexical semantics?": [30, 44, 42, 43, 39],
    "How do language models detect dementia from speech?": [33, 45, 46, 31, 19],
    "What infrastructure is needed for enterprise MT?": [9, 47, 48, 13, 28],
    "work with multilingual NMT": [14, 9, 49, 29, 20],
    "Why markup is important": [53, 27, 50, 51, 18],
}
# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Metrics:
    """Container for per-query IR metrics."""

    mrr: float
    p_at_1: float
    p_at_5: float
    r_at_5: float
    average_precision: float
    ndcg_at_5: float


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _extract_citation_key(url: str | None) -> str | None:
    """Return the ACL Anthology citation key encoded in *url* if possible."""

    if not url:
        return None

    # Expected patterns:
    #   https://aclanthology.org/C10-5001.pdf
    #   https://aclanthology.org/2025.wnut-1.15/
    match = re.search(r"aclanthology\.org/([^/?#]+)", url)
    if match:
        key = match.group(1)
        key = key.replace(".pdf", "")  # drop optional .pdf
        key = key.rstrip("/")  # drop trailing slash
        return key

    return None


def build_id_mapping(bib_path: Path) -> dict[int, str]:
    """Return 1-indexed mapping *position → citation_key* from *benchmark.bib*.

    The mapping is built from the *url* field of each BibTeX record – not from
    the citation key inside the ``@inproceedings{…}`` header – because the
    ACL Anthology identifier is the join key later used to match Elasticsearch
    documents.
    """

    logger.info("Parsing benchmark BibTeX at %s", bib_path)
    content = bib_path.read_text(encoding="utf-8")

    entries = [e.strip() for e in content.split("@") if e.strip()]
    mapping: dict[int, str] = {}
    url_pattern = re.compile(r"url\s*=\s*\"([^\"]+)\"")

    for idx, entry in enumerate(entries, 1):
        url_match = url_pattern.search(entry)
        if not url_match:
            continue
        url = url_match.group(1)
        citation_key = _extract_citation_key(url)
        if citation_key is None:
            continue
        mapping[idx] = citation_key

    logger.info("Loaded %d BibTeX entries", len(mapping))
    return mapping


def build_uuid_to_citation_map(
    client: Elasticsearch, index_name: str
) -> dict[str, str]:
    """Return mapping *es_uuid → citation_key* for all documents in *index_name*."""

    logger.info(
        "Scanning Elasticsearch index '%s' for UUID → citation mapping …", index_name
    )
    uuid_to_citation: dict[str, str] = {}

    for hit in scan(client, index=index_name, query={"_source": ["url"]}):
        uuid: str = hit.get("_id")
        url: str | None = hit.get("_source", {}).get("url")
        citation = _extract_citation_key(url)
        if citation:
            uuid_to_citation[uuid] = citation

    logger.info("Created mapping for %d indexed documents", len(uuid_to_citation))
    return uuid_to_citation


# ---------------------------------------------------------------------------
# Search helpers (BM25 + k-NN + PageRank → RRF)
# ---------------------------------------------------------------------------


def _bm25_body(query: str, size: int) -> dict:
    return {
        "query": {"match": {"text": {"query": query, "fuzziness": "AUTO"}}},
        "size": size,
        "_source": ["pagerank"],
    }


def _rrf(rank: int, k: int) -> float:
    """Reciprocal Rank Fusion helper: ``1 / (k + rank)``."""

    return 1.0 / (k + rank)


def fuse_rrf(
    bm25_hits: list[dict],
    window_size: int,
    top_k: int,
) -> list[dict]:
    """Fuse BM25 and PageRank rankings via RRF."""

    scores: dict[str, float] = defaultdict(float)
    doc_store: dict[str, dict] = {}

    bm25_rank: dict[str, int] = {}
    pr_rank: dict[str, int] = {}

    # Collect BM25 ranks
    for r, hit in enumerate(bm25_hits, 1):
        doc_id = hit["_id"]
        doc_store[doc_id] = hit
        bm25_rank[doc_id] = r

    # Compute PageRank ranks (smaller rank = higher score)
    pageranks = {
        doc_id: hit.get("_source", {}).get("pagerank")
        for doc_id, hit in doc_store.items()
        if hit.get("_source", {}).get("pagerank") is not None
    }
    for r, (doc_id, _score) in enumerate(
        sorted(pageranks.items(), key=lambda x: x[1], reverse=True),
        1,
    ):
        pr_rank[doc_id] = r

    # Combine
    for doc_id in doc_store:
        if doc_id in bm25_rank:
            scores[doc_id] += _rrf(bm25_rank[doc_id], window_size)
        if doc_id in pr_rank:
            scores[doc_id] += _rrf(pr_rank[doc_id], window_size)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [doc_store[d_id] for d_id, _ in sorted_docs]


def search_query(
    client: Elasticsearch,
    query: str,
    top_k: int = 5,
) -> list[str]:
    """Execute search for *query* using only BM25 and PageRank, returning list of Elasticsearch UUIDs."""

    window_size = max(top_k, 10)

    bm25_hits = client.search(
        index=settings.index_name, body=_bm25_body(query, window_size)
    )["hits"]["hits"]

    fused_hits = fuse_rrf(bm25_hits, window_size, top_k)
    return [h["_id"] for h in fused_hits]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def reciprocal_rank(relevant: set[str], retrieved: list[str]) -> float:  # MRR@ALL
    for rank, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    hit_count = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hit_count / k


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit_count = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hit_count / len(relevant)


def average_precision(relevant: set[str], retrieved: list[str]) -> float:
    if not relevant:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for rank, doc in enumerate(retrieved, 1):
        if doc in relevant:
            hits += 1
            sum_precisions += hits / rank
    return sum_precisions / len(relevant)


def dcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], 1):
        rel = 1 if doc in relevant else 0
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    dcg = dcg_at_k(relevant, retrieved, k)
    ideal_rels = [1] * min(len(relevant), k)
    idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal_rels, 1))
    return 0.0 if idcg == 0 else dcg / idcg


# ---------------------------------------------------------------------------
# Evaluation routine
# ---------------------------------------------------------------------------


def evaluate() -> None:
    """Run the full evaluation and print aggregated results."""

    project_root = Path(__file__).resolve().parents[2]
    bib_path = project_root / "data" / "benchmark.bib"

    id_mapping = build_id_mapping(bib_path)

    client = Elasticsearch(settings.es_host)
    if not client.ping():
        raise SystemExit(f"Cannot connect to Elasticsearch at {settings.es_host}")

    uuid_to_citation = build_uuid_to_citation_map(client, settings.index_name)

    # Aggregate containers
    all_metrics: list[Metrics] = []

    for idx, (query, relevant_indices) in enumerate(QUERY_ARTICLE_DICT.items(), 1):
        logger.info("[%d/%d] %s", idx, len(QUERY_ARTICLE_DICT), query)

        relevant_citations = {
            id_mapping[i] for i in relevant_indices if i in id_mapping
        }
        logger.debug("  Relevant citations: %s", relevant_citations)

        retrieved_uuids = search_query(client, query, top_k=settings.top_k)
        retrieved_citations: list[str] = [
            uuid_to_citation[uid] for uid in retrieved_uuids if uid in uuid_to_citation
        ]

        # Compute metrics
        metrics = Metrics(
            mrr=reciprocal_rank(relevant_citations, retrieved_citations),
            p_at_1=precision_at_k(relevant_citations, retrieved_citations, 1),
            p_at_5=precision_at_k(relevant_citations, retrieved_citations, 5),
            r_at_5=recall_at_k(relevant_citations, retrieved_citations, 5),
            average_precision=average_precision(
                relevant_citations, retrieved_citations
            ),
            ndcg_at_5=ndcg_at_k(relevant_citations, retrieved_citations, 5),
        )
        all_metrics.append(metrics)

        logger.info(
            "    MRR=%0.3f  P@1=%0.3f  P@5=%0.3f  R@5=%0.3f  MAP=%0.3f  NDCG@5=%0.3f",
            metrics.mrr,
            metrics.p_at_1,
            metrics.p_at_5,
            metrics.r_at_5,
            metrics.average_precision,
            metrics.ndcg_at_5,
        )

    # Aggregate across queries
    def _avg(attr: str) -> float:
        return sum(getattr(m, attr) for m in all_metrics) / len(all_metrics)

    logger.info("\n=== FINAL AVERAGED RESULTS (n=%d queries) ===", len(all_metrics))
    logger.info("MRR          : %.3f", _avg("mrr"))
    logger.info("Precision@1  : %.3f", _avg("p_at_1"))
    logger.info("Precision@5  : %.3f", _avg("p_at_5"))
    logger.info("Recall@5     : %.3f", _avg("r_at_5"))
    logger.info("MAP          : %.3f", _avg("average_precision"))
    logger.info("NDCG@5       : %.3f", _avg("ndcg_at_5"))


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate()
