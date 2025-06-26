"""Interactive search CLI – BM25 + vector similarity + PageRank with client-side fusion.

For every user query the tool:
1. Embeds the query with :class:`DenseEmbedder`.
2. Executes two ES searches (lexical + KNN).
3. Fuses the result sets locally using Reciprocal Rank Fusion (RRF) with PageRank as a third ranked list.
   - Final score = RRF(BM25 rank) + RRF(KNN rank) + RRF(PageRank rank).
4. Prints the top-k documents with basic metadata and PageRank.

All runtime parameters are supplied via :pyfile:`src.search.settings`.
"""

from __future__ import annotations

from collections import defaultdict
from elasticsearch import Elasticsearch
from src.common.dense_embedder import DenseEmbedder
from src.search.settings import settings


def _print_hit(rank: int, hit: dict) -> None:
    source = hit["_source"]
    title = source.get("title", "<no title>")
    year = source.get("year", "?")
    authors = ", ".join(source.get("author", []) or [])
    print(f"{rank}. {title} ({year})")
    if authors:
        print(f"   {authors}")
    if source.get("url"):
        print(f"   {source['url']}")


def _connect() -> Elasticsearch:
    """Create and validate an Elasticsearch connection."""

    client = Elasticsearch(settings.es_host)
    if not client.ping():
        raise SystemExit(f"Cannot connect to Elasticsearch at {settings.es_host}")
    return client


def _bm25_body(query: str, size: int) -> dict:
    return {
        "query": {"match": {"text": {"query": query, "fuzziness": "AUTO"}}},
        "size": size,
        "_source": ["title", "author", "year", "url", "pagerank"],
    }


def _knn_body(query_vector: list[float]) -> dict:
    return {
        "knn": {
            "field": "text_embedding",
            "query_vector": query_vector,
            "k": settings.knn_k,
            "num_candidates": settings.knn_candidates,
        },
        "size": settings.knn_k,
        "_source": ["title", "author", "year", "url", "pagerank"],
    }


def _rrf_fuse(
    lex_hits: list[dict], knn_hits: list[dict], window_size: int, top_k: int
) -> list[dict]:
    """
    Combine BM25, KNN, and PageRank using Reciprocal Rank Fusion (RRF).

    - Each list contributes via RRF: 1 / (k + rank).
    - PageRank contributes through its rank order (no score normalisation).
    - Final score: RRF(BM25) + RRF(KNN) + RRF(PageRank).
    - Returns the top_k documents by combined score.
    """

    def rrf(rank: int, k: int) -> float:
        return 1.0 / (k + rank)

    scores: dict[str, float] = defaultdict(float)
    doc_store: dict[str, dict] = {}

    # Store hits and prepare rank dictionaries
    bm25_rank: dict[str, int] = {}
    knn_rank: dict[str, int] = {}
    pagerank_values: dict[str, float] = {}

    for r, hit in enumerate(lex_hits, 1):
        doc_id = hit["_id"]
        doc_store[doc_id] = hit
        bm25_rank[doc_id] = r
        pr = hit["_source"].get("pagerank")
        if pr is not None:
            pagerank_values[doc_id] = pr

    for r, hit in enumerate(knn_hits, 1):
        doc_id = hit["_id"]
        doc_store.setdefault(doc_id, hit)
        knn_rank[doc_id] = r
        pr = hit["_source"].get("pagerank")
        if pr is not None:
            pagerank_values[doc_id] = pr

    # Determine PageRank ranking among all collected docs
    pr_sorted = sorted(pagerank_values.items(), key=lambda x: x[1], reverse=True)
    pr_rank = {doc_id: r + 1 for r, (doc_id, _) in enumerate(pr_sorted)}

    # Fuse using RRF formula
    for doc_id in doc_store:
        if doc_id in bm25_rank:
            scores[doc_id] += rrf(bm25_rank[doc_id], window_size)
        if doc_id in knn_rank:
            scores[doc_id] += rrf(knn_rank[doc_id], window_size)
        if doc_id in pr_rank:
            scores[doc_id] += rrf(pr_rank[doc_id], window_size)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [doc_store[d_id] for d_id, _ in sorted_docs]


def _interactive_loop(client: Elasticsearch, embedder: DenseEmbedder) -> None:
    """Prompt the user for queries and display fused results."""

    while True:
        try:
            query = input("query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not query or query.lower() in {"exit", "quit"}:
            break

        query_vector = embedder.embed_query(query)
        window_size = max(settings.knn_k, settings.top_k)

        lex_hits = client.search(
            index=settings.index_name,
            body=_bm25_body(query, window_size),
        )["hits"]["hits"]

        knn_hits = client.search(
            index=settings.index_name,
            body=_knn_body(query_vector),
        )["hits"]["hits"]

        if not lex_hits and not knn_hits:
            print("No matches\n")
            continue

        hits = _rrf_fuse(lex_hits, knn_hits, window_size, settings.top_k)

        for i, hit in enumerate(hits, 1):
            _print_hit(i, hit)
        print()


def main() -> None:
    client = _connect()
    embedder = DenseEmbedder()
    _interactive_loop(client, embedder)


if __name__ == "__main__":
    main()
