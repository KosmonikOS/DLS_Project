"""Enhanced interactive search CLI with typo correction.

This is an improved version of search_cli.py that includes:
- Automatic typo correction before search
- Query suggestions for ambiguous corrections
- Confidence scoring for corrections
- Option to use original query if correction confidence is low
"""

from __future__ import annotations

from collections import defaultdict
from elasticsearch import Elasticsearch
from src.common.dense_embedder import DenseEmbedder
from src.search.settings import settings
from src.search.query_processor import QueryProcessor
import logging

logger = logging.getLogger(__name__)


def _print_hit(rank: int, hit: dict) -> None:
    source = hit["_source"]
    title = source.get("title", "<no title>")
    year = source.get("year", "?")
    authors = ", ".join(source.get("author", []) or [])
    pagerank = source.get("pagerank", 0.0)

    print(f"{rank}. {title} ({year})")
    if authors:
        print(f"   Authors: {authors}")
    if source.get("url"):
        print(f"   URL: {source['url']}")
    if pagerank > 0:
        print(f"   PageRank: {pagerank:.6f}")


def _connect() -> Elasticsearch:
    """Create and validate an Elasticsearch connection."""
    client = Elasticsearch(settings.es_host)
    if not client.ping():
        raise SystemExit(f"Cannot connect to Elasticsearch at {settings.es_host}")
    return client


def _bm25_body(query: str, size: int) -> dict:
    """Construct a BM25 query that searches across *text*, *title*, and *author*.

    Title and author fields receive a higher boost compared to the main body
    text so that exact matches in metadata rank higher.
    """

    return {
        "query": {
            "multi_match": {
                "query": query,
                "fields": [
                    "title^2",  # titles are highly informative
                    "author",  # author names are somewhat informative
                    "text",  # full-text body
                ],
                "type": "best_fields",
                "fuzziness": "AUTO",
            }
        },
        "size": size,
        "_source": ["title", "author", "year", "url", "pagerank"],
    }


def _rrf_fuse(lex_hits: list[dict], window_size: int, top_k: int) -> list[dict]:
    """
    Combine BM25 and PageRank using Reciprocal Rank Fusion (RRF).
    """

    def rrf(rank: int, k: int) -> float:
        return 1.0 / (k + rank)

    scores: dict[str, float] = defaultdict(float)
    doc_store: dict[str, dict] = {}

    # Store hits and prepare rank dictionaries
    bm25_rank: dict[str, int] = {}
    pagerank_values: dict[str, float] = {}

    for r, hit in enumerate(lex_hits, 1):
        doc_id = hit["_id"]
        doc_store[doc_id] = hit
        bm25_rank[doc_id] = r
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
        if doc_id in pr_rank:
            scores[doc_id] += rrf(pr_rank[doc_id], window_size)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [doc_store[d_id] for d_id, _ in sorted_docs]


def _execute_search(
    client: Elasticsearch,
    query: str,
) -> list[dict]:
    """Execute the search with the given query using only BM25 and PageRank."""
    window_size = max(settings.top_k, 10)

    # Execute BM25 search
    lex_hits = client.search(
        index=settings.index_name,
        body=_bm25_body(query, window_size),
    )["hits"]["hits"]

    # Fuse results (BM25 + PageRank only)
    return _rrf_fuse(lex_hits, window_size, settings.top_k)


def _interactive_loop(client: Elasticsearch, query_processor: QueryProcessor) -> None:
    """Enhanced prompt loop with typo correction."""

    print("\n=== ACL Anthology Search Engine ===")
    print("Type 'help' for commands, 'exit' to quit\n")

    while True:
        try:
            query = input("query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        # Handle commands
        if query.lower() == "exit" or query.lower() == "quit":
            print("Goodbye!")
            break
        elif query.lower() == "help":
            print("\nCommands:")
            print("  help     - Show this help message")
            print("  exit/quit - Exit the search engine")
            print("  !<query>  - Skip typo correction for this query")
            print("\nJust type your search query to find papers.\n")
            continue

        # Check if user wants to skip correction (prefix with !)
        skip_correction = query.startswith("!")
        if skip_correction:
            query = query[1:].strip()

        # Process query with typo correction unless skipped
        if not skip_correction:
            corrected_query, corrections, confidence = query_processor.process_query(
                query
            )

            # Show corrections if any were made
            if corrections and corrected_query != query:
                print(f"\n📝 Query corrected: '{query}' → '{corrected_query}'")
                print(f"   Corrections: {', '.join(corrections)}")
                print(f"   Confidence: {confidence:.2%}")

                # If confidence is low, ask user
                if confidence < 0.8:
                    print("\n   Low confidence in corrections. Choose an option:")
                    print(f"   1) Use corrected: '{corrected_query}'")
                    print(f"   2) Use original: '{query}'")

                    # Show alternatives if available
                    alternatives = query_processor.suggest_alternatives(query)
                    if alternatives:
                        print("   3) Other suggestions:")
                        for i, alt in enumerate(alternatives, 1):
                            print(f"      3.{i}) '{alt}'")

                    choice = input("\n   Enter choice (1/2/3.x): ").strip()

                    if choice == "2":
                        corrected_query = query
                    elif choice.startswith("3."):
                        try:
                            alt_idx = int(choice[2:]) - 1
                            if 0 <= alt_idx < len(alternatives):
                                corrected_query = alternatives[alt_idx]
                        except ValueError:
                            pass

                query_to_search = corrected_query
            else:
                query_to_search = query
        else:
            query_to_search = query

        # Execute search
        print(f"\n🔍 Searching for: '{query_to_search}'...")
        hits = _execute_search(client, query_to_search)

        if not hits:
            print("No matches found.\n")

            # Suggest alternatives if no results
            if not skip_correction:
                alternatives = query_processor.suggest_alternatives(query_to_search)
                if alternatives:
                    print("Did you mean:")
                    for alt in alternatives:
                        print(f"   - {alt}")
                    print()
            continue

        # Display results
        print(f"\nFound {len(hits)} results:\n")
        for i, hit in enumerate(hits, 1):
            _print_hit(i, hit)
            print()  # Empty line between results


def main() -> None:
    """Enhanced main function with query processing."""
    # Initialize components
    client = _connect()

    # Initialize query processor with custom NLP vocabulary
    custom_vocab = [
        # Add conference-specific terms
        "emnlp",
        "naacl",
        "eacl",
        "coling",
        "lrec",
        "conll",
        "tacl",
        # Add more technical terms as needed
        "roberta",
        "xlnet",
        "albert",
        "electra",
        "t5",
        "bart",
        "squad",
        "glue",
        "superglue",
        "bleu",
        "rouge",
        "meteor",
    ]
    query_processor = QueryProcessor(custom_vocabulary=custom_vocab)

    # Run interactive loop
    _interactive_loop(client, query_processor)


if __name__ == "__main__":
    main()
