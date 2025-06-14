import argparse
from elasticsearch import Elasticsearch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive search over indexed papers.")
    parser.add_argument("--index-name", default="papers", help="Elasticsearch index name")
    parser.add_argument("--es-hosts", nargs="+", default=["http://localhost:9200"], help="Elasticsearch hosts")
    return parser.parse_args()


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


def main() -> None:
    args = _parse_args()
    client = Elasticsearch(args.es_hosts)

    if not client.ping():
        raise SystemExit(f"Cannot connect to Elasticsearch at {args.es_hosts}")

    while True:
        try:
            query = input("query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not query or query.lower() in {"exit", "quit"}:
            break

        body = {
            "query": {
                "match": {
                    "text": {
                        "query": query,
                        "fuzziness": "AUTO"
                    }
                }
            },
            "size": 5,
            "_source": ["title", "author", "year", "url"]
        }

        res = client.search(index=args.index_name, body=body)
        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            print("No matches\n")
            continue
        for i, hit in enumerate(hits, 1):
            _print_hit(i, hit)
        print()


if __name__ == "__main__":
    main() 