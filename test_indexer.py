from src.indexing.elastic_search_indexer import ElasticSearchIndexer
from src.indexing.dense_embedder import get_embeddings

texts = [
    "Example of the first text",
    "Second text for the embedding"
]
embeddings = get_embeddings(texts)
docs = [
    {
        "text": text,
        "title": f"Title {i}",
        "author": ["Author"],
        "year": "2024",
        "url": f"http://example.com/{i}",
        "text_embedding": emb
    }
    for i, (text, emb) in enumerate(zip(texts, embeddings))
]

indexer = ElasticSearchIndexer()
index_name = "test_dense_index"
indexer.create_index(index_name, force_delete=True, embedding_dim=1024)

indexer.index_documents(index_name, docs)
print("Indexing has finished.")