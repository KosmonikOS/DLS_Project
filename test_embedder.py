from src.indexing.dense_embedder import get_embeddings

texts = [
    "Example of the first text",
    "Second text for the embedding"
]
embeddings = get_embeddings(texts)
print("Embeddings:", embeddings)
print("Dimensionality:", len(embeddings[0]))