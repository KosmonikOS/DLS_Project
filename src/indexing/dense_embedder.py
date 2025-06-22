from langchain_community.embeddings import OllamaEmbeddings
from typing import List

def get_embeddings(texts: List[str], model_name: str = "dengcao/Qwen3-Embedding-0.6B:Q8_0") -> List[List[float]]:
    """
    Generates embeddings for a list of texts using models from Ollama.

    Arguments:
    texts: List of strings to embed.
    model_name: Name of the model in Ollama.

    Returns:
    List of embeddings (lists of floats).
    """
    
    ollama_embeddings = OllamaEmbeddings(
        model=model_name,
    )
    
    return ollama_embeddings.embed_documents(texts)

if __name__ == '__main__':
    # Example
    
    sample_texts = [
        "This is an example of text to get embedding.",
        "LangChain is very convenient for working with models.",
    ]
    
    try:
        embeddings = get_embeddings(sample_texts)
        print(f"Got {len(embeddings)} embeddings.")
        if embeddings:
            print(f"Dimensionality of the first embedding: {len(embeddings[0])}")
            # print("Example of embedding:", embeddings[0][:5])
    except Exception as e:
        print(f"Got error: {e}")
        print("Check that model is running with Ollama") 