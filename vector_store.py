import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    if not texts:
        return None

    embeddings = embedder.encode(texts)

    if embeddings is None or len(embeddings) == 0:
        return None

    return np.array(embeddings, dtype="float32")


def build_vector_store(chunks):
    if not chunks:
        raise ValueError("No chunks provided for vector store")

    embeddings = embed_texts(chunks)

    if embeddings is None or embeddings.ndim != 2:
        raise ValueError("Failed to generate valid embeddings")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks
