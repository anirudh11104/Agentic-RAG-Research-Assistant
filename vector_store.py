import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ingest import load_documents, chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts):
    embeddings = model.encode(texts)
    return np.array(embeddings, dtype="float32")


def build_vector_store(chunks):
    embeddings = embed_texts(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks


if __name__ == "__main__":
    text = load_documents("data/docs.txt")
    chunks = chunk_text(text)

    index, stored_chunks = build_vector_store(chunks)

    print(f"Vector DB created with {index.ntotal} embeddings")
