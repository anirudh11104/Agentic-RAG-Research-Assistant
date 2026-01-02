# rag.py

import numpy as np
from sentence_transformers import SentenceTransformer
from vector_store import build_vector_store
from ingest import load_documents, chunk_text
from transformers import pipeline
from evaluator import evaluate_answer


# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Local LLM (FREE)
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)


def embed_query(query):
    return np.array(embedder.encode([query]), dtype="float32")


def retrieve(query, index, chunks, k=1):
    query_embedding = embed_query(query)
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]


def generate_answer(query, context):
    prompt = f"""
You are a technical assistant.

Using ONLY the information in the context below,
answer the question in 2–3 clear sentences.

If the context does not contain the answer, reply exactly:
"I don't know".

Context:
{context}

Question:
{query}
"""

    result = qa_pipeline(prompt)
    return result[0]["generated_text"].strip()


if __name__ == "__main__":
    text = load_documents("data/docs.txt")
    chunks = chunk_text(text)
    index, chunks = build_vector_store(chunks)

    query = input("Ask a question: ")

    for attempt in range(2):  # agent retries
        retrieved_chunks = retrieve(query, index, chunks, k=attempt + 1)
        context = "\n".join(retrieved_chunks)[:1000]

        answer = generate_answer(query, context)

        print(f"\nAttempt {attempt + 1} Answer:\n{answer}")

        if evaluate_answer(answer):
            print("\n✅ Answer accepted by evaluator")
            break
        else:
            print("\n⚠️ Answer rejected — retrying with broader retrieval...\n")

