# rag.py

import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from vector_store import build_vector_store
from ingest import load_documents, chunk_text
from groq import Groq

load_dotenv()

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Embedding model (unchanged)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def embed_query(query):
    return np.array(embedder.encode([query]), dtype="float32")


def retrieve(query, index, chunks, k=3):
    query_embedding = embed_query(query)
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]


def generate_answer(query, context):
    prompt = f"""
You are a helpful study assistant.

Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()
