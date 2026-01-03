import streamlit as st
from ingest import chunk_text
from vector_store import build_vector_store
from rag import retrieve, generate_answer
from evaluator import evaluate_answer

st.set_page_config(page_title="Agentic RAG Assistant", layout="centered")

st.title("🧠 Agentic RAG Research Assistant")
st.write("Upload a document and ask questions grounded only in that document.")

uploaded_file = st.file_uploader(
    "Upload a text file",
    type=["txt"]
)

if uploaded_file:
    text = uploaded_file.read().decode("utf-8").strip()

    if not text:
        st.error("Uploaded file is empty. Please upload a valid document.")
        st.stop()

    st.success("File uploaded successfully!")

    chunks = chunk_text(text)

    if not chunks:
        st.error("No valid chunks could be created from this document.")
        st.stop()

    with st.expander("📄 View Generated Chunks"):
        for i, chunk in enumerate(chunks[:5]):
            st.write(f"**Chunk {i+1}:** {chunk[:300]}...")

    try:
        index, chunks = build_vector_store(chunks)
    except Exception as e:
        st.error(f"Vector store error: {e}")
        st.stop()

query = st.text_input("Ask a question about the document")

if uploaded_file and query:
    for attempt in range(2):
        retrieved_chunks = retrieve(query, index, chunks, k=attempt + 1)
        context = "\n".join(retrieved_chunks)[:1000]

        answer = generate_answer(query, context)

        st.subheader(f"🔁 Attempt {attempt + 1}")
        st.write("🧠 Generated Answer:")
        st.write(answer)

        accepted = evaluate_answer(answer)

        if accepted:
            st.success("✅ Final answer accepted by agent")
            break
        else:
            st.warning("❌ Answer rejected — retrying with broader retrieval")

    else:
        st.error("🚫 Agent could not produce a confident answer.")
