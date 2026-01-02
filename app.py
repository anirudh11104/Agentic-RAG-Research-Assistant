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
    text = uploaded_file.read().decode("utf-8")
    st.success("File uploaded successfully!")

query = st.text_input("Ask a question about the document")

if uploaded_file and query:
    with st.spinner("Processing..."):

        chunks = chunk_text(text)
        with st.expander("📄 View Generated Chunks"):
            for i, chunk in enumerate(chunks[:5]):
                st.write(f"**Chunk {i+1}:** {chunk[:300]}...")
        index, chunks = build_vector_store(chunks)

        for attempt in range(2):
            st.markdown(f"### 🔁 Attempt {attempt + 1}")

            retrieved_chunks = retrieve(query, index, chunks, k=attempt + 1)
            context = "\n".join(retrieved_chunks)[:1000]

            answer = generate_answer(query, context)

            st.write("🧠 **Generated Answer:**")
            st.write(answer)

            decision = evaluate_answer(answer)

            st.write(
            "📊 **Evaluator Decision:**",
            "ACCEPTED ✅" if decision else "REJECTED ❌"
            )

            if decision:
                st.success("Final answer accepted by agent")
                break
            else:
                st.warning("Answer rejected — retrying with broader retrieval")
        else:
            st.error("Agent could not produce a confident answer.")


