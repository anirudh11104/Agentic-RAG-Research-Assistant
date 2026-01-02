def load_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, max_words=120):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        words = para.split()
        if current_length + len(words) <= max_words:
            current_chunk.append(para)
            current_length += len(words)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_length = len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks





if __name__ == "__main__":
    text = load_documents("data/docs.txt")
    chunks = chunk_text(text)

    print(f"Total chunks created: {len(chunks)}")
    print("Sample chunk:\n")
    print(chunks[0])
