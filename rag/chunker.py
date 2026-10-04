def prepare_chunks(
    documents: List[Dict],
    chunk_size: int = 800,
    overlap: int = 100
) -> List[Dict]:
    """
    Split loaded document pages into overlapping text chunks.
    """

    chunks = []

    for document in documents:

        text = document.get("text", "").strip()
        metadata = document.get("metadata", {}).copy()

        if not text:
            continue

        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(
                start + chunk_size,
                text_length
            )

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunk_metadata = metadata.copy()

                chunk_metadata["chunk"] = len(chunks) + 1

                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })

            if end >= text_length:
                break

            start = end - overlap

    return chunks
