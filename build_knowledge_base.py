import os
import sys
import subprocess

from rag.document_loader import load_documents
from rag.chunker import prepare_chunks
from rag.vector_store import build_and_save_vector_store


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOWNLOAD_SCRIPT = os.path.join(
    BASE_DIR,
    "data",
    "download_sources.py"
)


def main():
    print("=" * 60)
    print("AwaamiAgent Knowledge Base Builder")
    print("=" * 60)

    # Step 1: Download official source documents
    print("\n[1/4] Downloading official civic documents...")

    if not os.path.exists(DOWNLOAD_SCRIPT):
        raise FileNotFoundError(
            f"Download script not found: {DOWNLOAD_SCRIPT}"
        )

    subprocess.run(
        [sys.executable, DOWNLOAD_SCRIPT],
        check=True
    )

    # Step 2: Load PDF documents
    print("\n[2/4] Loading PDF documents...")

    documents = load_documents()

    if not documents:
        raise RuntimeError(
            "No documents were loaded. "
            "Please check data/sources.csv and data/documents/."
        )

    print(f"Loaded {len(documents)} document pages.")

    # Step 3: Split documents into chunks
    print("\n[3/4] Creating text chunks...")

    chunks = prepare_chunks(documents)

    if not chunks:
        raise RuntimeError(
            "No text chunks were created."
        )

    print(f"Created {len(chunks)} chunks.")

    # Step 4: Create FAISS vector database
    print("\n[4/4] Creating FAISS vector database...")

    store = build_and_save_vector_store(chunks)

    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Documents/pages: {len(documents)}")
    print(f"Text chunks:     {len(chunks)}")
    print(f"FAISS vectors:   {store.index.ntotal}")
    print(f"Metadata records:{len(store.metadata)}")

    print("\nVector database location:")
    print("data/vector_store/")

    print("\nAwaamiAgent RAG knowledge base is ready.")


if __name__ == "__main__":
    main()
