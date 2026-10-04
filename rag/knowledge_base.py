import os
import subprocess
import sys

import streamlit as st

from rag.document_loader import load_documents
from rag.chunker import prepare_chunks
from rag.vector_store import build_and_save_vector_store


VECTOR_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "vector_store"
)

INDEX_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "civic.index"
)

METADATA_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "metadata.json"
)


def knowledge_base_exists():
    """Check whether the FAISS knowledge base already exists."""
    return (
        os.path.exists(INDEX_FILE)
        and os.path.exists(METADATA_FILE)
    )


def build_knowledge_base():
    """
    Download official sources, extract text, create chunks,
    generate embeddings, and build the FAISS vector database.
    """

    print("Starting AwaamiAgent knowledge-base build...")

    # Download official PDF sources
    download_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "download_sources.py"
    )

    if not os.path.exists(download_script):
        raise FileNotFoundError(
            "data/download_sources.py was not found."
        )

    subprocess.run(
        [sys.executable, download_script],
        check=True
    )

    # Load downloaded documents
    documents = load_documents()

    if not documents:
        raise RuntimeError(
            "No documents were loaded from data/documents."
        )

    print(f"Loaded {len(documents)} document pages.")

    # Create text chunks
    chunks = prepare_chunks(documents)

    if not chunks:
        raise RuntimeError(
            "No text chunks were created."
        )

    print(f"Created {len(chunks)} text chunks.")

    # Create FAISS vector database
    index, metadata = build_and_save_vector_store(chunks)

    print(
        f"FAISS knowledge base created with "
        f"{index.ntotal} vectors."
    )

    return index, metadata


@st.cache_resource(show_spinner=False)
def ensure_knowledge_base():
    """
    Load the existing knowledge base or build it once
    when it does not exist.
    """

    if knowledge_base_exists():
        return "existing"

    build_knowledge_base()

    return "created"
