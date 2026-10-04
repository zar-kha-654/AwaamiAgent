import csv
import os
from typing import List, Dict
from pypdf import PdfReader


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES_FILE = os.path.join(
    BASE_DIR,
    "data",
    "sources.csv"
)

DOCUMENTS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "documents"
)


def load_source_metadata() -> Dict[str, Dict]:
    """
    Load source information from sources.csv.
    """

    sources = {}

    if not os.path.exists(SOURCES_FILE):
        return sources

    with open(
        SOURCES_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            source_id = row.get("source_id", "").strip()

            if source_id:
                sources[source_id] = row

    return sources


def extract_pdf_text(pdf_path: str) -> List[Dict]:
    """
    Extract text from every page of a PDF.
    """

    documents = []

    source_metadata = load_source_metadata()

    filename = os.path.basename(pdf_path)

    source_id = os.path.splitext(filename)[0]

    metadata = source_metadata.get(
        source_id,
        {}
    )

    reader = PdfReader(pdf_path)

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        text = text.strip()

        if not text:
            continue

        page_metadata = metadata.copy()

        page_metadata.update({
            "source_id": source_id,
            "filename": filename,
            "page": page_number
        })

        documents.append({
            "text": text,
            "metadata": page_metadata
        })

    return documents


def load_documents() -> List[Dict]:
    """
    Load all PDF documents from data/documents.
    """

    documents = []

    if not os.path.exists(DOCUMENTS_DIR):
        return documents

    for filename in sorted(
        os.listdir(DOCUMENTS_DIR)
    ):

        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(
            DOCUMENTS_DIR,
            filename
        )

        try:
            documents.extend(
                extract_pdf_text(pdf_path)
            )

        except Exception as e:
            print(
                f"Could not load {filename}: {e}"
            )

    return documents


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

        text = document.get(
            "text",
            ""
        ).strip()

        metadata = document.get(
            "metadata",
            {}
        ).copy()

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

                chunk_metadata["chunk"] = (
                    len(chunks) + 1
                )

                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })

            if end >= text_length:
                break

            start = end - overlap

    return chunks


if __name__ == "__main__":

    documents = load_documents()

    print(
        f"Loaded {len(documents)} document pages."
    )

    chunks = prepare_chunks(documents)

    print(
        f"Created {len(chunks)} text chunks."
    )
