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
    Load source information from data/sources.csv.

    Returns:
        Dictionary where the key is source_id.
    """

    sources = {}

    if not os.path.exists(SOURCES_FILE):
        raise FileNotFoundError(
            f"Sources file not found: {SOURCES_FILE}"
        )

    with open(
        SOURCES_FILE,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            source_id = row.get("source_id", "").strip()

            if not source_id:
                continue

            sources[source_id] = {
                "source_id": source_id,
                "category": row.get("category", "").strip(),
                "source_name": row.get("source_name", "").strip(),
                "source_url": row.get("source_url", "").strip(),
                "authority": row.get("authority", "").strip(),
                "description": row.get("description", "").strip(),
                "last_verified": row.get("last_verified", "").strip(),
            }

    return sources


def extract_pdf_text(pdf_path: str) -> List[Dict]:
    """
    Extract text from every page of a PDF.

    Returns a list containing one dictionary per page.
    """

    pages = []

    reader = PdfReader(pdf_path)

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = text.strip()

        if text:
            pages.append({
                "page_number": page_number,
                "text": text
            })

    return pages


def load_documents() -> List[Dict]:
    """
    Load all PDF documents from data/documents/.

    Each page is returned with source metadata.
    """

    if not os.path.exists(DOCUMENTS_DIR):
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    source_metadata = load_source_metadata()

    documents = []

    for filename in sorted(os.listdir(DOCUMENTS_DIR)):

        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(
            DOCUMENTS_DIR,
            filename
        )

        # Expected filename format:
        # ELEC001_some_document_name.pdf

        source_id = filename.split("_", 1)[0].strip()

        metadata = source_metadata.get(
            source_id,
            {}
        )

        pages = extract_pdf_text(pdf_path)

        for page in pages:

            documents.append({
                "text": page["text"],

                "metadata": {
                    "source_id": source_id,
                    "source_name": metadata.get(
                        "source_name",
                        ""
                    ),
                    "category": metadata.get(
                        "category",
                        ""
                    ),
                    "authority": metadata.get(
                        "authority",
                        ""
                    ),
                    "source_url": metadata.get(
                        "source_url",
                        ""
                    ),
                    "description": metadata.get(
                        "description",
                        ""
                    ),
                    "last_verified": metadata.get(
                        "last_verified",
                        ""
                    ),
                    "document": filename,
                    "page": page["page_number"],
                }
            })

    return documents


if __name__ == "__main__":

    documents = load_documents()

    print(
        f"Loaded {len(documents)} document pages."
    )

    for document in documents[:3]:

        print("\n--- DOCUMENT ---")

        print(
            "Source:",
            document["metadata"]["source_id"]
        )

        print(
            "Name:",
            document["metadata"]["source_name"]
        )

        print(
            "Page:",
            document["metadata"]["page"]
        )

        print(
            "Text:",
            document["text"][:500]
        )
