import json
import os
from typing import Dict, List, Tuple

import faiss
import numpy as np

from .embeddings import EmbeddingModel


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

VECTOR_DIR = os.path.join(
    BASE_DIR,
    "data",
    "vector_store"
)

INDEX_FILE = os.path.join(
    VECTOR_DIR,
    "civic.index"
)

METADATA_FILE = os.path.join(
    VECTOR_DIR,
    "metadata.json"
)


class CivicVectorStore:
    """
    FAISS-based vector store for civic documents.

    The FAISS index stores vectors.
    metadata.json stores the original text and
    official source information for every vector.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.embedding_model = EmbeddingModel(
            model_name=model_name
        )

        self.index = None
        self.metadata: List[Dict] = []

    def build(
        self,
        chunks: List[Dict],
    ) -> None:
        """
        Build a FAISS index from document chunks.
        """

        if not chunks:
            raise ValueError(
                "Cannot build vector store from empty chunks."
            )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        vectors = self.embedding_model.encode(
            texts
        )

        dimension = vectors.shape[1]

        # Inner product works well with normalized embeddings.
        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            vectors
        )

        self.metadata = chunks

    def save(self) -> None:
        """
        Save FAISS index and metadata to disk.
        """

        if self.index is None:
            raise ValueError(
                "Vector index has not been built."
            )

        os.makedirs(
            VECTOR_DIR,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            INDEX_FILE
        )

        with open(
            METADATA_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2
            )

    def load(self) -> None:
        """
        Load an existing FAISS index and metadata.
        """

        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_FILE}"
            )

        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_FILE}"
            )

        self.index = faiss.read_index(
            INDEX_FILE
        )

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(
                file
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Search the vector store using a user query.
        """

        if self.index is None:
            raise ValueError(
                "Vector index has not been loaded or built."
            )

        if not query or not query.strip():
            return []

        query_vector = (
            self.embedding_model.encode_query(
                query
            )
        )

        number_of_results = min(
            top_k,
            self.index.ntotal
        )

        if number_of_results == 0:
            return []

        scores, indices = self.index.search(
            query_vector,
            number_of_results
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            if index >= len(self.metadata):
                continue

            result = self.metadata[index].copy()

            result["score"] = float(
                score
            )

            results.append(
                result
            )

        return results


def build_and_save_vector_store(
    chunks: List[Dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> CivicVectorStore:
    """
    Build and save a civic FAISS vector store.
    """

    store = CivicVectorStore(
        model_name=model_name
    )

    store.build(
        chunks
    )

    store.save()

    return store


def load_vector_store(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> CivicVectorStore:
    """
    Load an existing civic vector store.
    """

    store = CivicVectorStore(
        model_name=model_name
    )

    store.load()

    return store


if __name__ == "__main__":

    print(
        "Civic vector store module loaded successfully."
    )
