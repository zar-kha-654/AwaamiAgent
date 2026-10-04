from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer.

    This class converts civic document text into
    numerical vectors that can be searched using FAISS.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

    def encode(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Convert a list of texts into embedding vectors.
        """

        if not texts:
            return np.empty(
                (0, 384),
                dtype="float32"
            )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.astype(
            "float32"
        )

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Convert a user question into one embedding vector.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.astype(
            "float32"
        )


def create_embeddings(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """
    Convenience function for creating embeddings.
    """

    model = EmbeddingModel(
        model_name=model_name
    )

    return model.encode(texts)


if __name__ == "__main__":

    model = EmbeddingModel()

    sample_texts = [
        "A consumer may submit a complaint regarding an electricity bill.",
        "Electricity distribution companies must follow applicable regulations.",
    ]

    vectors = model.encode(
        sample_texts
    )

    print(
        "Embedding model:",
        model.model_name
    )

    print(
        "Number of texts:",
        len(sample_texts)
    )

    print(
        "Embedding shape:",
        vectors.shape
    )

    print(
        "Vector dimension:",
        vectors.shape[1]
    )
