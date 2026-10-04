from typing import Dict, List, Optional

from .vector_store import CivicVectorStore


class CivicRetriever:
    """
    Retrieval layer for AwaamiAgent.

    This class searches the FAISS vector store and returns
    relevant civic evidence together with source metadata.
    """

    def __init__(
        self,
        vector_store: CivicVectorStore,
    ):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant civic evidence.

        Args:
            query:
                User's civic question.

            top_k:
                Maximum number of results.

            min_score:
                Optional similarity threshold.

        Returns:
            List of relevant chunks with metadata and scores.
        """

        if not query or not query.strip():
            return []

        results = self.vector_store.search(
            query=query,
            top_k=top_k,
        )

        if min_score is not None:
            results = [
                result
                for result in results
                if result.get("score", 0.0)
                >= min_score
            ]

        return results

    def retrieve_for_category(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Dict]:
        """
        Retrieve evidence and optionally filter by civic category.
        """

        results = self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

        if not category:
            return results

        category = category.lower().strip()

        filtered_results = []

        for result in results:

            result_category = (
                result
                .get("metadata", {})
                .get("category", "")
                .lower()
                .strip()
            )

            if result_category == category:
                filtered_results.append(result)

        return filtered_results

    def format_context(
        self,
        results: List[Dict],
    ) -> str:
        """
        Convert retrieved evidence into a clean context block
        that can later be provided to the AI model.
        """

        if not results:
            return (
                "No relevant official civic evidence "
                "was retrieved."
            )

        context_parts = []

        for number, result in enumerate(
            results,
            start=1
        ):

            metadata = result.get(
                "metadata",
                {}
            )

            source_id = metadata.get(
                "source_id",
                "Unknown"
            )

            source_name = metadata.get(
                "source_name",
                "Unknown"
            )

            authority = metadata.get(
                "authority",
                "Unknown"
            )

            source_url = metadata.get(
                "source_url",
                ""
            )

            page = metadata.get(
                "page",
                ""
            )

            chunk = metadata.get(
                "chunk",
                ""
            )

            score = result.get(
                "score",
                0.0
            )

            text = result.get(
                "text",
                ""
            )

            context_parts.append(
                f"""
[EVIDENCE {number}]
Source ID: {source_id}
Source: {source_name}
Authority: {authority}
Page: {page}
Chunk: {chunk}
Similarity Score: {score:.4f}
Official URL: {source_url}

Evidence:
{text}
""".strip()
            )

        return "\n\n".join(
            context_parts
        )


def retrieve_civic_evidence(
    query: str,
    vector_store: CivicVectorStore,
    top_k: int = 5,
    min_score: Optional[float] = None,
) -> List[Dict]:
    """
    Simple function for AwaamiAgent integration.
    """

    retriever = CivicRetriever(
        vector_store
    )

    return retriever.retrieve(
        query=query,
        top_k=top_k,
        min_score=min_score,
    )


def build_grounding_context(
    query: str,
    vector_store: CivicVectorStore,
    top_k: int = 5,
    min_score: Optional[float] = None,
) -> str:
    """
    Retrieve evidence and return it as an AI-ready
    grounding context.
    """

    retriever = CivicRetriever(
        vector_store
    )

    results = retriever.retrieve(
        query=query,
        top_k=top_k,
        min_score=min_score,
    )

    return retriever.format_context(
        results
    )


if __name__ == "__main__":

    print(
        "Civic retrieval module loaded successfully."
    )
