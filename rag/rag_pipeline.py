from typing import Dict, List, Optional

from .retrieval import CivicRetriever
from .source_mapper import SourceMapper
from .vector_store import CivicVectorStore


class RAGPipeline:
    """
    Main civic RAG pipeline.

    Responsibilities:
    1. Load the FAISS vector store.
    2. Retrieve relevant civic evidence.
    3. Map retrieved results to verified sources.
    4. Build grounding context for the AI agent.
    """

    def __init__(
        self,
        vector_store: CivicVectorStore,
        source_mapper: Optional[SourceMapper] = None,
    ):
        self.vector_store = vector_store

        self.retriever = CivicRetriever(
            vector_store
        )

        self.source_mapper = (
            source_mapper
            if source_mapper is not None
            else SourceMapper()
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Dict]:
        """
        Retrieve and source-map relevant civic evidence.
        """

        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

        mapped_results = (
            self.source_mapper.map_results(
                results
            )
        )

        return mapped_results

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> str:
        """
        Retrieve evidence and format it as grounding
        context for an AI model.
        """

        results = self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

        return self._format_grounding_context(
            results
        )

    def _format_grounding_context(
        self,
        results: List[Dict],
    ) -> str:
        """
        Convert retrieved evidence into a structured
        context block for the AI agent.
        """

        if not results:
            return (
                "NO OFFICIAL CIVIC EVIDENCE WAS FOUND.\n"
                "Do not invent rules, procedures, deadlines, "
                "fees, or legal requirements."
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

            category = metadata.get(
                "category",
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

            verified = metadata.get(
                "last_verified",
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
Source Name: {source_name}
Authority: {authority}
Category: {category}
Page: {page}
Chunk: {chunk}
Similarity Score: {score:.4f}
Last Verified: {verified}
Official URL: {source_url}

Evidence Text:
{text}
""".strip()
            )

        return "\n\n".join(
            context_parts
        )

    def get_sources(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Dict]:
        """
        Return unique verified source records associated
        with the retrieved evidence.
        """

        results = self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

        sources = []

        seen = set()

        for result in results:

            metadata = result.get(
                "metadata",
                {}
            )

            source_id = metadata.get(
                "source_id"
            )

            if not source_id:
                continue

            if source_id in seen:
                continue

            source = self.source_mapper.get_source(
                source_id
            )

            if source:

                sources.append(
                    source
                )

                seen.add(
                    source_id
                )

        return sources

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> Dict:
        """
        Main integration method for AwaamiAgent.

        Returns:
            {
                "question": ...,
                "evidence": [...],
                "context": "...",
                "sources": [...]
            }
        """

        evidence = self.retrieve(
            query=question,
            top_k=top_k,
            min_score=min_score,
        )

        context = self._format_grounding_context(
            evidence
        )

        sources = self.get_sources(
            query=question,
            top_k=top_k,
            min_score=min_score,
        )

        return {
            "question": question,
            "evidence": evidence,
            "context": context,
            "sources": sources,
        }


def create_rag_pipeline() -> RAGPipeline:
    """
    Create a ready-to-use RAG pipeline from the saved
    FAISS vector store.
    """

    vector_store = CivicVectorStore()

    vector_store.load()

    return RAGPipeline(
        vector_store=vector_store
    )


def retrieve_civic_context(
    question: str,
    top_k: int = 5,
    min_score: Optional[float] = None,
) -> str:
    """
    Simple integration function.

    AwaamiAgent can call this function when it needs
    official civic evidence.
    """

    pipeline = create_rag_pipeline()

    return pipeline.build_context(
        query=question,
        top_k=top_k,
        min_score=min_score,
    )


def query_civic_rag(
    question: str,
    top_k: int = 5,
    min_score: Optional[float] = None,
) -> Dict:
    """
    Full RAG query interface for AwaamiAgent.
    """

    pipeline = create_rag_pipeline()

    return pipeline.query(
        question=question,
        top_k=top_k,
        min_score=min_score,
    )


if __name__ == "__main__":

    print(
        "Civic RAG pipeline module loaded successfully."
    )
