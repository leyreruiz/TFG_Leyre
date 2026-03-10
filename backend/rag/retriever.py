"""RAG Retriever: wraps the ChromaDB search.

Provides a simple `.search(query, k)` interface that returns
a list of strings (relevant documents).
"""

from backend.clients.bbdd_client import (
    obtain_client,
    prepare_collection,
    search_similars,
    search_by_source,
)


class ChromaDbRetriever:
    """Retriever that searches for similar documents in ChromaDB."""

    def search(self, query: str, k: int = 5) -> list[str]:
        """Search the k most relevant documents for the query.

        Returns:
            List of strings with the document contents.
            Empty list if no results or on error.
        """
        results = search_similars(query, n=k)

        if not results or not results.get("documents"):
            return []

        # results["documents"] is [[doc1, doc2, ...]]
        return results["documents"][0]

    def search_by_source(self, fuente: str) -> list[str]:
        """Retrieve all chunks for a specific source file, ordered by chunk index.

        Returns:
            List of strings with the full document contents.
            Empty list if no results or on error.
        """
        return search_by_source(fuente)


class DummyRetriever:
    """Test retriever that returns fake documents."""

    def search(self, query: str, k: int = 5) -> list[str]:
        return [
            "Test document 1: Introduction to the topic.",
            "Test document 2: Development of the content.",
            "Test document 3: Final conclusions.",
        ]
