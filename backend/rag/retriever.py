"""RAG Retriever: wraps the ChromaDB search.

Provides a simple `.search(query, k)` interface that returns
a list of strings (relevant documents).
"""

from backend.clients.bbdd_client import (
    search_similars,
    search_by_source,
)


class ChromaDbRetriever:
    """Retriever that searches for similar documents in ChromaDB."""

    def __init__(self, active_topic: str | None = None):
        self.active_topic = active_topic

    def set_topic(self, topic: str | None):
        """Set the default topic collection for subsequent searches."""
        self.active_topic = topic

    def search(self, query: str, k: int = 5, topic: str | None = None) -> list[str]:
        """Search the k most relevant documents for the query.

        Returns:
            List of strings with the document contents.
            Empty list if no results or on error.
        """
        target_topic = topic if topic is not None else self.active_topic
        results = search_similars(query, n=k, topic=target_topic)

        if not results or not results.get("documents"):
            return []

        # results["documents"] is [[doc1, doc2, ...]]
        return results["documents"][0]

    def search_by_source(self, fuente: str, topic: str | None = None) -> list[str]:
        """Retrieve all chunks for a specific source file, ordered by chunk index.

        Returns:
            List of strings with the full document contents.
            Empty list if no results or on error.
        """
        target_topic = topic if topic is not None else self.active_topic
        return search_by_source(fuente, topic=target_topic)


class DummyRetriever:
    """Test retriever that returns fake documents."""

    def search(self, query: str, k: int = 5) -> list[str]:
        return [
            "Test document 1: Introduction to the topic.",
            "Test document 2: Development of the content.",
            "Test document 3: Final conclusions.",
        ]
