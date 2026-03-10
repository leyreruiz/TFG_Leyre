"""RAG Retriever: wraps the ChromaDB search.

Provides a simple `.search(query, k)` interface that returns
a list of strings (relevant documents).
"""

from backend.clients.bbdd_client import (
    obtener_cliente,
    preparar_coleccion,
    buscar_similares,
)


class ChromaDbRetriever:
    """Retriever that searches for similar documents in ChromaDB."""

    def search(self, query: str, k: int = 5) -> list[str]:
        """Search the k most relevant documents for the query.

        Returns:
            List of strings with the document contents.
            Empty list if no results or on error.
        """
        resultados = buscar_similares(query, n=k)

        if not resultados or not resultados.get("documents"):
            return []

        # resultados["documents"] is [[doc1, doc2, ...]]
        return resultados["documents"][0]


class DummyRetriever:
    """Test retriever that returns fake documents."""

    def search(self, query: str, k: int = 5) -> list[str]:
        return [
            "Test document 1: Introduction to the topic.",
            "Test document 2: Development of the content.",
            "Test document 3: Final conclusions.",
        ]
