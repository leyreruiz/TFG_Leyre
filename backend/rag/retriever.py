"""RAG Retriever: envuelve la búsqueda en ChromaDB.

Ofrece una interfaz simple `.search(query, k)` que devuelve
una lista de strings (documentos relevantes).
"""

from backend.clients.bbdd_client import (
    obtener_cliente,
    preparar_coleccion,
    buscar_similares,
)


class ChromaDbRetriever:
    """Retriever que busca documentos similares en ChromaDB."""

    def search(self, query: str, k: int = 5) -> list[str]:
        """Busca los k documentos más relevantes para la query.

        Returns:
            Lista de strings con el contenido de los documentos.
            Lista vacía si no hay resultados o hay error.
        """
        resultados = buscar_similares(query, n=k)

        if not resultados or not resultados.get("documents"):
            return []

        # resultados["documents"] es [[doc1, doc2, ...]]
        return resultados["documents"][0]


class DummyRetriever:
    """Retriever de prueba que devuelve documentos falsos."""

    def search(self, query: str, k: int = 5) -> list[str]:
        return [
            "Documento de prueba 1: Introducción al tema.",
            "Documento de prueba 2: Desarrollo del contenido.",
            "Documento de prueba 3: Conclusiones finales.",
        ]
