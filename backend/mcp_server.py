"""MCP Server: exposes the University Assistant knowledge base as MCP tools.

Exposes the system's retrieval capabilities as standard MCP tools:
  - Semantic search in the knowledge base (ChromaDB)
  - Source-bound retrieval of a document's chunks
  - Wikipedia lookup as an external fallback

Usage:
    python -m backend.mcp_server
"""

import logging

import wikipediaapi
from mcp.server.fastmcp import FastMCP

from backend.rag.retriever import ChromaDbRetriever

_wiki_es = wikipediaapi.Wikipedia(user_agent="UniversityAssistant/1.0", language="es")
_wiki_en = wikipediaapi.Wikipedia(user_agent="UniversityAssistant/1.0", language="en")

mcp = FastMCP(
    "University Assistant",
    instructions=(
        "Tienes acceso a una base de conocimiento educativa construida a partir de documentos "
        "universitarios. Usa search_knowledge_base para buscar contenido relevante por similitud, "
        "search_by_source para recuperar todos los fragmentos de un fichero y search_wikipedia "
        "como apoyo externo cuando un concepto no esté suficientemente explicado."
    ),
)

_retriever = ChromaDbRetriever()


# ---------------------------------------------------------------------------
# Knowledge-base retrieval tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_knowledge_base(query: str, topic: str, k: int = 5) -> str:
    """Busca contenido relevante en la base de conocimiento para una consulta dada.

    Args:
        query: La consulta de búsqueda (concepto, pregunta o palabra clave).
        topic: El tema en el que buscar.
        k:     Número de fragmentos de documento a recuperar (por defecto 5, máximo 15).
    """
    k = min(max(1, k), 15)
    _retriever.set_topic(topic)
    docs = _retriever.search(query, k=k)
    if not docs:
        return f"No se encontraron documentos relevantes para '{query}' en el tema '{topic}'."
    return "\n\n---\n\n".join(docs)


@mcp.tool()
def search_by_source(source: str, topic: str) -> str:
    """Recupera todos los fragmentos de un fichero fuente específico, en orden.

    Args:
        source: Nombre del fichero tal como está almacenado en los metadatos (p.ej. 'redes.txt').
        topic:  El tema al que pertenece el fichero.
    """
    _retriever.set_topic(topic)
    docs = _retriever.search_by_source(source, topic=topic)
    if not docs:
        return f"No se encontraron fragmentos para el fichero '{source}' en el tema '{topic}'."
    return "\n\n---\n\n".join(docs)


# ---------------------------------------------------------------------------
# External knowledge tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_wikipedia(query: str, sentences: int = 4) -> str:
    """Busca en Wikipedia para complementar el contenido del documento cuando un concepto
    no está suficientemente explicado en la base de conocimiento.

    Busca primero en español; si no existe la página, lo intenta en inglés.

    Args:
        query:     Concepto o término a buscar (p.ej. 'red neuronal', 'backpropagation').
        sentences: Número de frases del resumen a devolver (por defecto 4).
    """
    for wiki, lang in ((_wiki_es, "es"), (_wiki_en, "en")):
        page = wiki.page(query)
        if page.exists():
            # Take only the first N sentences from the summary
            raw_sentences = [s.strip() for s in page.summary.split(". ") if s.strip()]
            summary = ". ".join(raw_sentences[:sentences])
            if summary and not summary.endswith("."):
                summary += "."
            return f"**{page.title}** (Wikipedia/{lang})\n\n{summary}\n\nFuente: {page.fullurl}"

    return f"No se encontró artículo en Wikipedia para '{query}'."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
