"""MCP Server: exposes the University Assistant knowledge base as MCP tools.

Expone las capacidades del sistema como herramientas MCP estándar:
  - Búsqueda semántica en la base de conocimiento (ChromaDB)
  - Consulta de clases guardadas, resúmenes y preguntas de examen

Ejecución:
    python -m backend.mcp_server
"""

import json
import logging

import wikipediaapi
from mcp.server.fastmcp import FastMCP

from backend.class_storage import get_class, list_classes
from backend.rag.retriever import ChromaDbRetriever

_wiki_es = wikipediaapi.Wikipedia(user_agent="UniversityAssistant/1.0", language="es")
_wiki_en = wikipediaapi.Wikipedia(user_agent="UniversityAssistant/1.0", language="en")

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "University Assistant",
    instructions=(
        "Tienes acceso a una base de conocimiento educativa construida a partir de documentos "
        "universitarios. Usa search_knowledge_base para buscar contenido relevante y "
        "get_class_data para obtener la clase completa con resúmenes y preguntas de examen."
    ),
)

_retriever = ChromaDbRetriever()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_topics() -> list[str]:
    """Lista todos los temas disponibles en la base de conocimiento."""
    return list_classes()


@mcp.tool()
def search_knowledge_base(query: str, topic: str, k: int = 5) -> str:
    """Busca contenido relevante en la base de conocimiento para una consulta dada.

    Args:
        query: La consulta de búsqueda (concepto, pregunta o palabra clave).
        topic: El tema en el que buscar (usar list_topics para ver los disponibles).
        k:     Número de fragmentos de documento a recuperar (por defecto 5, máximo 15).
    """
    k = min(max(1, k), 15)
    _retriever.set_topic(topic)
    docs = _retriever.search(query, k=k)
    if not docs:
        return f"No se encontraron documentos relevantes para '{query}' en el tema '{topic}'."
    return "\n\n---\n\n".join(docs)


@mcp.tool()
def get_class_data(topic: str) -> str:
    """Obtiene la clase completa almacenada para un tema: estructura, resúmenes y preguntas.

    Args:
        topic: El nombre del tema (usar list_topics para ver los disponibles).
    """
    data = get_class(topic)
    if not data:
        return f"No se encontraron datos de clase para el tema '{topic}'."
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
def get_section_summary(topic: str, section: str) -> str:
    """Obtiene el resumen de una sección específica dentro de una clase.

    Args:
        topic:   El nombre del tema.
        section: El título exacto de la sección (tal como aparece en la estructura).
    """
    data = get_class(topic)
    if not data:
        return f"No se encontraron datos de clase para el tema '{topic}'."

    sections_data = data.get("sections_data", {})
    if section not in sections_data:
        available = ", ".join(sections_data.keys()) or "ninguna"
        return f"Sección '{section}' no encontrada. Secciones disponibles: {available}"

    summary = sections_data[section].get("summary", "")
    return summary if summary else f"Aún no se ha generado resumen para '{section}'."


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


@mcp.tool()
def get_section_questions(topic: str, section: str) -> str:
    """Obtiene las preguntas de examen de una sección específica.

    Args:
        topic:   El nombre del tema.
        section: El título exacto de la sección.
    """
    data = get_class(topic)
    if not data:
        return f"No se encontraron datos de clase para el tema '{topic}'."

    sections_data = data.get("sections_data", {})
    if section not in sections_data:
        available = ", ".join(sections_data.keys()) or "ninguna"
        return f"Sección '{section}' no encontrada. Secciones disponibles: {available}"

    questions = sections_data[section].get("questions", [])
    if not questions:
        return f"Aún no se han generado preguntas para la sección '{section}'."
    return json.dumps(questions, ensure_ascii=False, indent=2)


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
            raw = page.summary
            frases = [s.strip() for s in raw.split(". ") if s.strip()]
            resumen = ". ".join(frases[:sentences])
            if resumen and not resumen.endswith("."):
                resumen += "."
            return f"**{page.title}** (Wikipedia/{lang})\n\n{resumen}\n\nFuente: {page.fullurl}"

    return f"No se encontró artículo en Wikipedia para '{query}'."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
