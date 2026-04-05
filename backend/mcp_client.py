"""MCP Client: retriever que se comunica con el MCP server via protocolo stdio.

Expone la misma interfaz que ChromaDbRetriever (search / search_by_source / set_topic)
para que los agentes existentes funcionen sin ningún cambio.

El servidor MCP se lanza automáticamente como subproceso al crear MCPRetriever().
"""

import asyncio
import logging
import os
import sys
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# Separador que usa el MCP server para unir fragmentos
_CHUNK_SEP = "\n\n---\n\n"


class MCPRetriever:
    """Retriever que delega las búsquedas al MCP server vía protocolo stdio.

    Crea un event loop en un hilo daemon y mantiene una sesión MCP persistente
    con el servidor, de modo que cada llamada a search() / search_by_source()
    es síncrona para el resto del código pero usa el protocolo MCP por debajo.
    """

    def __init__(self):
        self.active_topic: str | None = None

        # Event loop dedicado en hilo daemon
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        # Conectar de forma síncrona antes de devolver el objeto
        future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        future.result(timeout=30)
        logger.info("MCPRetriever: connected to MCP server")

    # ------------------------------------------------------------------
    # Conexión
    # ------------------------------------------------------------------

    async def _connect(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "backend.mcp_server"],
            cwd=root,
        )
        self._stdio_cm = stdio_client(server_params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

    # ------------------------------------------------------------------
    # Interfaz pública (síncrona, igual que ChromaDbRetriever)
    # ------------------------------------------------------------------

    def set_topic(self, topic: str | None):
        """Establece el tema por defecto para las búsquedas siguientes."""
        self.active_topic = topic

    def search(self, query: str, k: int = 5, topic: str | None = None) -> list[str]:
        """Búsqueda semántica: devuelve los k fragmentos más relevantes."""
        target = topic if topic is not None else (self.active_topic or "")
        future = asyncio.run_coroutine_threadsafe(
            self._async_search(query, target, k), self._loop
        )
        return future.result(timeout=30)

    def search_by_source(self, source: str, topic: str | None = None) -> list[str]:
        """Recupera todos los fragmentos de un fichero fuente, en orden."""
        target = topic if topic is not None else (self.active_topic or "")
        future = asyncio.run_coroutine_threadsafe(
            self._async_search_by_source(source, target), self._loop
        )
        return future.result(timeout=30)

    # ------------------------------------------------------------------
    # Corrutinas internas
    # ------------------------------------------------------------------

    async def _async_search(self, query: str, topic: str, k: int) -> list[str]:
        result = await self._session.call_tool(
            "search_knowledge_base",
            {"query": query, "topic": topic, "k": k},
        )
        return self._parse_chunks(result)

    async def _async_search_by_source(self, source: str, topic: str) -> list[str]:
        result = await self._session.call_tool(
            "search_by_source",
            {"source": source, "topic": topic},
        )
        return self._parse_chunks(result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def search_wikipedia(self, query: str, sentences: int = 4) -> str:
        """Busca en Wikipedia via MCP y devuelve un fragmento del artículo."""
        future = asyncio.run_coroutine_threadsafe(
            self._async_wikipedia(query, sentences), self._loop
        )
        return future.result(timeout=30)

    async def _async_wikipedia(self, query: str, sentences: int) -> str:
        result = await self._session.call_tool(
            "search_wikipedia",
            {"query": query, "sentences": sentences},
        )
        try:
            return result.content[0].text
        except (IndexError, AttributeError):
            return ""

    @staticmethod
    def _parse_chunks(result) -> list[str]:
        """Extrae la lista de fragmentos de la respuesta del tool MCP."""
        try:
            text = result.content[0].text
        except (IndexError, AttributeError):
            return []
        # El servidor devuelve un mensaje de error cuando no hay resultados
        if "No se encontraron" in text:
            return []
        return [chunk.strip() for chunk in text.split(_CHUNK_SEP) if chunk.strip()]
