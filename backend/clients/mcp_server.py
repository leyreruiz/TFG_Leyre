from fastmcp import FastMCP
from backend.clients.bbdd_client import buscar_similares

# Creación del servidor MCP llamado "Conocimiento_TFG"
mcp = FastMCP("Conocimiento_TFG")

@mcp.tool()
def consultar_base_conocimiento(pregunta: str) -> str:
    """
    Busca información técnica en los documentos locales cargados. 
    Se debe usar cuando el usuario pregunte por detalles específicos del TFG o PDFs.
    """
    print(f"[Servidor MCP] Procesando consulta: {pregunta}")
    
    # Se utiliza la función de búsqueda de ChromaDB ya existente
    resultados = buscar_similares(pregunta, n=3)
    
    if not resultados or not resultados.get("documents"):
        return "No se ha encontrado información relevante en los documentos locales."
    
    # Se extraen y formatean los fragmentos encontrados
    documentos = resultados["documents"][0]
    contexto = "\n---\n".join(documentos)
    
    return f"Información relevante encontrada:\n{contexto}"

if __name__ == "__main__":
    mcp.run()