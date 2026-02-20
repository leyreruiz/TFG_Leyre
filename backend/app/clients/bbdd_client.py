import chromadb
import os

# CONFIGURACIÓN
DB_PATH = "./chroma_db"  # Carpeta donde se guardarán los datos físicamente
COLLECTION_NAME = "tabla_nueva"

import uuid

print("[bbdd_client] Importación completada")

def obtener_cliente():
    """Establece la conexión con la base de datos persistente."""
    try:
        # Crea el cliente que guarda los datos en el disco local
        client = chromadb.PersistentClient(path=DB_PATH)
        return client
    except Exception as e:
        print(f" Error conectando a ChromaDB: {e}")
        return None

def preparar_coleccion(client):
    """
    Crea la colección (equivalente a la tabla) si no existe.
    Configura el modelo de embeddings automáticamente.
    """
    print("[DEBUG] Preparando colección de ChromaDB...")
    print("[DEBUG] Usando embeddings por defecto de ChromaDB (sin descargar modelos)...")
    
    # Usa los embeddings por defecto de ChromaDB sin descargar nada
    # Esto evita bloqueos al descargar modelos grandes
    coleccion = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Sin especificar embedding_function, usa embeddings por defecto
    )
    print(f"✓ Colección '{COLLECTION_NAME}' verificada o creada en {DB_PATH}.")
    return coleccion


def guardar_texto_chroma(texto, id=None, metadata=None):
    """Genera embedding localmente y guarda el texto en la colección de Chroma."""
    client = obtener_cliente()
    if not client:
        return None

    coleccion = preparar_coleccion(client)
    if id is None:
        id = str(uuid.uuid4())

    try:
        add_kwargs = {
            "ids": [id],
            "documents": [texto],
        }
        # Solo incluir metadatos si se ha pasado un dict no vacío
        if isinstance(metadata, dict) and len(metadata) > 0:
            add_kwargs["metadatas"] = [metadata]

        coleccion.add(**add_kwargs)
        print(f"Documento guardado en Chroma con id={id}")
        return id
    except Exception as e:
        print(f"Error guardando en Chroma: {e}")
        return None


def buscar_similares(texto, n=3):
    """Genera embedding del texto y consulta la colección en Chroma por similitud."""
    client = obtener_cliente()
    if not client:
        return None

    coleccion = preparar_coleccion(client)
    try:
        resultados = coleccion.query(
            query_texts=[texto],
            n_results=n,
            include=["metadatas", "documents", "distances"]
        )
        return resultados
    except Exception as e:
        print(f"Error consultando Chroma: {e}")
        return None

if __name__ == "__main__":
    # Prueba de inicialización
    cliente = obtener_cliente()
    if cliente:
        mi_coleccion = preparar_coleccion(cliente)
        # Inserción de ejemplo y consulta rápida
        ejemplo = "Este es un texto de prueba para almacenar en ChromaDB."
        doc_id = guardar_texto_chroma(ejemplo)
        if doc_id:
            print("Realizando búsqueda de prueba...")
            res = buscar_similares("texto de prueba")
            print(res)