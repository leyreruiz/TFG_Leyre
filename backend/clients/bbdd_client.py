import chromadb
import os

# CONFIGURATION
DB_PATH = "./chroma_db"  # Folder where data will be stored on disk
COLLECTION_NAME = "tabla_nueva"

import uuid

print("[bbdd_client] Import complete")

def obtener_cliente():
    """Establish a connection to the persistent ChromaDB database."""
    try:
        # Creates a client that persists data to local disk
        client = chromadb.PersistentClient(path=DB_PATH)
        return client
    except Exception as e:
        print(f" Error connecting to ChromaDB: {e}")
        return None

def preparar_coleccion(client):
    """
    Create the collection (equivalent to a table) if it does not exist.
    Configures the embeddings model automatically.
    """
    print("[DEBUG] Preparing ChromaDB collection...")
    print("[DEBUG] Using ChromaDB default embeddings (no model download required)...")
    
    # Use ChromaDB's default embeddings without downloading anything
    # This avoids blocking on large model downloads
    coleccion = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Without specifying embedding_function, uses default embeddings
    )
    print(f"✓ Collection '{COLLECTION_NAME}' verified or created at {DB_PATH}.")
    return coleccion


def guardar_texto_chroma(texto, id=None, metadata=None):
    """Generate embedding locally and save the text in the Chroma collection."""
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
        # Only include metadata if a non-empty dict was provided
        if isinstance(metadata, dict) and len(metadata) > 0:
            add_kwargs["metadatas"] = [metadata]

        coleccion.add(**add_kwargs)
        print(f"Document saved in Chroma with id={id}")
        return id
    except Exception as e:
        print(f"Error saving to Chroma: {e}")
        return None


def buscar_similares(texto, n=3):
    """Generate an embedding for the text and query the Chroma collection by similarity."""
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
        print(f"Error querying Chroma: {e}")
        return None

if __name__ == "__main__":
    # Initialization test
    cliente = obtener_cliente()
    if cliente:
        mi_coleccion = preparar_coleccion(cliente)
        # Sample insert and quick query
        ejemplo = "This is a test text to store in ChromaDB."
        doc_id = guardar_texto_chroma(ejemplo)
        if doc_id:
            print("Running test search...")
            res = buscar_similares("test text")
            print(res)