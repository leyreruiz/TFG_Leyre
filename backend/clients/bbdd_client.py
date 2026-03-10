import chromadb
import os

# CONFIGURATION
DB_PATH = "./chroma_db"  # Folder where data will be stored on disk
COLLECTION_NAME = "tabla_nueva"

import uuid

print("[bbdd_client] Import complete")

def obtain_client():
    """Establish a connection to the persistent ChromaDB database."""
    try:
        # Creates a client that persists data to local disk
        client = chromadb.PersistentClient(path=DB_PATH)
        return client
    except Exception as e:
        print(f" Error connecting to ChromaDB: {e}")
        return None

def prepare_collection(client):
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


def save_text_chroma(texto, id=None, metadata=None):
    """Generate embedding locally and save the text in the Chroma collection."""
    client = obtain_client()
    if not client:
        return None

    coleccion = prepare_collection(client)
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


def search_similars(texto, n=3):
    """Generate an embedding for the text and query the Chroma collection by similarity."""
    client = obtain_client()
    if not client:
        return None

    coleccion = prepare_collection(client)
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


def search_by_source(source: str) -> list[str]:
    """Retrieve all chunks from ChromaDB that belong to a specific source file.

    Args:
        source: filename as stored in metadata (e.g. 'redes_neuronales.txt')

    Returns:
        List of chunk strings ordered by chunk index, or [] on error.
    """
    client = obtain_client()
    if not client:
        return []

    coleccion = prepare_collection(client)
    try:
        results = coleccion.get(
            where={"source": source},
            include=["metadatas", "documents"],
        )
        if not results or not results.get("documents"):
            return []
        # Sort chunks by their index so context is ordered
        pares = list(zip(results["metadatas"], results["documents"]))
        pares.sort(key=lambda p: p[0].get("chunk", 0))
        return [doc for _, doc in pares]
    except Exception as e:
        print(f"Error retrieving documents by source: {e}")
        return []

if __name__ == "__main__":
    # Initialization test
    cliente = obtain_client()
    if cliente:
        mi_coleccion = prepare_collection(cliente)
        # Sample insert and quick query
        ejemplo = "This is a test text to store in ChromaDB."
        doc_id = save_text_chroma(ejemplo)
        if doc_id:
            print("Running test search...")
            res = search_similars("test text")
            print(res)