"""Ingesta: delega en ChromaDB para generar y almacenar embeddings.

La función `guardar_en_db` usa `guardar_texto_chroma` de `bbdd_client`.
"""

# Import robusto de la función que guarda en Chroma
try:
    from backend.app.clients.bbdd_client import guardar_texto_chroma
except Exception:
    try:
        from ..clients.bbdd_client import guardar_texto_chroma
    except Exception:
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
        from backend.app.clients.bbdd_client import guardar_texto_chroma


def guardar_en_db(texto, metadata=None):
    """Guarda `texto` en la colección de Chroma. Devuelve el id o None."""
    print(f"Guardando en Chroma: '{texto[:40]}...'")
    doc_id = guardar_texto_chroma(texto, metadata=metadata)
    if doc_id:
        print(f"Guardado OK — id={doc_id}")
    else:
        print("Error guardando en Chroma.")
    return doc_id


if __name__ == "__main__":
    ejemplo_texto = "El TFG sobre inteligencia artificial está progresando muy bien."
    guardar_en_db(ejemplo_texto, metadata={"origen": "prueba", "autor": "Leyre"})