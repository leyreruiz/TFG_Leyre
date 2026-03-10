"""Script to ingest all test topic files into ChromaDB.

Usage: python -m backend.ingest_topics
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.clients.bbdd_client import guardar_texto_chroma


def _dividir_en_chunks(texto: str, tamaño_chunk: int = 500, solapamiento: int = 50) -> list[str]:
    """Split a text into overlapping chunks."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamaño_chunk, len(texto))
        chunk = texto[inicio:fin].strip()
        if chunk:
            chunks.append(chunk)
        if fin == len(texto):
            break
        inicio = fin - solapamiento
    return chunks


def ingestar_archivo_txt(ruta_archivo: str, metadata: dict | None = None) -> list[str]:
    """Read a .txt file and ingest it into ChromaDB in chunks.

    Args:
        ruta_archivo: path to the .txt file
        metadata: optional dict with additional metadata

    Returns:
        List of saved IDs, or [] on error.
    """
    if not os.path.exists(ruta_archivo):
        print(f"Error: file not found: {ruta_archivo}")
        return []

    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            texto = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    print(f"Reading {ruta_archivo} ({len(texto)} characters)...")

    chunks = _dividir_en_chunks(texto)
    print(f"Split into {len(chunks)} chunks")

    ids_guardados = []
    nombre_archivo = os.path.basename(ruta_archivo)

    for i, chunk in enumerate(chunks):
        meta = dict(metadata or {})
        meta.update({"chunk": i, "fuente": nombre_archivo})
        doc_id = guardar_texto_chroma(chunk, metadata=meta)
        if doc_id:
            ids_guardados.append(doc_id)
            print(f"  Chunk {i + 1}/{len(chunks)} saved")

    print(f"Ingest complete: {len(ids_guardados)} chunks saved")
    return ids_guardados


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

TOPICS = [
    ("redes_neuronales.txt", {"tema": "redes_neuronales", "asignatura": "IA"}),
    ("bases_datos.txt", {"tema": "bases_datos", "asignatura": "BBDD"}),
    ("sistemas_operativos.txt", {"tema": "sistemas_operativos", "asignatura": "SO"}),
]


def main():
    print("=== Topic ingest ===\n")

    for filename, metadata in TOPICS:
        ruta = os.path.join(DATA_DIR, filename)
        if not os.path.exists(ruta):
            print(f"⚠ Not found: {ruta}")
            continue

        print(f"\n--- Ingesting: {filename} ---")
        ids = ingestar_archivo_txt(ruta, metadata=metadata)
        print(f"  → {len(ids)} chunks saved.\n")

    print("\n=== Ingest complete ===")


if __name__ == "__main__":
    main()
