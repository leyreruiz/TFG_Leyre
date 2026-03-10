"""Script to ingest all test topic files into ChromaDB.

Usage: python -m backend.ingest_topics
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.clients.bbdd_client import save_text_chroma


def _divide_in_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split a text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def ingest_file_txt(file_path: str, metadata: dict | None = None) -> list[str]:
    """Read a .txt file and ingest it into ChromaDB in chunks.

    Args:
        file_path: path to the .txt file
        metadata: optional dict with additional metadata

    Returns:
        List of saved IDs, or [] on error.
    """
    if not os.path.exists(file_path):
        print(f"Error: file not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    print(f"Reading {file_path} ({len(text)} characters)...")

    chunks = _divide_in_chunks(text)
    print(f"Split into {len(chunks)} chunks")

    ids_guardados = []
    nombre_archivo = os.path.basename(file_path)

    for i, chunk in enumerate(chunks):
        meta = dict(metadata or {})
        meta.update({"chunk": i, "source": nombre_archivo})
        doc_id = save_text_chroma(chunk, metadata=meta)
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
        ids = ingest_file_txt(ruta, metadata=metadata)
        print(f"  → {len(ids)} chunks saved.\n")

    print("\n=== Ingest complete ===")


if __name__ == "__main__":
    main()
