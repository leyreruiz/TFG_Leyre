"""Script para ingestar todos los archivos de prueba en ChromaDB.

Uso: python -m backend.ingest_topics
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.clients.bbdd_client import guardar_texto_chroma


def _dividir_en_chunks(texto: str, tamaño_chunk: int = 500, solapamiento: int = 50) -> list[str]:
    """Divide un texto en chunks con solapamiento."""
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
    """Lee un archivo .txt y lo ingesta en ChromaDB por chunks.

    Args:
        ruta_archivo: ruta al archivo .txt
        metadata: dict opcional con metadatos adicionales

    Returns:
        Lista de IDs guardados, o [] en caso de error.
    """
    if not os.path.exists(ruta_archivo):
        print(f"Error: archivo no encontrado: {ruta_archivo}")
        return []

    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            texto = f.read()
    except Exception as e:
        print(f"Error leyendo archivo: {e}")
        return []

    print(f"Leyendo {ruta_archivo} ({len(texto)} caracteres)...")

    chunks = _dividir_en_chunks(texto)
    print(f"Dividido en {len(chunks)} chunks")

    ids_guardados = []
    nombre_archivo = os.path.basename(ruta_archivo)

    for i, chunk in enumerate(chunks):
        meta = dict(metadata or {})
        meta.update({"chunk": i, "fuente": nombre_archivo})
        doc_id = guardar_texto_chroma(chunk, metadata=meta)
        if doc_id:
            ids_guardados.append(doc_id)
            print(f"  Chunk {i + 1}/{len(chunks)} guardado")

    print(f"Ingesta completada: {len(ids_guardados)} chunks guardados")
    return ids_guardados


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

TOPICS = [
    ("redes_neuronales.txt", {"tema": "redes_neuronales", "asignatura": "IA"}),
    ("bases_datos.txt", {"tema": "bases_datos", "asignatura": "BBDD"}),
    ("sistemas_operativos.txt", {"tema": "sistemas_operativos", "asignatura": "SO"}),
]


def main():
    print("=== Ingesta de temas de prueba ===\n")

    for filename, metadata in TOPICS:
        ruta = os.path.join(DATA_DIR, filename)
        if not os.path.exists(ruta):
            print(f"⚠ No encontrado: {ruta}")
            continue

        print(f"\n--- Ingiriendo: {filename} ---")
        ids = ingestar_archivo_txt(ruta, metadata=metadata)
        print(f"  → {len(ids)} chunks guardados.\n")

    print("\n=== Ingesta finalizada ===")


if __name__ == "__main__":
    main()
