"""Script para ingestar todos los archivos de prueba en ChromaDB.

Uso: python -m backend.ingest_topics
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.agents.rag import ingestar_archivo_txt


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
