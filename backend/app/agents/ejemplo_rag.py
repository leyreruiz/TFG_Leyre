"""Ejemplo de uso del RAG: ingestar documento y consultar con contexto."""

print("[DEBUG ejemplo_rag.py] Script iniciado")

import os
print("[DEBUG ejemplo_rag.py] os importado")
print("IMPORTS OK")  # ← añade esta línea temporal
# Importar funciones del RAG
print("[DEBUG ejemplo_rag.py] Intentando importar rag...")
try:
    from backend.app.agents.rag import ingestar_archivo_txt, consultar_con_contexto
    print("[DEBUG ejemplo_rag.py] ✓ Imports from backend.app.agents.rag successful")
except Exception as e:
    print(f"[DEBUG ejemplo_rag.py] Intento 1 falló: {e}")
    from rag import ingestar_archivo_txt, consultar_con_contexto
    print("[DEBUG ejemplo_rag.py] ✓ Imports from rag successful")

print("[DEBUG ejemplo_rag.py] Todos los imports completados")


def ejemplo_basico():
    """Ejemplo básico: ingestar y consultar."""
    
    print("=" * 60)
    print("EJEMPLO RAG: ChromaDB + Ollama")
    print("=" * 60)
    
    # Ruta del archivo de ejemplo (ubicado junto a este script)
    print("\n[DEBUG] Buscando archivo de ejemplo...")
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    archivo_ejemplo = os.path.join(directorio_actual, "ejemplo_ia.txt")
    
    print(f"\nArchivo: {archivo_ejemplo}")
    print(f"Existe: {os.path.exists(archivo_ejemplo)}\n")
    
    if not os.path.exists(archivo_ejemplo):
        print("No se encontró ejemplo_ia.txt. Descargalo o crea uno.")
        return
    
    # PASO 1: Ingestar el documento
    print("=" * 60)
    print("PASO 1: Ingesta de documento")
    print("=" * 60)
    print("[DEBUG] Iniciando ingesta...")
    
    print("[DEBUG] Llamando a ingestar_archivo_txt()...")
    ids = ingestar_archivo_txt(
        archivo_ejemplo,
        metadata={
            "tipo": "artículo",
            "tema": "inteligencia artificial",
            "fecha": "2026-02-19"
        }
    )
    print("[DEBUG] Ingesta completada.")
    
    print(f"\n✓ {len(ids)} chunks almacenados en ChromaDB\n")
    
    # PASO 2: Consultar con contexto
    print("=" * 60)
    print("PASO 2: Consultar con contexto RAG")
    print("=" * 60)
    
    preguntas = [
        "¿Qué es Machine Learning?",
        "¿Cuáles son los tipos principales de aprendizaje?",
        "¿Para qué se usa ChromaDB?",
        "¿Cuáles son los desafíos principales de la IA?"
    ]
    
    for pregunta in preguntas:
        print(f"\n{'─' * 60}")
        print(f" Pregunta: {pregunta}\n")
        print("[DEBUG] Llamando a consultar_con_contexto()...")
        
        respuesta = consultar_con_contexto(pregunta, n_resultados=3, temperatura=0.5)
        print("[DEBUG] Respuesta recibida.")
        
        if respuesta:
            print(f" Respuesta:\n{respuesta}")
        else:
            print("No se pudo obtener respuesta")
    
    print(f"\n{'=' * 60}\n✓ Ejemplo completado\n")


if __name__ == "__main__":
    print("INICIANDO EJEMPLO RAG...")
    ejemplo_basico()
