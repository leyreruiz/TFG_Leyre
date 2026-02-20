"""RAG (Retrieval Augmented Generation) Module.

Integra ChromaDB (búsqueda de documentos) con Ollama (generación de texto).
Flujo:
  1. Ingestar archivos txt en ChromaDB
  2. Consultar: buscar documentos relevantes + pasar al LLM con contexto
"""

import os
import sys

print("[DEBUG rag.py] Iniciando imports...")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
print("[DEBUG rag.py] Intentando import con sys.path modificado...")
from backend.app.clients.bbdd_client import guardar_texto_chroma, buscar_similares
from backend.app.clients.llm_client import chat_with_model
print("[DEBUG rag.py] ✓ Imports con sys.path successful")

print("[DEBUG rag.py] Configurando modelo...")
# Configuración de Ollama
OLLAMA_MODEL = "llama3.2"  # Cambia según tu modelo instalado
print(f"[DEBUG rag.py] OLLAMA_MODEL = {OLLAMA_MODEL}")


def dividir_en_chunks(texto, tamaño_chunk=500, solapamiento=50):
    chunks = []
    inicio = 0
    
    while inicio < len(texto):
        fin = min(inicio + tamaño_chunk, len(texto))
        chunk = texto[inicio:fin].strip()
        
        if chunk:
            chunks.append(chunk)
        
        # Si llegamos al final, salimos
        if fin == len(texto):
            break
            
        inicio = fin - solapamiento  # El solapamiento solo si hay más texto
    
    return chunks


def ingestar_archivo_txt(ruta_archivo, metadata=None):
    """Lee un archivo txt y lo ingesta en ChromaDB por chunks.
    
    Args:
        ruta_archivo: ruta al archivo .txt
        metadata: dict opcional con metadatos (ej: {"fuente":"documento.txt"})
    
    Returns:
        lista de ids guardados o [] en error
    """
    if not os.path.exists(ruta_archivo):
        print(f"Error: archivo no encontrado: {ruta_archivo}")
        return []
    
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            texto = f.read()
    except Exception as e:
        print(f"Error leyendo archivo: {e}")
        return []
    
    print(f"Leyendo {ruta_archivo} ({len(texto)} caracteres)...")
    
    # Dividir en chunks
    chunks = dividir_en_chunks(texto)
    print(f"Dividido en {len(chunks)} chunks")
    
    # Guardar cada chunk
    ids_guardados = []
    nombre_archivo = os.path.basename(ruta_archivo)
    
    for i, chunk in enumerate(chunks):
        meta = metadata or {}
        meta.update({"chunk": i, "fuente": nombre_archivo})
        
        doc_id = guardar_texto_chroma(chunk, metadata=meta)
        if doc_id:
            ids_guardados.append(doc_id)
            print(f"  Chunk {i+1}/{len(chunks)} guardado")
    
    print(f"Ingesta completada: {len(ids_guardados)} chunks guardados")
    return ids_guardados


def consultar_con_contexto(pregunta, n_resultados=3, temperatura=0.7):
    """Consulta el LLM usando documentos relevantes de ChromaDB.
    
    Args:
        pregunta: pregunta del usuario
        n_resultados: número de documentos a recuperar (contexto)
        temperatura: creatividad del modelo (0-1)
    
    Returns:
        respuesta del LLM (string)
    """
    print(f"\n🔍 Buscando documentos relevantes para: '{pregunta}'")
    
    # Buscar documentos similares
    resultados = buscar_similares(pregunta, n=n_resultados)
    
    if not resultados or not resultados.get("documents"):
        print("No se encontraron documentos. Respondiendo sin contexto...")
        documentos_contexto = ""
    else:
        documentos = resultados["documents"][0]
        distancias = resultados.get("distances", [[]])[0]
        
        print(f"✓ Encontrados {len(documentos)} documentos similares")
        
        # Construir contexto
        documentos_contexto = "\n---\n".join(
            f"[Relevancia: {1 - d:.2f}]\n{doc}"
            for doc, d in zip(documentos, distancias)
        )
    
    # Construir prompt con contexto
    sistema = """Eres un asistente experto basado en documentos.
Tu tarea es responder preguntas solo usando la información del contexto proporcionado.
Si la respuesta no está en el contexto, indica que no tienes esa información.
Sé conciso y preciso."""
    
    prompt_usuario = f"""Contexto de documentos:
---
{documentos_contexto}
---

Pregunta: {pregunta}

Por favor, responde basándote únicamente en el contexto anterior."""
    
    print(f"\nConsultando modelo {OLLAMA_MODEL}...\n")
    
    try:
        # Consultar Ollama via chat_with_model
        respuesta = chat_with_model(
            messages=[
                {"role": "system", "content": sistema},
                {"role": "user", "content": prompt_usuario}
            ],
            model=OLLAMA_MODEL,
            temperature=temperatura
        )
        
        if respuesta:
            print(f"✓ Respuesta generada\n")
            return respuesta
        else:
            print("Error: no se pudo obtener respuesta del modelo")
            return None
    
    except Exception as e:
        print(f"Error en RAG: {e}")
        return None


# CLI interactivo
def main():
    """Interfaz CLI para ingestar y consultar."""
    print("=== RAG Chat (ChromaDB + Ollama) ===\n")
    
    while True:
        print("\nOpciones:")
        print("1. Ingestar archivo txt")
        print("2. Consultar con contexto")
        print("3. Salir")
        
        opcion = input("\nSelecciona opción (1-3): ").strip()
        
        if opcion == "1":
            ruta = input("Ruta del archivo .txt: ").strip()
            ingestar_archivo_txt(ruta, metadata={"tipo": "manual"})
        
        elif opcion == "2":
            pregunta = input("Escribe tu pregunta: ").strip()
            if pregunta:
                respuesta = consultar_con_contexto(pregunta)
                if respuesta:
                    print(f"\n📝 Respuesta:\n{respuesta}")
        
        elif opcion == "3":
            print("¡Hasta luego!")
            break
        
        else:
            print("Opción no válida")


if __name__ == "__main__":
    main()
