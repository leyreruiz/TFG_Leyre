"""CLI principal del backend.

Uso:
  python -m backend.main "Tu pregunta aquí"
  python -m backend.main --ingest ruta/archivo.txt
  python -m backend.main                            (modo interactivo)
"""

import sys
import os

# Asegurar que el directorio raíz del proyecto está en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.orchestrator import Orchestrator
from backend.agents.summary_agent import SummaryAgent
from backend.agents.exam_agent import ExamAgent
from backend.rag.retriever import ChromaDbRetriever
from backend.ingest_topics import ingestar_archivo_txt


def build_orchestrator():
    """Construye el orquestador con todos los agentes disponibles."""
    retriever = ChromaDbRetriever()
    summary_agent = SummaryAgent(retriever)
    exam_agent = ExamAgent(retriever)

    orchestrator = Orchestrator(
        agents=[summary_agent, exam_agent]
    )
    return orchestrator


def main():
    # Modo ingesta: --ingest <archivo>
    if len(sys.argv) >= 3 and sys.argv[1] == "--ingest":
        ruta = sys.argv[2]
        print(f"\n=== INGESTA ===\nArchivo: {ruta}\n")
        ids = ingestar_archivo_txt(ruta)
        if ids:
            print(f"\nIngesta completada: {len(ids)} chunks guardados.")
        else:
            print("\nError durante la ingesta.")
        return

    # Modo pregunta directa: python -m backend.main "pregunta"
    if len(sys.argv) >= 2:
        question = " ".join(sys.argv[1:])
        orchestrator = build_orchestrator()
        result = orchestrator.route(question)
        _print_result(result)
        return

    # Modo interactivo
    print("=== Asistente Universitario ===")
    print("Escribe tu pregunta (o 'salir' para terminar)\n")

    orchestrator = build_orchestrator()

    while True:
        try:
            question = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta luego!")
            break

        if not question:
            continue
        if question.lower() in ("salir", "exit", "quit"):
            print("¡Hasta luego!")
            break

        result = orchestrator.route(question)
        _print_result(result)


def _print_result(result: dict):
    """Formatea y muestra el resultado del agente."""
    agent = result.get("agent", "unknown")

    print(f"\n=== RESPUESTA ({agent}) ===\n")

    if "error" in result:
        print(f"Error: {result['error']}\n")
        return

    if "content" in result:
        print(result["content"])

    # Si es examen, mostrar resumen de preguntas parseadas
    if agent == "exam" and result.get("questions"):
        n = result.get("num_questions", 0)
        print(f"\n--- {n} pregunta(s) generada(s) correctamente ---")

    print()


if __name__ == "__main__":
    main()
