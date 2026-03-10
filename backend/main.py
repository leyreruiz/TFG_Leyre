"""Main CLI entry point for the backend.

Usage:
  python -m backend.main "Your question here"         (summary - RAG search)
  python -m backend.main "examen redes_neuronales"   (exam for a specific file)
  python -m backend.main --ingest path/to/file.txt   (ingest a file into ChromaDB)
  python -m backend.main                              (interactive mode)

ExamAgent examples:
  - "examen redes_neuronales"
  - "exam bases_datos.txt"
  - "test sistemas_operativos"
"""

import sys
import os

# Ensure the project root directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.orchestrator import Orchestrator
from backend.agents.summary_agent import SummaryAgent
from backend.agents.exam_agent import ExamAgent
from backend.rag.retriever import ChromaDbRetriever
from backend.ingest_topics import ingestar_archivo_txt


def build_orchestrator():
    """Build the orchestrator with all available agents."""
    retriever = ChromaDbRetriever()
    summary_agent = SummaryAgent(retriever)
    exam_agent = ExamAgent()  # ExamAgent works with specific files, not RAG

    orchestrator = Orchestrator(
        agents=[summary_agent, exam_agent]
    )
    return orchestrator


def main():
    # Ingest mode: --ingest <file>
    if len(sys.argv) >= 3 and sys.argv[1] == "--ingest":
        ruta = sys.argv[2]
        print(f"\n=== INGEST ===\nFile: {ruta}\n")
        ids = ingestar_archivo_txt(ruta)
        if ids:
            print(f"\nIngest complete: {len(ids)} chunks saved.")
        else:
            print("\nError during ingest.")
        return

    # Direct question mode: python -m backend.main "question"
    if len(sys.argv) >= 2:
        question = " ".join(sys.argv[1:])
        orchestrator = build_orchestrator()
        result = orchestrator.route(question)
        _print_result(result)
        return

    # Interactive mode
    print("=== University Assistant ===")
    print("Type your question (or 'quit' to exit)\n")

    orchestrator = build_orchestrator()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("salir", "exit", "quit"):
            print("Goodbye!")
            break

        result = orchestrator.route(question)
        _print_result(result)


def _print_result(result: dict):
    """Format and display the agent result."""
    agent = result.get("agent", "unknown")

    print(f"\n=== RESPONSE ({agent}) ===\n")

    if "error" in result:
        print(f"Error: {result['error']}\n")
        return

    if "content" in result:
        print(result["content"])

    # For exams, show a summary of parsed questions
    if agent == "exam" and result.get("questions"):
        n = result.get("num_questions", 0)
        print(f"\n--- {n} question(s) generated successfully ---")

    print()


if __name__ == "__main__":
    main()
