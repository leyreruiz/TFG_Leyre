"""
Script para chatear con un LLM open source via Ollama.
"""
import sys
import ollama

MODEL = "llama3.2"   # cambia por cualquier modelo instalado con `ollama pull <modelo>`



def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        print(response.message.content)
        return

    print(f"Chat con {MODEL} (escribe 'salir' para terminar)\n")
    history = []

    while True:
        try:
            prompt = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("salir", "exit", "quit"):
            print("Hasta luego!")
            break

        history.append({"role": "user", "content": prompt})
        response = ollama.chat(model=MODEL, messages=history)
        answer = response.message.content
        history.append({"role": "assistant", "content": answer})

        print(f"\nModelo: {answer}\n")


if __name__ == "__main__":
    main()