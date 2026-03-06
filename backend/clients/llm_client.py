"""
Script para chatear con un LLM open source via Ollama.
"""
import sys
print("[llm_client] Importando ollama...")
import ollama
print("[llm_client] Ollama importado correctamente")

MODEL = "llama3.2"   # cambia por cualquier modelo instalado con `ollama pull <modelo>`
print("[llm_client] Importación completada")


def chat_with_model(messages, model=MODEL, temperature=0.7):
    """Consulta el modelo Ollama con un historial de mensajes.
    
    Args:
        messages: lista de dicts {"role": "user"|"assistant", "content": "..."}
        model: modelo a usar (default: llama3.2)
        temperature: creatividad (0-1)
    
    Returns:
        string con la respuesta del modelo o None en error
    """
    try:
        print(f"[DEBUG] Conectando con Ollama modelo={model}...")
        response = ollama.chat(
            model=model,
            messages=messages,
            stream=False,
            options={"temperature": temperature}
        )
        print(f"[DEBUG] Respuesta recibida de Ollama")
        return response.message.content
    except Exception as e:
        print(f"Error consultando {model}: {e}")
        return None


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