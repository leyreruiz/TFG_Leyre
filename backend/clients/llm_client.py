"""
Script to chat with an LLM via Groq.
"""
import sys
print("[llm_client] Importing groq...")
from groq import Groq
print("[llm_client] Groq imported successfully")

MODEL = "llama-3.3-70b-versatile"
client = Groq(api_key="")  # pega tu key aquí
print("[llm_client] Import complete")


def chat_with_model(messages, model=MODEL, temperature=0.7):
    """Query the Groq model with a message history.
    
    Args:
        messages: list of dicts {"role": "user"|"assistant", "content": "..."}
        model: model to use (default: llama-3.3-70b-versatile)
        temperature: creativity (0-1)
    
    Returns:
        string with the model response, or None on error
    """
    try:
        print(f"[DEBUG] Connecting to Groq model={model}...")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        print(f"[DEBUG] Response received from Groq")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying {model}: {e}")
        return None


def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        print(response.choices[0].message.content)
        return

    print(f"Chat with {MODEL} (type 'quit' to exit)\n")
    history = []

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("salir", "exit", "quit"):
            print("Goodbye!")
            break

        history.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(model=MODEL, messages=history)
        answer = response.choices[0].message.content
        history.append({"role": "assistant", "content": answer})

        print(f"\nModel: {answer}\n")


if __name__ == "__main__":
    main()