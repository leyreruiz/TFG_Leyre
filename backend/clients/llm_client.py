"""
Script to chat with an LLM via Groq.
"""
import sys
print("[llm_client] Importing groq...")
from groq import Groq
print("[llm_client] Groq imported successfully")

MODEL = "llama-3.1-8b-instant"
client = Groq(api_key="")
print("[llm_client] Import complete")


def chat_with_model(messages, model=MODEL, temperature=0.7):
    """Query the Groq model with a message history.
    
    Args:
        messages: list of dicts {"role": "user"|"assistant", "content": "..."}
        model: model to use (default: llama-3.1-8b-instant)
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
