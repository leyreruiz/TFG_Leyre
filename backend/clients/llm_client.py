"""
Script to chat with an LLM via Groq.
"""
import logging
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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
        logger.debug("Connecting to Groq model=%s", model)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        logger.debug("Response received from Groq")
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Error querying %s: %s", model, e)
        return None
