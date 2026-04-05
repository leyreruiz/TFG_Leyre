"""Shared utilities for the backend."""

import re


# Keywords stripped from user messages before sending queries to ChromaDB.
# Bilingual (ES/EN) to match how students naturally phrase requests.
_SUMMARY_KEYWORDS = [
    "hazme un resumen de", "hazme un resumen sobre",
    "resumen de", "resumen sobre", "resume", "resumen",
    "summary of", "summary about", "summary",
    "explícame", "explicame", "explain",
    "qué es", "que es", "what is",
]

_EXAM_KEYWORDS = ["examen", "exam", "test", "pregunta"]

_STRUCTURE_KEYWORDS = ["estructura", "structure", "preparar", "clase", "prepare"]


def normalize_topic(name: str) -> str:
    """Normalize a topic name to a safe identifier (lowercase, underscores).

    Used as the canonical key for ChromaDB collections and JSON class files.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "default"


def extract_search_term(message: str, intent: str = "summary") -> str | None:
    """Strip intent keywords from a user message to get the bare topic term.

    Args:
        message: raw user message
        intent: one of "summary", "exam", "structure" — selects which keyword
                list to strip

    Returns:
        Cleaned topic string, or None if nothing remains after stripping.
    """
    keyword_map = {
        "summary": _SUMMARY_KEYWORDS,
        "exam": _EXAM_KEYWORDS,
        "structure": _STRUCTURE_KEYWORDS,
    }
    keywords = keyword_map.get(intent, _SUMMARY_KEYWORDS)

    msg = message.strip()
    msg_lower = msg.lower()
    for kw in keywords:
        if msg_lower.startswith(kw):
            msg = msg[len(kw):].strip()
            msg_lower = msg.lower()
    return msg if msg else None
