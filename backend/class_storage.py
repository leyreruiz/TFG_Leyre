"""Class Storage: manages persisting generated classes to JSON."""

import logging
import os
import json
from datetime import datetime
from typing import Optional

from backend.utils import normalize_topic as _normalize_topic
from backend.clients.bbdd_client import delete_collection

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
CLASSES_DIR = os.path.join(BASE_DIR, "data", "classes")


def _ensure_storage_dir():
    os.makedirs(CLASSES_DIR, exist_ok=True)


def _topic_file_path(topic_key: str) -> str:
    return os.path.join(CLASSES_DIR, f"{topic_key}.json")


def _iter_topic_keys() -> list[str]:
    _ensure_storage_dir()
    return [name[:-5] for name in os.listdir(CLASSES_DIR) if name.endswith(".json")]


def _resolve_topic_key(topic: str) -> Optional[str]:
    """Find the stored file key that matches a topic, using normalized comparison."""
    normalized_target = _normalize_topic(topic)
    for key in _iter_topic_keys():
        if key == topic or _normalize_topic(key) == normalized_target:
            return key
    return None


def _write_topic_file(topic_key: str, class_obj: dict) -> bool:
    try:
        _ensure_storage_dir()
        with open(_topic_file_path(topic_key), "w", encoding="utf-8") as f:
            json.dump(class_obj, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error("Error writing topic file: %s", e)
        return False


def _load_class_and_section(topic: str, section_title: str) -> tuple[Optional[str], Optional[dict]]:
    """Load and validate class data for a given topic and section.

    Returns (topic_key, class_obj) if both exist, or (None, None) with a warning logged.
    This avoids repeating the same lookup + validation pattern across multiple functions.
    """
    topic_key = _resolve_topic_key(topic)
    class_obj = get_class(topic)
    if not class_obj:
        logger.warning("Class '%s' not found", topic)
        return None, None
    if section_title not in class_obj.get("sections_data", {}):
        logger.warning("Section '%s' not found in '%s'", section_title, topic)
        return None, None
    # Fallback: derive topic_key from stored data if the initial resolve failed
    if not topic_key:
        topic_key = _normalize_topic(class_obj.get("topic", topic))
    return topic_key, class_obj


def get_class(topic: str) -> Optional[dict]:
    """Retrieve a saved class by topic name."""
    _ensure_storage_dir()
    try:
        topic_key = _resolve_topic_key(topic)
        if not topic_key:
            return None
        with open(_topic_file_path(topic_key), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error reading class '%s': %s", topic, e)
        return None


def save_class(topic: str, structure: str, sections: list[str], sections_data: dict):
    """Save a complete class to JSON."""
    _ensure_storage_dir()
    try:
        topic_key = _resolve_topic_key(topic) or _normalize_topic(topic)
        previous = get_class(topic_key)

        class_obj = {
            "topic": topic_key,
            "structure": structure,
            "sections": sections,
            "sections_data": sections_data,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # Preserve original creation date if the class already existed
        if previous and previous.get("created_at"):
            class_obj["created_at"] = previous["created_at"]

        if not _write_topic_file(topic_key, class_obj):
            return False

        logger.info("Class '%s' saved", topic_key)
        return True
    except Exception as e:
        logger.error("Error saving class '%s': %s", topic, e)
        return False


def update_section_summary(topic: str, section_title: str, new_summary: str) -> bool:
    """Update the summary of a specific section."""
    try:
        topic_key, class_obj = _load_class_and_section(topic, section_title)
        if class_obj is None:
            return False
        class_obj["sections_data"][section_title]["summary"] = new_summary
        class_obj["updated_at"] = datetime.now().isoformat()
        if not _write_topic_file(topic_key, class_obj):
            return False
        logger.info("Summary updated for '%s' -> '%s'", topic_key, section_title)
        return True
    except Exception as e:
        logger.error("Error updating section summary: %s", e)
        return False


def update_section_questions(topic: str, section_title: str, questions: list) -> bool:
    """Update the questions of a specific section."""
    try:
        topic_key, class_obj = _load_class_and_section(topic, section_title)
        if class_obj is None:
            return False
        class_obj["sections_data"][section_title]["questions"] = questions
        class_obj["updated_at"] = datetime.now().isoformat()
        if not _write_topic_file(topic_key, class_obj):
            return False
        logger.info("Questions updated for '%s' -> '%s'", topic_key, section_title)
        return True
    except Exception as e:
        logger.error("Error updating section questions: %s", e)
        return False


def clear_section_conversation(topic: str, section_title: str) -> bool:
    """Remove the conversation log from a section."""
    try:
        topic_key, class_obj = _load_class_and_section(topic, section_title)
        if class_obj is None:
            return False
        class_obj["sections_data"][section_title]["conversation"] = []
        class_obj["updated_at"] = datetime.now().isoformat()
        return _write_topic_file(topic_key, class_obj)
    except Exception as e:
        logger.error("Error clearing conversation: %s", e)
        return False


def append_section_conversation_turn(topic: str, section_title: str, question: str, answer: str) -> bool:
    """Append a Q&A turn to the conversation log of a section."""
    try:
        topic_key, class_obj = _load_class_and_section(topic, section_title)
        if class_obj is None:
            return False
        section_data = class_obj["sections_data"][section_title]
        section_data.setdefault("conversation", []).append({"question": question, "answer": answer})
        class_obj["updated_at"] = datetime.now().isoformat()
        return _write_topic_file(topic_key, class_obj)
    except Exception as e:
        logger.error("Error appending conversation turn: %s", e)
        return False


def update_question_user_answer(topic: str, section_title: str, question_index: int, user_answer: str, user_correct: bool) -> bool:
    """Save the user's answer (letter + correctness) for a specific question."""
    try:
        topic_key, class_obj = _load_class_and_section(topic, section_title)
        if class_obj is None:
            return False
        questions = class_obj["sections_data"][section_title].get("questions", [])
        if question_index < 0 or question_index >= len(questions):
            logger.warning("Question index %d out of range in '%s'/'%s'", question_index, topic, section_title)
            return False
        questions[question_index]["user_answer"] = user_answer
        questions[question_index]["user_correct"] = user_correct
        class_obj["updated_at"] = datetime.now().isoformat()
        return _write_topic_file(topic_key, class_obj)
    except Exception as e:
        logger.error("Error updating question user answer: %s", e)
        return False


def list_classes() -> list[str]:
    """Return sorted list of all saved class topic keys."""
    _ensure_storage_dir()
    try:
        return sorted(_iter_topic_keys())
    except Exception as e:
        logger.error("Error listing classes: %s", e)
        return []


def delete_class(topic: str) -> bool:
    """Delete a saved class JSON file and its ChromaDB collection."""
    _ensure_storage_dir()
    try:
        topic_key = _resolve_topic_key(topic)
        if not topic_key:
            logger.warning("Class '%s' not found", topic)
            return False
        
        # Delete JSON file
        os.remove(_topic_file_path(topic_key))
        logger.info("Class '%s' deleted from JSON storage", topic_key)
        
        # Delete ChromaDB collection
        delete_collection(topic=topic_key)
        logger.info("ChromaDB collection for '%s' deleted", topic_key)
        
        return True
    except Exception as e:
        logger.error("Error deleting class '%s': %s", topic, e)
        return False
