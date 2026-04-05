"""Class Storage: manages persisting generated classes to JSON."""

import logging
import os
import json
from datetime import datetime
from typing import Optional

from backend.utils import normalize_topic as _normalize_topic

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
    """Find the existing file key that matches a topic, using normalized comparison."""
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
    _ensure_storage_dir()
    try:
        topic_key = _resolve_topic_key(topic)
        class_obj = get_class(topic)
        if not class_obj:
            logger.warning("Class '%s' not found", topic)
            return False

        if section_title not in class_obj.get("sections_data", {}):
            logger.warning("Section '%s' not found in '%s'", section_title, topic)
            return False

        class_obj["sections_data"][section_title]["summary"] = new_summary
        class_obj["updated_at"] = datetime.now().isoformat()

        if not topic_key:
            topic_key = _normalize_topic(class_obj.get("topic", topic))

        if not _write_topic_file(topic_key, class_obj):
            return False

        logger.info("Summary updated for '%s' -> '%s'", topic_key, section_title)
        return True
    except Exception as e:
        logger.error("Error updating section summary: %s", e)
        return False


def update_section_questions(topic: str, section_title: str, questions: list) -> bool:
    """Update the questions of a specific section."""
    _ensure_storage_dir()
    try:
        topic_key = _resolve_topic_key(topic)
        class_obj = get_class(topic)
        if not class_obj:
            logger.warning("Class '%s' not found", topic)
            return False

        if section_title not in class_obj.get("sections_data", {}):
            logger.warning("Section '%s' not found in '%s'", section_title, topic)
            return False

        class_obj["sections_data"][section_title]["questions"] = questions
        class_obj["updated_at"] = datetime.now().isoformat()

        if not topic_key:
            topic_key = _normalize_topic(class_obj.get("topic", topic))

        if not _write_topic_file(topic_key, class_obj):
            return False

        logger.info("Questions updated for '%s' -> '%s'", topic_key, section_title)
        return True
    except Exception as e:
        logger.error("Error updating section questions: %s", e)
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
    """Delete a saved class JSON file."""
    _ensure_storage_dir()
    try:
        topic_key = _resolve_topic_key(topic)
        if not topic_key:
            logger.warning("Class '%s' not found", topic)
            return False

        os.remove(_topic_file_path(topic_key))
        logger.info("Class '%s' deleted", topic_key)
        return True
    except Exception as e:
        logger.error("Error deleting class '%s': %s", topic, e)
        return False
