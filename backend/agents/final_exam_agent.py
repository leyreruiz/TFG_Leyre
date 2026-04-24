"""Final Exam Agent: analyzes weak points and generates a personalized final exam.

Uses the student's past performance (correct/incorrect answers per section and
number of questions asked to the Explainer) to build an adaptive exam that
focuses on weak areas and orders questions from easy to hard.
"""

import logging
import re

from backend.clients.llm_client import chat_with_model

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a university professor creating a personalized final exam. "
    "Generate multiple-choice questions based ONLY on the provided content. "
    "Each question must have exactly 4 options (a, b, c, d), ONLY ONE correct answer, "
    "and a difficulty level (easy, medium, or hard)."
)


def analyze_weak_points(class_data: dict) -> dict:
    """Analyze the student's performance per section and return a weakness report.

    Returns a dict with:
      - sections: list of {title, correct, total, conversations, score, level}
      - weak/medium/strong: lists of section titles grouped by score
    """
    sections_data = class_data.get("sections_data", {})
    sections_analysis = []

    for title, data in sections_data.items():
        questions = data.get("questions", [])
        answered = [q for q in questions if "user_correct" in q]
        correct = sum(1 for q in answered if q.get("user_correct"))
        total = len(answered)
        conversations = len(data.get("conversation", []))

        # Score: 70% accuracy + 30% conversation penalty
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 0.0  # no data = treat as weak

        # More conversations = more doubts = weaker understanding
        # Cap penalty at 0.3 (10+ questions to Explainer = max penalty)
        conversation_penalty = min(conversations / 10, 1.0)
        score = round((accuracy * 0.7) + ((1 - conversation_penalty) * 0.3), 2)

        if score >= 0.7:
            level = "strong"
        elif score >= 0.4:
            level = "medium"
        else:
            level = "weak"

        sections_analysis.append({
            "title": title,
            "correct": correct,
            "total": total,
            "conversations": conversations,
            "score": score,
            "level": level,
        })

    weak = [s["title"] for s in sections_analysis if s["level"] == "weak"]
    medium = [s["title"] for s in sections_analysis if s["level"] == "medium"]
    strong = [s["title"] for s in sections_analysis if s["level"] == "strong"]

    return {
        "sections": sections_analysis,
        "weak": weak,
        "medium": medium,
        "strong": strong,
    }


def _decide_question_counts(analysis: dict, target_total: int = 12) -> dict:
    """Decide how many questions to generate per section based on weakness.

    Weak sections get ~3, medium ~2, strong ~1. Adjusted to fit target_total.
    """
    weights = {"weak": 3, "medium": 2, "strong": 1}
    raw = {}
    for section in analysis["sections"]:
        raw[section["title"]] = weights[section["level"]]

    total_raw = sum(raw.values())
    if total_raw == 0:
        return {}

    counts = {}
    for title, w in raw.items():
        counts[title] = max(1, round(w / total_raw * target_total))

    return counts


def generate_final_exam(class_data: dict, analysis: dict) -> list[dict]:
    """Generate final exam questions, weighted by weakness and sorted by difficulty.

    Returns a list of question dicts, each with:
      question, options, correct_answer, explanation, difficulty, section
    """
    sections_data = class_data.get("sections_data", {})
    counts = _decide_question_counts(analysis)

    all_questions = []

    for section_title, num_questions in counts.items():
        summary = sections_data.get(section_title, {}).get("summary", "")
        if not summary:
            continue

        # Collect existing question texts to avoid repeats
        existing = sections_data.get(section_title, {}).get("questions", [])
        existing_texts = "\n".join(f"- {q['question']}" for q in existing)

        raw = _generate_section_questions(
            summary, section_title, existing_texts, num_questions
        )
        if not raw:
            continue

        parsed = _parse_questions(raw)
        for q in parsed:
            q["section"] = section_title
        all_questions.extend(parsed)

    # Sort by difficulty: easy first, then medium, then hard
    order = {"easy": 0, "medium": 1, "hard": 2}
    all_questions.sort(key=lambda q: order.get(q.get("difficulty", "medium"), 1))

    return all_questions


def _generate_section_questions(
    summary: str, section_title: str, existing_texts: str, num: int
) -> str | None:
    """Call the LLM to generate questions for one section with difficulty levels."""
    prompt = f"""Section: '{section_title}'
Content:
---
{summary}
---

The following questions already exist — do NOT repeat them or ask the same concepts:
{existing_texts}

Generate exactly {num} NEW multiple-choice questions about DIFFERENT aspects of the content.
Assign a difficulty level to each question: easy, medium, or hard.

For each question, use this EXACT format:

QUESTION 1: [question text]
DIFFICULTY: [easy/medium/hard]
a) [option a]
b) [option b]
c) [option c]
d) [option d]
RESPONSE: [correct letter]
EXPLANATION: [brief explanation of why it's correct]"""

    return chat_with_model(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )


def _parse_questions(raw_text: str) -> list[dict]:
    """Parse LLM response into structured question dicts with difficulty."""
    questions = []
    blocks = re.split(r"(?=\bQUESTION\s*\d+\s*:)", raw_text, flags=re.IGNORECASE)

    for block in blocks:
        if not re.search(r"\bQUESTION\s*\d+\s*:", block, flags=re.IGNORECASE):
            continue

        q_match = re.search(
            r"QUESTION\s*\d+\s*:\s*(.*?)\s*(?=\n\s*DIFFICULTY|\n\s*[aA][\)\.]\s)",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not q_match:
            continue

        diff_match = re.search(
            r"DIFFICULTY\s*:\s*(easy|medium|hard)", block, flags=re.IGNORECASE
        )
        difficulty = diff_match.group(1).lower() if diff_match else "medium"

        option_pattern = (
            r"\n\s*([a-dA-D])[\)\.]\s*(.*?)"
            r"(?=\n\s*[a-dA-D][\)\.]\s|\n\s*RESPONSE\s*:|$)"
        )
        option_matches = re.findall(
            option_pattern, block, flags=re.IGNORECASE | re.DOTALL
        )
        options_map = {letter.lower(): text.strip() for letter, text in option_matches}

        if not all(k in options_map for k in ("a", "b", "c", "d")):
            continue

        response_match = re.search(
            r"RESPONSE\s*:\s*([a-dA-D])", block, flags=re.IGNORECASE
        )
        explanation_match = re.search(
            r"EXPLANATION\s*:\s*(.*)", block, flags=re.IGNORECASE | re.DOTALL
        )

        if not response_match:
            continue

        questions.append({
            "question": q_match.group(1).strip(),
            "options": [
                options_map["a"],
                options_map["b"],
                options_map["c"],
                options_map["d"],
            ],
            "correct_answer": response_match.group(1).strip().lower(),
            "explanation": (
                explanation_match.group(1).strip() if explanation_match else ""
            ),
            "difficulty": difficulty,
        })

    return questions
