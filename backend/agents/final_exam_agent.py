"""Final Exam Agent: analyzes weak points and generates a personalized final exam.

Uses the student's past performance (correct/incorrect answers per section and
number of questions asked to the Explainer) to build an adaptive exam that
focuses on weak areas and orders questions from easy to hard.
"""

import logging
import re

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest
from backend.clients.llm_client import chat_with_model, MODEL

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a university professor creating a personalized final exam. "
    "Generate multiple-choice questions based ONLY on the provided content. "
    "Each question must have exactly 4 options (a, b, c, d), ONLY ONE correct answer, "
    "and a difficulty level (easy, medium, or hard)."
)


class FinalExamAgent(BaseAgent):
    """Generates an adaptive final exam weighted by the student's weak points."""

    def __init__(self, llm_model: str = MODEL, target_total: int = 12):
        self.llm_model = llm_model
        self.target_total = target_total

    def handle(self, request: StudentRequest) -> dict:
        """Generic entry point. The endpoints in api.py call analyze() and generate()
        directly because they need the parsed class JSON, not a free-text request."""
        return {
            "agent": "final_exam",
            "error": "Use FinalExamAgent.analyze(class_data) and .generate(class_data, analysis) directly.",
        }

    def analyze(self, class_data: dict) -> dict:
        """Compute the per-section mastery score and bucket sections into tiers."""
        sections_data = class_data.get("sections_data", {})
        sections_analysis = []

        for title, data in sections_data.items():
            questions = data.get("questions", [])
            answered = [q for q in questions if "user_correct" in q]
            correct = sum(1 for q in answered if q.get("user_correct"))
            total = len(questions)  # All questions in the section, not just answered ones
            conversations = len(data.get("conversation", []))

            accuracy = (correct / total) if total > 0 else 0.0
            # Penalty saturates at 10 conversational turns about a section
            conversation_penalty = min(conversations / 10, 1.0)
            score = max(round((accuracy - conversation_penalty * 0.2), 2), 0)

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
        print(f"Section analysis: {sections_analysis}")
        return {
            "sections": sections_analysis,
            "weak": weak,
            "medium": medium,
            "strong": strong,
        }

    def generate(self, class_data: dict, analysis: dict) -> list[dict]:
        """Generate the final exam questions weighted by weakness, ordered by difficulty."""
        sections_data = class_data.get("sections_data", {})
        counts = self._decide_question_counts(analysis, target_total=self.target_total)

        all_questions = []

        for section_title, num_questions in counts.items():
            summary = sections_data.get(section_title, {}).get("summary", "")
            if not summary:
                continue

            existing = sections_data.get(section_title, {}).get("questions", [])
            existing_texts = "\n".join(f"- {q['question']}" for q in existing)

            raw = self._generate_section_questions(
                summary, section_title, existing_texts, num_questions
            )
            if not raw:
                continue

            parsed = self._parse_questions(raw)
            for q in parsed:
                q["section"] = section_title
            all_questions.extend(parsed)

        order = {"easy": 0, "medium": 1, "hard": 2}
        all_questions.sort(key=lambda q: order.get(q.get("difficulty", "medium"), 1))

        return all_questions

    @staticmethod
    def _decide_question_counts(analysis: dict, target_total: int = 12) -> dict:
        """Allocate the target number of questions across sections by weakness tier.

        Weak sections get weight 3, medium 2, strong 1.
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

    def _generate_section_questions(
        self, summary: str, section_title: str, existing_texts: str, num: int
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
            model=self.llm_model,
            temperature=0.6,
        )

    @staticmethod
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
