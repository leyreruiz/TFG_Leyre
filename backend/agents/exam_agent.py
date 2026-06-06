"""Exam Agent: generates multiple-choice exam questions.

Two generation paths, invoked explicitly by the caller:
  - generate_from_summary(): used by the orchestration pipeline — the section
    summary is passed in directly as context and no retrieval is performed.
  - handle(): standalone path — retrieves context from ChromaDB for a topic
    or source file and generates the exam from it.
"""

import logging
import os
import re

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest
from backend.clients.llm_client import chat_with_model
from backend.utils import extract_search_term

logger = logging.getLogger(__name__)

# System prompt shared by both question-generation methods
_EXAM_SYSTEM_PROMPT = (
    "You are a university professor expert in creating exams. "
    "Your task is to generate multiple-choice questions based ONLY on the provided content. "
    "Each question must have exactly 4 options (a, b, c, d) and ONLY ONE correct answer."
)


class ExamAgent(BaseAgent):
    """Generates multiple-choice exam questions using ChromaDB as the only data source."""

    def __init__(self, retriever, llm_model="llama-3.1-8b-instant", num_questions=3, data_dir="backend/data"):
        self.retriever = retriever
        self.llm_model = llm_model
        self.num_questions = num_questions
        self.data_dir = data_dir

    def generate_from_summary(self, section_title: str, summary: str) -> dict:
        """Generate an exam directly from a section's summary, without retrieval.

        This is the entry point used by the orchestration pipeline: the summary the
        Summary agent has just produced is passed in as context, so the Exam agent
        performs no retrieval of its own and the questions stay aligned with what
        the student has just read.

        Returns a dict with content (raw text), questions (parsed list), mode and source info.
        """
        context = summary.strip()
        if not context:
            return {"agent": "exam", "error": f"Summary context is empty for '{section_title}'."}

        raw_answer = self._generate_questions(context, section_title)
        if raw_answer is None:
            return {"agent": "exam", "error": f"Could not generate questions for '{section_title}'."}

        questions = self._parse_questions(raw_answer)
        return {
            "agent": "exam",
            "content": raw_answer,
            "questions": questions,
            "num_questions": len(questions),
            "retrieval_mode": "summary",
            "source": section_title,
        }

    def handle(self, request: StudentRequest) -> dict:
        """Generate an exam by retrieving context from ChromaDB for a topic or file.

        This is the standalone retrieval path: the message names a topic or a
        source file, the agent retrieves the relevant chunks itself, and generates
        the exam from them. The orchestration pipeline does not use this path —
        it calls generate_from_summary() with the already-produced summary instead.

        Returns a dict with content (raw text), questions (parsed list), mode and source info.
        """
        term = extract_search_term(request.message, intent="exam")

        if not term:
            return {
                "agent": "exam",
                "error": (
                    "No file or topic specified in the message. "
                    "Examples: 'exam redes_neuronales', 'exam aprendizaje supervisado'."
                ),
            }

        candidate_file = term if term.endswith(".txt") else term + ".txt"
        known_files = self._list_known_files()

        if candidate_file in known_files:
            logger.debug("Mode: file (%s)", candidate_file)
            docs = self.retriever.search_by_source(candidate_file)
            mode = "file"
            source_info = candidate_file
        else:
            logger.debug("Mode: semantic (query: %s)", term)
            docs = self.retriever.search(term, k=10)
            mode = "semantic"
            source_info = term

        if not docs:
            return {
                "agent": "exam",
                "error": (
                    f"No information was found about '{term}'. "
                    f"Available files: {', '.join(known_files)}"
                ),
            }

        context = "\n\n".join(docs)

        raw_answer = self._generate_questions(context, source_info)
        if raw_answer is None:
            return {"agent": "exam", "error": f"Could not generate questions for '{source_info}'."}

        questions = self._parse_questions(raw_answer)
        return {
            "agent": "exam",
            "content": raw_answer,
            "questions": questions,
            "num_questions": len(questions),
            "retrieval_mode": mode,
            "source": source_info,
        }

    def add_questions(self, summary: str, section_title: str, existing_questions: list, num_new: int = 3) -> list[dict]:
        """Generate additional questions that don't repeat the existing ones."""
        existing_texts = "\n".join(f"- {q['question']}" for q in existing_questions)
        raw = self._generate_additional_questions(summary, section_title, existing_texts, num_new)
        if not raw:
            return []
        return self._parse_questions(raw)

    def _list_known_files(self) -> list[str]:
        """Return the list of .txt filenames available in the data directory."""
        try:
            return [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]
        except Exception:
            return []

    def _generate_questions(self, content: str, filename: str) -> str | None:
        """Call the LLM to generate multiple-choice questions based on the provided content."""
        prompt = f"""File content '{filename}':
---
{content}
---

Generate exactly {self.num_questions} multiple-choice questions.

For each question, use this exact format:

QUESTION 1: [question text]
a) [option a]
b) [option b]
c) [option c]
d) [option d]
RESPONSE: [correct letter]
EXPLANATION: [brief explanation of why it's correct]

Make sure:
- The questions cover different aspects of the content
- The incorrect options are plausible but clearly distinguishable
- The explanations are concise and useful for learning"""

        return chat_with_model(
            messages=[
                {"role": "system", "content": _EXAM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.llm_model,
            temperature=0.5,  # lower creativity for more precise exams
        )

    def _generate_additional_questions(self, summary: str, section_title: str, existing_texts: str, num_new: int) -> str | None:
        """Call the LLM to generate new questions, explicitly avoiding the existing ones."""
        prompt = f"""Section: '{section_title}'
Content summary:
---
{summary}
---

The following questions already exist — do NOT repeat them or ask about the same concepts:
{existing_texts}

Generate exactly {num_new} NEW multiple-choice questions about DIFFERENT aspects of the content.

For each question, use this exact format:

QUESTION 1: [question text]
a) [option a]
b) [option b]
c) [option c]
d) [option d]
RESPONSE: [correct letter]
EXPLANATION: [brief explanation of why it's correct]"""

        return chat_with_model(
            messages=[
                {"role": "system", "content": _EXAM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.llm_model,
            temperature=0.7,  # slightly higher for variety
        )

    def _parse_questions(self, raw_text: str) -> list[dict]:
        """Parse the LLM response into a structured list of question dicts.

        Uses block-based parsing to avoid brittle cross-question regex assumptions.
        Returns an empty list if parsing fails (raw text is still available in the response).
        """
        questions = []

        # Split into per-QUESTION blocks
        blocks = re.split(r"(?=\bQUESTION\s*\d+\s*:)", raw_text, flags=re.IGNORECASE)

        for block in blocks:
            if not re.search(r"\bQUESTION\s*\d+\s*:", block, flags=re.IGNORECASE):
                continue

            q_match = re.search(
                r"QUESTION\s*\d+\s*:\s*(.*?)\s*(?=\n\s*[aA][\)\.]\s)",
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not q_match:
                continue

            # Accept both "a)" and "a." formats, with optional spaces
            option_pattern = r"\n\s*([a-dA-D])[\)\.]\s*(.*?)(?=\n\s*[a-dA-D][\)\.]\s|\n\s*RESPONSE\s*:|$)"
            option_matches = re.findall(option_pattern, block, flags=re.IGNORECASE | re.DOTALL)
            options_map = {letter.lower(): text.strip() for letter, text in option_matches}

            if not all(k in options_map for k in ("a", "b", "c", "d")):
                continue

            response_match = re.search(r"RESPONSE\s*:\s*([a-dA-D])", block, flags=re.IGNORECASE)
            explanation_match = re.search(r"EXPLANATION\s*:\s*(.*)", block, flags=re.IGNORECASE | re.DOTALL)

            if not response_match:
                continue

            questions.append({
                "question": q_match.group(1).strip(),
                "options": [options_map["a"], options_map["b"], options_map["c"], options_map["d"]],
                "correct_answer": response_match.group(1).strip().lower(),
                "explanation": explanation_match.group(1).strip() if explanation_match else "",
            })

        return questions
