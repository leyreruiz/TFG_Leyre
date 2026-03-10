"""Exam Agent: generates multiple-choice exam questions.

Two retrieval modes selected automatically:
  - File mode:     query matches a known .txt file  → all chunks of that file (ChromaDB)
  - Semantic mode: query is a topic/concept         → similarity search across all sources
"""

import os
import re

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest
from backend.clients.llm_client import chat_with_model


class ExamAgent(BaseAgent):
    """Generates multiple-choice exam questions using ChromaDB as the only data source."""

    def __init__(self, retriever, llm_model="llama-3.3-70b-versatile", num_questions=3, data_dir="backend/data"):
        self.retriever = retriever
        self.llm_model = llm_model
        self.num_questions = num_questions
        self.data_dir = data_dir

    def can_handle(self, intent: str) -> bool:
        return intent == "exam"

    def handle(self, request: StudentRequest) -> dict:
        """Generate an exam choosing the retrieval mode automatically.

        - If the term extracted from the message matches a known .txt file,
          ALL chunks of that file are retrieved from ChromaDB (file mode).
        - Otherwise, a semantic similarity search is performed across all
          ingested sources (semantic mode), useful for cross-topic queries.

        Returns:
            dict with content (raw text), questions (parsed), mode and source info.
        """
        term = self._extract_term(request.message)

        if not term:
            return {
                "agent": "exam",
                "error": (
                    "No file or topic specified in the message. "
                    "Examples: 'exam redes_neuronales', 'exam aprendizaje supervisado'."
                ),
            }

        # Detect retrieval mode
        candidate_file = term if term.endswith(".txt") else term + ".txt"
        known_files = self._list_known_files()
        
        if candidate_file in known_files:
            # FILE MODE: retrieve all chunks of that specific file
            print(f"[ExamAgent] Mode: file ({candidate_file})")
            docs = self.retriever.search_by_source(candidate_file)
            mode = "file"
            source_info = candidate_file
        else:
            # SEMANTIC MODE: similarity search across all sources
            print(f"[ExamAgent] Mode: semantic (query: {term})")
            docs = self.retriever.search(term, k=10)
            mode = "semantic"
            source_info = term

        if not docs:
            return {
                "agent": "exam",
                "error": (
                    f"No information was found about '{term}' "
                    f"Available files: {', '.join(known_files)}"
                ),
            }

        context = "\n\n".join(docs)

        # Generate questions based on the retrieved context
        raw_answer = self._generate_questions(context, source_info)

        if raw_answer is None:
            return {
                "agent": "exam",
                "error": "No information was found about '{term}' "
                         f"Available files: {', '.join(known_files)}"
            }

        questions = self._parse_questions(raw_answer)

        return {
            "agent": "exam",
            "content": raw_answer,
            "questions": questions,
            "num_questions": len(questions),
            "retrieval_mode": mode,
            "source": source_info,
        }

    def _extract_term(self, message: str) -> str | None:
        """Extract the topic/filename term from the message, stripping exam keywords.

        Examples:
          "examen redes_neuronales"        → "redes_neuronales"
          "exam bases_datos.txt"           → "bases_datos.txt"
          "examen aprendizaje supervisado" → "aprendizaje supervisado"
          "test backpropagation"           → "backpropagation"
        """
        keywords = ["examen", "exam", "test", "pregunta tipo", "pregunta", "tipo"]
        message_lower = message.lower().strip()
        for kw in keywords:
            if message_lower.startswith(kw):
                message = message[len(kw):].strip()
                message_lower = message.lower().strip()
        return message if message else None

    def _list_known_files(self) -> list[str]:
        """Return the list of .txt filenames available in the data directory."""
        try:
            return [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]
        except Exception:
            return []

    def _generate_questions(self, content: str, filename: str) -> str | None:
        """Call the LLM to generate multiple-choice questions based on the file content."""

        sistema = (
            "You are a univeristy professor expert in creating exams. "
            "Your task is to generate multiple-choice questions (multiple options) "
            "based ONLY on the provided content. "
            "Each question must have exactly 4 options (a, b, c, d) "
            "and only one correct answer."
        )

        prompt_usuario = f"""File content '{filename}':
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
                {"role": "system", "content": sistema},
                {"role": "user", "content": prompt_usuario},
            ],
            model=self.llm_model,
            temperature=0.5,  # Lower creativity for more precise exams
        )

    def _parse_questions(self, raw_text: str) -> list[dict]:
        """Attempt to parse the LLM response into a structured list.

        Uses regex to extract each question with its options and answer.
        Returns an empty list if parsing fails (raw text is still available).
        """
        questions = []

        # Pattern to match each question block
        pattern = (
            r"QUESTION\s*\d+\s*:\s*(.+?)\n"  # question text
            r"\s*a\)\s*(.+?)\n"                # option a
            r"\s*b\)\s*(.+?)\n"                # option b
            r"\s*c\)\s*(.+?)\n"                # option c
            r"\s*d\)\s*(.+?)\n"                # option d
            r"\s*RESPONSE\s*:\s*([abcd])\s*\n?"   # correct answer
            r"\s*EXPLANATION\s*:\s*(.+?)(?=\n\s*QUESTION|\Z)"  # explanation
        )

        matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            q_text, opt_a, opt_b, opt_c, opt_d, correct, explanation = match
            questions.append({
                "question": q_text.strip(),
                "options": [
                    opt_a.strip(),
                    opt_b.strip(),
                    opt_c.strip(),
                    opt_d.strip(),
                ],
                "correct_answer": correct.strip().lower(),
                "explanation": explanation.strip(),
            })

        return questions
