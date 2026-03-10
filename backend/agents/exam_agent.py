"""Exam Agent: generates multiple-choice exam questions based on a specific file."""

import os
import re

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest
from backend.clients.llm_client import chat_with_model


class ExamAgent(BaseAgent):
    """Generates multiple-choice exam questions based on a specific file."""

    def __init__(self, llm_model="llama-3.3-70b-versatile", num_questions=3, data_dir="backend/data"):
        self.llm_model = llm_model
        self.num_questions = num_questions
        self.data_dir = data_dir

    def can_handle(self, intent: str) -> bool:
        return intent == "exam"

    def handle(self, request: StudentRequest) -> dict:
        """Generate an exam based on the specified file.
        
        Args:
            request.message: Message with keywords + filename
                           e.g. "examen redes_neuronales" or "test bases_datos.txt"
        
        Returns:
            dict with content (raw text) and questions (parsed)
        """
        # Extract the filename from the message, ignoring keywords
        filename = self._extract_filename(request.message)
        
        if not filename:
            return {
                "agent": "exam",
                "error": f"No file specified. Usage: 'examen redes_neuronales' or 'exam bases_datos'. Available: {self._list_available_files()}",
            }
        
        # Ensure .txt extension
        if not filename.endswith(".txt"):
            filename = filename + ".txt"
        
        # Build the file path
        filepath = os.path.join(self.data_dir, filename)
        
        # Check that the file exists
        if not os.path.exists(filepath):
            return {
                "agent": "exam",
                "error": f"File not found: {filename}. Available: {self._list_available_files()}",
            }
        
        # Read the file contents
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return {
                "agent": "exam",
                "error": f"Error reading file: {e}",
            }
        
        if not content.strip():
            return {
                "agent": "exam",
                "error": f"File {filename} is empty.",
            }
        
        # Generate questions based on the full file content
        raw_answer = self._generate_questions(content, filename)
        
        if raw_answer is None:
            return {
                "agent": "exam",
                "error": "Could not generate the exam.",
            }
        
        # Parse questions into structured format
        questions = self._parse_questions(raw_answer)
        
        return {
            "agent": "exam",
            "content": raw_answer,
            "questions": questions,
            "num_questions": len(questions),
            "source_file": filename,
        }

    def _extract_filename(self, message: str) -> str | None:
        """Extract the filename from the message, ignoring keywords.
        
        Examples:
          "examen redes_neuronales" → "redes_neuronales"
          "exam bases_datos.txt"    → "bases_datos.txt"
          "test sistemas_operativos" → "sistemas_operativos"
          "pregunta tipo redes_neuronales" → "redes_neuronales"
        """
        # Keywords to strip
        keywords = ["examen", "exam", "test", "pregunta tipo", "pregunta", "tipo"]
        
        # Lowercase for matching
        message_lower = message.lower().strip()
        
        # Remove the keyword from the start of the message
        for kw in keywords:
            if message_lower.startswith(kw):
                # Strip the keyword and surrounding whitespace
                message = message[len(kw):].strip()
                message_lower = message.lower().strip()
        
        # The remainder should be the filename
        if message:
            return message
        return None

    def _list_available_files(self) -> str:
        """List available files in the data directory."""
        try:
            files = [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]
            return ", ".join(files)
        except:
            return "Could not list available files"

    def _generate_questions(self, content: str, filename: str) -> str | None:
        """Call the LLM to generate multiple-choice questions based on the file content."""

        sistema = (
            "Eres un profesor universitario experto en crear exámenes. "
            "Tu tarea es generar preguntas de tipo test (opción múltiple) "
            "basándote ÚNICAMENTE en el contenido proporcionado. "
            "Cada pregunta debe tener exactamente 4 opciones (a, b, c, d) "
            "y solo una respuesta correcta."
        )

        prompt_usuario = f"""Contenido del archivo '{filename}':
---
{content}
---

Genera exactamente {self.num_questions} preguntas de examen tipo test.

Para CADA pregunta usa este formato exacto:

PREGUNTA 1: [texto de la pregunta]
a) [opción a]
b) [opción b]
c) [opción c]
d) [opción d]
RESPUESTA: [letra correcta]
EXPLICACIÓN: [breve explicación de por qué es correcta]

Asegúrate de que:
- Las preguntas cubran diferentes aspectos del contenido
- Las opciones incorrectas sean plausibles pero claramente distinguibles
- Las explicaciones sean concisas y útiles para el aprendizaje"""

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
            r"PREGUNTA\s*\d+\s*:\s*(.+?)\n"  # question text
            r"\s*a\)\s*(.+?)\n"                # option a
            r"\s*b\)\s*(.+?)\n"                # option b
            r"\s*c\)\s*(.+?)\n"                # option c
            r"\s*d\)\s*(.+?)\n"                # option d
            r"\s*RESPUESTA\s*:\s*([abcd])\s*\n?"   # correct answer
            r"\s*EXPLICACI[\xd3O]N\s*:\s*(.+?)(?=\n\s*PREGUNTA|\Z)"  # explanation
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
