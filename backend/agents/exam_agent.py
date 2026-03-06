"""Exam Agent: genera preguntas de examen tipo test usando RAG + LLM."""

import json
import re

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest, ExamQuestion
from backend.clients.llm_client import chat_with_model


class ExamAgent(BaseAgent):
    """Genera preguntas de examen tipo test basadas en documentos recuperados."""

    def __init__(self, retriever, llm_model="llama3.2", num_questions=3):
        self.retriever = retriever
        self.llm_model = llm_model
        self.num_questions = num_questions

    def can_handle(self, intent: str) -> bool:
        return intent == "exam"

    def handle(self, request: StudentRequest) -> dict:
        # Para exámenes usamos menos documentos pero más focalizados
        docs = self.retriever.search(request.message, k=3)
        context = "\n\n".join(docs) if docs else "(No se encontraron documentos relevantes)"

        if not docs:
            return {
                "agent": "exam",
                "error": "No se encontraron documentos para generar el examen.",
            }

        raw_answer = self._generate_questions(context, request.message)

        if raw_answer is None:
            return {
                "agent": "exam",
                "error": "No se pudo generar el examen.",
            }

        # Intentamos parsear las preguntas a formato estructurado
        questions = self._parse_questions(raw_answer)

        return {
            "agent": "exam",
            "content": raw_answer,
            "questions": questions,
            "num_questions": len(questions),
        }

    def _generate_questions(self, context: str, topic: str) -> str | None:
        """Llama al LLM para generar preguntas tipo test."""

        sistema = (
            "Eres un profesor universitario experto en crear exámenes. "
            "Tu tarea es generar preguntas de tipo test (opción múltiple) "
            "basándote ÚNICAMENTE en el contexto proporcionado. "
            "Cada pregunta debe tener exactamente 4 opciones (a, b, c, d) "
            "y solo una respuesta correcta."
        )

        prompt_usuario = f"""Contexto de documentos:
---
{context}
---

Tema solicitado por el estudiante:
{topic}

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
- Las preguntas cubran diferentes aspectos del tema
- Las opciones incorrectas sean plausibles pero claramente distinguibles
- Las explicaciones sean concisas y útiles para el aprendizaje"""

        return chat_with_model(
            messages=[
                {"role": "system", "content": sistema},
                {"role": "user", "content": prompt_usuario},
            ],
            model=self.llm_model,
            temperature=0.5,  # Menos creatividad para exámenes más precisos
        )

    def _parse_questions(self, raw_text: str) -> list[dict]:
        """Intenta parsear la respuesta del LLM a una lista estructurada.

        Usa regex para extraer cada pregunta con sus opciones y respuesta.
        Si falla el parseo, devuelve lista vacía (el texto raw sigue disponible).
        """
        questions = []

        # Patrón para encontrar cada bloque de pregunta
        pattern = (
            r"PREGUNTA\s*\d+\s*:\s*(.+?)\n"  # pregunta
            r"\s*a\)\s*(.+?)\n"                # opción a
            r"\s*b\)\s*(.+?)\n"                # opción b
            r"\s*c\)\s*(.+?)\n"                # opción c
            r"\s*d\)\s*(.+?)\n"                # opción d
            r"\s*RESPUESTA\s*:\s*([abcd])\s*\n?"   # respuesta correcta
            r"\s*EXPLICACI[ÓO]N\s*:\s*(.+?)(?=\n\s*PREGUNTA|\Z)"  # explicación
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
