from pydantic import BaseModel
from typing import Optional


class StudentRequest(BaseModel):
    message: str
    intent: Optional[str] = None


class ExamQuestion(BaseModel):
    """Una pregunta de examen tipo test."""
    question: str
    options: list[str]         # 4 opciones (a, b, c, d)
    correct_answer: str        # letra correcta: "a", "b", "c" o "d"
    explanation: str           # breve explicación de por qué es correcta
