from pydantic import BaseModel
from typing import Optional


class StudentRequest(BaseModel):
    message: str
    intent: Optional[str] = None


class ExamQuestion(BaseModel):
    """A multiple-choice exam question."""
    question: str
    options: list[str]         # 4 options (a, b, c, d)
    correct_answer: str        # correct letter: "a", "b", "c" or "d"
    explanation: str           # brief explanation of why the answer is correct
