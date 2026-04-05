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


# ── API request models ────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    document: str

class RegenerateSummaryRequest(BaseModel):
    topic: Optional[str] = None
    section_title: str

class RegenerateExamRequest(BaseModel):
    topic: Optional[str] = None
    section_title: str
    section_summary: Optional[str] = None

class AskRequest(BaseModel):
    question:        str
    section_title:   str
    section_summary: str
    conversation_id: Optional[str] = None

class AddQuestionsRequest(BaseModel):
    topic:              str
    section_title:      str
    section_summary:    str
    existing_questions: list = []
    num_questions:      int  = 3

class UpdateQuestionsRequest(BaseModel):
    topic:         str
    section_title: str
    questions:     list
