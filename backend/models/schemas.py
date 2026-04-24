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
    topic:           Optional[str] = None

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

class SubmitAnswerRequest(BaseModel):
    topic:          str
    section_title:  str
    question_index: int
    user_answer:    str   # letter: "a", "b", "c" or "d"
    user_correct:   bool

class RestartWithSuggestionsRequest(BaseModel):
    document:    str
    suggestions: str


class FinalExamGenerateRequest(BaseModel):
    topic: str
    mode:  str = "study"  # "study" or "exam"


class FinalExamSubmitRequest(BaseModel):
    topic:   str
    answers: list  # [{question_index: int, user_answer: str}]
