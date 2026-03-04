from pydantic import BaseModel
from typing import Optional


class StudentRequest(BaseModel):
    message: str
    intent: Optional[str] = None
