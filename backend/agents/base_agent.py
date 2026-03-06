from abc import ABC, abstractmethod
from backend.models.schemas import StudentRequest


class BaseAgent(ABC):

    @abstractmethod
    def can_handle(self, intent: str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def handle(self, request: StudentRequest) -> dict:
        raise NotImplementedError()
