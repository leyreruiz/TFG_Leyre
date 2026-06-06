from abc import ABC, abstractmethod
from backend.models.schemas import StudentRequest


class BaseAgent(ABC):

    @abstractmethod
    def handle(self, request: StudentRequest) -> dict:
        raise NotImplementedError()
