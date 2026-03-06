"""Summary Agent: genera resúmenes estructurados usando RAG + LLM."""

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest
from backend.clients.llm_client import chat_with_model


class SummaryAgent(BaseAgent):

    def __init__(self, retriever, llm_model="llama3.2"):
        self.retriever = retriever
        self.llm_model = llm_model

    def can_handle(self, intent: str) -> bool:
        return intent == "summary"

    def handle(self, request: StudentRequest) -> dict:
        # Buscar documentos relevantes
        docs = self.retriever.search(request.message, k=5)
        context = "\n\n".join(docs) if docs else "(No se encontraron documentos relevantes)"

        sistema = (
            "Eres un tutor universitario experto. "
            "Tu tarea es construir un resumen estructurado y claro "
            "basándote únicamente en el contexto proporcionado. "
            "Si la información no está en el contexto, indícalo."
        )

        prompt_usuario = f"""Contexto de documentos:
---
{context}
---

Pregunta del estudiante:
{request.message}

Por favor, genera un resumen estructurado y claro basándote en el contexto anterior."""

        answer = chat_with_model(
            messages=[
                {"role": "system", "content": sistema},
                {"role": "user", "content": prompt_usuario},
            ],
            model=self.llm_model,
            temperature=0.7,
        )

        if answer:
            return {
                "agent": "summary",
                "content": answer,
            }
        else:
            return {
                "agent": "summary",
                "error": "No se pudo generar la respuesta del modelo.",
            }
