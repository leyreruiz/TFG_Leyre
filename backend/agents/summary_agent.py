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
        # Para resúmenes usamos más documentos para tener contexto amplio
        docs = self.retriever.search(request.message, k=5)
        context = "\n\n".join(docs) if docs else "(No se encontraron documentos relevantes)"

        sistema = (
            "Eres un tutor universitario experto y pedagógico. "
            "Tu objetivo es ayudar al estudiante a entender el tema en profundidad.\n\n"
            "Reglas:\n"
            "1. Basa tu respuesta ÚNICAMENTE en el contexto proporcionado.\n"
            "2. Estructura el resumen con secciones claras usando encabezados.\n"
            "3. Incluye los conceptos clave y sus definiciones.\n"
            "4. Si es relevante, menciona relaciones entre conceptos.\n"
            "5. Si la información no está en el contexto, dilo explícitamente.\n"
            "6. Usa un tono claro y accesible para un estudiante universitario."
        )

        prompt_usuario = f"""Contexto de documentos:
---
{context}
---

Pregunta del estudiante:
{request.message}

Genera un resumen estructurado y claro basándote en el contexto anterior.
Usa encabezados, listas y definiciones cuando sea apropiado."""

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
