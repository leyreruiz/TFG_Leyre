"""Summary Agent: generates structured summaries using RAG + LLM."""

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest
from backend.clients.llm_client import chat_with_model


class SummaryAgent(BaseAgent):

    def __init__(self, retriever, llm_model="llama-3.3-70b-versatile"):
        self.retriever = retriever
        self.llm_model = llm_model

    def can_handle(self, intent: str) -> bool:
        return intent == "summary"

    def handle(self, request: StudentRequest) -> dict:
        # For summaries, use more documents to have broad context
        docs = self.retriever.search(request.message, k=5)
        context = "\n\n".join(docs) if docs else "(No relevant documents found)"

        sistema = (
            "You are a university professor expert and pedagogic "
            "Your objective is to help a student to learn a topic deeply.\n\n"
            "Rules:\n"
            "1. Base your answer ONLY on the context provided\n"
            "2. Structure the exam with clear sections using headings.\n"
            "3. Include the key concepts and their definitions.\n"
            "4. If relevant, mention relationships between concepts.\n"
            "5. If the information is not in context, state this explicitly.\n"
            "6. Use a clear and accessible tone for a university student."
        )

        prompt_usuario = f"""Documents' context:
---
{context}
---

Student's question:
{request.message}

Generate a clear, structured summary based on the preceding context.
Use headings, lists, and definitions where appropriate."""

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
                "error": "Could not generate a response from the model.",
            }
