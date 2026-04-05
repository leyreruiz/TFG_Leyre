"""Summary Agent: generates structured summaries using RAG + LLM."""

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest
from backend.clients.llm_client import chat_with_model
from backend.utils import extract_search_term


class SummaryAgent(BaseAgent):

    def __init__(self, retriever, llm_model="llama-3.1-8b-instant"):
        self.retriever = retriever
        self.llm_model = llm_model

    def can_handle(self, intent: str) -> bool:
        return intent == "summary"

    def handle(self, request: StudentRequest) -> dict:
        # Strip intent words so ChromaDB gets a clean topic query
        search_query = extract_search_term(request.message, intent="summary") or request.message
        # For summaries, use more documents to have broad context
        docs = self.retriever.search(search_query, k=5)
        context = "\n\n".join(docs) if docs else "(No relevant documents found)"

        system = (
            "You are a university professor who excels at explaining complex topics "
            "in a clear, engaging, and pedagogical way.\n\n"
            "Your goal is to help a student truly understand a topic — not just memorize it.\n\n"
            "Rules:\n"
            "1. Base your answer ONLY on the context provided.\n"
            "2. Alternate between prose and lists naturally: explain the 'why' and context in paragraphs, "
            "then use bullet points or numbered lists to summarize, enumerate steps, or highlight key terms.\n"
            "3. Never write more than 3–4 consecutive bullet points without a short explanatory paragraph before or after.\n"
            "4. Never write more than 3–4 consecutive paragraphs without breaking with a list, heading, or visual element.\n"
            "5. Define key concepts inline within prose the first time they appear.\n"
            "6. Use analogies or simple examples to make abstract ideas concrete.\n"
            "7. If the information is not in context, say so explicitly.\n"
            "8. Keep the tone approachable and direct — like a good teacher, not a textbook."
        )

        user_prompt = f"""Documents' context:
        ---
        {context}
        ---

        Student's question:
        {request.message}

        Write a summary based on the context above that mixes prose and lists naturally.

        Formatting guidelines:
        - Use `#` for the main title and `##` for sections.
        - Introduce each section with 1–2 sentences of context or explanation (prose).
        - Follow with a bullet list or numbered list to highlight key points, steps, or terms.
        - Add a closing sentence or short paragraph after lists when the idea needs connecting to the bigger picture.
        - Bold (**bold**) key terms the first time they appear.
        - End with a "## Key Takeaway" section: 2–3 sentences summarizing the core idea."""

        answer = chat_with_model(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
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
