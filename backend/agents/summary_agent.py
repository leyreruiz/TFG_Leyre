"""Summary Agent: generates structured summaries using RAG + LLM."""

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest
from backend.clients.llm_client import chat_with_model
from backend.utils import extract_search_term


class SummaryAgent(BaseAgent):

    def __init__(self, retriever, llm_model="llama-3.1-8b-instant"):
        self.retriever = retriever
        self.llm_model = llm_model

    def handle(self, request: StudentRequest) -> dict:
        """Entry point used by the orchestrator (no outline available)."""
        search_query = extract_search_term(request.message, intent="summary") or request.message
        return self.summarize(section_title=search_query, section_outline="")

    def summarize(self, section_title: str, section_outline: str) -> dict:
        """Generate a summary guided by the section's outline from the class index.

        Args:
            section_title:   Title of the section to summarise.
            section_outline: The block from the class index that belongs to this section
                             (e.g. the bullet points listed under the ### heading).
                             Pass an empty string if not available.
        """
        # Retrieve relevant document chunks for this section
        docs = self.retriever.search(section_title, k=5)
        context = "\n\n".join(docs) if docs else "(No relevant documents found)"

        system = (
            "You are a university professor who excels at explaining complex topics "
            "in a clear, engaging, and pedagogical way.\n\n"
            "Your goal is to help a student truly understand a topic — not just memorize it.\n\n"
            "Rules:\n"
            "1. Base your answer ONLY on the context provided.\n"
            "2. Follow the structure of the outline when one is provided.\n"
            "3. Alternate between prose and lists naturally.\n"
            "4. Define key concepts inline within prose the first time they appear.\n"
            "5. Use analogies or simple examples to make abstract ideas concrete.\n"
            "6. If the information is not in context, say so explicitly.\n"
            "7. Keep the tone approachable and direct — like a good teacher, not a textbook."
        )

        # Include the outline block only when one is available
        outline_block = (
            f"\nClass index for this section (follow this structure):\n{section_outline}\n"
            if section_outline else ""
        )

        user_prompt = f"""Documents' context:
---
{context}
---
{outline_block}
Write a summary of '{section_title}' based on the context above.
{"Follow the outline structure above: cover each point listed in the same order." if section_outline else ""}

Formatting guidelines:
- Use `##` for each subsection that appears in the outline (or for natural topic breaks if no outline).
- Introduce each section with 1–2 sentences of prose, then use bullet points for key ideas.
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
            return {"agent": "summary", "content": answer}
        else:
            return {"agent": "summary", "error": "Could not generate a response from the model."}
