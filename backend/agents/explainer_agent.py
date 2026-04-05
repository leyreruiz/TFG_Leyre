"""Explainer Agent: answers student questions grounded in the current section's summary.

It does NOT perform RAG — it reuses the summary already generated for the section,
so it is fast and contextually accurate.
"""

from typing import Optional

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest
from backend.clients.llm_client import chat_with_model


class ExplainerAgent(BaseAgent):

    def __init__(self, llm_model="llama-3.1-8b-instant", max_history_turns: int = 6, max_conversations: int = 200):
        self.llm_model = llm_model
        self.max_history_messages = max(2, max_history_turns * 2)
        self.max_conversations = max(1, max_conversations)
        self.histories: dict[str, list[dict[str, str]]] = {}

    def can_handle(self, intent: str) -> bool:
        return intent == "explain"

    def _prune_conversations(self) -> None:
        while len(self.histories) > self.max_conversations:
            oldest_key = next(iter(self.histories))
            self.histories.pop(oldest_key, None)

    def _append_turn(self, conversation_id: str, user_message: str, assistant_message: str) -> None:
        history = self.histories.setdefault(conversation_id, [])
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})
        if len(history) > self.max_history_messages:
            self.histories[conversation_id] = history[-self.max_history_messages :]

        self._prune_conversations()

    def handle(
        self,
        request: StudentRequest,
        section_context: str = "",
        conversation_id: Optional[str] = None,
    ) -> dict:
        conversation_key = (conversation_id or "default").strip() or "default"

        system = (
            "You are a university professor explaining concepts to a student during class. "
            "Your explanations are conversational and flowing — you write in natural paragraphs, "
            "like you're speaking directly to someone, not writing a report. "
            "Avoid bullet points, numbered lists, or headers. Instead, build your explanation "
            "step by step through connected sentences and paragraphs. "
            "Use analogies and plain language when helpful. Be thorough but don't over-explain. "
            "If the question goes beyond the covered material, acknowledge it briefly and answer "
            "from general knowledge, making clear it's outside the section."
        )

        messages = [{"role": "system", "content": system}]

        if section_context.strip():
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Current section context that must be used as primary reference:\n"
                        f"---\n{section_context}\n---"
                    ),
                }
            )

        history = self.histories.get(conversation_key, [])
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": request.message})

        answer = chat_with_model(
            messages=messages,
            model=self.llm_model,
            temperature=0.6,
        )

        if answer:
            self._append_turn(conversation_key, request.message, answer)
            return {"agent": "explainer", "content": answer}
        else:
            return {"agent": "explainer", "error": "Could not generate a response."}