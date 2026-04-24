"""Explainer Agent: answers student questions grounded in the current section's summary.

It does NOT perform RAG — it reuses the summary already generated for the section,
so it is fast and contextually accurate.
"""

import logging
import re
from typing import Optional

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest
from backend.clients.llm_client import chat_with_model

logger = logging.getLogger(__name__)


class ExplainerAgent(BaseAgent):

    def __init__(self, llm_model="llama-3.1-8b-instant", max_history_turns: int = 6, max_conversations: int = 200):
        self.llm_model = llm_model
        # Each turn = 1 user message + 1 assistant message → multiply by 2
        self.max_history_messages = max(2, max_history_turns * 2)
        self.max_conversations = max(1, max_conversations)
        # conversation_id → list of {"role": ..., "content": ...} messages
        self.histories: dict[str, list[dict[str, str]]] = {}

    def can_handle(self, intent: str) -> bool:
        return intent == "explain"

    def _prune_conversations(self) -> None:
        """Remove the oldest conversation if the limit is exceeded."""
        while len(self.histories) > self.max_conversations:
            oldest_key = next(iter(self.histories))
            self.histories.pop(oldest_key, None)

    def clear_conversation(self, conversation_id: str) -> None:
        self.histories.pop(conversation_id, None)

    def _append_turn(self, conversation_id: str, user_message: str, assistant_message: str) -> None:
        """Add a turn to the history and trim to the maximum window size."""
        history = self.histories.setdefault(conversation_id, [])
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})
        # Keep only the most recent N messages
        if len(history) > self.max_history_messages:
            self.histories[conversation_id] = history[-self.max_history_messages:]
        self._prune_conversations()

    def handle(
        self,
        request: StudentRequest,
        section_context: str = "",
        conversation_id: Optional[str] = None,
        db_context: str = "",
        wikipedia_context: str = "",
        topic: str = "",
        section: str = "",
    ) -> dict:
        conversation_key = (conversation_id or "default").strip() or "default"

        # Build hints so the LLM knows what subject/section is active
        topic_hint = f" The subject being studied is: '{topic}'." if topic.strip() else ""
        section_hint = f" The active section is: '{section}'." if section.strip() else ""

        system = (
            "You are a university professor answering student questions during class. "
            f"{topic_hint}{section_hint} "
            "You should answer questions that are related to the subject being studied, "
            "including tangential or broader questions that connect to the topic in a meaningful way. "
            "For example, if the topic is 'neural networks', questions about related math concepts, "
            "historical context, practical applications, or comparisons with other methods are all welcome. "
            "Only refuse questions that are clearly and completely unrelated to the academic subject "
            "(e.g., cooking recipes, sports scores, personal advice). "
            "If you must refuse, respond with a short friendly sentence suggesting the student ask about the course material, "
            "and end your response with exactly: [REFUSED] "
            "When the student uses vague references such as 'the examples', 'the theorem', 'that concept', 'explain it', etc., "
            "always interpret them as referring to the content of the current section summary provided below. "
            "IMPORTANT: Always respond in the same language the student used in their question. "
            "When the question is relevant, explain in a conversational and flowing way, in natural paragraphs, "
            "as if speaking directly to the student. Avoid bullet points, numbered lists, and headers. "
            "Use analogies and plain language. Be thorough but don't over-explain. "
            "When answering, prioritize sources in this order: "
            "1) the section summary, 2) the course knowledge base, 3) Wikipedia. "
            "Only use Wikipedia if the higher-priority sources are insufficient. "
            "At the very end of your response, on a new line, add exactly: "
            "[Source: section] if you only used the section material or knowledge base, "
            "[Source: wikipedia] if you only used Wikipedia, or "
            "[Source: both] if you used both."
        )

        # Start with the system instruction
        messages = [{"role": "system", "content": system}]

        # Append available context blocks (section summary, DB results, Wikipedia)
        if section_context.strip():
            messages.append({
                "role": "system",
                "content": (
                    "Current section context that must be used as primary reference:\n"
                    f"---\n{section_context}\n---"
                ),
            })

        if db_context.strip():
            messages.append({
                "role": "system",
                "content": (
                    "Additional context retrieved from the course knowledge base "
                    "(use this as a high-priority reference, after the section summary):\n"
                    f"---\n{db_context}\n---"
                ),
            })

        if wikipedia_context.strip():
            messages.append({
                "role": "system",
                "content": (
                    "Additional context from Wikipedia "
                    "(use only if the section summary and knowledge base do not sufficiently cover the question):\n"
                    f"---\n{wikipedia_context}\n---"
                ),
            })

        # Append conversation history, then the current user question
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
            # Detect refusal — do not persist refused answers to history
            if re.search(r"\[REFUSED\]", answer, re.IGNORECASE):
                answer = re.sub(r"\s*\[REFUSED\]", "", answer, flags=re.IGNORECASE).rstrip()
                logger.info("Explainer refused off-topic question: '%s'", request.message[:80])
                return {"agent": "explainer", "content": answer, "refused": True}

            # Extract and strip the [Source: ...] tag before storing/returning
            match = re.search(r"\[Source:\s*(section|wikipedia|both)\]", answer, re.IGNORECASE)
            if match:
                source = match.group(1).lower()
                logger.info("Explainer source for '%s': %s", request.message[:80], source)
                answer = answer[: match.start()].rstrip()
            else:
                source = "section"
                logger.warning("Explainer did not include a source tag for: '%s'", request.message[:80])

            self._append_turn(conversation_key, request.message, answer)
            return {"agent": "explainer", "content": answer, "source": source}
        else:
            return {"agent": "explainer", "error": "Could not generate a response."}
