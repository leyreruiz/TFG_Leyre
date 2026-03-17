"""Structurer Agent: analyzes a document and produces a class structure/outline.

Given a document, retrieves all its chunks from ChromaDB and asks the LLM
to generate a structured class outline with sections and key topics.
"""

import os

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest
from backend.clients.llm_client import chat_with_model


class StructurerAgent(BaseAgent):
    """Analyzes a document and produces a structured class outline."""

    def __init__(self, retriever, llm_model="llama-3.1-8b-instant", data_dir="backend/data"):
        self.retriever = retriever
        self.llm_model = llm_model
        self.data_dir = data_dir

    def can_handle(self, intent: str) -> bool:
        return intent == "structure"

    def handle(self, request: StudentRequest) -> dict:
        term = self._extract_term(request.message)

        if not term:
            return {
                "agent": "structurer",
                "error": "No file specified.",
            }

        candidate_file = term if term.endswith(".txt") else term + ".txt"
        known_files = self._list_known_files()

        if candidate_file in known_files:
            docs = self.retriever.search_by_source(candidate_file)
            source = candidate_file
        else:
            docs = self.retriever.search(term, k=15)
            source = term

        if not docs:
            return {
                "agent": "structurer",
                "error": f"No information found about '{term}'. "
                         f"Available files: {', '.join(known_files)}",
            }

        context = "\n\n".join(docs)
        structure = self._generate_structure(context, source)

        if structure is None:
            return {
                "agent": "structurer",
                "error": f"No information found about '{term}'.",
            }

        return {
            "agent": "structurer",
            "content": structure,
            "source": source,
        }

    def _extract_term(self, message: str) -> str | None:
        keywords = ["estructura", "structure", "preparar", "clase", "prepare"]
        msg = message.strip()
        msg_lower = msg.lower()
        for kw in keywords:
            if msg_lower.startswith(kw):
                msg = msg[len(kw):].strip()
                msg_lower = msg.lower()
        return msg if msg else None

    def _list_known_files(self) -> list[str]:
        try:
            return [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]
        except Exception:
            return []

    def _generate_structure(self, content: str, source: str) -> str | None:
        system = (
            "You are a university professor expert in preparing classes. "
            "Your task is to analyze the provided content and generate a structured "
            "class outline with sections, subsections, and key concepts for each part. "
            "The structure should serve as a guide for delivering a comprehensive class on the topic. "
            "IMPORTANT: Only generate the main structure. Do not include any additional outlines, "
            "schedules, or supplementary information."
        )

        prompt = f"""Content of the document '{source}':
---
{content}
---

Generate ONLY a class structure based on the previous content with this format:

## Class Title

### 1. Main Section Title
- Key Concept 1
- Key Concept 2
- Key Concept 3

### 2. Next Section Title
- Key Concept 4
- Key Concept 5

### 3. Another Section
- Key Concept 6
- Key Concept 7

Continue until you have covered all the main topics in the content.

IMPORTANT INSTRUCTIONS:
- Each section must start with "### " followed by a number and title
- Use bullet points for key concepts under each section
- Do NOT include timing information
- Do NOT include reading assignments
- Stop after the last main section - do not add any additional content"""

        return chat_with_model(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model=self.llm_model,
            temperature=0.4,
        )
