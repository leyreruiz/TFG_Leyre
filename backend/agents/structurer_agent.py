"""Structurer Agent: analyzes a document and produces a class structure/outline.

Given a document, retrieves all its chunks from ChromaDB and asks the LLM
to generate a structured class outline with sections and key topics.
"""

import os

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest
from backend.clients.llm_client import chat_with_model
from backend.utils import extract_search_term


class StructurerAgent(BaseAgent):
    """Analyzes a document and produces a structured class outline."""

    def __init__(self, retriever, llm_model="llama-3.1-8b-instant", data_dir="backend/data"):
        self.retriever = retriever
        self.llm_model = llm_model
        self.data_dir = data_dir

    def handle(self, request: StudentRequest, suggestions: str = "") -> dict:
        """Generate the class outline, optionally incorporating user suggestions.

        Args:
            request: StudentRequest with the document/topic to structure.
            suggestions: Optional user feedback to incorporate into the outline.
        """
        term = extract_search_term(request.message, intent="structure")

        if not term:
            return {"agent": "structurer", "error": "No file specified."}

        # Resolve the term to a known source file (any supported format);
        # use full-file retrieval if found, otherwise fall back to similarity search.
        candidate_file = self._resolve_source_file(term)

        if candidate_file:
            docs = self.retriever.search_by_source(candidate_file)
            source = candidate_file
        else:
            docs = self.retriever.search(term, k=15)
            source = term

        if not docs:
            return {
                "agent": "structurer",
                "error": f"No information found about '{term}'. "
                         f"Available files: {', '.join(self._list_known_files())}",
            }

        context = "\n\n".join(docs)
        structure = self._generate_structure(context, source, suggestions=suggestions)

        if structure is None:
            return {"agent": "structurer", "error": f"No information found about '{term}'."}

        return {"agent": "structurer", "content": structure, "source": source}

    # Document formats that can be resolved to a source file for full-file retrieval
    _SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".pptx", ".docx")

    def _list_known_files(self) -> list[str]:
        """Return the list of supported source filenames available in the data directory."""
        try:
            return [
                f for f in os.listdir(self.data_dir)
                if os.path.splitext(f)[1].lower() in self._SUPPORTED_EXTENSIONS
            ]
        except Exception:
            return []

    def _resolve_source_file(self, term: str) -> str | None:
        """Find the source file matching a term, regardless of its format.

        The term may already include an extension, or be the bare topic name
        (e.g. 'neural_networks' → 'neural_networks.pdf'). Returns the matching
        filename as stored in ChromaDB metadata, or None if no file matches.
        """
        known_files = self._list_known_files()
        if term in known_files:
            return term
        for extension in self._SUPPORTED_EXTENSIONS:
            if f"{term}{extension}" in known_files:
                return f"{term}{extension}"
        return None

    def _generate_structure(self, content: str, source: str, suggestions: str = "") -> str | None:
        """Call the LLM to produce a class outline from the document content."""
        system = (
            "You are a university professor creating a class outline. "
            "Generate a clean, high-level structure that serves as a table of contents — "
            "section titles and brief topic names only. "
            "No explanations, no definitions, no details. Those come later when each section is taught."
        )

        suggestions_block = (
            f"\n\nUSER SUGGESTIONS TO INCORPORATE:\n{suggestions.strip()}"
            if suggestions and suggestions.strip()
            else ""
        )

        prompt = f"""Document: '{source}'
    ---
    {content}
    ---{suggestions_block}

    Generate a high-level class outline using this format:

    ## [Class Title]

    ### 1. [Section Title]
    - [Topic name, no explanation]
    - [Topic name, no explanation]

    ### 2. [Section Title]
    - [Topic name, no explanation]

    RULES:
    - Topic names should be short: 2-5 words max
    - No definitions, no descriptions, no "how" or "why" phrases
    - This is a table of contents, not a lesson plan
    - Cover all main topics in the document
    {"- Apply the user suggestions listed above" if suggestions and suggestions.strip() else ""}"""

        return chat_with_model(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model=self.llm_model,
            temperature=0.4,
        )
