"""LangGraph workflow for class preparation.

Pipeline:
  1. StructurerAgent  → generates a class outline with numbered sections
  2. Parse sections   → extracts each section title from the outline
  3. For each section (loop):
     a. SummaryAgent  → summarizes the section topic
     b. ExamAgent     → generates exam questions for the section

Results are cached in backend/cache/<document>.json so the LLM is only called once per document.
"""

import json
import operator
import os
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from backend.agents.structurer_agent import StructurerAgent
from backend.agents.summary_agent import SummaryAgent
from backend.agents.exam_agent import ExamAgent
from backend.models.schemas import StudentRequest
from backend.rag.retriever import ChromaDbRetriever


#State definition

#Information shared across the pipeline steps is stored in the state, which is passed from node to node.
class PrepareClassState(TypedDict):
    document: str                                          # input topic / filename
    structure: str                                         # raw outline from StructurerAgent
    sections: list[str]                                    # parsed section titles
    current_index: int                                     # loop counter
    current_summary: str                                   # summary of the section being processed
    section_results: Annotated[list[dict], operator.add]   # per-section results (appended)
    error: str                                             # error message if any


#Shared agents

_retriever = ChromaDbRetriever()
_structurer = StructurerAgent(_retriever)
_summary = SummaryAgent(_retriever)
_exam = ExamAgent(_retriever, num_questions=2)


#Node functions

def generate_structure(state: PrepareClassState) -> dict:
    """Call the StructurerAgent to produce the class outline."""
    print("[Graph] Step 1: generating class structure...")
    request = StudentRequest(message=state["document"], intent="structure")
    result = _structurer.handle(request)

    if "error" in result:
        return {"error": result["error"]}
    return {"structure": result["content"]}


def parse_sections(state: PrepareClassState) -> dict:
    """Extract individual section titles from the markdown outline."""
    structure = state.get("structure", "")

    # Match lines like "### 1. Section title" or "### Section title"
    pattern = r"###\s*(?:\d+\.\s*)?(.+)"
    matches = re.findall(pattern, structure)
    sections = [m.strip() for m in matches if m.strip()]

    if not sections:
        return {"sections": [], "error": "No sections could be parsed from the structure."}

    print(f"[Graph] Parsed {len(sections)} sections: {sections}")
    return {"sections": sections, "current_index": 0}


def summarize_section(state: PrepareClassState) -> dict:
    """Generate a summary for the current section."""
    idx = state["current_index"]
    section = state["sections"][idx]
    total = len(state["sections"])

    print(f"[Graph] Summarizing section {idx + 1}/{total}: {section}")

    summary_req = StudentRequest(message=f"resume {section}", intent="summary")
    summary_result = _summary.handle(summary_req)
    content = summary_result.get("content", summary_result.get("error", ""))

    return {"current_summary": content}


def examine_section(state: PrepareClassState) -> dict:
    """Generate exam questions based on the summary of the current section."""
    idx = state["current_index"]
    section = state["sections"][idx]
    summary = state["current_summary"]
    total = len(state["sections"])

    print(f"[Graph] Generating exam for section {idx + 1}/{total}: {section}")

    exam_req = StudentRequest(
        message=f"exam {section}\n\nusing the following summary context:\n{summary}",
        intent="exam",
    )
    exam_result = _exam.handle(exam_req)

    section_data = {
        "title": section,
        "summary": summary,
        "exam": exam_result.get("content", exam_result.get("error", "")),
        "questions": exam_result.get("questions", []),
    }

    return {"section_results": [section_data], "current_index": idx + 1}


# Routing functions

def after_structure(state: PrepareClassState) -> str:
    """Stop on error, otherwise continue to parsing."""
    if state.get("error"):
        return END
    return "parse_sections"


def after_parse(state: PrepareClassState) -> str:
    """Stop if no sections were found, otherwise start summarizing."""
    if not state.get("sections"):
        return END
    return "summarize_section"


def next_section(state: PrepareClassState) -> str:
    """Loop back if more sections remain, otherwise finish."""
    if state["current_index"] < len(state["sections"]):
        return "summarize_section"
    return END


# Graph assembly

def build_prepare_class_graph():
    """Build and compile the LangGraph state machine."""
    graph = StateGraph(PrepareClassState)

    graph.add_node("generate_structure", generate_structure)
    graph.add_node("parse_sections", parse_sections)
    graph.add_node("summarize_section", summarize_section)
    graph.add_node("examine_section", examine_section)

    graph.add_edge(START, "generate_structure")
    graph.add_conditional_edges(
        "generate_structure", after_structure, ["parse_sections", END],
    )
    graph.add_conditional_edges(
        "parse_sections", after_parse, ["summarize_section", END],
    )
    graph.add_edge("summarize_section", "examine_section")
    graph.add_conditional_edges(
        "examine_section", next_section, ["summarize_section", END],
    )

    return graph.compile()


# Pre-compiled graph instance
prepare_class_graph = build_prepare_class_graph()

# Cache helpers
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _cache_path(document: str) -> str:
    """Return the cache file path for a document (sanitised)."""
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', document)
    return os.path.join(CACHE_DIR, f"{safe_name}.json")


def _load_cache(document: str) -> dict | None:
    """Load cached result if it exists, otherwise return None."""
    path = _cache_path(document)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[Cache] Loaded cached result for '{document}'")
            return data
        except Exception as e:
            print(f"[Cache] Error reading cache: {e}")
    return None


def _save_cache(document: str, data: dict) -> None:
    """Save pipeline result to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(document)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Cache] Saved result for '{document}' → {path}")
    except Exception as e:
        print(f"[Cache] Error saving cache: {e}")



def prepare_class(document: str) -> dict:
    """Run the full class-preparation pipeline (with cache).

    Args:
        document: Topic name or filename (e.g. "redes_neuronales").

    Returns:
        dict with keys: structure, sections (per-section results), error.
    """
    cached = _load_cache(document)
    if cached is not None:
        return cached

    result = prepare_class_graph.invoke({
        "document": document,
        "structure": "",
        "sections": [],
        "current_index": 0,
        "current_summary": "",
        "section_results": [],
        "error": "",
    })

    output = {
        "structure": result.get("structure", ""),
        "sections": result.get("section_results", []),
        "error": result.get("error", ""),
    }

    if not output["error"]:
        _save_cache(document, output)

    return output



