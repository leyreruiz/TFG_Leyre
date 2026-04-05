"""FastAPI: university assistant API. No LangGraph — plain SSE generator."""

import logging
import os, sys, json
from typing import Optional

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.agents.structurer_agent import StructurerAgent
from backend.agents.summary_agent import SummaryAgent
from backend.agents.exam_agent import ExamAgent
from backend.agents.explainer_agent import ExplainerAgent
from backend.models.schemas import StudentRequest
from backend.rag.retriever import ChromaDbRetriever
from backend.ingest_topics import ingest_file
from backend.class_storage import (
    get_class,
    save_class,
    update_section_summary,
    update_section_questions,
    list_classes,
    delete_class,
)
import re

app = FastAPI(title="University Assistant API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_retriever  = ChromaDbRetriever()
_structurer = StructurerAgent(_retriever)
_summary    = SummaryAgent(_retriever)
_exam       = ExamAgent(_retriever, num_questions=2)
_explainer  = ExplainerAgent()


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _parse_sections(structure: str) -> list[str]:
    matches = re.findall(r"###\s*(?:\d+\.\s*)?(.+)", structure)
    return [m.strip() for m in matches if m.strip()]


def _run_pipeline(document: str):
    """Generator: runs the full pipeline and yields SSE strings.
    
    At the end, saves the complete class to JSON.
    """
    # 1. Structure
    req = StudentRequest(message=document, intent="structure")
    result = _structurer.handle(req)
    if "error" in result:
        yield _sse({"type": "error", "message": result["error"]})
        return

    structure = result["content"]
    sections  = _parse_sections(structure)
    if not sections:
        yield _sse({"type": "error", "message": "No sections found in structure."})
        return

    # Persist class skeleton immediately so partial progress survives interruptions.
    save_class(
        document,
        structure,
        sections,
        {section: {"summary": "", "questions": []} for section in sections},
    )

    yield _sse({"type": "structure", "structure": structure, "sections": sections})

    # 2. For each section: summary then exam
    total = len(sections)
    for idx, section in enumerate(sections):
        # Summary
        req    = StudentRequest(message=f"resume {section}", intent="summary")
        result = _summary.handle(req)
        summary = result.get("content", result.get("error", ""))
        update_section_summary(document, section, summary)
        yield _sse({
            "type":           "summary",
            "section_index":  idx,
            "section_title":  section,
            "total_sections": total,
            "summary":        summary,
        })

        # Exam
        req = StudentRequest(
            message=f"exam {section}\n\nusing the following summary context:\n{summary}",
            intent="exam",
        )
        result    = _exam.handle(req)
        questions = result.get("questions", [])
        update_section_questions(document, section, questions)
        yield _sse({
            "type":           "exam",
            "section_index":  idx,
            "section_title":  section,
            "total_sections": total,
            "questions":      questions,
        })
    
    yield _sse({"type": "done"})


def _run_saved_pipeline(class_data: dict):
    """Generator: replays a previously saved class as SSE events."""
    structure = class_data.get("structure", "")
    sections = class_data.get("sections", [])
    sections_data = class_data.get("sections_data", {})

    if not structure or not sections:
        yield _sse({"type": "error", "message": "Saved class is incomplete."})
        return

    yield _sse({"type": "structure", "structure": structure, "sections": sections})

    total = len(sections)
    for idx, section in enumerate(sections):
        section_info = sections_data.get(section, {})
        yield _sse({
            "type": "summary",
            "section_index": idx,
            "section_title": section,
            "total_sections": total,
            "summary": section_info.get("summary", ""),
        })
        yield _sse({
            "type": "exam",
            "section_index": idx,
            "section_title": section,
            "total_sections": total,
            "questions": section_info.get("questions", []),
        })

    yield _sse({"type": "done"})


# ── Endpoints ────────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    document: str

class RegenerateSummaryRequest(BaseModel):
    topic: Optional[str] = None
    section_title: str


class RegenerateExamRequest(BaseModel):
    topic: Optional[str] = None
    section_title: str
    section_summary: Optional[str] = None

class AskRequest(BaseModel):
    question:        str
    section_title:   str
    section_summary: str
    conversation_id: Optional[str] = None


@app.post("/start")
def start(request: StartRequest):
    _retriever.set_topic(request.document)
    saved_class = get_class(request.document)
    if saved_class:
        return StreamingResponse(_run_saved_pipeline(saved_class), media_type="text/event-stream")

    return StreamingResponse(_run_pipeline(request.document), media_type="text/event-stream")


@app.post("/regenerate-summary")
def regenerate_summary(request: RegenerateSummaryRequest):
    # Topic is required to update the JSON storage
    if not request.topic:
        return {"error": "topic is required to save the regenerated content"}

    _retriever.set_topic(request.topic)
    
    # Generate new summary
    req    = StudentRequest(message=f"resume {request.section_title}", intent="summary")
    result = _summary.handle(req)
    if "error" in result:
        return {"error": result["error"]}
    
    summary = result["content"]
    
    # Generate new exam based on updated summary
    req_exam = StudentRequest(
        message=f"exam {request.section_title}\n\nusing the following summary context:\n{summary}",
        intent="exam"
    )
    exam_result = _exam.handle(req_exam)
    questions = exam_result.get("questions", [])
    
    # Update in JSON storage
    summary_updated = update_section_summary(request.topic, request.section_title, summary)
    questions_updated = update_section_questions(request.topic, request.section_title, questions)

    if not summary_updated or not questions_updated:
        return {
            "error": "Could not persist regenerated content for this topic/section"
        }
    
    return {
        "summary": summary,
        "questions": questions,
    }


@app.post("/regenerate-exam")
def regenerate_exam(request: RegenerateExamRequest):
    if not request.topic:
        return {"error": "topic is required to save the regenerated exam"}

    _retriever.set_topic(request.topic)

    summary_context = (request.section_summary or "").strip()
    if not summary_context:
        class_obj = get_class(request.topic)
        if not class_obj:
            return {"error": "Could not load class to regenerate exam"}
        summary_context = (
            class_obj
            .get("sections_data", {})
            .get(request.section_title, {})
            .get("summary", "")
            .strip()
        )

    req_exam = StudentRequest(
        message=(
            f"exam {request.section_title}\n\n"
            f"using the following summary context:\n{summary_context}"
        ),
        intent="exam",
    )
    exam_result = _exam.handle(req_exam)
    questions = exam_result.get("questions", [])

    questions_updated = update_section_questions(request.topic, request.section_title, questions)
    if not questions_updated:
        return {"error": "Could not persist regenerated exam for this topic/section"}

    return {"questions": questions}


@app.post("/ask")
def ask(request: AskRequest):
    conversation_id = (request.conversation_id or "").strip()
    if not conversation_id:
        normalized_section = "_".join((request.section_title or "section").strip().lower().split())
        conversation_id = f"default::{normalized_section}"

    req    = StudentRequest(message=request.question, intent="explain")
    result = _explainer.handle(
        req,
        section_context=request.section_summary,
        conversation_id=conversation_id,
    )
    logger.debug("Explainer answered for section: %s", request.section_title)
    return result


@app.post("/ingest-and-prepare")
async def ingest_and_prepare(file: UploadFile = File(...)):
    try:
        extension = os.path.splitext(file.filename)[1].lower()
        if extension not in {".txt", ".pdf", ".pptx"}:
            return {"error": "Unsupported file type. Use .txt, .pdf or .pptx"}

        contents = await file.read()
        data_dir  = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        file_name = os.path.splitext(file.filename)[0]
        ids = ingest_file(file_path, metadata={"topic": file_name, "source": "upload"})
        if not ids:
            return {"error": "Could not extract content from file"}
        return {"topic": file_name}
    except Exception as e:
        return {"error": str(e)}


@app.get("/topics")
def list_topics():
    try:
        topics = []
        for f in os.listdir("backend/data"):
            base, ext = os.path.splitext(f)
            if ext.lower() in {".txt", ".pdf", ".pptx"}:
                topics.append(base)
        return {"topics": sorted(set(topics))}
    except Exception:
        return {"topics": []}


@app.get("/classes")
def get_classes():
    """List all saved classes."""
    classes = list_classes()
    return {"classes": classes}


@app.get("/classes/{topic}")
def get_saved_class(topic: str):
    """Retrieve a saved class by topic."""
    class_data = get_class(topic)
    if not class_data:
        return {"error": f"Class '{topic}' not found"}
    return class_data


@app.delete("/classes/{topic}")
def remove_class(topic: str):
    """Delete a saved class and its source .txt/.pdf/.pptx topic file."""
    class_deleted = delete_class(topic)

    normalized_topic = "_".join((topic or "").strip().lower().split())
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    source_deleted = False
    for candidate in {topic, normalized_topic}:
        for extension in (".txt", ".pdf", ".pptx"):
            topic_file = os.path.join(data_dir, f"{candidate}{extension}")
            if os.path.exists(topic_file):
                try:
                    os.remove(topic_file)
                    source_deleted = True
                except Exception:
                    pass

    if class_deleted or source_deleted:
        return {
            "message": f"Class '{topic}' deleted",
            "class_deleted": class_deleted,
            "source_deleted": source_deleted,
        }
    return {"error": f"Could not delete class '{topic}'"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html", headers={"Cache-Control": "no-store"})