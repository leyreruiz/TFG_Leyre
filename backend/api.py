import logging, os, sys, json

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from backend.agents.structurer_agent import StructurerAgent
from backend.agents.summary_agent import SummaryAgent
from backend.agents.exam_agent import ExamAgent
from backend.agents.explainer_agent import ExplainerAgent
from backend.models.schemas import (
    StudentRequest,
    StartRequest,
    RegenerateSummaryRequest,
    RegenerateExamRequest,
    AskRequest,
    AddQuestionsRequest,
    UpdateQuestionsRequest,
    SubmitAnswerRequest,
    RestartWithSuggestionsRequest,
)
from backend.mcp_client import MCPRetriever
from backend.ingest_topics import ingest_file
from backend.class_storage import (
    get_class,
    save_class,
    update_section_summary,
    update_section_questions,
    append_section_conversation_turn,
    clear_section_conversation,
    update_question_user_answer,
    list_classes,
    delete_class,
)
import re

app = FastAPI(title="University Assistant API")

# Allow all origins so the frontend (served from the same server) can call the API
# without CORS issues during local development.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Module-level singletons — agents are stateless except for the shared retriever,
# so creating them once at startup is safe and avoids repeated initialisation cost.
_retriever  = MCPRetriever()
_structurer = StructurerAgent(_retriever)
_summary    = SummaryAgent(_retriever)
_exam       = ExamAgent(_retriever, num_questions=2)
_explainer  = ExplainerAgent()


def _sse(data: dict) -> str:
    """Encode a dict as a Server-Sent Events (SSE) data frame.

    The frontend's EventSource listener receives these strings and parses them
    with JSON.parse(event.data).
    """
    return f"data: {json.dumps(data)}\n\n"


def _parse_sections(structure: str) -> list[str]:
    """Extract section titles from the structurer's markdown output.

    The structurer formats each section as '### [optional-number.] Title'.
    Returns a list of plain title strings (numbers and leading/trailing
    whitespace removed).

    Example input line:  '### 2. Neural Networks' ; Example output item: 'Neural Networks'
    """
    matches = re.findall(r"###\s*(?:\d+\.\s*)?(.+)", structure)
    return [m.strip() for m in matches if m.strip()]


def _extract_section_outline(structure: str, section_title: str) -> str:
    """Return the block of the index that belongs to a specific section.

    Iterates the structure line by line, collecting lines that belong to the
    given section header until the next '###' header is reached.

    Example — given this structure:
        ### 1. Intro
        - What is X
        - History
        ### 2. Methods
        - Algorithm A

    Calling with section_title="Intro" returns:
        ### 1. Intro
        - What is X
        - History

    Used by _run_pipeline to pass each section's sub-outline to the summary
    agent so that the generated summary follows the document's structure.
    """
    lines = structure.split("\n")
    in_section = False
    section_lines = []
    for line in lines:
        is_section_header = re.match(r"###\s*(?:\d+\.\s*)?" + re.escape(section_title), line, re.IGNORECASE)
        if is_section_header:
            in_section = True
        elif in_section and re.match(r"###", line):
            break  # reached the next section
        if in_section:
            section_lines.append(line)
    return "\n".join(section_lines).strip()


def _run_pipeline(document: str, suggestions: str = ""):
    """Generator that runs the full learning pipeline and yields SSE strings.

    Steps:
      1. Call StructurerAgent to produce a section index from the ingested document.
      2. Persist a skeleton class immediately so progress survives interruptions.
      3. For each section, call SummaryAgent then ExamAgent and stream each result.
      4. Yield a final 'done' event.

    If `suggestions` is provided (from /restart-with-suggestions), they are
    forwarded to the structurer so the LLM can incorporate user feedback when
    rebuilding the outline.

    Yielded SSE event types: 'structure', 'summary', 'exam', 'done', 'error'.
    """
    # 1. Structure — with or without user suggestions
    req = StudentRequest(message=document, intent="structure")
    result = _structurer.handle(req, suggestions=suggestions)
    if "error" in result:
        yield _sse({"type": "error", "message": result["error"]})
        return

    structure = result["content"]
    sections  = _parse_sections(structure)
    if not sections:
        yield _sse({"type": "error", "message": "No sections found in structure."})
        return

    # Persist class skeleton immediately so partial progress survives interruptions
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
        # Pass the section's outline so the summary follows the index structure
        section_outline = _extract_section_outline(structure, section)
        result = _summary.summarize(section_title=section, section_outline=section_outline)
        summary = result.get("content", result.get("error", ""))
        update_section_summary(document, section, summary)
        yield _sse({
            "type":           "summary",
            "section_index":  idx,
            "section_title":  section,
            "total_sections": total,
            "summary":        summary,
        })

        # Generate exam questions using the freshly created summary as context
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
    """Generator that replays a previously saved class as SSE events.

    Instead of calling the LLM again, it reads the persisted JSON and emits the
    same event sequence that _run_pipeline would produce. This lets the frontend
    restore a session without any LLM cost.

    Yielded SSE event types: 'structure', 'summary', 'exam', 'conversation' (if
    a saved Q&A history exists for a section), 'done', 'error'.
    """
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
        # Restore saved conversation history for this section, if any
        conversation = section_info.get("conversation", [])
        if conversation:
            yield _sse({
                "type": "conversation",
                "section_index": idx,
                "qa": conversation,
            })

    yield _sse({"type": "done"})


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/ingest-and-prepare")
async def ingest_and_prepare(file: UploadFile = File(...)):
    """Upload a file and ingest its content into ChromaDB.

    Accepts .txt, .pdf, and .pptx files. The file is saved to backend/data/
    and then chunked and embedded into a topic-specific ChromaDB collection.
    Returns the derived topic name (filename without extension) which the
    frontend uses as the document identifier for subsequent /start calls.
    """
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


@app.post("/start")
def start(request: StartRequest):
    """Start a learning session for the given document/topic.

    If a saved class already exists for the topic, it is replayed from JSON
    (no LLM calls). Otherwise the full pipeline is executed from scratch.
    The response is an SSE stream consumed by the frontend's EventSource.
    """
    _retriever.set_topic(request.document)
    saved_class = get_class(request.document)
    if saved_class:
        return StreamingResponse(_run_saved_pipeline(saved_class), media_type="text/event-stream")

    return StreamingResponse(_run_pipeline(request.document), media_type="text/event-stream")


@app.post("/restart-with-suggestions")
def restart_with_suggestions(request: RestartWithSuggestionsRequest):
    """Delete the existing class and re-run the pipeline with user suggestions.

    The user can provide feedback (e.g. 'include more detail on X') that the
    structurer agent will take into account when rebuilding the outline.
    The existing saved class is deleted first so the fresh result is stored.
    """
    _retriever.set_topic(request.document)
    delete_class(request.document)
    return StreamingResponse(
        _run_pipeline(request.document, suggestions=request.suggestions),
        media_type="text/event-stream",
    )


@app.post("/regenerate-summary")
def regenerate_summary(request: RegenerateSummaryRequest):
    """Regenerate the summary and exam questions for a single section.

    Called when the user clicks 'Regenerate' on a section card. Generates a
    new summary via SummaryAgent and then immediately regenerates the exam so
    the questions stay consistent with the updated content. Both results are
    persisted to the class JSON and returned to the frontend.
    """
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

    # Persist both to JSON storage
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
    """Regenerate only the exam questions for a single section.

    If the request does not include a summary, it is loaded from the saved class
    JSON so ExamAgent always has context to produce relevant questions.
    The new question list replaces the old one in the JSON store.
    """
    if not request.topic:
        return {"error": "topic is required to save the regenerated exam"}

    _retriever.set_topic(request.topic)

    # Use the summary supplied in the request, or fall back to the stored one
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


@app.post("/add-questions")
def add_questions(request: AddQuestionsRequest):
    """Generate additional questions for a section without repeating existing ones.

    Delegates to ExamAgent.add_questions(), which is aware of the questions
    already shown to the user and avoids duplicates. The new questions are
    appended to the stored list and returned to the frontend.
    """
    new_questions = _exam.add_questions(
        summary=request.section_summary,
        section_title=request.section_title,
        existing_questions=request.existing_questions,
        num_new=request.num_questions,
    )
    if not new_questions:
        return {"error": "Could not generate additional questions"}

    # Append new questions to the ones already stored
    class_obj = get_class(request.topic)
    if class_obj:
        stored = class_obj.get("sections_data", {}).get(request.section_title, {}).get("questions", [])
        update_section_questions(request.topic, request.section_title, stored + new_questions)

    return {"questions": new_questions}


@app.post("/submit-answer")
def submit_answer(request: SubmitAnswerRequest):
    """Persist the user's answer for a specific question.

    Stores the chosen option and whether it was correct in the class JSON so
    the frontend can restore answer state when a saved class is reloaded.
    """
    updated = update_question_user_answer(
        request.topic, request.section_title,
        request.question_index, request.user_answer, request.user_correct,
    )
    if not updated:
        return {"error": "Could not save answer"}
    return {"ok": True}


@app.post("/update-questions")
def update_questions(request: UpdateQuestionsRequest):
    """Persist the current questions array after the user deletes a question.

    The frontend sends the full updated list; this endpoint overwrites the
    stored list so deletions survive a page reload.
    """
    updated = update_section_questions(request.topic, request.section_title, request.questions)
    if not updated:
        return {"error": "Could not update questions"}
    return {"ok": True}


@app.post("/ask")
def ask(request: AskRequest):
    """Answer a student question using ExplainerAgent.

    Enriches the agent's context with:
      - Relevant chunks from ChromaDB (RAG retrieval).
      - A Wikipedia excerpt for the question (via MCPRetriever).
      - The section's pre-generated summary.
      - The conversation history for the session (keyed by conversation_id).

    If the agent does not refuse the question (off-topic guard), the turn is
    appended to the class's persistent conversation history.
    """
    conversation_id = (request.conversation_id or "").strip()
    if not conversation_id:
        # Build a stable default ID from the section title so the same section
        # always maps to the same conversation history within a session.
        normalized_section = "_".join((request.section_title or "section").strip().lower().split())
        conversation_id = f"default::{normalized_section}"

    db_chunks = _retriever.search(request.question)
    db_context = "\n\n".join(db_chunks) if db_chunks else ""
    wikipedia_context = _retriever.search_wikipedia(request.question)

    req    = StudentRequest(message=request.question, intent="explain")
    result = _explainer.handle(
        req,
        section_context=request.section_summary,
        conversation_id=conversation_id,
        db_context=db_context,
        wikipedia_context=wikipedia_context,
        section=request.section_title,
    )
    logger.info("Explainer answered for section: %s", request.section_title)

    # Only persist if the agent actually answered (not refused)
    if request.topic and "content" in result and not result.get("refused"):
        append_section_conversation_turn(
            request.topic, request.section_title, request.question, result["content"]
        )

    return result


@app.post("/clear-conversation")
def clear_conversation(request: AskRequest):
    """Clear the in-memory and persisted conversation history for a section.

    Called when the user resets the chat panel. Removes the history from both
    ExplainerAgent's in-memory store and the class JSON.
    """
    conversation_id = (request.conversation_id or "").strip()
    if conversation_id:
        _explainer.clear_conversation(conversation_id)
    if request.topic and request.section_title:
        clear_section_conversation(request.topic, request.section_title)
    return {"ok": True}




@app.get("/topics")
def list_topics():
    """List all topics that have a source file in backend/data/.

    Scans for .txt, .pdf, and .pptx files and returns their base names.
    Used by the frontend to populate the topic selector dropdown.
    """
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
    """List all saved classes (topic names only).

    Returns the names of all classes stored in backend/data/classes/.
    The frontend uses this list to show previously generated sessions.
    """
    classes = list_classes()
    return {"classes": classes}


@app.get("/classes/{topic}")
def get_saved_class(topic: str):
    """Retrieve the full saved class JSON for a topic.

    Returns structure, sections, sections_data (summaries, questions,
    conversation history), created_at, and updated_at fields.
    """
    class_data = get_class(topic)
    if not class_data:
        return {"error": f"Class '{topic}' not found"}
    return class_data


@app.delete("/classes/{topic}")
def remove_class(topic: str):
    """Delete a saved class and its source file from disk.

    Removes the class JSON from backend/data/classes/ and the original
    .txt/.pdf/.pptx file from backend/data/. Both the original and
    normalised (lowercase, underscored) forms of the topic name are tried
    when looking for the source file.
    """
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
    """Health check endpoint. Returns {'status': 'ok'} when the server is up."""
    return {"status": "ok"}


@app.get("/")
def serve_frontend():
    """Serve the single-page frontend application.

    Cache-Control is set to no-store so the browser always fetches the latest
    version of index.html during development.
    """
    return FileResponse("frontend/index.html", headers={"Cache-Control": "no-store"})
