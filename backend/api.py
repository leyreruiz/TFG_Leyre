"""FastAPI: exposes the orchestrator as an HTTP service for the frontend."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.orchestrator import Orchestrator
from backend.agents.summary_agent import SummaryAgent
from backend.agents.exam_agent import ExamAgent
from backend.agents.structurer_agent import StructurerAgent
from backend.rag.retriever import ChromaDbRetriever
from backend.graph import prepare_class
from backend.ingest_topics import ingest_file_txt

app = FastAPI(title="University Assistant API")

# CORS so the frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the orchestrator once at startup
retriever = ChromaDbRetriever()
orchestrator = Orchestrator(
    agents=[
        SummaryAgent(retriever),
        ExamAgent(retriever),
        StructurerAgent(retriever),
    ]
)


class ChatRequest(BaseModel):
    message: str


class PrepareRequest(BaseModel):
    document: str


@app.post("/chat")
def chat(request: ChatRequest):
    """Receive a message and return the response from the appropriate agent."""
    result = orchestrator.route(request.message)
    return result


@app.post("/prepare")
def prepare_class_endpoint(request: PrepareRequest):
    """Run the LangGraph pipeline: structure → per-section summary + exam."""
    result = prepare_class(request.document)
    return result


@app.post("/ingest-and-prepare")
async def ingest_and_prepare(file: UploadFile = File(...)):
    """Upload a .txt file, ingest it, and run the preparation pipeline."""
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Save to backend/data directory
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, file.filename)
        
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        # Ingest the file into ChromaDB
        file_name = os.path.splitext(file.filename)[0]
        metadata = {"topic": file_name, "source": "upload"}
        ingest_file_txt(file_path, metadata=metadata)
        
        # Run the preparation pipeline
        result = prepare_class(file_name)
        
        return result
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}


@app.get("/topics")
def list_topics():
    """List available files for generating exams."""
    data_dir = "backend/data"
    try:
        files = [f.replace(".txt", "") for f in os.listdir(data_dir) if f.endswith(".txt")]
        return {"topics": files}
    except Exception:
        return {"topics": []}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_frontend():
    """Serve the frontend directly from the API (no cache)."""
    return FileResponse(
        "frontend/index.html",
        headers={"Cache-Control": "no-store"},
    )
