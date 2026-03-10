"""FastAPI: exposes the orchestrator as an HTTP service for the frontend."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.orchestrator import Orchestrator
from backend.agents.summary_agent import SummaryAgent
from backend.agents.exam_agent import ExamAgent
from backend.rag.retriever import ChromaDbRetriever

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
    ]
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    """Receive a message and return the response from the appropriate agent."""
    result = orchestrator.route(request.message)
    return result


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
    """Serve the frontend directly from the API."""
    return FileResponse("frontend/index.html")
