"""API FastAPI: expone el orquestador como servicio HTTP para el frontend."""

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

app = FastAPI(title="Asistente Universitario API")

# CORS para que el frontend pueda llamar a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Construir el orquestador una sola vez al arrancar
retriever = ChromaDbRetriever()
orchestrator = Orchestrator(
    agents=[
        SummaryAgent(retriever),
        ExamAgent(),
    ]
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    """Recibe un mensaje y devuelve la respuesta del agente correspondiente."""
    result = orchestrator.route(request.message)
    return result


@app.get("/topics")
def list_topics():
    """Lista los archivos disponibles para generar exámenes."""
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
    """Sirve el frontend directamente desde la API."""
    return FileResponse("frontend/index.html")
