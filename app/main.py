"""
main.py — FastAPI backend for the recruiter chatbot.

Endpoints:
  GET  /            → serve the chat UI (static/index.html)
  GET  /health      → health check (Chroma collection + Ollama reachability)
  POST /chat        → streamed chat answer (text/event-stream)

Run:
  cd _projects/recruiter-chatbot
  python -m uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .ingest import COLLECTION_NAME
from .llm import OLLAMA_MODEL, health_check, stream_chat
from .prompt import SYSTEM_PROMPT, build_user_message
from .retriever import _get_collection, retrieve


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI(title="David Wirfs — Recruiter Chatbot", version="0.1.0")

# CORS open for local dev; tighten before public deploy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    top_k: int = 5


@app.get("/")
def index():
    """Serve the recruiter chat UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    """Sanity-check Chroma + Ollama before the recruiter sends a message."""
    try:
        coll = _get_collection()
        chroma_ok = True
        chunk_count = coll.count()
        chroma_error = None
    except Exception as e:
        chroma_ok = False
        chunk_count = 0
        chroma_error = str(e)

    ollama = health_check()

    return {
        "ok": chroma_ok and ollama.get("ollama_reachable", False),
        "chroma": {
            "ok": chroma_ok,
            "collection": COLLECTION_NAME,
            "chunks": chunk_count,
            "error": chroma_error,
        },
        "ollama": ollama,
        "model": OLLAMA_MODEL,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """Stream a recruiter-aware answer back to the client."""
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="empty message")

    # Retrieve relevant context from the recruiter knowledge base
    chunks = retrieve(req.message, top_k=req.top_k)
    user_message = build_user_message(req.message, chunks)

    citations = [
        {
            "source_label": c.source_label,
            "source_file": c.source_file,
            "chunk_index": c.chunk_index,
            "score": c.score,
        }
        for c in chunks
    ]

    def event_stream():
        # First event carries the citations so the UI can show sources
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
        try:
            for token in stream_chat(SYSTEM_PROMPT, user_message):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
