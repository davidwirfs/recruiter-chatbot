"""
main.py — FastAPI backend for the recruiter chatbot (public-deploy version).

Endpoints:
  GET  /            → serve the chat UI (static/index.html)
  GET  /health      → health check (Chroma collection + Gemini config)
  POST /chat        → streamed chat answer (text/event-stream)

Public-deploy hardening (Phase 2b/3):
- Per-IP rate limit on /chat (20 requests/hour) via slowapi.
- Real-IP detection through X-Forwarded-For so HF Spaces' reverse proxy
  doesn't make every request look like it's coming from the same gateway.
- Pydantic-enforced message length cap (500 chars) — rejects oversized
  payloads before they reach the LLM.
- CORS deliberately permissive: the chatbot has no auth, no cookies, no
  cross-origin state. Wildcard origins are harmless here and let any
  embed-it-in-an-iframe use case work without further config.

Local development of this public-deploy version still works as long as
GEMINI_API_KEY is set:
  GEMINI_API_KEY=... python -m uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .ingest import COLLECTION_NAME
from .llm import GEMINI_MODEL, health_check, stream_chat
from .prompt import SYSTEM_PROMPT, build_user_message
from .retriever import _get_collection, retrieve


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

# Hard cap on incoming message length. 500 chars covers any realistic
# recruiter question; longer payloads are either a prompt-injection
# attempt or noise.
MAX_MESSAGE_CHARS = 500

# Per-IP rate limit on /chat. Recruiters typically ask 1-5 questions per
# session; 20/hour gives plenty of headroom while shutting down scripted
# abuse and accidental tight-loop refreshes.
CHAT_RATE_LIMIT = "20/hour"


def real_client_ip(request: Request) -> str:
    """Extract the real client IP behind HF Spaces' reverse proxy.

    HF Spaces (like any cloud host) terminates TLS at a load balancer and
    forwards the request to our container. Without this helper, slowapi
    would see every request as coming from the load balancer's internal IP
    and rate-limit everyone in aggregate. X-Forwarded-For carries the
    original client IP as the first comma-separated entry.
    """
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=real_client_ip)

app = FastAPI(title="David Wirfs — Recruiter Chatbot", version="0.5.2")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: see module docstring on why this stays permissive.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    top_k: int = Field(default=5, ge=1, le=20)


@app.get("/")
def index():
    """Serve the recruiter chat UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    """Sanity-check Chroma + Gemini configuration before the recruiter sends a message."""
    try:
        coll = _get_collection()
        chroma_ok = True
        chunk_count = coll.count()
        chroma_error = None
    except Exception as e:
        chroma_ok = False
        chunk_count = 0
        chroma_error = str(e)

    llm = health_check()

    return {
        "ok": chroma_ok and llm.get("api_key_set", False),
        "chroma": {
            "ok": chroma_ok,
            "collection": COLLECTION_NAME,
            "chunks": chunk_count,
            "error": chroma_error,
        },
        "llm": llm,
        "model": GEMINI_MODEL,
    }


@app.post("/chat")
@limiter.limit(CHAT_RATE_LIMIT)
def chat(request: Request, req: ChatRequest):
    """Stream a recruiter-aware answer back to the client.

    Rate-limited per real client IP (X-Forwarded-For-aware) to
    `CHAT_RATE_LIMIT`. Message length is pydantic-validated to
    `MAX_MESSAGE_CHARS`; oversized payloads return 422 before the
    handler body runs.
    """
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
