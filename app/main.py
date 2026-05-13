"""
main.py — FastAPI backend for the recruiter chatbot (public-deploy version).

Endpoints:
  GET  /            → serve the chat UI (static/index.html)
  GET  /health      → health check (Chroma collection + Gemini config + cache)
  POST /chat        → streamed chat answer (text/event-stream)

Public-deploy hardening (Phase 2b/3 + v0.6 stability pass):
- Per-IP rate limit on /chat (20 requests/hour) via slowapi.
- Real-IP detection through X-Forwarded-For so HF Spaces' reverse proxy
  doesn't make every request look like it's coming from the same gateway.
- Pydantic-enforced message length cap (500 chars) — rejects oversized
  payloads before they reach the LLM.
- CORS deliberately permissive: the chatbot has no auth, no cookies, no
  cross-origin state. Wildcard origins are harmless here and let any
  embed-it-in-an-iframe use case work without further config.
- In-memory TTL response cache (cachetools): repeated recruiter questions
  ("tell me about David", "what does he do?") never re-hit the LLM.
  Biggest single source of free-tier 429s eliminated.
- Multi-provider retry+failover lives in llm.py; this module wires the
  on_retry callback into an SSE `retry` event so the UI can show a
  subtle indicator while the system self-heals.

Local development of this public-deploy version still works as long as
GEMINI_API_KEY is set:
  GEMINI_API_KEY=... python -m uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .ingest import COLLECTION_NAME
from .llm import GEMINI_MODEL, health_check, stream_chat_with_failover
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

# Response cache: 100 entries × 24h TTL. Sized to comfortably hold a
# full day of LinkedIn-driven recruiter traffic (typically <50 distinct
# question phrasings/day). Eviction is automatic — old entries fall off
# by TTL, hot entries by LRU when the cap is hit. In-memory only:
# HF Spaces restarts are rare and a cache warms in minutes.
RESPONSE_CACHE_MAXSIZE = 100
RESPONSE_CACHE_TTL_SECONDS = 24 * 60 * 60

_response_cache: TTLCache = TTLCache(
    maxsize=RESPONSE_CACHE_MAXSIZE,
    ttl=RESPONSE_CACHE_TTL_SECONDS,
)


def _cache_key(message: str, top_k: int) -> str:
    """Normalize a recruiter question into a stable cache key.

    Normalization steps (each addresses a real recruiter phrasing
    variant we want to treat as identical):
      - lowercase           → "Tell me about David" == "tell me about david"
      - collapse whitespace → "tell   me  about" == "tell me about"
      - strip punctuation   → "what does he do?" == "what does he do"

    top_k is included because a different retrieval breadth produces
    a different answer; we don't want a top_k=3 answer served to a
    top_k=10 request.
    """
    normalized = message.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return hashlib.sha256(f"{normalized}|{top_k}".encode("utf-8")).hexdigest()


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

app = FastAPI(title="David Wirfs — Recruiter Chatbot", version="0.6.2")
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
    """Sanity-check Chroma + LLM configuration before the recruiter sends a message.

    Reports configured providers (without probing them — that would burn
    rate-limit quota) and current cache occupancy. `ok` requires the
    Chroma vector store to be loadable AND at least one LLM provider
    to have a configured API key.
    """
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
        "cache": {
            "size": len(_response_cache),
            "maxsize": _response_cache.maxsize,
            "ttl_seconds": RESPONSE_CACHE_TTL_SECONDS,
        },
    }


@app.post("/chat")
@limiter.limit(CHAT_RATE_LIMIT)
def chat(request: Request, req: ChatRequest):
    """Stream a recruiter-aware answer back to the client.

    Caching: identical (normalized) questions return instantly from the
    in-memory TTL cache as a single SSE event. Cache miss path runs
    retrieval + LLM-with-failover, accumulates the full response, then
    stores it under the cache key.

    Rate-limited per real client IP (X-Forwarded-For-aware) to
    `CHAT_RATE_LIMIT`. Message length is pydantic-validated to
    `MAX_MESSAGE_CHARS`; oversized payloads return 422 before the
    handler body runs.
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="empty message")

    key = _cache_key(req.message, req.top_k)
    cached = _response_cache.get(key)

    if cached is not None:
        # Cache hit: replay the answer as a single SSE event. We don't
        # fake token-by-token streaming — instant response is a feature,
        # not a bug. The frontend handles a one-shot 'token' event
        # identically to a streamed one.
        def cached_stream():
            yield f"data: {json.dumps({'type': 'citations', 'citations': cached['citations']})}\n\n"
            yield f"data: {json.dumps({'type': 'cache', 'hit': True})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': cached['text']})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    # Cache miss: retrieve relevant context from the recruiter knowledge base
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

        # Collect retry events from the LLM layer and re-emit them as
        # SSE events. The frontend uses these to flash a subtle amber
        # tint on the typing indicator — no scary error text.
        pending_retries: list[dict] = []

        def on_retry(provider_name: str, attempt: int, reason: object) -> None:
            pending_retries.append(
                {
                    "type": "retry",
                    "provider": provider_name,
                    "attempt": attempt,
                    "reason": str(reason),
                }
            )

        accumulated = []
        try:
            for token in stream_chat_with_failover(
                SYSTEM_PROMPT, user_message, on_retry=on_retry
            ):
                # Flush any retry events that the LLM layer queued up
                # before the first token arrived.
                while pending_retries:
                    yield f"data: {json.dumps(pending_retries.pop(0))}\n\n"
                accumulated.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Stream completed successfully — store the full response so
            # the next identical question is a cache hit.
            full_text = "".join(accumulated)
            if full_text.strip():
                _response_cache[_cache_key(req.message, req.top_k)] = {
                    "text": full_text,
                    "citations": citations,
                }
        except Exception as e:
            # Flush any retry events we accumulated before the failure,
            # so the UI saw the system tried. Then surface the error.
            while pending_retries:
                yield f"data: {json.dumps(pending_retries.pop(0))}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
