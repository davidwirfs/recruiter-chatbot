"""
llm.py — thin Ollama client.

Talks to the local Ollama daemon at http://localhost:11434/api/chat using
the stdlib only (no extra SDK dependency). Streams tokens back so the
frontend can render them as they arrive.

Default model: llama3.1 (already on David's machine).
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Generator

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")


def stream_chat(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
) -> Generator[str, None, None]:
    """Yield response tokens from Ollama as they arrive."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "options": {"temperature": temperature},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = obj.get("message", {})
            content = msg.get("content", "")
            if content:
                yield content
            if obj.get("done"):
                break


def health_check() -> dict:
    """Return whether Ollama is reachable and which models are installed."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
        models = [m.get("name") for m in data.get("models", [])]
        return {
            "ollama_reachable": True,
            "configured_model": OLLAMA_MODEL,
            "configured_model_present": any(
                OLLAMA_MODEL in (m or "") for m in models
            ),
            "models_available": models,
        }
    except Exception as e:
        return {
            "ollama_reachable": False,
            "configured_model": OLLAMA_MODEL,
            "error": str(e),
        }
