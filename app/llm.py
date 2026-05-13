"""
llm.py — Groq LLM client (HuggingFace Spaces / public-deploy version).

Replaces the local Ollama client used in the development version with a call
to Groq's OpenAI-compatible streaming API. Groq serves Llama 3.1/3.3 models
with very fast inference (~700 tok/sec on the 8B model) on a generous free
tier that's more than enough for a LinkedIn-facing recruiter chatbot.

Requires GROQ_API_KEY in the environment. On HuggingFace Spaces this is set
once via:  Space → Settings → Variables and secrets → New secret → GROQ_API_KEY.

Default model: `llama-3.1-8b-instant` — matches the local-Ollama llama3.1 8B
behavior for parity with the dev experience. Free-tier limits as of mid-2026:
30 req/min, 14,400 tok/min, 500,000 tok/day.

Upgrade path: set GROQ_MODEL=llama-3.3-70b-versatile via Space env var for
higher answer quality. Tighter rate limits (6,000 tok/min, 100,000 tok/day)
but more than enough for a LinkedIn-link's recruiter traffic.

Uses stdlib urllib only — no extra SDK dependency.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Generator

GROQ_URL = os.environ.get("GROQ_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def stream_chat(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
) -> Generator[str, None, None]:
    """Yield response tokens from Groq's OpenAI-compatible streaming API."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not configured. Set it in HuggingFace Space → "
            "Settings → Variables and secrets, or in your local .env "
            "if testing the public-deploy version outside HF."
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{GROQ_URL}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            # SSE protocol: skip empty keepalive lines and non-data events
            if not line or not line.startswith("data: "):
                continue
            payload_str = line[6:]
            if payload_str == "[DONE]":
                break
            try:
                obj = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content


def health_check() -> dict:
    """Return whether Groq credentials are configured.

    Note: we deliberately do NOT make a probe call to Groq from /health
    — that would consume rate-limit quota on every health hit (which the
    frontend pings on every page load). Configuration check is enough;
    the first /chat request surfaces any actual API issue.
    """
    return {
        "provider": "groq",
        "configured_model": GROQ_MODEL,
        "api_key_set": bool(GROQ_API_KEY),
        "endpoint": GROQ_URL,
    }
