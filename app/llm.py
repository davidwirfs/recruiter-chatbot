"""
llm.py — Gemini LLM client (HuggingFace Spaces / public-deploy version).

Replaces the local Ollama client used in the development version with a call
to Google's Gemini API via its OpenAI-compatible streaming endpoint.

Why Gemini (and not Groq, the previous default): Groq's account/key system
caused repeated `invalid_api_key` rejections during the v0.4.x deploy despite
freshly-created valid keys and correct env-var passing through HF Spaces.
Gemini's OpenAI-compatible endpoint accepts the same streaming code with
minimal changes.

Requires GEMINI_API_KEY in the environment. On HuggingFace Spaces this is set
once via:  Space → Settings → Variables and secrets → New secret → GEMINI_API_KEY.
Get a free key at https://aistudio.google.com/apikey (no billing setup needed
for the free tier).

Default model: `gemini-3.1-flash-lite` — Google's free-tier flash-lite model
as of May 2026. The earlier `gemini-2.0-flash` (v0.5.0 default) was moved to
paid-tier-only sometime in 2026, returning `429 limit:0` for free-tier keys —
hence the v0.5.1 swap. Free-tier cap on 3.1-flash-lite: ~250 requests per
user per day, more than enough for a LinkedIn-driven recruiter chatbot.

Override via the GEMINI_MODEL env var:
  - `gemini-flash-lite-latest` — alias that auto-tracks the current
    free-tier flash-lite (may silently update when Google releases new
    versions; trade-off between freshness and predictability).
  - `gemini-2.5-flash` — higher quality but requires a paid account.

Uses stdlib urllib only — no extra SDK dependency.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Generator

GEMINI_URL = os.environ.get(
    "GEMINI_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai",
)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def stream_chat(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
) -> Generator[str, None, None]:
    """Yield response tokens from Gemini's OpenAI-compatible streaming API."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not configured. Set it in HuggingFace Space → "
            "Settings → Variables and secrets, or in your local .env "
            "if testing the public-deploy version outside HF."
        )

    payload = {
        "model": GEMINI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{GEMINI_URL}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GEMINI_API_KEY}",
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
    """Return whether Gemini credentials are configured.

    Note: we deliberately do NOT make a probe call to Gemini from /health
    — that would consume rate-limit quota on every health hit (which the
    frontend pings on every page load). Configuration check is enough;
    the first /chat request surfaces any actual API issue.
    """
    return {
        "provider": "gemini",
        "configured_model": GEMINI_MODEL,
        "api_key_set": bool(GEMINI_API_KEY),
        "endpoint": GEMINI_URL,
    }
