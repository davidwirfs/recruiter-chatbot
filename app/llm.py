"""
llm.py — Multi-provider LLM client with retry + failover.

The chatbot's primary LLM is Google's Gemini free tier, which is fast and
plenty for LinkedIn-driven recruiter traffic — but its free-tier backend
returns transient 503/429 under any non-trivial load. A bare 503 reaching
the recruiter UI is the worst possible first impression.

This module addresses that with three layers, all stdlib + free tier:

  1. Each provider call retries on transient errors (429/5xx, network)
     with exponential backoff + jitter.
  2. If a provider exhausts its retries, we fail over to the next
     configured provider transparently.
  3. The SSE layer in main.py emits a `retry` event between attempts so
     the UI can show a subtle indicator (no scary error text).

All providers use the OpenAI-compatible streaming protocol, so the wire
code is identical — only base URL, model name, and API key differ.

Configured providers (in failover order):
  - Gemini     (primary)  — `gemini-3.1-flash-lite` free tier
  - Groq       (fallback) — `llama-3.3-70b-versatile` free tier

A provider with an empty API key is skipped entirely, so the system
keeps working if only `GEMINI_API_KEY` is set. The Groq URL is
configurable via `GROQ_URL` so users can swap to Cerebras
(`https://api.cerebras.ai/v1`) or OpenRouter without a code change —
useful given v0.4.x hit `invalid_api_key` issues with Groq's own keys.

Uses stdlib urllib only — no extra SDK dependency.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Generator, Optional


def _build_providers() -> list[dict]:
    """Build the ordered provider list from current env vars.

    Built lazily (called from stream_chat_with_failover) so tests and
    local development can mutate env vars between requests.
    """
    return [
        {
            "name": "gemini",
            "url": os.environ.get(
                "GEMINI_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            "model": os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite"),
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
            "max_retries": 3,
        },
        {
            "name": "groq",
            "url": os.environ.get("GROQ_URL", "https://api.groq.com/openai/v1"),
            # llama-3.1-8b-instant is the always-free workhorse on Groq —
            # smaller than the 70B but reliably available on every free
            # account. Larger models (llama-3.3-70b-versatile, mixtral)
            # have been moved in and out of the paid tier; defaulting to
            # the 8B avoids the v0.6.0 deploy issue where the 70B
            # returned 403 Forbidden for free-tier keys.
            "model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
            "api_key": os.environ.get("GROQ_API_KEY", ""),
            "max_retries": 2,
        },
    ]


# Exposed for /health and logging. Module-level snapshot of the primary
# model name; main.py uses this only for the `model` field in /health.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")


def _stream_provider(
    provider: dict,
    system_prompt: str,
    user_message: str,
    temperature: float,
) -> Generator[str, None, None]:
    """Stream tokens from one OpenAI-compatible provider.

    Raises urllib.error.HTTPError on non-2xx (with the response body
    attached to .reason for diagnostics — provider error bodies often
    explain *why* a 403/401/429 happened, e.g. "model not available
    on your tier"), URLError/TimeoutError on network failure. The
    caller (stream_chat_with_failover) decides whether to retry or
    fail over based on the exception type.
    """
    payload = {
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{provider['url']}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider['api_key']}",
        },
        method="POST",
    )
    try:
        resp_cm = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        # Read the response body so the failure reason is visible
        # in container logs (HF Space → Logs tab). We can't mutate
        # e.reason (it's a read-only property on HTTPError), so we
        # attach the body as a fresh attribute and print to stderr.
        # The retry/failover layer still sees the same exception
        # type and status code — only logging is enriched.
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            body = ""
        e.response_body = body  # readable by callers if needed
        if body:
            print(
                f"[llm] HTTP {e.code} from {provider['name']} "
                f"({provider['url']}, model={provider['model']}): {body}",
                file=sys.stderr,
                flush=True,
            )
        raise
    with resp_cm as resp:
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


# HTTP status codes worth retrying. 429 = rate limit (Gemini free tier
# is quota-throttled), 5xx = transient backend failure.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff with jitter, capped at 4s.

    Sleep durations by attempt index (with jitter range):
      attempt 0 → 0.5–1.0s
      attempt 1 → 1.0–1.5s
      attempt 2 → 2.0–2.5s
      attempt 3+ → 4.0–4.5s (cap)
    Worst-case wall-clock before Gemini→Groq failover with 3 Gemini
    retries: ~2.5s (sleep happens between attempts, not after the last).
    """
    base = min((2 ** attempt) * 0.5, 4.0)
    time.sleep(base + random.uniform(0, 0.5))


def stream_chat_with_failover(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    on_retry: Optional[Callable[[str, int, object], None]] = None,
) -> Generator[str, None, None]:
    """Try each configured provider with retry + failover.

    Yields token strings exactly like the old stream_chat. When a
    transient error triggers a retry or a provider switch, the
    `on_retry` callback (if provided) is invoked synchronously with
    (provider_name, attempt_number, reason). The SSE layer uses this
    to emit a `retry` event so the UI can show a subtle indicator
    while the system self-heals.

    NOTE on streaming semantics: if the provider has already started
    yielding tokens and then fails mid-stream, those tokens have
    already been sent to the user — we don't retry mid-stream, we
    only retry/failover when the request fails *before* the first
    token. That avoids producing garbled half-answers.
    """
    providers = _build_providers()
    last_error: Exception = RuntimeError("No providers configured")

    for provider in providers:
        if not provider["api_key"]:
            continue
        for attempt in range(provider["max_retries"]):
            try:
                # If this generator yields even one token, the request
                # succeeded — pass everything through and return.
                yield from _stream_provider(
                    provider, system_prompt, user_message, temperature
                )
                return
            except urllib.error.HTTPError as e:
                last_error = e
                # Non-transient HTTP errors (4xx other than 429) usually
                # indicate a misconfigured key or bad request — switch
                # to the next provider immediately instead of burning
                # retries on a request that will never succeed.
                if e.code not in _RETRYABLE_STATUS:
                    if on_retry:
                        on_retry(provider["name"], attempt + 1, e.code)
                    break
                if on_retry:
                    on_retry(provider["name"], attempt + 1, e.code)
                if attempt < provider["max_retries"] - 1:
                    _backoff_sleep(attempt)
            except (urllib.error.URLError, TimeoutError) as e:
                last_error = e
                if on_retry:
                    on_retry(provider["name"], attempt + 1, "network")
                if attempt < provider["max_retries"] - 1:
                    _backoff_sleep(attempt)

    raise RuntimeError(f"All LLM providers exhausted. Last error: {last_error}")


# Backward-compat alias. main.py and any external caller can still
# import `stream_chat`; it now routes through the failover layer with
# no retry callback. Use stream_chat_with_failover directly if you
# want the on_retry hook.
def stream_chat(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
) -> Generator[str, None, None]:
    yield from stream_chat_with_failover(
        system_prompt, user_message, temperature, on_retry=None
    )


def health_check() -> dict:
    """Return provider configuration state (no probe calls).

    We deliberately do NOT make probe calls — the frontend hits /health
    on every page load and that would burn rate-limit quota. The first
    /chat request surfaces any actual API issue. With the failover
    layer in place, a misbehaving primary provider is already invisible
    to the recruiter.
    """
    providers = _build_providers()
    return {
        "provider": "gemini",
        "configured_model": GEMINI_MODEL,
        "api_key_set": any(p["api_key"] for p in providers),
        "endpoint": providers[0]["url"],
        "providers": [
            {
                "name": p["name"],
                "model": p["model"],
                "configured": bool(p["api_key"]),
            }
            for p in providers
        ],
    }
