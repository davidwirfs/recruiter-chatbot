---
title: David Wirfs — Recruiter Chatbot
emoji: 🎸
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Recruiter Chatbot — David Wirfs

A retrieval-augmented chatbot that lets recruiters, headhunters, and hiring
managers ask questions about David's career, skills, and operating style —
grounded in his CV and his "How I Work" operating manifesto.

Built as a personal-portfolio asset by
[David Wirfs](https://www.linkedin.com/in/blitzscaleit), Finance × AI
Systems Builder.

> Looking for a Europe-based role at the intersection of Finance and AI,
> with hybrid or remote flexibility. Reach out at **david@wirfs.me**.

---

## Live

The public-hosted version runs on HuggingFace Spaces:
**https://huggingface.co/spaces/davidwirfs/recruiter-chatbot** (link goes
live once the Space is deployed — first time may take ~30s to wake up
after 48h of inactivity).

---

## What it does

Ask anything about David's background. The bot retrieves the most relevant
chunks from his CV and manifesto, then synthesizes a tight answer
(two sentences by default) grounded only in those chunks. It speaks about
David in the third person, refuses salary or sensitive questions, and
ends where the answer ends — no appended call-to-action.

Three rotating example questions seed the experience:

- *What has he shipped at scale?*
- *What is he building now?*
- *What does he need from his next role?*

---

## Architecture

```
Browser (recruiter)
     │
     ▼
┌──────────────────────────────────────┐
│  FastAPI backend                     │  ← Python, /chat (SSE stream) + /health
│  + slowapi rate limit (20 req/hr/IP) │
│  + cachetools TTL response cache     │  ← 100 × 24h, instant on repeat questions
└─────┬──────────────────────────┬─────┘
      │                          │
      ▼                          ▼
┌──────────┐    ┌────────────────────────────────────┐
│ ChromaDB │    │ LLM with retry+failover            │
│ (in-     │    │  1. Gemini   (3 retries, exp bkoff)│
│  image)  │    │  2. Groq     (2 retries, exp bkoff)│  ← free tier failover
└────┬─────┘    │  All providers OpenAI-compatible.  │
     │          └────────────────────────────────────┘
     ▼
data/  ←  markdown source files (CV + manifesto)
chroma/ ← embedded chunks, built INTO the Docker image
```

**Stack:** FastAPI · ChromaDB · sentence-transformers/all-MiniLM-L6-v2 ·
Google Gemini API (`gemini-3.1-flash-lite` by default) with optional
provider-neutral failover (default target: OpenRouter Llama 3.3 70B
free; configurable to any OpenAI-compatible endpoint) · cachetools
for response caching · slowapi for rate limiting. Open source. Free tier (~250 req/user/day on Gemini
plus Groq's free tier as failover) covers any realistic LinkedIn-recruiter
volume.

**Stability (v0.6.0):** transient `503`/`429` from Gemini's free-tier
backend are absorbed by three layers — retry with exponential backoff,
in-memory TTL response cache for repeated questions, and transparent
failover to a free Groq endpoint. The recruiter never sees a raw API
error unless every layer is exhausted.

There is a separate **local development version** that swaps the Gemini
LLM call for a local Ollama daemon — that lives in the parent project
folder, not in this repo. The version in THIS repo is the public-deploy
variant.

> **Why Gemini, not Groq (primary):** v0.4.x used Groq as the only LLM
> provider. During public deploy we hit repeated `invalid_api_key`
> rejections from Groq despite freshly-created keys and correctly-passed
> env vars. We swapped to Gemini in v0.5.0 — same OpenAI-compatible
> interface, more reliable free tier.
>
> **Why both, now (v0.6.0):** Gemini's free-tier backend started
> returning transient 503s under LinkedIn-driven recruiter traffic. Rather
> than swap providers again, we added Groq back as a failover-only path
> with a configurable `GROQ_URL` (escape hatch: swap to Cerebras or
> OpenRouter without a code change if Groq's keys misbehave again).
>
> **Why `gemini-3.1-flash-lite`, not `gemini-2.0-flash`:** v0.5.0 used
> `gemini-2.0-flash` as the default. By May 2026 Google had moved that
> model to paid-tier-only (free-tier keys get `429 limit:0`). v0.5.1
> swapped to `gemini-3.1-flash-lite`, Google's current free-tier model.

---

## Deployment (HuggingFace Spaces)

This repo is configured to auto-deploy to HuggingFace Spaces via the
YAML frontmatter at the top of this README. Steps:

1. Create a new Space at https://huggingface.co/new-space:
   - Owner: `davidwirfs`
   - Space name: `recruiter-chatbot`
   - License: MIT
   - SDK: **Docker** (matches `sdk: docker` in frontmatter)
   - Public visibility
2. In the new Space's Settings → Variables and secrets, add:
   - `GEMINI_API_KEY` = your key from
     [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (free tier,
     no billing setup required)
   - **(Recommended)** Failover provider — set all three together:
     - `FALLBACK_API_KEY` = your OpenRouter key from
       [openrouter.ai/keys](https://openrouter.ai/keys) (free signup,
       no payment info)
     - `FALLBACK_URL` = `https://openrouter.ai/api/v1`
     - `FALLBACK_MODEL` = `meta-llama/llama-3.3-70b-instruct:free`
       (or any model from [openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0))

     OpenRouter's free-model list is publicly documented and consistent
     across accounts — unlike Groq's per-account gating which we burned
     several hours on. If you skip the failover, the chatbot still
     works on Gemini-only, but loses one stability layer.

     Want to use a different provider? Set `FALLBACK_URL` to any
     OpenAI-compatible endpoint (Cerebras: `https://api.cerebras.ai/v1`,
     Groq: `https://api.groq.com/openai/v1`, your own self-hosted
     vLLM, etc.) — the code is provider-neutral.
3. Connect the Space to this GitHub repo, OR add the Space as a second
   git remote and push:
   ```bash
   git remote add hf https://huggingface.co/spaces/davidwirfs/recruiter-chatbot
   git push hf main
   ```
4. HF auto-builds the Docker image (~3–4 min on first build, faster on
   re-builds via layer cache).
5. Once built, the Space is live at
   `https://davidwirfs-recruiter-chatbot.hf.space`.

---

## Local development of the public-deploy version

Useful if you want to test changes before pushing to HF Spaces.

### Prerequisites
- macOS or Linux, Python 3.11+
- A free Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Three commands

```bash
git clone https://github.com/davidwirfs/recruiter-chatbot.git
cd recruiter-chatbot
./scripts/1-setup.sh
./scripts/2-ingest.sh
GEMINI_API_KEY=... ./scripts/3-serve.sh
```

Open http://localhost:8000. Page is empty by default — start typing.

---

## Source documents

Two files in `data/`:

- `2026-05-11-cv-david-wirfs.md` — clean markdown CV (derived from the
  canonical master, trimmed to recruiter-relevant content)
- `2026-05-11-how-i-work-manifesto.md` — David's operating manifesto

To update either, edit the file, then re-run `./scripts/2-ingest.sh` to
refresh the Chroma vector store. On HF Spaces, push to main and the
container rebuilds with the new content baked in.

---

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | _(required)_ | Your Gemini API key. Set in HF Space → Settings → Variables and secrets. |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Google's free-tier flash-lite as of May 2026. Alternatives: `gemini-flash-lite-latest` (alias, auto-tracks current free-tier) or `gemini-2.5-flash` (paid account, higher quality). |
| `GEMINI_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` | Rarely changed. |
| `FALLBACK_API_KEY` | _(empty → fallback disabled)_ | **Recommended.** Free OpenRouter key from [openrouter.ai/keys](https://openrouter.ai/keys). When set, the configured fallback provider is used whenever Gemini exhausts its retries. |
| `FALLBACK_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible chat-completions endpoint. Examples: OpenRouter (default), `https://api.cerebras.ai/v1`, `https://api.groq.com/openai/v1`, your own self-hosted vLLM. |
| `FALLBACK_MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | Model name as the chosen provider expects it. OpenRouter free-model catalog: [openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0). |
| `FALLBACK_REFERER` | _(auto)_ | OpenRouter analytics header. Harmless against endpoints that ignore it. Override if you want branded usage. |
| `FALLBACK_TITLE` | _(auto)_ | OpenRouter analytics header. Same as above. |
| `GROQ_API_KEY` / `GROQ_URL` / `GROQ_MODEL` | _(backward compat)_ | Old (v0.6.0–0.6.4) env var names. Still read for backward compatibility — but `FALLBACK_*` always wins if both are set. New deployments should use `FALLBACK_*`. |
| `ANONYMIZED_TELEMETRY` | `false` (set in Dockerfile + scripts) | Disables ChromaDB anonymous telemetry. |
| `HF_HUB_OFFLINE` | `1` (set in Dockerfile + scripts after model cache) | Stops huggingface_hub from pinging HF servers at runtime. |
| `TRANSFORMERS_OFFLINE` | `1` (set in Dockerfile + scripts after model cache) | Stops transformers from version checks at runtime. |

---

## Privacy

The chatbot's stack is open-source and runs on infrastructure you (or
your hosting provider) controls. The retrieval side (Chroma + embeddings)
runs entirely inside the deployed container — no outbound calls during
normal operation.

The **LLM inference call** goes out to Google's Gemini API (one inference
call per recruiter question). Gemini sees the question text and the retrieved
context chunks. Per Google's published policy for free-tier AI Studio usage,
prompts and responses may be used to improve Google's products — if that's
a concern, the local development version (parent project, not this repo)
keeps inference fully local via Ollama and never sends data to any cloud.

A small privacy notice on the UI reminds users not to share confidential
information.

---

## Tech notes

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
  ~80 MB). Pre-cached inside the Docker image at build time, so the
  running container makes no outbound HF Hub calls.
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`, 800 chars
  with 100 overlap. Markdown source files preserved structural boundaries
  through chunking far better than PDFs did.
- **LLM:** `gemini-3.1-flash-lite` via Google's OpenAI-compatible Gemini
  streaming API. Free tier as of May 2026: ~250 requests per user per day
  on `gemini-3.1-flash-lite` — far more than enough for a LinkedIn-driven
  recruiter chatbot. (Google moved `gemini-2.0-flash` and earlier models
  to paid-tier-only sometime in 2026, hence the explicit flash-lite
  default.)
- **Rate limiting:** slowapi, 20 req/hour per real client IP (parsed
  from `X-Forwarded-For` to defeat HF Spaces' reverse-proxy IP collapse).
- **Input length cap:** 500 chars, enforced by pydantic before the
  handler runs.
- **Response cache:** in-memory TTL cache (cachetools), 100 entries ×
  24h TTL. Identical (normalized) recruiter questions return instantly
  without re-hitting the LLM — by far the biggest source of free-tier
  rate-limit pressure removed.
- **Multi-provider failover:** every `/chat` request is wrapped in a
  retry+failover layer ([app/llm.py](app/llm.py)). Gemini is tried first
  with 3 retries (exponential backoff with jitter: ~0.75s, ~1.25s — sleep
  only happens between attempts, so total ~2s wall-clock max before
  failover). On exhaustion, Groq is tried with 2 retries. The frontend
  shows a subtle amber tint on the typing dots during retries so the
  recruiter sees activity, not a frozen UI.

---

## License

[MIT](./LICENSE) — David Wirfs, 2026.

---

## Contact

David Wirfs · **david@wirfs.me** ·
[linkedin.com/in/blitzscaleit](https://www.linkedin.com/in/blitzscaleit)
· Küssnacht (CH) / Cologne (DE)
