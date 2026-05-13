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
┌─────────────────────────────┐
│  FastAPI backend            │  ← Python, /chat (SSE stream) + /health
│  + slowapi rate limit       │     20 req/hour per IP
└─────┬─────────────────┬─────┘
      │                 │
      ▼                 ▼
┌──────────┐      ┌──────────────────────────┐
│ ChromaDB │      │ Gemini API               │  ← gemini-2.0-flash
│ (in-     │      │ (Google, free tier,      │     low latency
│  image)  │      │  OpenAI-compatible)      │     overridable via env
└────┬─────┘      └──────────────────────────┘
     │
     ▼
data/  ←  markdown source files (CV + manifesto)
chroma/ ← embedded chunks, built INTO the Docker image
```

**Stack:** FastAPI · ChromaDB · sentence-transformers/all-MiniLM-L6-v2 ·
Google Gemini API (`gemini-2.0-flash` by default) · slowapi for rate limiting.
Open source. Free tier covers any realistic LinkedIn-recruiter volume.

There is a separate **local development version** that swaps the Gemini
LLM call for a local Ollama daemon — that lives in the parent project
folder, not in this repo. The version in THIS repo is the public-deploy
variant.

> **Why Gemini, not Groq:** v0.4.x used Groq's inference API. During public
> deploy we hit repeated `invalid_api_key` rejections from Groq despite
> freshly-created keys and correctly-passed env vars. We swapped to Gemini
> in v0.5.0 — same OpenAI-compatible interface, more reliable free tier.

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
| `GEMINI_MODEL` | `gemini-2.0-flash` | Alternatives: `gemini-2.5-flash` (newer, higher quality) or `gemini-1.5-flash` (older fallback). |
| `GEMINI_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` | Rarely changed. |
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
- **LLM:** `gemini-2.0-flash` via Google's OpenAI-compatible Gemini
  streaming API. Free tier as of mid-2026: ~15 req/min, ~1M tok/min,
  ~1,500 req/day — far more than enough for a LinkedIn-driven recruiter
  chatbot.
- **Rate limiting:** slowapi, 20 req/hour per real client IP (parsed
  from `X-Forwarded-For` to defeat HF Spaces' reverse-proxy IP collapse).
- **Input length cap:** 500 chars, enforced by pydantic before the
  handler runs.

---

## License

[MIT](./LICENSE) — David Wirfs, 2026.

---

## Contact

David Wirfs · **david@wirfs.me** ·
[linkedin.com/in/blitzscaleit](https://www.linkedin.com/in/blitzscaleit)
· Küssnacht (CH) / Cologne (DE)
