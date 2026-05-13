# Recruiter Chatbot — David Wirfs

A local, open-source retrieval-augmented chatbot that lets recruiters,
headhunters, and hiring managers ask questions about David's career,
skills, and operating style — grounded in his CV and his "How I Work"
operating manifesto.

Built as a personal-portfolio asset by [David Wirfs](https://www.linkedin.com/in/blitzscaleit),
Finance × AI Systems Builder.

> Looking for a Europe-based role at the intersection of Finance and AI,
> with hybrid or remote flexibility. Reach out at **david@wirfs.me**.

---

## Live

<!-- Replace once deployed:
**Try it:** https://davidwirfs-recruiter-chatbot.hf.space
-->

A public-hosted version will be linked here once the HuggingFace Spaces
deployment is live. In the meantime, the bot can be run locally — see
[Quick Start](#quick-start) below.

---

## What it does

Ask anything about David's background. The bot retrieves the most
relevant chunks from his CV and manifesto, then synthesizes a tight
answer (two sentences default) grounded only in those chunks. It
speaks about David in the third person, refuses salary or sensitive
questions, and ends where the answer ends — no appended call-to-action.

Three rotating example questions seed the experience:

- *What has he shipped at scale?*
- *What is he building now?*
- *What does he need from his next role?*

---

## Architecture

```
Browser
   │
   ▼
┌─────────────────────────────┐
│  FastAPI backend            │  ← Python, /chat (SSE stream) + /health
└─────┬─────────────────┬─────┘
      │                 │
      ▼                 ▼
┌──────────┐      ┌──────────────────┐
│ ChromaDB │      │ Local Ollama     │  ← llama3.1 by default
│ (local)  │      │ (or remote API   │     swappable via OLLAMA_URL +
│ vectors  │      │  via env vars)   │     OLLAMA_MODEL env vars
└────┬─────┘      └──────────────────┘
     │
     ▼
data/  ←  markdown source files (CV + manifesto)
chroma/ ← embedded chunks (derived data, regenerable)
```

**Stack:** FastAPI · ChromaDB · sentence-transformers/all-MiniLM-L6-v2 ·
Ollama (local) — all open source, no API keys required, zero outbound
network traffic during normal operation (see § Privacy).

---

## Quick Start

### Prerequisites

- macOS or Linux
- Python 3.10+
- [Ollama](https://ollama.com) installed locally with a model pulled
  (default: `llama3.1`)

### Three commands

```bash
git clone https://github.com/davidwirfs/recruiter-chatbot.git
cd recruiter-chatbot

./scripts/1-setup.sh    # one-time: create venv, install deps, cache embedding model
./scripts/2-ingest.sh   # one-time per source-doc update: chunk + embed both MDs
./scripts/3-serve.sh    # start the backend; leave running
```

Then open **http://localhost:8000** in your browser. The page is
empty by default — start typing.

Make sure Ollama is running (`ollama serve`) before step 3. If you see
"address already in use," that means Ollama is already running — fine.

---

## Source documents

Two files in `data/`:

- `2026-05-11-cv-david-wirfs.md` — clean markdown CV (mirrors the
  canonical master, trimmed to recruiter-relevant content)
- `2026-05-11-how-i-work-manifesto.md` — David's operating manifesto

To update, edit the markdown directly, then re-run
`./scripts/2-ingest.sh` to refresh the Chroma vector store.

---

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama daemon address |
| `OLLAMA_MODEL` | `llama3.1` | Which model to use |
| `ANONYMIZED_TELEMETRY` | `false` (set in `2-ingest.sh` and `3-serve.sh`) | Disables ChromaDB anonymous telemetry |
| `HF_HUB_OFFLINE` | `1` (set in run scripts) | Stops huggingface_hub from pinging HF servers |
| `TRANSFORMERS_OFFLINE` | `1` (set in run scripts) | Stops transformers from version checks |

---

## Privacy

The local stack makes **zero outbound network calls** during normal
operation. All inference (the LLM, the embeddings, the vector search)
runs on the machine where you start it. The only outbound traffic in
the system's lifecycle is during `./scripts/1-setup.sh`:

- `pip install` from PyPI (one-time)
- Embedding-model download from HuggingFace Hub (one-time, ~80 MB,
  cached at `~/.cache/huggingface/`)

After setup completes, the system never reaches out again.

ChromaDB's anonymous telemetry, HuggingFace Hub revision-check pings,
and Transformers version pings are all explicitly disabled in the run
scripts via env vars.

---

## Tech notes

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
  ~80 MB, runs comfortably on CPU)
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`, 800 chars
  with 100 overlap. Markdown source files preferred over PDFs because
  the splitter respects structural boundaries (headings, blank lines)
  far more reliably than character-level splits on flattened PDF text.
- **LLM:** Ollama-served `llama3.1` by default. Any Ollama-supported
  model works via `OLLAMA_MODEL` env var.

---

## License

[MIT](./LICENSE) — David Wirfs, 2026.

---

## Contact

David Wirfs · **david@wirfs.me** · [linkedin.com/in/blitzscaleit](https://www.linkedin.com/in/blitzscaleit) · Küssnacht (CH) / Cologne (DE)
