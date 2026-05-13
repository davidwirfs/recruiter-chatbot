#!/usr/bin/env bash
# 3-serve.sh — start the FastAPI backend on http://localhost:8000.
# Make sure Ollama is running (usually auto-started); if needed: `ollama serve`
#
# Runs in zero-outbound-traffic mode: ChromaDB telemetry off, HuggingFace
# Hub offline, Transformers offline. The embedding model must already be
# cached locally (1-setup.sh handles that).
set -e
cd "$(dirname "$0")/.."

# Zero-outbound mode (see ADR-012)
export ANONYMIZED_TELEMETRY=false       # ChromaDB anonymous telemetry off
export HF_HUB_OFFLINE=1                 # huggingface_hub: no update pings
export TRANSFORMERS_OFFLINE=1           # transformers: no version checks

source .venv/bin/activate
echo "[serve] starting FastAPI on http://localhost:8000"
echo "[serve] zero-outbound mode: telemetry off, HF/Transformers offline"
echo "[serve] open http://localhost:8000 in your browser"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
