#!/usr/bin/env bash
# 2-ingest.sh — load CV + How I Work into the local Chroma vector store.
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
python -m app.ingest
