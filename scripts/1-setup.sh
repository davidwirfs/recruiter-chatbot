#!/usr/bin/env bash
# 1-setup.sh — one-time install of Python dependencies into a local venv.
#
# Also pre-downloads the embedding model so subsequent runs of 2-ingest.sh
# and 3-serve.sh can operate in zero-outbound-traffic mode (HF_HUB_OFFLINE
# and TRANSFORMERS_OFFLINE require the model to already be cached locally).
#
# This is the ONLY script that requires outbound network access (pip from
# PyPI + sentence-transformers model from HuggingFace Hub). After this
# completes, the chatbot runs fully offline.
set -e
cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "[setup] creating venv at .venv/"
  python3 -m venv .venv
fi

echo "[setup] activating venv and installing requirements"
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "[setup] pre-downloading embedding model so subsequent runs are offline"
echo "[setup] (~80 MB to ~/.cache/huggingface/ — one-time download)"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('[setup] embedding model cached.')"

echo
echo "[setup] done."
echo "[setup] Next: ./scripts/2-ingest.sh (or ./scripts/3-serve.sh if Chroma already populated)"
echo "[setup] From this point on, the chatbot runs with ZERO outbound network calls."
