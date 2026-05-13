# Recruiter chatbot — HuggingFace Spaces Docker SDK image.
#
# Build: HF Spaces auto-builds on each push to the Space's git remote
# (which mirrors github.com/davidwirfs/recruiter-chatbot main).
# Runtime: uvicorn on port 7860 (HF Spaces convention).
#
# Required env vars (set in Space → Settings → Variables and secrets):
#   GEMINI_API_KEY  — your Gemini API key (free tier, aistudio.google.com/apikey)
# Optional overrides:
#   GEMINI_MODEL    — default "gemini-3.1-flash-lite" (Google's free-tier
#                     flash-lite as of May 2026; ~250 req/user/day cap).
#                     Alternatives: "gemini-flash-lite-latest" (alias,
#                     auto-tracks current free-tier), "gemini-2.5-flash"
#                     (paid account, higher quality).
#   GEMINI_URL      — default "https://generativelanguage.googleapis.com/v1beta/openai";
#                     rarely changed.

FROM python:3.11-slim

# Non-root user (HF Spaces security convention — UID 1000 is the HF default)
RUN useradd -m -u 1000 user

# Minimal build toolchain for native wheels (chromadb / sentence-transformers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=false

WORKDIR /home/user/app

# Install Python deps first so this layer caches across source-only changes
COPY --chown=user:user requirements.txt /home/user/app/
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image. This runs while we still
# have outbound network access; subsequent layers turn offline mode on so
# the runtime container never reaches out to HuggingFace Hub.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && echo "Embedding model cached at ~/.cache/huggingface/"

# From here on: full offline mode for HF hub + transformers.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Copy source code
COPY --chown=user:user app /home/user/app/app
COPY --chown=user:user data /home/user/app/data
COPY --chown=user:user static /home/user/app/static

# Build the Chroma vector store at image-build time so the running container
# starts instantly with retrieval ready — no ingestion lag on first request.
# Uses the cached embedding model (no network needed).
RUN python -m app.ingest

# HF Spaces convention
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
