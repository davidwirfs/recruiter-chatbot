"""
retriever.py — query the recruiter Chroma collection.

Returns top-k chunks ranked by semantic similarity, each with a source label
so the LLM can cite which document the answer came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .ingest import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


@dataclass
class Chunk:
    text: str
    score: float | None
    source_label: str
    source_file: str
    chunk_index: int


_collection = None


def _get_collection():
    """Lazy singleton — load Chroma once per process."""
    global _collection
    if _collection is None:
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=embedder
        )
    return _collection


def retrieve(query: str, top_k: int = 5) -> list[Chunk]:
    if not query or not query.strip():
        return []
    coll = _get_collection()
    raw = coll.query(query_texts=[query], n_results=top_k)

    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[None] * len(docs)])[0]

    chunks: list[Chunk] = []
    for text, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        score = (1.0 - float(dist)) if dist is not None else None
        chunks.append(
            Chunk(
                text=text,
                score=score,
                source_label=str(meta.get("source_label", "Unknown")),
                source_file=str(meta.get("source_file", "")),
                chunk_index=int(meta.get("chunk_index", -1)),
            )
        )
    return chunks


def format_context_block(chunks: list[Chunk]) -> str:
    """Format retrieved chunks into a context block for the LLM prompt."""
    if not chunks:
        return "(no relevant context retrieved)"
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"--- Source [{i}]: {c.source_label} ---")
        lines.append(c.text.strip())
        lines.append("")
    return "\n".join(lines)
