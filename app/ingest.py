"""
ingest.py — load source documents from data/, chunk, embed, store in local Chroma.

Re-runnable. Deletes and rebuilds the recruiter collection each time so an
updated CV / manifesto fully replaces the old one (no stale chunks).

Usage:
    cd _projects/recruiter-chatbot
    python -m app.ingest

v0.3.0 (2026-05-11): now ingests BOTH markdown (*.md) AND PDF (*.pdf) files.
Markdown preferred for new sources because the chunker can preserve logical
units (each role, each degree, each section) far more reliably than PDF-extracted
plain text, which loses visual hierarchy at extraction time.

Pattern reused from _ecosystem/rag/embedding_pipeline.py:
- LangChain RecursiveCharacterTextSplitter, 800 chars / 100 overlap
- sentence-transformers/all-MiniLM-L6-v2 embeddings (384-dim, ~80 MB, free)
- Per-source metadata so the retriever can cite which doc a chunk came from
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma"

# --- Config ------------------------------------------------------------------
COLLECTION_NAME = "recruiter_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Map filename → human-readable source label that the LLM can cite.
# v0.2.4 (Jobs round 6): cut to two-word minimum. The recruiter trusts you
# to know what these are; no need to spell out "Operating Manifesto" or
# tag the CV with a date.
SOURCE_LABELS = {
    "2026-05-11-cv-david-wirfs.md": "CV",
    "2026-05-11-how-i-work-manifesto.md": "Manifesto",
    # Archived PDF source labels retained so older references (or
    # accidentally-restored archives) still cite correctly.
    "2026-05-05-cv-david-wirfs.pdf": "CV",
    "2026-05-05-how-i-work-manifesto.pdf": "Manifesto",
}


# --- Source loading ----------------------------------------------------------
def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF as a single string."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def extract_md_text(md_path: Path) -> str:
    """Read a markdown file as text. No transformation — the markdown
    structure is what makes ingestion reliable; we pass it through verbatim
    so the splitter can use blank lines and headings as natural boundaries."""
    return md_path.read_text(encoding="utf-8").strip()


def extract_text(path: Path) -> str:
    """Dispatch on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix == ".md":
        return extract_md_text(path)
    raise ValueError(f"Unsupported source file format: {path.name}")


# --- Chunking ----------------------------------------------------------------
def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def chunk_id(source_file: str, chunk_index: int) -> str:
    """Stable id: hash of source path + chunk index."""
    digest = hashlib.sha1(
        f"{source_file}::{chunk_index}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{Path(source_file).stem}-{chunk_index:04d}-{digest}"


# --- Main --------------------------------------------------------------------
def ingest() -> dict:
    """Wipe and rebuild the recruiter Chroma collection from data/*.md + data/*.pdf."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Expected data/ folder at {DATA_DIR}")

    # Gather sources — markdown first (preferred), then any PDFs
    sources = sorted(DATA_DIR.glob("*.md")) + sorted(DATA_DIR.glob("*.pdf"))
    if not sources:
        raise FileNotFoundError(f"No .md or .pdf sources found in {DATA_DIR}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Wipe + recreate so re-ingestion fully replaces the old corpus
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  wiped existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "schema_version": "2026-05-11-recruiter-chatbot-v1",
        },
    )

    total_chunks = 0
    file_summary = []
    for src in sources:
        print(f"\n[{src.name}]")
        text = extract_text(src)
        if not text:
            print("  (empty extract — skipping)")
            continue

        chunks = chunk_text(text)
        if not chunks:
            print("  (no chunks produced — skipping)")
            continue

        source_label = SOURCE_LABELS.get(src.name, src.name)
        ids = [chunk_id(src.name, i) for i in range(len(chunks))]
        metadatas = [
            {
                "source_file": src.name,
                "source_label": source_label,
                "chunk_index": i,
                "chunk_total": len(chunks),
            }
            for i in range(len(chunks))
        ]

        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        print(f"  embedded {len(chunks)} chunks ({len(text)} chars)")
        total_chunks += len(chunks)
        file_summary.append({"file": src.name, "chunks": len(chunks)})

    print(f"\n[done] {total_chunks} chunks across {len(file_summary)} file(s)")
    print(f"[chroma] {CHROMA_DIR}")
    return {
        "total_chunks": total_chunks,
        "files": file_summary,
        "chroma_dir": str(CHROMA_DIR),
    }


if __name__ == "__main__":
    ingest()
