"""Build the Chroma collection from the markdown knowledge base.

Run directly to (re)build:  python -m triage.index
"""

from __future__ import annotations

import pathlib
import re

import chromadb
from chromadb.utils import embedding_functions

KB_DIR = pathlib.Path(__file__).parent / "kb"
DB_DIR = pathlib.Path(__file__).parent.parent / "chroma_db"
COLLECTION = "meridian_kb"
# Chroma's default embedding function is all-MiniLM-L6-v2 served through
# onnxruntime. Same model as the sentence-transformers build, but ~80MB of
# dependency instead of ~2.5GB of torch, and it runs entirely offline once the
# model file is cached.
EMBED_MODEL = "all-MiniLM-L6-v2 (onnx)"

# Chunks target this many words. The KB docs are short and single-topic, so
# paragraph-level chunks keep each one tightly scoped while staying long enough
# to ground a reply.
TARGET_WORDS = 130


def _title(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def chunk_document(text: str, title: str) -> list[str]:
    """Split into chunks of roughly TARGET_WORDS, never splitting a paragraph.

    The title is prepended to every chunk so an isolated chunk still carries the
    context of which document it came from -- without it, a chunk like "after
    30 days the data is purged" retrieves well but reads ambiguously.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    paragraphs = [p for p in paragraphs if not p.startswith("# ")]

    chunks: list[str] = []
    current: list[str] = []
    count = 0

    for para in paragraphs:
        words = len(para.split())
        if current and count + words > TARGET_WORDS:
            chunks.append("\n\n".join(current))
            current, count = [], 0
        current.append(para)
        count += words

    if current:
        chunks.append("\n\n".join(current))

    return [f"{title}\n\n{chunk}" for chunk in chunks]


def _embedding_function():
    return embedding_functions.DefaultEmbeddingFunction()


def _create_collection(client):
    """Create the collection with cosine distance.

    Chroma moved this setting from `metadata` to `configuration` across versions;
    try the current shape first and fall back.
    """
    kwargs = {"name": COLLECTION, "embedding_function": _embedding_function()}
    try:
        return client.create_collection(
            **kwargs, configuration={"hnsw": {"space": "cosine"}}
        )
    except TypeError:
        return client.create_collection(**kwargs, metadata={"hnsw:space": "cosine"})


def build_index(verbose: bool = True) -> int:
    """Rebuild the collection from scratch. Returns the chunk count."""
    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # first run, nothing to delete

    collection = _create_collection(client)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text()
        title = _title(text, path.stem)
        for i, chunk in enumerate(chunk_document(text, title)):
            ids.append(f"{path.stem}::{i}")
            documents.append(chunk)
            metadatas.append(
                {"source": path.name, "title": title, "chunk_index": i}
            )

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    if verbose:
        print(f"indexed {len(ids)} chunks from {len(list(KB_DIR.glob('*.md')))} documents")
        print(f"collection: {COLLECTION}  at  {DB_DIR}")

    return len(ids)


def get_collection():
    """Open the existing collection for querying."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_collection(
        name=COLLECTION, embedding_function=_embedding_function()
    )


if __name__ == "__main__":
    build_index()
