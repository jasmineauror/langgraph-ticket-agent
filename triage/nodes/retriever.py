"""Retriever node: similarity search over the Chroma knowledge base.

Returns chunks *with their distances* and records max_similarity. Exposing the
score is the point: it turns "the knowledge base has no answer to this" into a
measurable property of retrieval, instead of something the language model has to
introspect about and report honestly.
"""

from __future__ import annotations

from ..index import get_collection
from ..state import Chunk, TicketState

TOP_K = 4

_collection = None


def _get():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection


def set_collection(collection) -> None:
    """Install a collection to use instead of opening a new one.

    Exists for long-lived hosts. `get_collection()` constructs a fresh Chroma
    client AND a fresh ONNX embedding function -- and each embedding function
    lazily builds its own onnxruntime InferenceSession, ~100-150MB resident. One
    is the memory budget on a small container, so a host that also needs the
    collection for its own purposes must share this one rather than open a
    second.
    """
    global _collection
    _collection = collection


def retrieve(state: TicketState) -> TicketState:
    results = _get().query(
        query_texts=[state["ticket_text"]],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks: list[Chunk] = [
        Chunk(text=doc, source=meta["source"], distance=float(dist))
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]

    # Cosine distance in [0, 2]; similarity = 1 - distance.
    max_similarity = max((1.0 - c["distance"] for c in chunks), default=0.0)

    scores = ", ".join(f"{c['source']}={1.0 - c['distance']:.3f}" for c in chunks)

    return {
        "retrieved": chunks,
        "max_similarity": max_similarity,
        "trace": [
            f"[retriever] {len(chunks)} chunks, max_similarity={max_similarity:.3f} "
            f"({scores})"
        ],
    }
