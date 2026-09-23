"""Every cached resource in the app, in one file so there is one file to audit.

The governing rule: **`st.cache_resource` is cross-session shared mutable
state.** Anything placed there is a module global with nicer ergonomics and
exactly the same hazards. Things belong there because sharing is *correct*,
never because it is faster.

Two hard constraints on every cached function below:

  * it must not read `st.session_state`, a widget value, or per-user secrets --
    a cached body captures its first caller's values and serves them to every
    session for the life of the process;
  * it must not run a graph. A cached "fresh measurement" is the self-measuring
    harness bug this project has already catalogued five instances of.
"""

from __future__ import annotations

import threading

import ui.bootstrap as bootstrap  # noqa: F401  (must precede any triage import)

import streamlit as st

from triage import index as triage_index
from triage.graph import build_graph
from triage.index import chunk_document
from triage.nodes import retriever

# Guards the ONNX model download. Not optional, and not merely an optimisation:
# chromadb's downloader streams to a FIXED path and then untars to a FIXED path
# with no locking of its own, so two cold sessions racing corrupt each other's
# model. The window is the app's first ~40 seconds, which is exactly when two
# people click a shared link at once.
_WARM_LOCK = threading.Lock()


@st.cache_resource(show_spinner=False)
def get_graph():
    """The compiled graph, shared across sessions.

    Safe to share: no checkpointer, no mutable attributes, all run state lives
    in the channels. Concurrent `.stream()` calls with different inputs are
    independent.
    """
    return build_graph()


@st.cache_resource(show_spinner=False)
def get_collection():
    """The ONE Chroma collection for this process.

    This is the memory-critical resource. Every `triage.index.get_collection()`
    call constructs a fresh embedding function, and each of those lazily builds
    its own onnxruntime InferenceSession at ~100-150MB resident. On a ~1GB
    container one is the budget, so a second live embedder is not slow, it is an
    OOM.

    Installing it into `triage.nodes.retriever` is what makes the graph and the
    Knowledge Base page share this instance instead of each opening their own.

    Never pass a custom `chromadb.Settings` anywhere: chromadb memoises its
    System by path and raises "an instance of Chroma already exists ... with
    different settings" if a second client asks for the same path with different
    settings. Defaults everywhere means the memo hits.
    """
    try:
        collection = triage_index.get_collection()
    except Exception:
        # The committed index is missing (a partial checkout, a wiped volume).
        # Rebuild once, inside this cached body, so it can happen at most once
        # per process and under the cache's own computation lock -- build_index()
        # deletes the collection before re-adding it, and that must never race a
        # session holding a live handle.
        triage_index.build_index(verbose=False)
        collection = triage_index.get_collection()

    retriever.set_collection(collection)
    return collection


@st.cache_resource(show_spinner=False)
def warm_embedder() -> int:
    """Force the embedding model to download and load. Returns its dimension.

    chromadb downloads the ~79MB ONNX model lazily inside `__call__`, not at
    construction, and reports progress with a tqdm bar written to stderr --
    invisible in a browser. Left alone, a cold container's first ticket simply
    hangs for 20-40 seconds with no explanation.

    Doing it here, once, behind a caller-supplied status message turns that into
    a labelled wait.
    """
    with _WARM_LOCK:
        collection = get_collection()
        collection.query(query_texts=["warmup"], n_results=1)
    return 384


@st.cache_resource(show_spinner=False)
def eval_gate() -> threading.BoundedSemaphore:
    """Process-wide single-flight for the eval suite.

    One run is ~21 requests. Two concurrent runs are ~42, which against a
    20-requests-per-model-per-day free tier guarantees daily-quota 429s -- and a
    daily-quota trip sets an 86,400 second circuit cooldown, bricking the app
    for every later visitor. A second user is told a run is already in progress
    rather than being queued behind it.
    """
    return threading.BoundedSemaphore(1)


@st.cache_data(show_spinner=False)
def load_kb_docs() -> list[dict]:
    """The knowledge base as documents and chunks.

    Pure file reads plus `chunk_document()`. Needs neither Chroma nor ONNX,
    which is what lets the Knowledge Base page render fully on a cold container
    before the embedder has finished loading.
    """
    docs = []
    for path in sorted(triage_index.KB_DIR.glob("*.md")):
        text = path.read_text()
        title = triage_index._title(text, path.stem)
        chunks = chunk_document(text, title)
        docs.append(
            {
                "source": path.name,
                "title": title,
                "text": text,
                "chunks": chunks,
                "words": len(text.split()),
            }
        )
    return docs


@st.cache_data(show_spinner=False)
def load_fixtures() -> list[dict]:
    from evals.runner import load_fixtures as _load

    return _load()


@st.cache_data(show_spinner=False, ttl=30)
def index_manifest() -> list[str]:
    """Chunk ids currently in the collection.

    `include=[]` means Chroma returns ids only and performs no embedding, so
    this is a millisecond sqlite read rather than a model call.
    """
    return sorted(get_collection().get(include=[])["ids"])


def expected_manifest() -> list[str]:
    """Chunk ids recomputed from the markdown, for the staleness check.

    The id scheme is `f"{path.stem}::{i}"`, so comparing against the collection
    is exact and costs nothing.
    """
    ids = []
    for doc in load_kb_docs():
        stem = doc["source"].removesuffix(".md")
        ids.extend(f"{stem}::{i}" for i in range(len(doc["chunks"])))
    return sorted(ids)
