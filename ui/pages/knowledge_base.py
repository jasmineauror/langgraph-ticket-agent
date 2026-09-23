"""Knowledge Base: what the retriever actually has to work with.

The point of this page is to make "the knowledge base cannot answer this"
inspectable rather than a claim. You can see every chunk, and probe any query
against them.
"""

from __future__ import annotations

import ui.bootstrap as bootstrap  # noqa: F401  (must precede any triage import)

import streamlit as st

from triage.index import TARGET_WORDS
from triage.nodes.judge import GROUNDING_FLOOR
from ui.auth import require_auth
from ui.render import INK_2, INK_MUTED, pill, similarity_bars, stat_tile
from ui.resources import (
    expected_manifest,
    get_collection,
    index_manifest,
    load_kb_docs,
    warm_embedder,
)

require_auth()

st.title("Knowledge Base")
st.caption(
    "Fifteen hand-written documents, deliberately incomplete. There is no "
    "refund policy and nothing about self-hosted deployment — which is what "
    "makes the no-hallucination fixtures testable."
)

# Rendering the documents needs neither Chroma nor the embedding model, so this
# page is fully useful on a cold container before the embedder has loaded.
docs = load_kb_docs()
chunk_total = sum(len(d["chunks"]) for d in docs)

cols = st.columns(4, gap="small")
with cols[0]:
    st.markdown(stat_tile("Documents", str(len(docs))), unsafe_allow_html=True)
with cols[1]:
    st.markdown(
        stat_tile("Chunks", str(chunk_total), f"~{TARGET_WORDS} words each"),
        unsafe_allow_html=True,
    )
with cols[2]:
    st.markdown(
        stat_tile("Embedding", "384-dim", "all-MiniLM-L6-v2, local"),
        unsafe_allow_html=True,
    )
with cols[3]:
    st.markdown(
        stat_tile("Words", f"{sum(d['words'] for d in docs):,}"),
        unsafe_allow_html=True,
    )

# --- staleness -------------------------------------------------------------

st.markdown("")
try:
    live_ids = index_manifest()
    expected_ids = expected_manifest()
except Exception as exc:
    st.warning(f"Could not read the index: {exc}")
    live_ids, expected_ids = [], []

if live_ids and live_ids != expected_ids:
    missing = set(expected_ids) - set(live_ids)
    extra = set(live_ids) - set(expected_ids)
    st.error(
        f"The committed index does not match `triage/kb/`: "
        f"{len(missing)} chunk(s) missing, {len(extra)} stale. "
        f"Run `python -m triage.index` and commit the result. "
        f"Serving stale vectors would produce results that look like prompt "
        f"behaviour and are not."
    )
elif live_ids:
    st.markdown(
        pill("good", "✓", f"index matches the source documents ({len(live_ids)} chunks)"),
        unsafe_allow_html=True,
    )

# --- probe -----------------------------------------------------------------

st.markdown("")
with st.container(border=True):
    st.markdown("**Probe the retriever**")
    st.caption(
        "The same cosine search the pipeline runs. Try something the docs do "
        "not cover — the top score stays well above zero, because similarity "
        "measures topical relatedness, not whether an answer is present."
    )
    query = st.text_input(
        "Query",
        label_visibility="collapsed",
        placeholder="e.g. do you offer on-premise deployment?",
    )
    if query.strip():
        warm_embedder()
        result = get_collection().query(
            query_texts=[query.strip()],
            n_results=4,
            include=["documents", "metadatas", "distances"],
        )
        chunks = [
            {"source": m["source"], "distance": d, "text": t}
            for t, m, d in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]
        st.markdown(
            similarity_bars(chunks, GROUNDING_FLOOR), unsafe_allow_html=True
        )
        best = 1.0 - min(result["distances"][0])
        if best < GROUNDING_FLOOR:
            st.markdown(
                pill("critical", "✕", f"below the floor — nothing on this subject"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                pill("good", "✓", f"on-subject material found ({best:.3f})"),
                unsafe_allow_html=True,
            )
            st.caption(
                "On-subject is not the same as answerable. Whether these "
                "passages actually contain the answer is the judge's grounding "
                "check, which reads the text."
            )
        with st.expander("Retrieved text"):
            for chunk in chunks:
                st.markdown(
                    f'<div style="color:{INK_MUTED};font-size:11px;'
                    f'font-family:monospace;">{chunk["source"]} · '
                    f'{1.0 - chunk["distance"]:.3f}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="color:{INK_2};font-size:13px;margin:4px 0 14px;">'
                    f'{chunk["text"]}</div>',
                    unsafe_allow_html=True,
                )

# --- documents -------------------------------------------------------------

st.markdown("")
st.markdown("### Documents")
for doc in docs:
    with st.expander(f"{doc['source']}  ·  {len(doc['chunks'])} chunk(s)"):
        st.caption(doc["title"])
        for i, chunk in enumerate(doc["chunks"]):
            st.markdown(
                f'<div style="color:{INK_MUTED};font-size:11px;'
                f'font-family:monospace;">chunk {i} · '
                f'{len(chunk.split())} words</div>',
                unsafe_allow_html=True,
            )
            st.code(chunk, language="text")
