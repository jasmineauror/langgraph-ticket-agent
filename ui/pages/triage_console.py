"""Triage Console: submit a ticket and watch the pipeline decide."""

from __future__ import annotations

import ui.bootstrap as bootstrap  # noqa: F401  (must precede any triage import)

import streamlit as st

from triage.nodes.judge import GROUNDING_FLOOR
from ui.auth import require_auth
from ui.pipeline import stream_run
from ui.render import (
    HAIRLINE,
    INK_2,
    INK_MUTED,
    STATUS,
    judge_rubric,
    pill,
    similarity_bars,
    stage_strip,
    stat_tile,
    trace_block,
)
from ui.resources import load_fixtures, warm_embedder
from ui.sidebar import active_backend

require_auth()

PRESET_BLURBS = {
    "normal-billing": "Answerable from the knowledge base",
    "technical-in-kb": "Specific figures, must come from the docs",
    "abusive": "Abusive — routed to a human before any retrieval",
    "refund-request-action": "Asks for an account action",
    "refund-policy-question": "Sensitive topic, undocumented policy",
    "no-kb-answer": "The knowledge base cannot answer this",
    "ambiguous-two-issues": "Two unrelated problems in one ticket",
}

st.title("Triage Console")
st.caption(
    "Every routing decision, retrieval score and judge check is shown. The "
    "interesting case is a ticket the knowledge base cannot answer."
)

# The embedder is warmed before any input is offered, so the ~79MB first-run
# model download happens inside a labelled wait rather than as an unexplained
# hang on the user's first ticket.
if not st.session_state.get("warmed"):
    with st.status("Preparing the retrieval index…", expanded=True) as status:
        st.write("Opening the vector store")
        st.write("Loading the embedding model (~79 MB on first run only)")
        warm_embedder()
        status.update(label="Retrieval index ready", state="complete", expanded=False)
    st.session_state.warmed = True

fixtures = {f["id"]: f for f in load_fixtures()}
busy = st.session_state.get("pending_ticket") is not None


def _load_preset(fixture_id: str) -> None:
    """Populate the ticket box from a fixture.

    Must be an on_click callback rather than a branch after the button. A
    widget's session_state key cannot be assigned once that widget has been
    instantiated in the current run, and the text area is built above the
    preset buttons. Callbacks run before the next rerun constructs its widgets,
    which is the window where the assignment is legal.
    """
    st.session_state.ticket_text = fixtures[fixture_id]["ticket"].strip()


def _submit() -> None:
    """Hand the ticket to the next run.

    A widget cannot be disabled and used in the same run -- the disable only
    takes effect on the following rerun -- and a rerun landing mid-stream
    aborts the graph, discarding model calls already paid for out of a daily
    quota. So the ticket goes into session_state here and the streaming happens
    on the next run, with every input disabled.
    """
    st.session_state.pending_ticket = (st.session_state.get("ticket_text") or "").strip()

# --- input ------------------------------------------------------------------

with st.container(border=True):
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("**Ticket**")
        text = st.text_area(
            "Ticket",
            key="ticket_text",
            height=170,
            label_visibility="collapsed",
            placeholder="Paste a customer support ticket…",
            disabled=busy,
        )
        st.button(
            "Run triage",
            type="primary",
            disabled=busy or not (text or "").strip(),
            on_click=_submit,
        )

    with right:
        st.markdown("**Or load an example**")
        st.caption("The seven tickets the eval suite asserts behaviour against.")
        for fid, blurb in PRESET_BLURBS.items():
            if fid not in fixtures:
                continue
            st.button(
                blurb,
                key=f"preset_{fid}",
                use_container_width=True,
                disabled=busy,
                on_click=_load_preset,
                args=(fid,),
            )

# --- run --------------------------------------------------------------------

if busy:
    ticket = st.session_state.pending_ticket
    strip = st.empty()
    progress = None
    for progress in stream_run(ticket, backend=active_backend()):
        strip.markdown(stage_strip(progress), unsafe_allow_html=True)
    st.session_state.last_run = progress
    st.session_state.pending_ticket = None
    st.rerun()

progress = st.session_state.get("last_run")
if progress is None:
    st.info("Submit a ticket, or load one of the examples above.")
    st.stop()

state = progress.state or {}
st.markdown(stage_strip(progress), unsafe_allow_html=True)

if progress.error:
    st.error(progress.error)
    st.stop()

# Demo mode replays responses recorded per prompt. A ticket nobody recorded
# falls back to a deliberately conservative canned reply, which means it always
# escalates. Saying so matters: an unlabelled escalation here would read as the
# pipeline's judgement about this ticket, which it is not.
if any("generic" in m for m in (state.get("served_by") or {}).values()):
    st.warning(
        "**This ticket was not part of the recorded demo set.** Demo mode has "
        "no recorded reply for it, so it returned a conservative canned "
        "response and the outcome below is not a real judgement about this "
        "text. Retrieval scores and routing *are* genuine. Turn off Demo mode "
        "in the sidebar for a live answer, or try one of the examples."
    )

# --- outcome ----------------------------------------------------------------

st.markdown("")
sent = state.get("verdict") == "SEND"
decided_by = state.get("decided_by", "?")

cols = st.columns(4, gap="small")
outcome_status, outcome_glyph, outcome_word = (
    ("good", "✓", "AUTO REPLY") if sent else ("serious", "→", "ESCALATED")
)
with cols[0]:
    st.markdown(
        stat_tile(
            "Outcome",
            outcome_word,
            f"decided by the {decided_by}",
        ),
        unsafe_allow_html=True,
    )
with cols[1]:
    st.markdown(
        stat_tile("Category", str(state.get("category", "—")).replace("_", " ")),
        unsafe_allow_html=True,
    )
with cols[2]:
    sim = state.get("max_similarity", 0.0)
    st.markdown(
        stat_tile(
            "Best match",
            f"{sim:.3f}",
            "above floor" if sim >= GROUNDING_FLOOR else "below floor",
        ),
        unsafe_allow_html=True,
    )
with cols[3]:
    st.markdown(
        stat_tile("Elapsed", f"{progress.elapsed:.1f}s", f"{len(progress.order)} steps"),
        unsafe_allow_html=True,
    )

st.markdown("")
body, side = st.columns([3, 2], gap="large")

with body:
    if sent:
        with st.container(border=True):
            st.markdown("**Reply sent to the customer**")
            st.markdown(state.get("draft_reply", ""))
            sources = state.get("cited_sources") or []
            st.caption(
                "Cited: " + (", ".join(f"`{s}`" for s in sources) if sources else "none")
            )
    else:
        with st.container(border=True):
            st.markdown("**Escalated to a human**")
            st.markdown(
                f'<div style="color:{STATUS["serious"]};font-size:14px;">'
                f"{state.get('escalation_reason', '')}</div>",
                unsafe_allow_html=True,
            )
            draft = state.get("draft_reply")
            if draft:
                st.markdown("")
                st.markdown(
                    f'<div style="color:{INK_MUTED};font-size:12px;">'
                    f"The draft the judge rejected, included so the human has "
                    f"context:</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="border-left:2px solid {HAIRLINE};padding-left:12px;'
                    f'color:{INK_2};font-size:13px;margin-top:8px;">{draft}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption(
                    "No draft was produced: the classifier routed this to a "
                    "human before any retrieval or drafting."
                )

    with st.container(border=True):
        st.markdown("**Retrieval**")
        st.caption(
            "Cosine similarity against the 24-chunk knowledge base. Length is "
            "the score; the dashed line is the floor below which nothing is "
            "considered on-subject."
        )
        st.markdown(
            similarity_bars(state.get("retrieved") or [], GROUNDING_FLOOR),
            unsafe_allow_html=True,
        )

with side:
    with st.container(border=True):
        st.markdown("**Judge**")
        st.caption(
            "Two code gates run before any model call; the rest is the model "
            "reading the draft against the sources."
        )
        st.markdown(judge_rubric(state.get("judge_checks") or {}), unsafe_allow_html=True)
        if state.get("retry_count"):
            st.markdown("")
            st.markdown(
                pill("warning", "◐", f"{state['retry_count']} redraft requested"),
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.markdown("**Served by**")
        served = state.get("served_by") or {}
        if served:
            for role, model in served.items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'gap:8px;padding:3px 0;font-size:12px;">'
                    f'<span style="color:{INK_2};">{role}</span>'
                    f'<code style="font-size:11px;">{model}</code></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No model calls were made.")

with st.expander("Execution trace"):
    trace_block(state.get("trace") or [])
