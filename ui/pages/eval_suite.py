"""Eval Suite: run the seven fixtures and show the score.

This page drives `evals/runner.py` -- the same module `pytest evals/` drives.
It does not reimplement grading, the outcome label, the PASS/FAIL/ERROR
trichotomy, or the score denominator. That is deliberate: a UI-side copy of the
assertion vocabulary would be a second, untested assertion engine, and this
project has already shipped a reporter that overstated its own result.

If this page and `pytest evals/ -v` ever disagree, that is a bug in one shared
code path rather than a discrepancy between two.
"""

from __future__ import annotations

import ui.bootstrap as bootstrap  # noqa: F401  (must precede any triage import)

import streamlit as st

from evals.runner import preflight, run_fixture, score
from ui.auth import require_auth
from ui.pipeline import run_to_state
from ui.render import HAIRLINE, INK_2, INK_MUTED, STATUS, pill, stat_tile
from ui.resources import eval_gate, load_fixtures, warm_embedder
from ui.sidebar import active_backend

require_auth()

STATUS_STYLE = {
    "PASS": ("good", "✓", "pass"),
    "FAIL": ("critical", "✕", "failed assertions"),
    "ERROR": ("warning", "!", "not evaluated"),
}

st.title("Eval Suite")
st.caption(
    "Seven hand-written tickets with declarative assertions. Each asserts not "
    "just the outcome but which node produced it — an earlier version reported "
    "5/6 passing while four fixtures never reached the retriever at all."
)

fixtures = load_fixtures()
declared = len(fixtures)

# --- run control ------------------------------------------------------------

backend = active_backend()
demo = backend == "demo"

with st.container(border=True):
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Run the suite**")
        st.caption(
            f"{declared} fixtures, three model calls each plus retries — about "
            f"21 requests."
            + (
                " Demo mode is on, so no API calls are made."
                if demo
                else " This consumes live quota."
            )
        )
    with right:
        start = st.button("Run all fixtures", type="primary", use_container_width=True)

if start:
    # Single-flight, process-wide. Two concurrent runs are ~42 requests, which
    # against a 20-per-model-per-day free tier guarantees a daily-quota 429 --
    # and that trips an 86,400s circuit cooldown which would brick the app for
    # every later visitor.
    gate = eval_gate()
    if not gate.acquire(blocking=False):
        st.warning(
            "An eval run is already in progress in another session. Two "
            "concurrent runs would exhaust the daily quota and disable the "
            "models for everyone."
        )
    else:
        try:
            if message := preflight(backend):
                # Refuse to produce a number at all. A score attributed to the
                # prompts when the cause was a broken connection is worse than
                # no score, because it looks like a measurement.
                st.error(message)
            else:
                warm_embedder()
                results = []
                placeholder = st.empty()
                bar = st.progress(0.0, text="Starting…")
                for i, fixture in enumerate(fixtures, start=1):
                    bar.progress(
                        (i - 1) / declared, text=f"Running {fixture['id']} ({i}/{declared})"
                    )
                    results.append(
                        run_fixture(
                            fixture,
                            run_fn=lambda t: run_to_state(t, backend=backend),
                        )
                    )
                    with placeholder.container():
                        for r in results:
                            status, glyph, word = STATUS_STYLE[r.status]
                            st.markdown(
                                f'<div style="display:flex;gap:10px;'
                                f'align-items:center;padding:3px 0;">'
                                f"{pill(status, glyph, word)}"
                                f'<code style="font-size:12px;">{r.id}</code></div>',
                                unsafe_allow_html=True,
                            )
                bar.progress(1.0, text="Complete")
                st.session_state.eval_results = results
        finally:
            gate.release()
        st.rerun()

results = st.session_state.get("eval_results")
if not results:
    st.info("Run the suite to see results.")
    st.stop()

# --- score ------------------------------------------------------------------

s = score(results, declared=declared)

st.markdown("")
cols = st.columns(4, gap="small")
with cols[0]:
    st.markdown(
        stat_tile("Score", s.headline, "passed / declared"), unsafe_allow_html=True
    )
with cols[1]:
    st.markdown(
        stat_tile("Failed assertions", str(s.failed), "real results"),
        unsafe_allow_html=True,
    )
with cols[2]:
    st.markdown(
        stat_tile("Not evaluated", str(s.unmeasured), "infrastructure"),
        unsafe_allow_html=True,
    )
with cols[3]:
    st.markdown(
        stat_tile(
            "Elapsed", f"{sum(r.duration_s for r in results):.0f}s"
        ),
        unsafe_allow_html=True,
    )

if s.unmeasured:
    st.warning(
        f"{s.unmeasured} fixture(s) could not be evaluated. The score above is "
        f"a **lower bound** — the denominator is fixtures declared, never "
        f"fixtures that happened to survive."
    )

# --- per-fixture ------------------------------------------------------------

st.markdown("")
by_id = {f["id"]: f for f in fixtures}

for result in results:
    status, glyph, word = STATUS_STYLE[result.status]
    state = result.state or {}
    with st.container(border=True):
        head, meta = st.columns([3, 2])
        with head:
            st.markdown(
                f'<div style="display:flex;gap:10px;align-items:center;">'
                f"{pill(status, glyph, word)}"
                f'<code style="font-size:13px;">{result.id}</code></div>',
                unsafe_allow_html=True,
            )
        with meta:
            st.markdown(
                f'<div style="text-align:right;color:{INK_MUTED};font-size:12px;">'
                f"{result.outcome or '—'} · decided by "
                f"{state.get('decided_by', '—')} · {result.duration_s:.1f}s</div>",
                unsafe_allow_html=True,
            )

        if result.failures:
            for failure in result.failures:
                st.markdown(
                    f'<div style="color:{STATUS["critical"]};font-size:13px;'
                    f'padding:2px 0;">— {failure}</div>',
                    unsafe_allow_html=True,
                )

        st.caption((by_id.get(result.id, {}).get("why") or "").strip())

        with st.expander("Assertions and trace"):
            st.markdown("**Asserted**")
            st.json(by_id.get(result.id, {}).get("assert", {}), expanded=True)
            if state.get("trace"):
                st.markdown("**Trace**")
                st.code("\n".join(state["trace"]), language="text")
            if state.get("draft_reply"):
                st.markdown("**Draft**")
                st.markdown(
                    f'<div style="color:{INK_2};font-size:13px;border-left:2px '
                    f'solid {HAIRLINE};padding-left:12px;">'
                    f'{state["draft_reply"]}</div>',
                    unsafe_allow_html=True,
                )
