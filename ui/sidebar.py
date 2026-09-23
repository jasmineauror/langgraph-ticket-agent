"""Sidebar: run configuration and operational state.

Rendered from the entry point so it is identical on every page.
"""

from __future__ import annotations

import ui.bootstrap as bootstrap  # noqa: F401  (must precede any triage import)

import streamlit as st

from triage import llm
from triage.nodes.judge import GROUNDING_FLOOR
from ui.render import HAIRLINE, INK_2, INK_MUTED, STATUS, pill


def active_backend() -> str | None:
    """The backend for this session's runs, or None to follow TRIAGE_LLM.

    Demo Mode is per-session because it is carried in TicketState, not in the
    environment. An environment flag would be process-wide: one visitor
    switching to Demo Mode would silently serve canned replies to everyone else
    connected at the time, which is a wrong answer presented as a real one.
    """
    return "demo" if st.session_state.get("demo_mode") else None


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Run configuration")


        live = bootstrap.has_live_credentials()
        # Default Demo Mode ON when no key exists, so the app is usable and
        # honest on a fresh clone rather than erroring on every ticket.
        st.session_state.setdefault("demo_mode", not live)

        st.toggle(
            "Demo mode",
            key="demo_mode",
            help=(
                "Runs the real graph, the real vector search and the real "
                "grounding gates, with canned model replies instead of API "
                "calls. Nothing leaves this machine and no quota is consumed."
            ),
        )

        if st.session_state.demo_mode:
            st.markdown(
                pill("warning", "◆", "canned model replies"),
                unsafe_allow_html=True,
            )
            st.caption(
                "Routing, retrieval scores and judge gates are genuine. Only "
                "the model's wording is fixed."
            )
        elif not live:
            st.markdown(pill("critical", "✕", "no API key"), unsafe_allow_html=True)
            st.caption("Set GROQ_API_KEY in secrets, or use Demo mode.")
        else:
            st.markdown(
                pill("good", "✓", f"live · {llm.backend_name()}"),
                unsafe_allow_html=True,
            )
            st.caption(f"Models: {llm.model_name()}")

        st.markdown(
            f'<hr style="border:none;border-top:1px solid {HAIRLINE};'
            f'margin:16px 0;">',
            unsafe_allow_html=True,
        )

        st.markdown("### Pipeline")
        st.markdown(
            f'<div style="color:{INK_2};font-size:13px;line-height:1.7;">'
            f"Grounding floor <code>{GROUNDING_FLOOR:.2f}</code><br>"
            f"Judge retries <code>1</code> max<br>"
            f"Retrieval <code>top-4</code>"
            f"</div>",
            unsafe_allow_html=True,
        )

        _render_breaker()


def _render_breaker() -> None:
    """Circuit-breaker state, and the only way to clear it.

    `circuit_state()` is read live on every render -- never cached. A cached
    safety signal is a stale one: it would show "all clear" while every call
    failed.
    """
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {HAIRLINE};'
        f'margin:16px 0;">',
        unsafe_allow_html=True,
    )
    st.markdown("### Model circuit")

    tripped = llm.circuit_state()
    if not tripped:
        st.markdown(pill("good", "✓", "all models available"), unsafe_allow_html=True)
        return

    for model, remaining in sorted(tripped.items()):
        # A daily-quota trip is an 86,400s cooldown; a capacity trip is 120s.
        daily = remaining > 3600
        status = "critical" if daily else "warning"
        detail = "daily quota" if daily else f"{remaining:.0f}s"
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;gap:8px;'
            f'align-items:center;padding:4px 0;">'
            f'<code style="font-size:11px;color:{INK_MUTED};">{model}</code>'
            f"{pill(status, '✕', detail)}</div>",
            unsafe_allow_html=True,
        )

    st.caption(
        "A daily-quota trip lasts 24h and would otherwise persist for the life "
        "of this server. Clearing re-probes each model, costing one fast 429 or "
        "one slow 503 each."
    )
    if st.button("Reset circuit", use_container_width=True):
        cleared = llm.clear_circuit()
        st.success(f"Cleared {len(cleared)} model(s).")
        st.rerun()
