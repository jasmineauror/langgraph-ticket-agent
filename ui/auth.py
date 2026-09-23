"""The passcode gate.

Honest about what it is: a shared secret sent over HTTPS, with no server-side
rate limiting beyond a per-session attempt counter. It keeps crawlers and a
casually shared link from burning a daily model quota. It is not
authentication, and the README says so.
"""

from __future__ import annotations

import hmac

import streamlit as st

import ui.bootstrap as bootstrap

MAX_ATTEMPTS = 8


def _authed() -> bool:
    return bool(st.session_state.get("authed"))


def render_gate() -> None:
    """Render the gate and stop the script unless this session is authorised.

    `authed` lives in session_state and MUST NOT be cached. `st.cache_data` and
    `st.cache_resource` are process-wide, so caching this flag would
    authenticate every future session in the process -- the highest-severity
    mistake available in this app.
    """
    if _authed():
        return

    expected = bootstrap.passcode()

    if expected is None:
        # Fail CLOSED. A missing secret is a misconfiguration, never a reason to
        # open the door.
        st.title("Ticket Triage Console")
        st.error(
            "This app is not configured: no `APP_PASSCODE` secret is set. "
            "Add one under **Advanced settings → Secrets** (see "
            "`.streamlit/secrets.toml.example`)."
        )
        st.stop()

    attempts = st.session_state.get("auth_attempts", 0)

    st.title("Ticket Triage Console")
    st.caption(
        "A LangGraph support-triage pipeline with retrieval grounding and an "
        "escalation judge."
    )

    if attempts >= MAX_ATTEMPTS:
        st.error(
            f"Too many attempts in this session ({attempts}). Reload the page "
            f"to try again."
        )
        st.stop()

    with st.form("gate", border=True):
        st.markdown("**Access code**")
        entered = st.text_input(
            "Access code",
            type="password",
            label_visibility="collapsed",
            placeholder="Enter the access code",
        )
        submitted = st.form_submit_button("Continue", type="primary")

    if submitted:
        # compare_digest, not ==, so the comparison does not leak length or a
        # common prefix through timing.
        if hmac.compare_digest(entered or "", expected):
            st.session_state.authed = True
            st.session_state.auth_attempts = 0
            st.rerun()
        else:
            st.session_state.auth_attempts = attempts + 1
            st.error("Incorrect access code.")

    with st.expander("What is this?"):
        st.markdown(
            """
A support ticket goes through four specialised stages:

1. **Classifier** — tags the ticket, and routes abuse or account actions
   straight to a human
2. **Retriever** — semantic search over a 24-chunk knowledge base, returning
   similarity scores rather than just text
3. **Responder** — drafts a reply **only** from retrieved text, or explicitly
   declines
4. **Escalation Judge** — two code gates plus a four-part rubric decide whether
   the draft may be sent

The interesting case is the ticket the knowledge base cannot answer. The
correct behaviour is to refuse and escalate, not to produce a confident,
plausible, ungrounded reply.
            """
        )

    st.stop()


def require_auth() -> None:
    """Belt-and-braces for page modules.

    Pages are only handed to `st.navigation` after the gate passes, so an
    unauthorised page is not routable at all. This is the second line, not the
    first -- and `st.stop()` rather than `return`, because a return would let
    the caller carry on rendering.
    """
    if not _authed():
        st.error("Not authorised.")
        st.stop()
