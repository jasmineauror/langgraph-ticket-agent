"""Ticket Triage Console -- entry point.

The import order in this file is load-bearing:

  1. `ui.bootstrap` FIRST, so Streamlit secrets reach os.environ before
     `triage.llm`'s module body runs and reads its import-time configuration.
  2. `st.set_page_config` before anything renders.
  3. The gate, which `st.stop()`s unless this session is authorised.
  4. `st.navigation` -- and only here are the pages named. A page never handed
     to st.navigation is not routable, which is why there is no top-level
     `pages/` directory: Streamlit auto-discovers that one and makes every file
     in it URL-addressable, which would be a structural bypass of the gate.
"""

from __future__ import annotations

import ui.bootstrap as bootstrap  # noqa: F401  # must be the first project import

import streamlit as st

st.set_page_config(
    page_title="Ticket Triage Console",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.auth import render_gate  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402

render_gate()

render_sidebar()

pages = [
    st.Page(
        "ui/pages/triage_console.py",
        title="Triage Console",
        icon=":material/support_agent:",
        default=True,
    ),
    st.Page("ui/pages/eval_suite.py", title="Eval Suite", icon=":material/science:"),
    st.Page(
        "ui/pages/knowledge_base.py",
        title="Knowledge Base",
        icon=":material/library_books:",
    ),
]

st.navigation(pages).run()
