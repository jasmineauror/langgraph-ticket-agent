"""Promote Streamlit secrets into the process environment.

Import this FIRST, before anything that touches `triage`. The ordering is
load-bearing, not stylistic.

`triage.llm` reads configuration at three different moments:

  * at IMPORT           -- MIN_REQUEST_INTERVAL_SECONDS
  * at FIRST MODEL CALL -- GEMINI_API_KEY / GROQ_API_KEY, then the client is
                           memoised for the life of the process
  * per call            -- TRIAGE_LLM, TRIAGE_MODEL_<ROLE>

The import-time one is the tight deadline, and it is tight in a way that is easy
to miss: Streamlit re-runs the *script* on every interaction, but
`import triage.llm` executes its module body exactly ONCE per server process, on
whichever browser session happens to arrive first. You cannot know which session
that is, so the promotion cannot live in a function that a page might forget to
call, nor in a cached resource that might first execute after some other module
has already imported `triage.llm`.

Two mechanisms pin it:

  1. The promotion runs in THIS MODULE'S BODY, so it happens at import.
  2. The last statement in this module imports `triage.llm`, forcing that
     module's body to execute inside this import, with the environment already
     correct. Python completes an imported module's body before the importer
     continues, which is what makes the ordering a guarantee rather than a hope.

`ui/__init__.py` must stay EMPTY for this to hold. If it imported anything that
reached `triage`, then `import ui.bootstrap` would trigger `ui/__init__.py`
first, and `triage.llm` would be imported before this body ever ran.
"""

from __future__ import annotations

import os

import streamlit as st

# Promoted to os.environ because that is the interface triage/ already reads.
# APP_PASSCODE is deliberately absent: nothing in triage/ needs it, and the
# process environment is visible in a traceback and to any dependency that dumps
# it. It stays in st.secrets and is read only by ui/auth.py.
PROMOTED_KEYS = (
    "TRIAGE_LLM",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "TRIAGE_MIN_INTERVAL",
    "TRIAGE_MODEL_CLASSIFIER",
    "TRIAGE_MODEL_RESPONDER",
    "TRIAGE_MODEL_JUDGE",
)


def _promote() -> list[str]:
    """Copy known secrets into os.environ. Returns the keys that were set.

    Secrets OVERRIDE the existing environment rather than deferring to it
    (`os.environ[k] = v`, not `setdefault`). On Streamlit Cloud there is no
    shell and no .env, so secrets are the only source; locally, a secrets.toml
    you wrote is a deliberate act and should beat a stale `export` in the shell
    that launched the server. Either choice is defensible, but a silently wrong
    API key is expensive, so the rule is written down rather than implied.
    """
    promoted: list[str] = []
    try:
        available = st.secrets
    except Exception:
        # No secrets file at all is a normal local state, not an error: the
        # environment or a Demo Mode run may supply everything needed.
        return promoted

    for key in PROMOTED_KEYS:
        try:
            value = available[key]
        except Exception:
            continue
        if value not in (None, ""):
            os.environ[key] = str(value)
            promoted.append(key)
    return promoted


PROMOTED = _promote()


def passcode() -> str | None:
    """The configured passcode, or None when the app is not configured.

    Never promoted to the environment. Returning None rather than raising lets
    the gate fail CLOSED with a clear message instead of a traceback.
    """
    try:
        value = st.secrets["APP_PASSCODE"]
    except Exception:
        return None
    return str(value) if value else None


def has_live_credentials() -> bool:
    """Whether any real backend could actually run, for the Demo Mode default."""
    return bool(
        os.environ.get("GROQ_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )


# LAST STATEMENT, and the reason this module exists. Forces triage.llm's module
# body -- including its import-time environment read -- to run now, with the
# environment already populated above.
import triage.llm  # noqa: E402,F401  (import ordering is the point)
