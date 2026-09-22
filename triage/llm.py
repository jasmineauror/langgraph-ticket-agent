"""Single entry point for every model call in the graph.

Two backends, selected by the TRIAGE_LLM env var:

    gemini  (default) Gemini via the Google AI API
    stub    deterministic canned responses, for unit tests

Every call names the *role* making it, and each role is bound to its own model
and thinking level. The classifier chooses among four fixed labels, so it gets
the cheapest model at minimal thinking. The judge weighs four competing
conditions and its mistakes are the ones that reach a customer, so it gets the
strongest model at high thinking.

Because the role is a parameter, the same eval suite can be rerun with a
different model in one role -- which turns "does this fixture fail because my
prompt is wrong, or because the model is too weak" into an answerable question.

Structured output is enforced by the API: passing `response_schema` constrains
decoding, so malformed JSON is not a failure mode we have to defend against in
prompt text.
"""

from __future__ import annotations

import json
import os
from typing import Any

# Per-role model assignment. Override any of these from the environment to
# benchmark a role on a different model without touching the graph.
MODELS = {
    "classifier": os.environ.get("TRIAGE_MODEL_CLASSIFIER", "gemini-3.5-flash-lite"),
    "responder": os.environ.get("TRIAGE_MODEL_RESPONDER", "gemini-3.8-flash"),
    "judge": os.environ.get("TRIAGE_MODEL_JUDGE", "gemini-3.8-flash"),
}

# Thinking effort per role, matched to how much the decision benefits from it.
THINKING_LEVEL = {
    "classifier": "MINIMAL",
    "responder": "LOW",
    "judge": "HIGH",
}

# Thinking tokens are drawn from the same budget as the visible response. A
# ceiling sized only for the answer therefore returns EMPTY text rather than an
# error when the model thinks: the budget is spent before it writes anything.
# These are sized for thinking level plus answer, not answer alone.
MAX_OUTPUT_TOKENS = {"classifier": 1024, "responder": 4096, "judge": 8192}


class LLMError(RuntimeError):
    pass


# --- stub backend, for unit tests -------------------------------------------

_STUB_RESPONSES: dict[str, Any] = {}


def set_stub(key: str, value: Any) -> None:
    """Register a canned response. `key` is matched as a substring of the prompt."""
    _STUB_RESPONSES[key] = value


def clear_stub() -> None:
    _STUB_RESPONSES.clear()


def _call_stub(role: str, system: str, user: str, schema: dict | None) -> Any:
    for key, value in _STUB_RESPONSES.items():
        if key in system or key in user:
            return value
    raise LLMError(f"stub has no response registered for prompt: {user[:80]!r}")


# --- gemini backend ---------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise LLMError(
                "no API key found: set GEMINI_API_KEY (free key from "
                "https://aistudio.google.com/apikey)"
            )
        _client = genai.Client()
    return _client


def _call_gemini(role: str, system: str, user: str, schema: dict | None) -> Any:
    from google.genai import types

    config_kwargs: dict[str, Any] = {
        "system_instruction": system,
        "max_output_tokens": MAX_OUTPUT_TOKENS[role],
        "thinking_config": types.ThinkingConfig(
            thinking_level=THINKING_LEVEL[role]
        ),
    }

    if schema is not None:
        # Constrains decoding to the schema, so the shape is guaranteed rather
        # than requested.
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = schema

    response = _get_client().models.generate_content(
        model=MODELS[role],
        contents=user,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    text = response.text
    if not text:
        raise LLMError(
            f"empty response in role {role!r} "
            f"(finish_reason={_finish_reason(response)!r}); "
            f"if MAX_TOKENS, raise MAX_OUTPUT_TOKENS[{role!r}]"
        )

    return _parse(text, schema)


def _finish_reason(response) -> str | None:
    try:
        return str(response.candidates[0].finish_reason)
    except (AttributeError, IndexError, TypeError):
        return None


# --- shared -----------------------------------------------------------------


def _parse(content: str, schema: dict | None) -> Any:
    if schema is None:
        return content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Should not happen with response_schema set, but a code fence or
        # trailing prose is cheap to recover from.
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise LLMError(f"could not parse JSON from model output: {content[:200]!r}")


_BACKENDS = {
    "gemini": _call_gemini,
    "stub": _call_stub,
}


def backend_name() -> str:
    return os.environ.get("TRIAGE_LLM", "gemini").lower()


def model_name(role: str | None = None) -> str:
    """The model in play, for the eval log and the CLI header."""
    if backend_name() == "stub":
        return "stub"
    if role:
        return MODELS[role]
    return " + ".join(sorted(set(MODELS.values())))


def call(role: str, system: str, user: str, schema: dict | None = None) -> Any:
    """Run one model call on behalf of `role`.

    With `schema`, returns a parsed dict. Without, returns stripped text.
    Retries once on a transient failure before giving up.
    """
    if role not in MODELS:
        raise LLMError(f"unknown role {role!r}; expected one of {sorted(MODELS)}")

    name = backend_name()
    if name not in _BACKENDS:
        raise LLMError(
            f"unknown TRIAGE_LLM backend {name!r}; expected one of {sorted(_BACKENDS)}"
        )
    fn = _BACKENDS[name]

    try:
        return fn(role, system, user, schema)
    except LLMError:
        if name == "stub":
            raise
        return fn(role, system, user, schema)
