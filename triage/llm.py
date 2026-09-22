"""Single entry point for every model call in the graph.

Two backends, selected by the TRIAGE_LLM env var:

    anthropic  (default) Claude via API
    stub       deterministic canned responses, for unit tests

Every call names the *role* making it, and each role is bound to a specific
model. Routing a mechanical classification to Haiku and reserving Sonnet for
the two calls that need judgment is a deliberate cost choice, not an oversight:
the classifier decides among four fixed labels, while the judge is the node
whose mistakes are expensive.

Because the role is a parameter, the same eval suite can be rerun with a
different model per role -- which is what makes "does this fixture fail because
my prompt is wrong, or because the model is too weak" an answerable question
instead of a guess.
"""

from __future__ import annotations

import json
import os
from typing import Any

# Per-role model assignment. Override any of these from the environment to
# benchmark a role on a different model without touching the graph.
MODELS = {
    "classifier": os.environ.get("TRIAGE_MODEL_CLASSIFIER", "claude-haiku-4-5"),
    "responder": os.environ.get("TRIAGE_MODEL_RESPONDER", "claude-sonnet-5"),
    "judge": os.environ.get("TRIAGE_MODEL_JUDGE", "claude-sonnet-5"),
}

# Output ceilings sized to each role's job. Kept modest to bound cost, but
# generous enough that a reply is never truncated mid-sentence.
MAX_TOKENS = {"classifier": 512, "responder": 1536, "judge": 768}

# Thinking is enabled only where reasoning actually changes the answer. The
# classifier picks one of four labels and the responder rewrites retrieved text;
# the judge weighs four competing conditions, so it gets adaptive thinking.
THINKING = {
    "classifier": None,  # Haiku 4.5 without a budget: no thinking
    "responder": {"type": "disabled"},
    "judge": "adaptive",  # omit the param -> Sonnet 5 runs adaptive
}


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


# --- anthropic backend ------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def _call_anthropic(role: str, system: str, user: str, schema: dict | None) -> Any:
    if schema is not None:
        system = (
            f"{system}\n\nRespond with a single JSON object matching this schema, "
            f"and nothing else -- no prose, no code fence:\n{json.dumps(schema)}"
        )

    kwargs: dict[str, Any] = {
        "model": MODELS[role],
        "max_tokens": MAX_TOKENS[role],
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    thinking = THINKING[role]
    if isinstance(thinking, dict):
        kwargs["thinking"] = thinking
    # `None` and "adaptive" both mean "send no thinking param"; the difference is
    # the model's own default, which the table in THINKING documents.

    response = _get_client().messages.create(**kwargs)

    if response.stop_reason == "refusal":
        raise LLMError(f"model declined the request in role {role!r}")

    content = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return _parse(content, schema)


# --- shared -----------------------------------------------------------------


def _parse(content: str, schema: dict | None) -> Any:
    if schema is None:
        return content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Recover a JSON object wrapped in prose or a code fence.
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise LLMError(f"could not parse JSON from model output: {content[:200]!r}")


_BACKENDS = {
    "anthropic": _call_anthropic,
    "stub": _call_stub,
}


def backend_name() -> str:
    return os.environ.get("TRIAGE_LLM", "anthropic").lower()


def model_name(role: str | None = None) -> str:
    """The model in play, for the eval log and the CLI header."""
    if backend_name() == "stub":
        return "stub"
    if role:
        return MODELS[role]
    distinct = sorted(set(MODELS.values()))
    return " + ".join(distinct)


def call(role: str, system: str, user: str, schema: dict | None = None) -> Any:
    """Run one model call on behalf of `role`.

    With `schema`, returns a parsed dict. Without, returns stripped text.
    Retries once on unparseable output before giving up.
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
