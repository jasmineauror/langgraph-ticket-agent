"""Single entry point for every model call in the graph.

Backend is chosen by the TRIAGE_LLM env var:

    ollama     (default) local model, free, offline, frozen weights
    anthropic  Claude via API, for isolating "small model" failures from
               "bad prompt" failures when an eval fixture fails
    stub       deterministic canned responses, for unit tests

Keeping every call behind one signature is what makes the model a variable in
the eval suite rather than a hardcoded assumption. When a fixture fails, being
able to rerun the identical suite on a stronger model tells you whether the
prompt is wrong or the model is too small -- otherwise that signal is ambiguous.
"""

from __future__ import annotations

import json
import os
from typing import Any

OLLAMA_MODEL = os.environ.get("TRIAGE_OLLAMA_MODEL", "qwen3:14b")
ANTHROPIC_MODEL = os.environ.get("TRIAGE_ANTHROPIC_MODEL", "claude-sonnet-5")


class LLMError(RuntimeError):
    pass


# --- stub backend, for unit tests -------------------------------------------

_STUB_RESPONSES: dict[str, Any] = {}


def set_stub(key: str, value: Any) -> None:
    """Register a canned response. `key` is matched as a substring of the prompt."""
    _STUB_RESPONSES[key] = value


def clear_stub() -> None:
    _STUB_RESPONSES.clear()


def _call_stub(system: str, user: str, schema: dict | None) -> Any:
    for key, value in _STUB_RESPONSES.items():
        if key in system or key in user:
            return value
    raise LLMError(f"stub has no response registered for prompt: {user[:80]!r}")


# --- ollama backend ---------------------------------------------------------


def _call_ollama(system: str, user: str, schema: dict | None) -> Any:
    import ollama

    kwargs: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # qwen3 emits <think> blocks by default, which corrupt structured output
        # and waste tokens on decisions this simple.
        "think": False,
        "options": {"temperature": 0.0},
    }
    if schema is not None:
        # Ollama constrains decoding to the schema, so the shape is reliable
        # even on a small model. Label *correctness* is still the model's job.
        kwargs["format"] = schema

    response = ollama.chat(**kwargs)
    content = response["message"]["content"]
    return _parse(content, schema)


# --- anthropic backend ------------------------------------------------------


def _call_anthropic(system: str, user: str, schema: dict | None) -> Any:
    import anthropic

    client = anthropic.Anthropic()
    if schema is not None:
        system = (
            f"{system}\n\nRespond with a single JSON object matching this schema, "
            f"and nothing else:\n{json.dumps(schema)}"
        )

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    content = "".join(b.text for b in response.content if b.type == "text")
    return _parse(content, schema)


# --- shared -----------------------------------------------------------------


def _parse(content: str, schema: dict | None) -> Any:
    if schema is None:
        return content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Recover a JSON object embedded in prose, which is the usual failure
        # mode when a backend does not hard-constrain decoding.
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise LLMError(f"could not parse JSON from model output: {content[:200]!r}")


_BACKENDS = {
    "ollama": _call_ollama,
    "anthropic": _call_anthropic,
    "stub": _call_stub,
}


def backend_name() -> str:
    return os.environ.get("TRIAGE_LLM", "ollama").lower()


def model_name() -> str:
    """The specific model in play, for the eval log."""
    name = backend_name()
    return {
        "ollama": OLLAMA_MODEL,
        "anthropic": ANTHROPIC_MODEL,
        "stub": "stub",
    }.get(name, name)


def call(system: str, user: str, schema: dict | None = None) -> Any:
    """Run one model call.

    With `schema`, returns a parsed dict. Without, returns stripped text.
    Retries once on unparseable output before giving up.
    """
    name = backend_name()
    if name not in _BACKENDS:
        raise LLMError(f"unknown TRIAGE_LLM backend {name!r}; expected one of {sorted(_BACKENDS)}")
    fn = _BACKENDS[name]

    try:
        return fn(system, user, schema)
    except LLMError:
        if name == "stub":
            raise
        # One retry. Malformed JSON from a small model is usually transient.
        return fn(system, user, schema)
