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
import logging
import os
import random
import time
from typing import Any

# The SDK warns about automatic function calling on every generate_content call
# even when no tools are declared. It is noise that would bury real eval output.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# Free-tier capacity is shared, so 503 and 429 are routine rather than
# exceptional. An eval suite that dies on the first blip is not measuring the
# prompts, so transient failures are retried with exponential backoff and
# jitter; 4xx other than 429 is a real bug and fails immediately.
TRANSIENT_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 1.5

# Per-role model preference, best first. Free-tier capacity is shared and
# genuinely volatile -- measured over two minutes, gemini-3.5-flash-lite went
# from serving to 503 and gemini-3.8-flash was saturated throughout -- so a
# single pinned model per role cannot keep an eval suite runnable. Each role
# instead walks its chain until one model answers.
#
# Set TRIAGE_MODEL_<ROLE> to pin a role to exactly one model, which is what you
# want when benchmarking: a fallback that silently swaps the model would make
# the eval result ambiguous.
MODEL_CHAINS = {
    "classifier": ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3-flash-preview"],
    "responder": ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3-flash-preview"],
    "judge": ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3-flash-preview"],
}


def _chain(role: str) -> list[str]:
    pinned = os.environ.get(f"TRIAGE_MODEL_{role.upper()}")
    return [pinned] if pinned else MODEL_CHAINS[role]


# Which model actually served each role on the last call. Fallback means the
# configured preference is not necessarily what produced a result, and an eval
# score attributed to the wrong model is worse than no score -- so the node
# traces record this, and it shows up in the eval output.
_LAST_MODEL: dict[str, str] = {}


def last_model(role: str) -> str:
    return _LAST_MODEL.get(role, "?")

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
    """Walk the role's model chain until one answers."""
    chain = _chain(role)
    errors_seen: list[str] = []

    for index, model in enumerate(chain):
        try:
            result = _generate(model, role, system, user, schema)
            _LAST_MODEL[role] = model
            if index > 0:
                logging.getLogger(__name__).warning(
                    "role=%s fell back to %s after %s", role, model, "; ".join(errors_seen)
                )
            return result
        except Exception as exc:
            if not (isinstance(exc, LLMError) or _is_transient(exc) or _is_unavailable(exc)):
                raise
            errors_seen.append(f"{model}: {type(exc).__name__}")
            continue

    raise LLMError(
        f"every model for role {role!r} failed -- {'; '.join(errors_seen)}"
    )


def _generate(model: str, role: str, system: str, user: str, schema: dict | None) -> Any:
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

    response = _retry_transient(
        lambda: _get_client().models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(**config_kwargs),
        ),
        label=f"{role}/{model}",
    )

    text = response.text
    if not text:
        raise LLMError(
            f"empty response from {model} in role {role!r} "
            f"(finish_reason={_finish_reason(response)!r}); "
            f"if MAX_TOKENS, raise MAX_OUTPUT_TOKENS[{role!r}]"
        )

    return _parse(text, schema)


def _is_transient(exc: Exception) -> bool:
    """Worth retrying the same model."""
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        return getattr(exc, "code", None) in TRANSIENT_STATUS
    return False


def _is_unavailable(exc: Exception) -> bool:
    """Not worth retrying, but worth trying the next model in the chain.

    A 404 means this key cannot reach that model at all -- several models the
    list endpoint advertises return 404 on generateContent.
    """
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        return getattr(exc, "code", None) in {403, 404}
    return False


def _retry_transient(thunk, label: str):
    """Retry `thunk` on transient API failures with exponential backoff."""
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return thunk()
        except Exception as exc:
            if not _is_transient(exc):
                raise
            last = exc
            if attempt == MAX_ATTEMPTS - 1:
                raise
            delay = BACKOFF_BASE_SECONDS * (2**attempt) + random.uniform(0, 0.5)
            logging.getLogger(__name__).info(
                "%s attempt %d/%d: %s; retrying in %.1fs",
                label, attempt + 1, MAX_ATTEMPTS, type(exc).__name__, delay,
            )
            time.sleep(delay)
    raise last  # unreachable, kept for type clarity


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
    """The preferred model, for the CLI header. Use last_model() for what ran."""
    if backend_name() == "stub":
        return "stub"
    if role:
        return _chain(role)[0]
    return " + ".join(dict.fromkeys(_chain(r)[0] for r in MODEL_CHAINS))


def call(role: str, system: str, user: str, schema: dict | None = None) -> Any:
    """Run one model call on behalf of `role`.

    With `schema`, returns a parsed dict. Without, returns stripped text.

    Transient API failures (429, 5xx) are retried against the same model with
    exponential backoff; a model that stays unavailable is skipped for the next
    one in the role's chain. `last_model(role)` reports which one answered.
    """
    if role not in MODEL_CHAINS:
        raise LLMError(f"unknown role {role!r}; expected one of {sorted(MODEL_CHAINS)}")

    name = backend_name()
    if name not in _BACKENDS:
        raise LLMError(
            f"unknown TRIAGE_LLM backend {name!r}; expected one of {sorted(_BACKENDS)}"
        )

    if name == "stub":
        _LAST_MODEL[role] = "stub"
    return _BACKENDS[name](role, system, user, schema)
