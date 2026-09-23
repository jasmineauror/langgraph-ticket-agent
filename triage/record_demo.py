"""Record a live run of every eval fixture, for Demo Mode to replay.

    TRIAGE_LLM=groq python -m triage.record_demo

Writes `triage/demo_responses.json`: for each model call the pipeline makes,
the exact response the live model gave, keyed by the prompt that produced it.

Demo Mode then replays these. The graph, the vector search, both code gates and
the judge rubric all run for real; only the model's wording is fixed. That
means a demo cannot fail because a quota ran out or a container was cold, and
what it shows is still a genuine run rather than a mock.

Re-record whenever the prompts, the knowledge base or the chunker change --
the key is a hash of the full prompt, so drift causes a miss and a visible
"generic" label rather than a silently mismatched reply.
"""

from __future__ import annotations

import json
from typing import Any

from evals.runner import load_fixtures
from triage import llm
from triage.graph import run_ticket

_recorded: dict[str, Any] = {}
_original = llm.call_with_model


def _recording_call(role, system, user, schema=None, backend=None):
    result, model = _original(role, system, user, schema, backend)
    _recorded[llm._demo_key(role, user)] = result
    return result, model


def main() -> None:
    if llm.backend_name() in ("demo", "stub"):
        raise SystemExit(
            "Recording needs a live backend. Run with TRIAGE_LLM=groq (or gemini)."
        )

    llm.call_with_model = _recording_call
    # Nodes call through the module attribute, so patching it here is enough.
    try:
        fixtures = load_fixtures()
        fallback: dict[str, Any] = {}

        for fixture in fixtures:
            state = run_ticket(fixture["ticket"].strip())
            print(
                f"  recorded {fixture['id']:26} "
                f"{state.get('verdict'):9} via {state.get('decided_by')}"
            )
            # The last classifier/responder/judge reply seen becomes the generic
            # fallback for a ticket nobody recorded. A conservative choice would
            # be better than a confident one, so prefer an escalating fixture.
            if fixture["id"] == "no-kb-answer":
                for key, value in _recorded.items():
                    fallback[key.split(":", 1)[0]] = value
    finally:
        llm.call_with_model = _original

    payload = {
        "_note": (
            "Recorded by `python -m triage.record_demo`. Keys are "
            "sha256(role + prompt)[:16]. Re-record when prompts, the knowledge "
            "base, or the chunker change."
        ),
        "responses": _recorded,
        "fallback": fallback,
    }
    llm.DEMO_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(
        f"\nwrote {llm.DEMO_PATH} "
        f"({len(_recorded)} responses, {len(fallback)} fallbacks)"
    )


if __name__ == "__main__":
    main()
