"""Responder node: draft a reply from the retrieved context."""

from __future__ import annotations

from .. import llm, prompts
from ..state import TicketState

REFUSAL_MARKER = "INSUFFICIENT_CONTEXT"


def _format_context(state: TicketState) -> str:
    if not state.get("retrieved"):
        return "(no knowledge base excerpts were retrieved)"
    return "\n\n---\n\n".join(
        f"[source: {c['source']}]\n{c['text']}" for c in state["retrieved"]
    )


def respond(state: TicketState) -> TicketState:
    user = (
        f"Ticket:\n\n{state['ticket_text']}\n\n"
        f"Knowledge base excerpts:\n\n{_format_context(state)}"
    )

    result = llm.call(
        system=prompts.RESPONDER_SYSTEM,
        user=user,
        schema=prompts.RESPONDER_SCHEMA,
    )

    reply = result["reply"]
    refused = REFUSAL_MARKER in reply

    return {
        "draft_reply": reply,
        "cited_sources": result.get("cited_sources", []),
        "responder_refused": refused,
        "trace": [
            f"[responder] drafted {len(reply)} chars, "
            f"cited={result.get('cited_sources', [])}, refused={refused}"
        ],
    }
