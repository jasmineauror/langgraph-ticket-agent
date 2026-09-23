"""Classifier node: tag the ticket and decide whether it is auto-answerable."""

from __future__ import annotations

from .. import llm, prompts
from ..state import TicketState


def classify(state: TicketState) -> TicketState:
    result, model = llm.call_with_model(
        role="classifier",
        system=prompts.CLASSIFIER_SYSTEM,
        user=f"Ticket:\n\n{state['ticket_text']}",
        schema=prompts.CLASSIFIER_SCHEMA,
        backend=state.get("backend") or None,
    )

    category = result["category"]
    auto_answerable = bool(result["auto_answerable"])

    return {
        "category": category,
        "auto_answerable": auto_answerable,
        "classifier_reasoning": result.get("reasoning", ""),
        "served_by": {"classifier": model},
        "trace": [
            f"[classifier] ({model}) category={category} "
            f"auto_answerable={auto_answerable} -- {result.get('reasoning', '')}"
        ],
    }
