"""Classifier node: tag the ticket and decide whether it is auto-answerable."""

from __future__ import annotations

from .. import llm, prompts
from ..state import TicketState


def classify(state: TicketState) -> TicketState:
    result = llm.call(
        system=prompts.CLASSIFIER_SYSTEM,
        user=f"Ticket:\n\n{state['ticket_text']}",
        schema=prompts.CLASSIFIER_SCHEMA,
    )

    category = result["category"]
    auto_answerable = bool(result["auto_answerable"])

    return {
        "category": category,
        "auto_answerable": auto_answerable,
        "classifier_reasoning": result.get("reasoning", ""),
        "trace": [
            f"[classifier] category={category} auto_answerable={auto_answerable} "
            f"-- {result.get('reasoning', '')}"
        ],
    }
