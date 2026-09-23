"""The graph's state contract.

Every node reads and writes only the keys declared here. Nodes never reach into
one another; the state is the interface between them.
"""

from typing import Annotated, Literal, TypedDict

import operator

Category = Literal["billing", "technical", "account", "abusive_or_out_of_scope"]
Verdict = Literal["SEND", "RETRY", "ESCALATE"]


class Chunk(TypedDict):
    """One retrieved knowledge-base passage."""

    text: str
    source: str
    distance: float  # cosine distance from Chroma; lower is more similar


class TicketState(TypedDict, total=False):
    # --- input ---
    ticket_text: str

    # --- classifier ---
    category: Category
    auto_answerable: bool
    classifier_reasoning: str

    # --- retriever ---
    retrieved: list[Chunk]
    max_similarity: float  # 1 - min(distance); 0.0 when nothing was retrieved

    # --- responder ---
    draft_reply: str
    cited_sources: list[str]
    responder_refused: bool  # the model itself declined to answer

    # --- judge ---
    verdict: Verdict
    # Which node produced the terminal decision. Recorded rather than inferred,
    # because an eval that checks the outcome without checking the mechanism
    # passes when the right answer arrives from the wrong node -- and then
    # silently stops testing anything the day that node changes its mind.
    decided_by: str
    escalation_reason: str
    judge_checks: dict[str, bool]

    # --- control ---
    retry_count: int

    # --- observability ---
    # operator.add so each node appends rather than clobbering the log
    trace: Annotated[list[str], operator.add]


def new_state(ticket_text: str) -> TicketState:
    """Build the initial state for a ticket."""
    return TicketState(
        ticket_text=ticket_text,
        retrieved=[],
        max_similarity=0.0,
        cited_sources=[],
        responder_refused=False,
        retry_count=0,
        judge_checks={},
        trace=[],
    )
