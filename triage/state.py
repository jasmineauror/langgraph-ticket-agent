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
    # Backend for this run only, overriding TRIAGE_LLM. Carried in state rather
    # than read from the environment because the environment is process-wide: a
    # UI that flipped it to "stub" for one session would silently serve stub
    # replies to every other session connected at the time.
    backend: str

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

    # Which model actually answered, per role. Populated from the return value
    # of llm.call_with_model rather than read from a module global, which is
    # shared across threads and would attribute one run's model to another.
    # operator.or_ merges each node's single entry; the judge's retry overwrites
    # its own key, which is the behaviour we want.
    served_by: Annotated[dict[str, str], operator.or_]

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


def new_state(ticket_text: str, backend: str | None = None) -> TicketState:
    """Build the initial state for a ticket.

    `backend` pins this run to one backend ("groq", "gemini", "stub"); omitted,
    every call falls back to TRIAGE_LLM.
    """
    return TicketState(
        ticket_text=ticket_text,
        backend=backend or "",
        retrieved=[],
        max_similarity=0.0,
        cited_sources=[],
        responder_refused=False,
        retry_count=0,
        judge_checks={},
        served_by={},
        trace=[],
    )
