"""Escalation Judge node: the last check before a draft reaches a customer.

The decision is split deliberately between code and model judgment, because the
two are good at different things.

**Code gates run first.** Two conditions are facts about state, not matters of
opinion: the responder explicitly declined, or retrieval found nothing on the
subject at all. Asking a model to re-derive either would be slower, cost a
request, and could be argued out of.

**The model handles what code cannot.** Whether a specific sentence is
traceable to a specific excerpt, and whether a topic commits the company to
something, are reading-comprehension problems. That is the model's job.

The floor is set low on purpose. Round 0 measured only 0.070 of separation
between the worst answerable fixture (0.587) and a question the knowledge base
genuinely cannot answer (0.517), because cosine similarity captures topical
relatedness rather than whether a passage contains the answer. A threshold
placed in that gap would be fitted to six examples. At 0.40 the gate makes only
the claim the scores actually support -- "nothing retrieved is even on this
subject" -- and the harder on-topic-but-unanswered case is left to the grounding
check, which reads the text.
"""

from __future__ import annotations

from .. import llm, prompts
from ..state import TicketState

MAX_RETRIES = 1

# Below this, nothing retrieved is on the subject at all. Deliberately not
# placed in the narrow gap between answerable and unanswerable fixtures.
GROUNDING_FLOOR = 0.40

# Failures a NEW DRAFT could plausibly fix. Only these are worth a retry.
REDRAFTABLE_CHECKS = (
    "addresses_ticket",
    "grounded_in_sources",
    "cites_sources",
)


def _escalate(
    state: TicketState, reason: str, checks: dict, redraftable: bool = True
) -> TicketState:
    """Escalate, or spend one retry on the responder first.

    `redraftable=False` skips the retry entirely. Some failures are facts about
    the TICKET rather than faults in the draft -- a seat-billing dispute does
    not stop being about money because the reply was reworded -- and retrying
    them is worse than useless: it hands the judge a second independent chance
    to reach the opposite conclusion.

    Measured, on the strong model: the judge flagged touches_sensitive, the
    responder reworded, and the judge then reported touches_sensitive=False and
    approved the reply. Retrying a ticket-level fact converts one correct
    escalation into a coin flip.
    """
    if redraftable and state.get("retry_count", 0) < MAX_RETRIES:
        return {
            "verdict": "RETRY",
            "decided_by": "judge",
            "escalation_reason": reason,
            "judge_checks": checks,
            "retry_count": state.get("retry_count", 0) + 1,
            "trace": [
                f"[judge] RETRY (attempt {state.get('retry_count', 0) + 1}) -- {reason}"
            ],
        }
    return {
        "verdict": "ESCALATE",
        "decided_by": "judge",
        "escalation_reason": reason,
        "judge_checks": checks,
        "trace": [f"[judge] ESCALATE -- checks={checks} -- {reason}"],
    }


def judge(state: TicketState) -> TicketState:
    max_similarity = state.get("max_similarity", 0.0)

    # --- code gates: facts about state, decided without a model call ---

    if state.get("responder_refused"):
        return _escalate(
            state,
            "the responder declined to answer: the retrieved excerpts do not "
            "contain what the customer asked for",
            {"responder_refused": True},
        )

    if max_similarity < GROUNDING_FLOOR:
        return _escalate(
            state,
            f"nothing in the knowledge base is on this subject "
            f"(max_similarity={max_similarity:.3f} < {GROUNDING_FLOOR})",
            {"above_grounding_floor": False},
        )

    # --- model judgment: reading comprehension over the excerpts ---

    sources = "\n\n---\n\n".join(
        f"[source: {c['source']}]\n{c['text']}" for c in state.get("retrieved", [])
    )
    user = (
        f"Ticket:\n\n{state['ticket_text']}\n\n"
        f"Drafted reply:\n\n{state['draft_reply']}\n\n"
        f"Sources the draft cited: {state.get('cited_sources') or '(none)'}\n\n"
        f"Knowledge base excerpts the draft was given:\n\n{sources}"
    )

    result = llm.call(
        role="judge",
        system=prompts.JUDGE_SYSTEM,
        user=user,
        schema=prompts.JUDGE_SCHEMA,
    )

    checks = {
        "addresses_ticket": bool(result.get("addresses_ticket")),
        "grounded_in_sources": bool(result.get("grounded_in_sources")),
        "touches_sensitive": bool(result.get("touches_sensitive")),
        "cites_sources": bool(result.get("cites_sources")),
        "above_grounding_floor": True,
    }
    reason = result.get("reason", "")

    # The rubric is enforced here rather than trusted from the model's verdict.
    # A model that reports grounded_in_sources=False and then returns SEND has
    # contradicted itself, and the checks are the more reliable signal.
    # Sensitive is decided once and never revisited.
    if checks["touches_sensitive"]:
        return _escalate(
            state,
            f"{reason} (failed: touches_sensitive)",
            checks,
            redraftable=False,
        )

    failed = [name for name in REDRAFTABLE_CHECKS if not checks[name]]
    if failed:
        return _escalate(state, f"{reason} (failed: {', '.join(failed)})", checks)

    return {
        "verdict": "SEND",
        "decided_by": "judge",
        "escalation_reason": "",
        "judge_checks": checks,
        "trace": [f"[judge] ({llm.last_model('judge')}) SEND -- checks={checks}"],
    }
