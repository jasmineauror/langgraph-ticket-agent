"""Escalation Judge node: decide whether the draft can be sent."""

from __future__ import annotations

from .. import llm, prompts
from ..state import TicketState

MAX_RETRIES = 1


def judge(state: TicketState) -> TicketState:
    user = (
        f"Ticket:\n\n{state['ticket_text']}\n\n"
        f"Drafted reply:\n\n{state['draft_reply']}"
    )

    result = llm.call(
        role="judge",
        system=prompts.JUDGE_SYSTEM,
        user=user,
        schema=prompts.JUDGE_SCHEMA,
    )

    checks = {"addresses_ticket": bool(result.get("addresses_ticket", False))}
    verdict = result["verdict"]
    reason = result.get("reason", "")

    # A failed draft gets one more attempt at the Responder before we give up
    # and hand it to a human.
    if verdict == "ESCALATE" and state.get("retry_count", 0) < MAX_RETRIES:
        return {
            "verdict": "RETRY",
            "decided_by": "judge",
            "escalation_reason": reason,
            "judge_checks": checks,
            "retry_count": state.get("retry_count", 0) + 1,
            "trace": [
                f"[judge] ({llm.last_model('judge')}) RETRY "
                f"(attempt {state.get('retry_count', 0) + 1}) -- {reason}"
            ],
        }

    return {
        "verdict": verdict,
        "decided_by": "judge",
        "escalation_reason": reason if verdict == "ESCALATE" else "",
        "judge_checks": checks,
        "trace": [
            f"[judge] ({llm.last_model('judge')}) {verdict} "
            f"-- checks={checks} -- {reason}"
        ],
    }
