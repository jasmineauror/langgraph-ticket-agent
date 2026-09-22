"""The assertion engine for eval fixtures.

Kept separate from the pytest wiring so it can be unit-tested against
hand-built states without invoking a model.
"""

from __future__ import annotations

from typing import Any

from triage.state import TicketState


def outcome(state: TicketState) -> str:
    return "AUTO_REPLY" if state.get("verdict") == "SEND" else "ESCALATE"


def check(assertions: dict[str, Any], state: TicketState) -> list[str]:
    """Return a list of failure messages. Empty means the fixture passed."""
    failures: list[str] = []
    result = outcome(state)
    reply = state.get("draft_reply", "") or ""
    reply_lower = reply.lower()
    cited = state.get("cited_sources") or []

    if "any_of" in assertions:
        branches = assertions["any_of"]
        branch_failures = [check(branch, state) for branch in branches]
        if all(bf for bf in branch_failures):
            detail = "; ".join(
                f"branch {i}: {', '.join(bf)}" for i, bf in enumerate(branch_failures)
            )
            failures.append(f"no any_of branch held ({detail})")

    if assertions.get("must_auto_reply") and result != "AUTO_REPLY":
        failures.append(
            f"expected AUTO_REPLY, got ESCALATE "
            f"(reason: {state.get('escalation_reason', '')!r})"
        )

    if assertions.get("must_not_auto_reply") and result == "AUTO_REPLY":
        failures.append("expected NOT to auto-reply, but a reply was sent")

    if assertions.get("must_escalate") and result != "ESCALATE":
        failures.append("expected ESCALATE, got AUTO_REPLY")

    if expected := assertions.get("expected_category"):
        actual = state.get("category")
        if actual != expected:
            failures.append(f"expected category {expected!r}, got {actual!r}")

    if expected := assertions.get("expected_reason"):
        actual = (state.get("escalation_reason") or "").lower()
        if expected.lower() not in actual:
            failures.append(f"expected reason containing {expected!r}, got {actual!r}")

    for source in assertions.get("must_cite_source", []):
        if source not in cited:
            failures.append(f"expected {source!r} in cited_sources, got {cited}")

    for needle in assertions.get("must_contain", []):
        if needle.lower() not in reply_lower:
            failures.append(f"reply missing required substring {needle!r}")

    # Checked against the draft whenever one exists, even if the judge
    # escalated it. A draft that says "I have processed your refund" is a
    # Responder bug regardless of whether the Judge caught it downstream --
    # and keeping the check unconditional is what tells you that fixing the
    # Judge alone did not fix the Responder.
    for needle in assertions.get("must_not_contain", []):
        if reply and needle.lower() in reply_lower:
            failures.append(f"reply contains forbidden substring {needle!r}")

    return failures
