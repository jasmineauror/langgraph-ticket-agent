"""Graph routing and state transitions, against a stubbed model.

These cover the control flow -- which is the part of a LangGraph app that breaks
in ways a model can't be blamed for. No model, no vector store, no network, so
they run in milliseconds and belong in CI.
"""

from __future__ import annotations

from triage import llm
from triage.graph import build_graph
from triage.state import new_state

CLASSIFIER_KEY = "triage incoming customer support tickets"
RESPONDER_KEY = "helpful customer support agent"
JUDGE_KEY = "review draft support replies"


def _classified(category: str, auto_answerable: bool = True):
    llm.set_stub(
        CLASSIFIER_KEY,
        {
            "category": category,
            "auto_answerable": auto_answerable,
            "reasoning": "stubbed",
        },
    )


def test_abusive_ticket_short_circuits_before_retrieval():
    """An abusive ticket must never reach retrieval or drafting.

    If the short-circuit edge regresses, the stub raises on the unregistered
    responder prompt, so this test fails loudly rather than silently spending
    work on a ticket a human has to handle anyway.
    """
    _classified("abusive_or_out_of_scope")

    state = build_graph().invoke(new_state("you people are useless"))

    assert state["verdict"] == "ESCALATE"
    assert state["retrieved"] == []
    assert not state.get("draft_reply")
    assert state["escalation_reason"]


def test_not_auto_answerable_escalates_without_drafting():
    _classified("billing", auto_answerable=False)

    state = build_graph().invoke(new_state("please close my account and refund me"))

    assert state["verdict"] == "ESCALATE"
    assert not state.get("draft_reply")


def test_happy_path_reaches_auto_reply(fake_kb):
    fake_kb([("Invoices live under Settings.", "billing-invoices.md", 0.12)])
    _classified("billing")
    llm.set_stub(
        RESPONDER_KEY,
        {"reply": "Invoices are under Settings.", "cited_sources": ["billing-invoices.md"]},
    )
    llm.set_stub(
        JUDGE_KEY, {"verdict": "SEND", "addresses_ticket": True, "reason": "grounded"}
    )

    state = build_graph().invoke(new_state("where are my invoices"))

    assert state["verdict"] == "SEND"
    assert state["cited_sources"] == ["billing-invoices.md"]
    assert state["max_similarity"] == 0.88
    assert state["retry_count"] == 0


def test_judge_rejection_retries_once_then_escalates(fake_kb):
    """A rejected draft gets exactly one more attempt, never an infinite loop."""
    fake_kb([("Something unrelated.", "tech-webhooks.md", 0.91)])
    _classified("technical")
    llm.set_stub(RESPONDER_KEY, {"reply": "A vague answer.", "cited_sources": []})
    llm.set_stub(
        JUDGE_KEY,
        {"verdict": "ESCALATE", "addresses_ticket": False, "reason": "does not address"},
    )

    state = build_graph().invoke(new_state("why did my webhook stop"))

    assert state["verdict"] == "ESCALATE"
    assert state["retry_count"] == 1, "retry must be capped at one attempt"
    assert "does not address" in state["escalation_reason"]


def test_trace_accumulates_across_nodes(fake_kb):
    fake_kb([("Invoices live under Settings.", "billing-invoices.md", 0.12)])
    _classified("billing")
    llm.set_stub(RESPONDER_KEY, {"reply": "x", "cited_sources": []})
    llm.set_stub(JUDGE_KEY, {"verdict": "SEND", "addresses_ticket": True, "reason": "ok"})

    state = build_graph().invoke(new_state("where are my invoices"))

    stages = [line.split("]")[0].lstrip("[") for line in state["trace"]]
    assert stages == ["classifier", "retriever", "responder", "judge", "outcome"]
