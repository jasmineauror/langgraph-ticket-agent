"""Graph routing and state transitions, against a stubbed model.

These cover the control flow -- which is the part of a LangGraph app that breaks
in ways a model can't be blamed for. No model, no vector store, no network, so
they run in milliseconds and belong in CI.
"""

from __future__ import annotations

from triage import llm
from triage.graph import build_graph
from triage.state import new_state

CLASSIFIER_KEY = "classifier"
RESPONDER_KEY = "responder"
JUDGE_KEY = "judge"


def _judge(
    verdict: str = "SEND",
    addresses_ticket: bool = True,
    grounded_in_sources: bool = True,
    touches_sensitive: bool = False,
    cites_sources: bool = True,
    reason: str = "ok",
):
    """A judge response with all four checks, defaulting to a clean pass."""
    return {
        "verdict": verdict,
        "addresses_ticket": addresses_ticket,
        "grounded_in_sources": grounded_in_sources,
        "touches_sensitive": touches_sensitive,
        "cites_sources": cites_sources,
        "reason": reason,
    }


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
    llm.set_stub(JUDGE_KEY, _judge(verdict="SEND"))

    state = build_graph().invoke(new_state("where are my invoices"))

    assert state["verdict"] == "SEND"
    assert state["cited_sources"] == ["billing-invoices.md"]
    assert state["max_similarity"] == 0.88
    assert state["retry_count"] == 0


def test_judge_rejection_retries_once_then_escalates(fake_kb):
    """A rejected draft gets exactly one more attempt, never an infinite loop."""
    # distance 0.45 -> similarity 0.55, above the 0.40 grounding floor, so the
    # judge's model call runs rather than the code gate short-circuiting it
    fake_kb([("Something unrelated.", "tech-webhooks.md", 0.45)])
    _classified("technical")
    llm.set_stub(
        RESPONDER_KEY, {"reply": "A vague answer.", "cited_sources": ["tech-webhooks.md"]}
    )
    llm.set_stub(
        JUDGE_KEY, _judge(verdict="ESCALATE", addresses_ticket=False,
                          reason="does not address")
    )

    state = build_graph().invoke(new_state("why did my webhook stop"))

    assert state["verdict"] == "ESCALATE"
    assert state["retry_count"] == 1, "retry must be capped at one attempt"
    assert "does not address" in state["escalation_reason"]


def test_trace_accumulates_across_nodes(fake_kb):
    fake_kb([("Invoices live under Settings.", "billing-invoices.md", 0.12)])
    _classified("billing")
    llm.set_stub(RESPONDER_KEY, {"reply": "x", "cited_sources": ["billing-invoices.md"]})
    llm.set_stub(JUDGE_KEY, _judge(verdict="SEND"))

    state = build_graph().invoke(new_state("where are my invoices"))

    stages = [line.split("]")[0].lstrip("[") for line in state["trace"]]
    assert stages == ["classifier", "retriever", "responder", "judge", "outcome"]


def test_grounding_floor_escalates_without_a_model_call(fake_kb):
    """Below the floor, nothing retrieved is on the subject.

    No judge stub is registered, so if the code gate regresses and the model is
    consulted anyway, the stub raises and this test fails loudly.
    """
    fake_kb([("Unrelated text.", "tech-sso-saml.md", 0.75)])  # similarity 0.25
    _classified("technical")
    llm.set_stub(RESPONDER_KEY, {"reply": "Some answer.", "cited_sources": []})

    state = build_graph().invoke(new_state("do you support on-premise deployment"))

    assert state["verdict"] == "ESCALATE"
    assert state["judge_checks"] == {"above_grounding_floor": False}
    assert "max_similarity" in state["escalation_reason"]


def test_responder_refusal_is_honoured_without_a_model_call(fake_kb):
    """An explicit refusal is a fact about state, not a matter of opinion."""
    fake_kb([("Invoices live under Settings.", "billing-invoices.md", 0.12)])
    _classified("billing")
    llm.set_stub(
        RESPONDER_KEY,
        {"reply": "INSUFFICIENT_CONTEXT - no refund policy in the excerpts.",
         "cited_sources": []},
    )

    state = build_graph().invoke(new_state("what is your refund policy"))

    assert state["verdict"] == "ESCALATE"
    assert state["judge_checks"] == {"responder_refused": True}


def test_sensitive_topic_escalates_even_when_well_grounded(fake_kb):
    fake_kb([("Refunds are case by case.", "billing-plan-changes.md", 0.2)])
    _classified("billing")
    llm.set_stub(RESPONDER_KEY, {"reply": "Grounded answer.",
                                 "cited_sources": ["billing-plan-changes.md"]})
    llm.set_stub(JUDGE_KEY, _judge(verdict="SEND", touches_sensitive=True))

    state = build_graph().invoke(new_state("can I get a refund"))

    assert state["verdict"] == "ESCALATE", "sensitive topics must not auto-reply"
    assert "touches_sensitive" in state["escalation_reason"]


def test_rubric_overrides_a_self_contradicting_verdict(fake_kb):
    """A model reporting grounded=False and then SEND has contradicted itself.

    The checks are the more reliable signal, so the rubric is enforced in code
    rather than trusted from the verdict field.
    """
    fake_kb([("Some text.", "tech-webhooks.md", 0.2)])
    _classified("technical")
    llm.set_stub(RESPONDER_KEY, {"reply": "Ungrounded claim.",
                                 "cited_sources": ["tech-webhooks.md"]})
    llm.set_stub(JUDGE_KEY, _judge(verdict="SEND", grounded_in_sources=False))

    state = build_graph().invoke(new_state("why did my webhook stop"))

    assert state["verdict"] == "ESCALATE"
    assert "grounded_in_sources" in state["escalation_reason"]
