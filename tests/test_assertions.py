"""The eval assertion engine, checked against hand-built states.

The eval suite is only trustworthy if its assertions actually fail when they
should. These are the tests for the tests.
"""

from __future__ import annotations

from evals.assertions import check, outcome
from triage.state import TicketState


def _state(**kwargs) -> TicketState:
    base = TicketState(
        verdict="SEND",
        draft_reply="Invoices are under Settings.",
        cited_sources=["billing-invoices.md"],
        category="billing",
        escalation_reason="",
        max_similarity=0.8,
    )
    base.update(kwargs)
    return base


def test_outcome_maps_verdict_to_terminal_state():
    assert outcome(_state(verdict="SEND")) == "AUTO_REPLY"
    assert outcome(_state(verdict="ESCALATE")) == "ESCALATE"


def test_passing_fixture_yields_no_failures():
    assertions = {
        "must_auto_reply": True,
        "must_cite_source": ["billing-invoices.md"],
        "must_contain": ["settings"],
        "expected_category": "billing",
    }
    assert check(assertions, _state()) == []


def test_must_escalate_fails_on_auto_reply():
    failures = check({"must_escalate": True}, _state(verdict="SEND"))
    assert len(failures) == 1
    assert "expected ESCALATE" in failures[0]


def test_must_not_contain_flags_a_hallucinated_specific():
    state = _state(
        draft_reply="Yes, we offer on-premise starting at $40,000 per year."
    )
    failures = check({"must_not_contain": ["$", "per year"]}, state)
    assert len(failures) == 2


def test_must_not_contain_applies_even_when_escalated():
    """A bad draft the judge caught is still a Responder bug worth reporting."""
    state = _state(verdict="ESCALATE", draft_reply="I have processed your refund.")
    failures = check({"must_not_contain": ["i have processed"]}, state)
    assert failures, "an escalated bad draft must still fail the content assertion"


def test_missing_citation_is_reported():
    failures = check({"must_cite_source": ["tech-webhooks.md"]}, _state())
    assert "tech-webhooks.md" in failures[0]


def test_any_of_passes_when_one_branch_holds():
    assertions = {
        "any_of": [
            {"must_escalate": True},
            {"must_auto_reply": True, "must_cite_source": ["billing-invoices.md"]},
        ]
    }
    assert check(assertions, _state(verdict="SEND")) == []


def test_any_of_fails_when_no_branch_holds():
    assertions = {
        "any_of": [
            {"must_escalate": True},
            {"must_auto_reply": True, "must_cite_source": ["tech-webhooks.md"]},
        ]
    }
    failures = check(assertions, _state(verdict="SEND"))
    assert len(failures) == 1
    assert "no any_of branch held" in failures[0]


def test_expected_reason_matches_case_insensitively():
    state = _state(verdict="ESCALATE", escalation_reason="Abusive language detected")
    assert check({"expected_reason": "abusive"}, state) == []


def test_terminated_by_catches_the_right_answer_from_the_wrong_node():
    """The regression this assertion exists for.

    An escalation is correct here, but it came from the classifier when the
    fixture requires the judge's grounding check to have run. Before
    terminated_by existed, this passed.
    """
    state = _state(verdict="ESCALATE", decided_by="classifier")
    failures = check({"must_escalate": True, "terminated_by": "judge"}, state)
    assert len(failures) == 1
    assert "classifier" in failures[0] and "judge" in failures[0]


def test_terminated_by_passes_when_the_mechanism_matches():
    state = _state(verdict="ESCALATE", decided_by="judge")
    assert check({"must_escalate": True, "terminated_by": "judge"}, state) == []


def test_must_reach_retriever_catches_a_short_circuit():
    state = _state(verdict="ESCALATE", decided_by="classifier", retrieved=[])
    failures = check({"must_reach_retriever": True}, state)
    assert len(failures) == 1
    assert "short-circuited" in failures[0]


def test_must_reach_retriever_passes_when_chunks_were_retrieved():
    state = _state(retrieved=[{"text": "x", "source": "a.md", "distance": 0.3}])
    assert check({"must_reach_retriever": True}, state) == []
