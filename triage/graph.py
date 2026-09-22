"""Graph wiring.

    ticket -> classifier -> [abusive/out-of-scope?] ------------> escalate
                                |
                                v
                            retriever -> responder -> judge -> [SEND?] -> reply
                                             ^          |
                                             +-- RETRY --+
                                                        |
                                                        +-----------> escalate

The two conditional edges are what make this a graph rather than a chain: the
classifier short-circuit (never spend retrieval and drafting on an abusive
ticket) and the judge's retry-or-escalate branch.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes.classifier import classify
from .nodes.judge import judge
from .nodes.responder import respond
from .nodes.retriever import retrieve
from .state import TicketState, new_state


def _auto_reply(state: TicketState) -> TicketState:
    return {
        "verdict": "SEND",
        "trace": ["[outcome] AUTO_REPLY"],
    }


def _escalate(state: TicketState) -> TicketState:
    reason = state.get("escalation_reason") or "flagged by classifier"
    return {
        "verdict": "ESCALATE",
        "escalation_reason": reason,
        "trace": [f"[outcome] ESCALATE -- {reason}"],
    }


def _route_after_classifier(state: TicketState) -> str:
    if state["category"] == "abusive_or_out_of_scope":
        return "escalate"
    if not state.get("auto_answerable", True):
        return "escalate"
    return "retrieve"


def _route_after_judge(state: TicketState) -> str:
    verdict = state["verdict"]
    if verdict == "SEND":
        return "auto_reply"
    if verdict == "RETRY":
        return "respond"
    return "escalate"


def build_graph():
    graph = StateGraph(TicketState)

    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("respond", respond)
    graph.add_node("judge", judge)
    graph.add_node("auto_reply", _auto_reply)
    graph.add_node("escalate", _escalate)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        _route_after_classifier,
        {"retrieve": "retrieve", "escalate": "escalate"},
    )
    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", "judge")
    graph.add_conditional_edges(
        "judge",
        _route_after_judge,
        {"auto_reply": "auto_reply", "respond": "respond", "escalate": "escalate"},
    )
    graph.add_edge("auto_reply", END)
    graph.add_edge("escalate", END)

    return graph.compile()


_compiled = None


def run_ticket(ticket_text: str) -> TicketState:
    """Run one ticket through the graph and return the final state."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled.invoke(new_state(ticket_text))
