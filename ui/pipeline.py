"""Driving the graph so a host can watch it happen.

`triage.graph.run_ticket()` is a single blocking call: correct for a CLI and
useless for a progress display. This module drives the compiled graph with
`.stream()` instead.

The load-bearing property, verified rather than assumed:

    graph.stream(s, stream_mode=["updates","values"])'s final `values` payload
    == graph.invoke(s)

So a streamed run produces exactly the state an invoked run would, which means
`evals.assertions.check()` grades a streamed run unchanged, and the eval page
can show per-stage progress without introducing a second execution path whose
results might not match the suite's.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Iterator

import ui.bootstrap as bootstrap  # noqa: F401  (must precede any triage import)

from triage.state import TicketState, new_state
from ui.resources import get_graph

# Display order for the stage strip. Terminal nodes are outcomes, not stages.
STAGES: tuple[str, ...] = ("classify", "retrieve", "respond", "judge")
TERMINALS: tuple[str, ...] = ("auto_reply", "escalate")

STAGE_LABELS = {
    "classify": "Classifier",
    "retrieve": "Retriever",
    "respond": "Responder",
    "judge": "Escalation Judge",
}


def _assert_stages_match_graph() -> None:
    """Fail loudly at import if the graph grew a node this module does not know.

    Same discipline `triage/diagram.py` applies to the diagram: derive from the
    compiled graph rather than maintaining a parallel hand-written list that can
    quietly fall behind. Without this, adding a node to graph.py would simply
    not render, with no error.
    """
    actual = {
        n
        for n in get_graph().get_graph().nodes
        if not n.startswith("__")
    }
    known = set(STAGES) | set(TERMINALS)
    if actual != known:
        raise RuntimeError(
            f"ui/pipeline.py is out of step with the graph. "
            f"Graph has {sorted(actual)}; this module knows {sorted(known)}. "
            f"Update STAGES/TERMINALS."
        )


@dataclasses.dataclass
class RunProgress:
    """Accumulates a run as it streams, so a partial run still renders."""

    ticket_text: str
    backend: str | None
    started_at: float = dataclasses.field(default_factory=time.monotonic)
    order: list[str] = dataclasses.field(default_factory=list)
    updates: dict[str, list[dict[str, Any]]] = dataclasses.field(default_factory=dict)
    state: TicketState = dataclasses.field(default_factory=dict)
    finished: bool = False
    error: str | None = None

    def record(self, node: str, update: dict[str, Any]) -> None:
        self.order.append(node)
        self.updates.setdefault(node, []).append(update)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def status_of(self, node: str) -> str:
        """done | skipped | running | pending, for the stage strip."""
        if node in self.updates:
            return "done"
        if self.finished and not self.error:
            # The run completed without ever reaching this node: the classifier
            # short-circuit routed past it. That is a decision the pipeline
            # made, not a stage that failed, and the strip says so.
            return "skipped"
        if self.error:
            return "pending"
        # Mid-run: the earliest stage not yet done is the one in flight.
        for candidate in STAGES:
            if candidate not in self.updates:
                return "running" if candidate == node else "pending"
        return "pending"

    def attempts(self, node: str) -> int:
        """How many times a node ran. `respond` and `judge` can fire twice."""
        return len(self.updates.get(node, ()))


def stream_run(
    ticket_text: str, backend: str | None = None
) -> Iterator[RunProgress]:
    """Yield the same RunProgress after every graph event, then once at the end.

    Yielding the accumulator rather than raw events means an interrupted rerun
    leaves a coherent partial view instead of a blank page: whatever was yielded
    last is a valid, if incomplete, picture of the run.
    """
    progress = RunProgress(ticket_text=ticket_text, backend=backend)
    graph = get_graph()

    try:
        for mode, chunk in graph.stream(
            dict(new_state(ticket_text, backend=backend)),
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                for node, update in chunk.items():
                    progress.record(node, update or {})
                    yield progress
            elif mode == "values":
                # The last `values` payload is the final state -- equal to what
                # invoke() would have returned.
                progress.state = chunk
    except Exception as exc:
        progress.error = f"{type(exc).__name__}: {exc}"

    progress.finished = True
    yield progress


def run_to_state(ticket_text: str, backend: str | None = None) -> TicketState:
    """Drive the graph by streaming, but return only the final state.

    This is the `RunFn` handed to `evals.runner`, so the eval page executes the
    streaming path while being graded by the identical assertion code the pytest
    suite uses. Exceptions propagate: `run_fixture` owns the ERROR boundary, and
    swallowing them here would let infrastructure failures be graded as results.
    """
    progress = None
    for progress in stream_run(ticket_text, backend=backend):
        pass
    if progress is None:  # pragma: no cover - stream_run always yields
        raise RuntimeError("stream produced no events")
    if progress.error:
        raise RuntimeError(progress.error)
    return progress.state
