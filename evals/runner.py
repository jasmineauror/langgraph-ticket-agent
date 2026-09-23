"""The eval suite's execution engine, independent of pytest.

Both `pytest evals/` and any other host (a UI, a script, CI) drive the suite
through this module. That is the entire reason it exists: a second
implementation of "run a fixture and grade it" would be a second, untested
assertion path, and this project has already shipped a reporter that overstated
its own result. One code path, one score.

Everything here that could plausibly be reimplemented by a caller is
deliberately NOT reimplementable:

  * grading is `assertions.check()`, verbatim, and the raw failure strings are
    passed through
  * the outcome label is `assertions.outcome()`, never `verdict == "SEND"`
    inlined somewhere
  * the ticket is `.strip()`ed HERE, so every host feeds the graph an identical
    string (YAML block scalars carry a trailing newline, and that one character
    is enough to make a result unreproducible)
  * PASS / FAIL / ERROR is one trichotomy with one try/except boundary
  * `score()` divides by fixtures DECLARED, never by fixtures that survived
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import Any, Callable, Iterator

import yaml

from evals.assertions import check, outcome
from triage.graph import run_ticket
from triage.state import TicketState

FIXTURES_PATH = pathlib.Path(__file__).parent / "fixtures.yaml"

# A run function takes ticket text and returns the final TicketState. The graph's
# streamed final `values` payload is equal to what `invoke()` returns, so a
# streaming host can inject its own and be graded identically.
RunFn = Callable[[str], TicketState]


@dataclasses.dataclass
class FixtureResult:
    id: str
    status: str  # "PASS" | "FAIL" | "ERROR"
    failures: list[str]
    state: TicketState | None = None
    outcome: str | None = None
    duration_s: float = 0.0

    @property
    def is_measured(self) -> bool:
        """False for ERROR: infrastructure failed, so nothing was measured."""
        return self.status in ("PASS", "FAIL")


def load_fixtures(path: pathlib.Path = FIXTURES_PATH) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text())


def preflight(backend: str | None = None) -> str | None:
    """One cheap probe. Returns an error message, or None when reachable.

    A long run that reports prompt failures which were really one dropped
    connection is worse than no run: the score looks like a measurement, so it
    gets believed. Check once, up front, and refuse to produce a number that
    would be attributed to the prompts.
    """
    from triage import llm

    effective = (backend or llm.backend_name()).lower()
    if effective in ("stub", "demo"):
        # Neither touches the network, so there is nothing to probe.
        return None

    try:
        llm.call(
            role="classifier",
            system="Reply with JSON.",
            user="Ticket:\n\ntest",
            schema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "billing",
                            "technical",
                            "account",
                            "abusive_or_out_of_scope",
                        ],
                    },
                    "auto_answerable": {"type": "boolean"},
                    "reasoning": {"type": "string"},
                },
                "required": ["category", "auto_answerable", "reasoning"],
            },
            backend=backend,
        )
    except Exception as exc:
        return (
            f"preflight failed, refusing to run the eval suite: {exc}\n"
            f"The fixtures measure prompt behavior; they cannot measure it "
            f"through a broken connection."
        )
    return None


def run_fixture(fixture: dict[str, Any], run_fn: RunFn | None = None) -> FixtureResult:
    """Run one fixture and grade it."""
    run_fn = run_fn or run_ticket
    started = time.monotonic()

    try:
        state = run_fn(fixture["ticket"].strip())
    except Exception as exc:
        return FixtureResult(
            id=fixture["id"],
            status="ERROR",
            failures=[f"{type(exc).__name__}: {exc}"],
            duration_s=time.monotonic() - started,
        )

    failures = check(fixture["assert"], state)
    return FixtureResult(
        id=fixture["id"],
        status="FAIL" if failures else "PASS",
        failures=failures,
        state=state,
        outcome=outcome(state),
        duration_s=time.monotonic() - started,
    )


def run_suite(
    fixtures: list[dict[str, Any]] | None = None,
    run_fn: RunFn | None = None,
) -> Iterator[FixtureResult]:
    """Yield one FixtureResult at a time, so a host can render progress.

    Yielding rather than returning a list means an aborted run leaves a partial
    set that `score()` reports honestly as partial, instead of nothing.
    """
    for fixture in fixtures if fixtures is not None else load_fixtures():
        yield run_fixture(fixture, run_fn)


@dataclasses.dataclass
class Score:
    passed: int
    failed: int
    errored: int
    unreported: int
    declared: int

    @property
    def headline(self) -> str:
        return f"{self.passed}/{self.declared}"

    @property
    def unmeasured(self) -> int:
        return self.errored + self.unreported


def score(results: list[FixtureResult], declared: int) -> Score:
    """Summarise, dividing by fixtures DECLARED.

    Never `len(results)`. An earlier reporter appended only after grading, so
    fixtures killed by a rate limit left the list entirely and it printed
    "3/3 passed" while four of seven had never run. A denominator that silently
    shrinks to whatever survived turns an outage into a perfect score.
    """
    return Score(
        passed=sum(1 for r in results if r.status == "PASS"),
        failed=sum(1 for r in results if r.status == "FAIL"),
        errored=sum(1 for r in results if r.status == "ERROR"),
        unreported=declared - len(results),
        declared=declared,
    )
