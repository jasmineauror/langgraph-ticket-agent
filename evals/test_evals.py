"""Fixture-based eval suite.

    pytest evals/ -v

Each fixture is one ticket run through the whole graph, with declarative
assertions about the behavior we require. Failures print the graph trace so a
failing eval tells you *where* it went wrong, not just that it did.

The running and grading live in `evals/runner.py`, not here. This module is a
thin pytest adapter over it, so that pytest and any other host (the UI) execute
the identical code path and cannot report different scores for the same run.
"""

from __future__ import annotations

import pytest

from evals.runner import load_fixtures, run_fixture

FIXTURES = load_fixtures()

# (fixture_id, status, detail) where status is PASS | FAIL | ERROR.
#
# ERROR is a separate status on purpose. An earlier version appended only after
# the assertions ran, so a fixture that died on a rate limit never reached the
# list -- and the summary printed "3/3 passed" while four of seven fixtures had
# not run at all. A score whose denominator silently shrinks to the tests that
# happened to survive is worse than no score.
RESULTS: list[tuple[str, str, list[str]]] = []


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["id"])
def test_fixture(fixture, record_property):
    result = run_fixture(fixture)
    RESULTS.append((result.id, result.status, result.failures))

    if result.status == "ERROR":
        pytest.fail(
            f"fixture {result.id!r} could not be evaluated -- "
            f"{result.failures[0]}\n"
            f"This is an infrastructure failure, not a result. It says nothing "
            f"about the prompts.",
            pytrace=False,
        )

    record_property("outcome", result.outcome)

    if result.status == "FAIL":
        state = result.state or {}
        trace = "\n".join(f"    {line}" for line in state.get("trace", []))
        reply = (state.get("draft_reply") or "(no draft)")[:400]
        pytest.fail(
            "\n".join(
                [
                    f"fixture {result.id!r} failed {len(result.failures)} assertion(s):",
                    *(f"  - {f}" for f in result.failures),
                    "",
                    f"  outcome: {result.outcome}",
                    f"  category: {state.get('category')}",
                    f"  max_similarity: {state.get('max_similarity', 0.0):.3f}",
                    f"  cited: {state.get('cited_sources')}",
                    f"  served_by: {state.get('served_by')}",
                    "",
                    "  trace:",
                    trace,
                    "",
                    "  draft:",
                    f"    {reply}",
                ]
            ),
            pytrace=False,
        )
