"""Fixture-based eval suite.

    pytest evals/ -v

Each fixture is one ticket run through the whole graph, with declarative
assertions about the behavior we require. Failures print the graph trace so a
failing eval tells you *where* it went wrong, not just that it did.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from evals.assertions import check, outcome
from triage.graph import run_ticket

FIXTURES_PATH = pathlib.Path(__file__).parent / "fixtures.yaml"
FIXTURES = yaml.safe_load(FIXTURES_PATH.read_text())

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
    try:
        state = run_ticket(fixture["ticket"].strip())
    except Exception as exc:
        RESULTS.append((fixture["id"], "ERROR", [f"{type(exc).__name__}: {exc}"]))
        pytest.fail(
            f"fixture {fixture['id']!r} could not be evaluated -- "
            f"{type(exc).__name__}: {exc}\n"
            f"This is an infrastructure failure, not a result. It says nothing "
            f"about the prompts.",
            pytrace=False,
        )

    failures = check(fixture["assert"], state)
    RESULTS.append((fixture["id"], "FAIL" if failures else "PASS", failures))
    record_property("outcome", outcome(state))

    if failures:
        trace = "\n".join(f"    {line}" for line in state.get("trace", []))
        reply = (state.get("draft_reply") or "(no draft)")[:400]
        pytest.fail(
            "\n".join(
                [
                    f"fixture {fixture['id']!r} failed {len(failures)} assertion(s):",
                    *(f"  - {f}" for f in failures),
                    "",
                    f"  outcome: {outcome(state)}",
                    f"  category: {state.get('category')}",
                    f"  max_similarity: {state.get('max_similarity', 0.0):.3f}",
                    f"  cited: {state.get('cited_sources')}",
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
