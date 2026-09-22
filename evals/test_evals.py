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

RESULTS: list[tuple[str, bool, list[str]]] = []


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["id"])
def test_fixture(fixture, record_property):
    state = run_ticket(fixture["ticket"].strip())
    failures = check(fixture["assert"], state)

    RESULTS.append((fixture["id"], not failures, failures))
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
