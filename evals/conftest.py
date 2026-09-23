"""Score line and preflight gate for a pytest eval run.

Both delegate to `evals/runner.py` so the numbers here and the numbers any other
host reports come from one implementation.
"""

from __future__ import annotations


def pytest_sessionfinish(session, exitstatus):
    from evals.runner import FixtureResult, score
    from evals.test_evals import FIXTURES, RESULTS

    if not RESULTS:
        return

    from triage import llm

    results = [
        FixtureResult(id=i, status=st, failures=d) for i, st, d in RESULTS
    ]
    s = score(results, declared=len(FIXTURES))

    print()
    print("=" * 68)
    print(f"EVAL SCORE: {s.headline} passed   (model: {llm.model_name()})")
    if s.failed:
        print(f"  {s.failed} failed on assertions  <- these are real results")
    if s.unmeasured:
        print(
            f"  {s.unmeasured} could not be evaluated  <- infrastructure, "
            f"NOT a result"
        )
        print("  The score above is a lower bound; those fixtures are unmeasured.")
    print("=" * 68)
    for name, status, detail in RESULTS:
        print(f"  [{status:5}] {name}")
        for line in detail:
            print(f"           {line}")
    print("=" * 68)


def pytest_collection_modifyitems(session, config, items):
    """Fail fast if the API is unreachable, using the runner's own probe."""
    import pytest

    from evals.runner import preflight

    if message := preflight():
        pytest.exit(message, returncode=2)
