"""Prints a score line at the end of an eval run, for the eval log."""

from __future__ import annotations


def pytest_sessionfinish(session, exitstatus):
    from evals.test_evals import RESULTS

    if not RESULTS:
        return

    from triage import llm

    from evals.test_evals import FIXTURES

    passed = sum(1 for _, status, _ in RESULTS if status == "PASS")
    failed = sum(1 for _, status, _ in RESULTS if status == "FAIL")
    errored = sum(1 for _, status, _ in RESULTS if status == "ERROR")
    declared = len(FIXTURES)
    unreported = declared - len(RESULTS)

    print()
    print("=" * 68)
    # Denominator is the number of fixtures DECLARED, never the number that
    # survived. Anything else lets infrastructure failures inflate the score.
    print(f"EVAL SCORE: {passed}/{declared} passed   (model: {llm.model_name()})")
    if failed:
        print(f"  {failed} failed on assertions  <- these are real results")
    if errored or unreported:
        print(
            f"  {errored + unreported} could not be evaluated  <- infrastructure, "
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
    """Fail fast if the API is unreachable.

    A four-minute eval run that reports six prompt failures which were actually
    one dropped connection is worse than no run: the score looks like a
    measurement, so it gets believed. Check connectivity once, up front, and
    refuse to produce a score that would be attributed to the prompts.
    """
    import os

    if os.environ.get("TRIAGE_LLM", "gemini").lower() == "stub":
        return

    import pytest

    from triage import llm

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
                        "enum": ["billing", "technical", "account",
                                 "abusive_or_out_of_scope"],
                    },
                    "auto_answerable": {"type": "boolean"},
                    "reasoning": {"type": "string"},
                },
                "required": ["category", "auto_answerable", "reasoning"],
            },
        )
    except Exception as exc:
        pytest.exit(
            f"preflight failed, refusing to run the eval suite: {exc}\n"
            f"The fixtures measure prompt behavior; they cannot measure it "
            f"through a broken connection.",
            returncode=2,
        )
