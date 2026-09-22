"""Prints a score line at the end of an eval run, for the eval log."""

from __future__ import annotations


def pytest_sessionfinish(session, exitstatus):
    from evals.test_evals import RESULTS

    if not RESULTS:
        return

    from triage import llm

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)

    print()
    print("=" * 62)
    print(f"EVAL SCORE: {passed}/{total} passed   (model: {llm.model_name()})")
    print("=" * 62)
    for name, ok, failures in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}")
        for failure in failures:
            print(f"         {failure}")
    print("=" * 62)
