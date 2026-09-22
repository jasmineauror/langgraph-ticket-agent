import os

import pytest

from triage import llm


@pytest.fixture(autouse=True)
def stub_backend(monkeypatch):
    """Every unit test runs against the deterministic stub, never a real model."""
    monkeypatch.setenv("TRIAGE_LLM", "stub")
    llm.clear_stub()
    yield
    llm.clear_stub()


@pytest.fixture
def fake_kb(monkeypatch):
    """Replace the Chroma collection with a canned result.

    The retriever calls `_get()` at request time rather than holding a module
    global, so swapping it here needs no running vector store.
    """

    def _install(chunks: list[tuple[str, str, float]]):
        class FakeCollection:
            def query(self, query_texts, n_results, include):
                return {
                    "documents": [[text for text, _, _ in chunks]],
                    "metadatas": [[{"source": src} for _, src, _ in chunks]],
                    "distances": [[dist for _, _, dist in chunks]],
                }

        from triage.nodes import retriever

        monkeypatch.setattr(retriever, "_get", lambda: FakeCollection())

    return _install
