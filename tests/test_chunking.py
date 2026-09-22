"""Chunking behavior. Pure function, no model or vector store involved."""

from __future__ import annotations

from triage.index import TARGET_WORDS, chunk_document


def test_title_is_prepended_to_every_chunk():
    """Without the title, an isolated chunk retrieves well but reads ambiguously."""
    text = "# Deleting your workspace\n\n" + "\n\n".join(
        " ".join(["word"] * 100) for _ in range(3)
    )
    chunks = chunk_document(text, "Deleting your workspace")

    assert len(chunks) > 1
    assert all(c.startswith("Deleting your workspace") for c in chunks)


def test_paragraphs_are_never_split_mid_paragraph():
    paragraph_a = " ".join(["alpha"] * 80)
    paragraph_b = " ".join(["beta"] * 80)
    chunks = chunk_document(f"# T\n\n{paragraph_a}\n\n{paragraph_b}", "T")

    # 160 words exceeds the target, so they land in separate chunks -- but each
    # paragraph stays whole.
    bodies = [c.replace("T\n\n", "") for c in chunks]
    assert paragraph_a in " ".join(bodies)
    assert paragraph_b in " ".join(bodies)
    for body in bodies:
        assert not (("alpha" in body) and ("beta" in body))


def test_short_document_stays_one_chunk():
    chunks = chunk_document("# T\n\nA short paragraph about invoices.", "T")
    assert len(chunks) == 1


def test_chunks_respect_target_size_where_paragraphs_allow():
    paragraphs = "\n\n".join(" ".join(["w"] * 40) for _ in range(10))
    chunks = chunk_document(f"# T\n\n{paragraphs}", "T")

    for chunk in chunks:
        body_words = len(chunk.split()) - 1  # minus the title token
        # One oversized paragraph can exceed the target; a run of small ones
        # must not accumulate far past it.
        assert body_words <= TARGET_WORDS + 40
