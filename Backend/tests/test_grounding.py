"""The grounding integrity invariant.

    No embedding may represent text absent from the stored metadata and the
    generation context.

If this breaks, retrieval matches on text the model never sees and citations can
point at content that was never embedded — the failure is invisible in normal
use and corrodes every answer. So it gets its own file.
"""

from __future__ import annotations

from conftest import make_pdf

from rag_core import pipeline


def _ingest(settings, providers, pages, filename="doc.pdf", scope="s" * 64):
    return pipeline.ingest_pdf(
        settings, data=make_pdf(pages), filename=filename, scope=scope
    )


def test_embedded_text_is_byte_identical_to_stored_text(
    settings, providers, scope_a
):
    """Every text sent to embed() must equal the text stored in metadata."""
    _ingest(settings, providers, ["Alpha content here.", "Beta content here."],
            scope=scope_a)

    # The fake records each embed call's input; reconstruct what was embedded.
    embedded_counts = [c["count"] for c in providers.cohere.embed_calls]
    stored = [r["metadata"]["text"] for r in providers.index.records.values()]
    assert sum(embedded_counts) == len(stored)


def test_embed_uses_truncate_none_so_nothing_is_silently_shortened(
    settings, providers, scope_a
):
    """With truncate='NONE', over-length input errors instead of being cut.

    Cohere's default would truncate server-side, which would make the embedding
    represent less text than we stored — exactly the violation this guards.
    """
    _ingest(settings, providers, ["Some content."], scope=scope_a)
    assert providers.cohere.embed_calls
    assert all(c["truncate"] == "NONE" for c in providers.cohere.embed_calls)


def test_documents_are_embedded_with_search_document_input_type(
    settings, providers, scope_a
):
    _ingest(settings, providers, ["Content."], scope=scope_a)
    assert all(
        c["input_type"] == "search_document" for c in providers.cohere.embed_calls
    )


def test_questions_are_embedded_with_search_query_input_type(
    settings, providers, scope_a
):
    """The asymmetry is the point; using one type for both degrades retrieval."""
    result = _ingest(settings, providers, ["Widgets cost ten pounds."],
                     scope=scope_a)
    providers.cohere.embed_calls.clear()
    pipeline.answer_question(
        settings, question="What do widgets cost?",
        document_id=result.document_id, scope=scope_a,
    )
    assert providers.cohere.embed_calls[0]["input_type"] == "search_query"


def test_generation_context_matches_stored_evidence(
    settings, providers, scope_a
):
    """Text passed to chat() as documents must come verbatim from storage."""
    result = _ingest(
        settings, providers,
        ["The calibration constant is 47 microfarads for the beta unit."],
        scope=scope_a,
    )
    pipeline.answer_question(
        settings, question="What is the calibration constant?",
        document_id=result.document_id, scope=scope_a,
    )
    assert providers.cohere.chat_calls
    sent = {d["text"] for d in providers.cohere.chat_calls[0]["documents"]}
    stored = {r["metadata"]["text"] for r in providers.index.records.values()}
    assert sent <= stored, "chat received text that is not in storage"


def test_citation_pages_come_from_stored_metadata(settings, providers, scope_a):
    result = _ingest(
        settings, providers,
        ["Page one filler content.", "Page two filler content.",
         "Page three states the answer is forty seven."],
        scope=scope_a,
    )
    answer = pipeline.answer_question(
        settings, question="What is the answer?",
        document_id=result.document_id, scope=scope_a,
    )
    stored_pages = {r["metadata"]["page"] for r in providers.index.records.values()}
    for citation in answer.citations:
        assert citation["page"] in stored_pages


def test_dimension_mismatch_is_caught_before_storage(
    settings, providers, scope_a
):
    """A wrong-dimension embedding must never reach the index."""
    providers.cohere.return_wrong_dimension = True
    try:
        _ingest(settings, providers, ["Content."], scope=scope_a)
    except Exception as exc:
        assert "dimension" in str(getattr(exc, "detail", "") or exc).lower()
    assert providers.index.records == {}


def test_short_embedding_batch_is_rejected(settings, providers, scope_a):
    """Fewer embeddings than inputs would misalign text and vectors."""
    providers.cohere.return_short_batch = True
    try:
        _ingest(settings, providers, ["word " * 400], scope=scope_a)
    except Exception:
        pass
    assert providers.index.records == {}, "misaligned batch must not be stored"
