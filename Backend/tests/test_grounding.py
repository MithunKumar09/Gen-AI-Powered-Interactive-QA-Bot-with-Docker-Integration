"""Grounding integrity invariants.

No embedding may represent text absent from stored metadata or from the
generation context.

If this contract breaks, retrieval may match text the model never receives and
citations may point to content that was never embedded. Those failures are
difficult to detect during normal use, so they are tested explicitly here.
"""

from __future__ import annotations

from typing import Any

import pytest

from conftest import make_pdf
from rag_core import pipeline
from rag_core.errors import ProviderError


def _ingest(
    settings,
    pages: list[str],
    *,
    filename: str = "doc.pdf",
    scope: str = "s" * 64,
):
    """Ingest a text-bearing PDF through the real pipeline."""
    return pipeline.ingest_pdf(
        settings,
        data=make_pdf(pages),
        filename=filename,
        scope=scope,
    )


def _stored_records(providers) -> list[dict[str, Any]]:
    """Return stored records in deterministic chunk order."""
    return sorted(
        providers.index.records.values(),
        key=lambda record: int(record["metadata"]["chunk_index"]),
    )


def _stored_metadata_by_chunk(providers) -> dict[int, dict[str, Any]]:
    """Index stored metadata by the chunk id used in Cohere documents."""
    return {
        int(record["metadata"]["chunk_index"]): record["metadata"]
        for record in providers.index.records.values()
    }


def _last_chat_documents(providers) -> list[dict[str, Any]]:
    """Return and validate the most recent Cohere Chat V2 document payload."""
    calls = providers.cohere.chat_calls
    assert calls, "expected at least one Cohere chat call"

    documents = calls[-1]["documents"]
    assert isinstance(documents, list), "chat documents must be a list"
    assert documents, "grounded generation must receive evidence documents"

    for index, document in enumerate(documents):
        assert isinstance(document, dict), (
            f"documents[{index}] must be a mapping"
        )

        document_id = document.get("id")
        assert isinstance(document_id, str) and document_id.strip(), (
            f"documents[{index}].id must be a non-empty string"
        )

        data = document.get("data")
        assert isinstance(data, dict) and data, (
            f"documents[{index}].data is required"
        )

        text = data.get("text")
        assert isinstance(text, str) and text.strip(), (
            f"documents[{index}].data.text is required"
        )

        page = data.get("page")
        assert isinstance(page, str) and page.strip(), (
            f"documents[{index}].data.page is required"
        )

    return documents


def test_embedded_text_is_byte_identical_to_stored_text(
    settings,
    providers,
    scope_a,
):
    """Every document text sent to embed() must equal stored metadata text."""
    _ingest(
        settings,
        ["Alpha content here.", "Beta content here."],
        scope=scope_a,
    )

    document_calls = [
        call
        for call in providers.cohere.embed_calls
        if call["input_type"] == "search_document"
    ]
    assert document_calls, "expected at least one document embedding call"

    embedded_texts = [
        text
        for call in document_calls
        for text in call["texts"]
    ]
    stored_texts = [
        record["metadata"]["text"]
        for record in _stored_records(providers)
    ]

    assert embedded_texts == stored_texts, (
        "embedded document text differs from stored metadata text"
    )


def test_embed_uses_truncate_none_so_nothing_is_silently_shortened(
    settings,
    providers,
    scope_a,
):
    """Over-length inputs must fail rather than being silently truncated."""
    _ingest(settings, ["Some content."], scope=scope_a)

    document_calls = [
        call
        for call in providers.cohere.embed_calls
        if call["input_type"] == "search_document"
    ]
    assert document_calls
    assert all(call["truncate"] == "NONE" for call in document_calls)


def test_documents_are_embedded_with_search_document_input_type(
    settings,
    providers,
    scope_a,
):
    _ingest(settings, ["Content."], scope=scope_a)

    assert providers.cohere.embed_calls
    assert all(
        call["input_type"] == "search_document"
        for call in providers.cohere.embed_calls
    )


def test_questions_are_embedded_with_search_query_input_type(
    settings,
    providers,
    scope_a,
):
    """Document and query embeddings must use their asymmetric input types."""
    result = _ingest(
        settings,
        ["Widgets cost ten pounds."],
        scope=scope_a,
    )

    providers.cohere.embed_calls.clear()

    question = "What do widgets cost?"
    pipeline.answer_question(
        settings,
        question=question,
        document_id=result.document_id,
        scope=scope_a,
    )

    query_calls = [
        call
        for call in providers.cohere.embed_calls
        if call["input_type"] == "search_query"
    ]

    assert len(query_calls) == 1
    assert query_calls[0]["texts"] == [question]


def test_generation_context_matches_stored_evidence(
    settings,
    providers,
    scope_a,
):
    """Every Cohere document must map exactly to stored chunk metadata."""
    result = _ingest(
        settings,
        ["The calibration constant is 47 microfarads for the beta unit."],
        scope=scope_a,
    )

    pipeline.answer_question(
        settings,
        question="What is the calibration constant?",
        document_id=result.document_id,
        scope=scope_a,
    )

    documents = _last_chat_documents(providers)
    stored_by_chunk = _stored_metadata_by_chunk(providers)

    sent_chunk_ids: list[int] = []

    for index, document in enumerate(documents):
        chunk_index = int(document["id"])
        sent_chunk_ids.append(chunk_index)

        assert chunk_index in stored_by_chunk, (
            f"documents[{index}] references unknown chunk {chunk_index}"
        )

        stored = stored_by_chunk[chunk_index]
        data = document["data"]

        assert data["text"] == stored["text"], (
            f"documents[{index}].data.text differs from stored chunk text"
        )
        assert data["page"] == str(stored["page"]), (
            f"documents[{index}].data.page differs from stored chunk page"
        )

    assert len(sent_chunk_ids) == len(set(sent_chunk_ids)), (
        "duplicate evidence chunks were sent to Cohere Chat"
    )


def test_citation_pages_come_from_stored_metadata(
    settings,
    providers,
    scope_a,
):
    result = _ingest(
        settings,
        [
            "Page one filler content.",
            "Page two filler content.",
            "Page three states the answer is forty seven.",
        ],
        scope=scope_a,
    )

    answer = pipeline.answer_question(
        settings,
        question="What is the answer?",
        document_id=result.document_id,
        scope=scope_a,
    )

    assert answer.citations, "grounded answer must expose provenance"

    stored_by_chunk = _stored_metadata_by_chunk(providers)

    for citation in answer.citations:
        chunk_index = citation["chunk_index"]

        assert chunk_index in stored_by_chunk, (
            f"citation references unknown chunk {chunk_index}"
        )
        assert citation["page"] == stored_by_chunk[chunk_index]["page"], (
            "citation page does not match the cited chunk's stored page"
        )


def test_dimension_mismatch_is_caught_before_storage(
    settings,
    providers,
    scope_a,
):
    """A wrong-dimension embedding must never reach Pinecone."""
    providers.cohere.return_wrong_dimension = True

    with pytest.raises(ProviderError) as exc_info:
        _ingest(settings, ["Content."], scope=scope_a)

    assert "dimension" in (exc_info.value.detail or "").lower()
    assert providers.index.records == {}


def test_short_embedding_batch_is_rejected(
    settings,
    providers,
    scope_a,
):
    """A short provider batch must fail before text/vector alignment is lost."""
    providers.cohere.return_short_batch = True

    with pytest.raises(ProviderError) as exc_info:
        _ingest(settings, ["word " * 400], scope=scope_a)

    detail = (exc_info.value.detail or "").lower()
    assert "embeddings" in detail
    assert "inputs" in detail
    assert providers.index.records == {}, (
        "misaligned embedding batch must not be stored"
    )