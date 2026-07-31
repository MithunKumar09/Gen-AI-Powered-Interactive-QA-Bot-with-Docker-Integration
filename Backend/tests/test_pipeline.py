"""Ingest transactionality, isolation, idempotency, and abstention.

The load-bearing claim: **a failed upload never destroys the working document.**
The pre-audit flow cleared the previous document before ingesting the new one, so
any failure left the user with nothing.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from conftest import make_pdf

from cohere.errors import TooManyRequestsError
from rag_core import pipeline
from rag_core.errors import (
    DocumentNotFoundError,
    IngestVerificationError,
    InvalidPdfError,
    ProviderError,
    ValidationError,
)

PAGES = [
    "Alpha widgets cost ten pounds.",
    "Beta gears weigh two kilograms.",
    "The calibration constant is 47 microfarads.",
]


def _last_chat_documents(providers) -> list[dict[str, Any]]:
    """Return the document payload sent in the most recent Cohere chat call.

    The helper validates the Cohere Chat V2 contract so test failures clearly
    identify payload drift instead of failing later with an unrelated KeyError.
    """
    calls = providers.cohere.chat_calls
    assert calls, "expected at least one Cohere chat call"

    documents = calls[-1]["documents"]
    assert isinstance(documents, list), "chat documents must be a list"
    assert documents, "expected at least one grounded document"

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

    return documents


def _last_chat_context(providers) -> str:
    """Join text from the most recent Cohere Chat V2 document payload."""
    return " ".join(
        document["data"]["text"]
        for document in _last_chat_documents(providers)
    )


def _ingest(settings, scope, pages=None, filename="doc.pdf"):
    return pipeline.ingest_pdf(
        settings, data=make_pdf(pages or PAGES), filename=filename, scope=scope
    )


# --- Happy path --------------------------------------------------------------


def test_ingest_stores_all_chunks(settings, providers, scope_a):
    result = _ingest(settings, scope_a)
    assert result.reused is False
    assert result.chunk_count == len(providers.index.records)
    assert result.page_count == 3


def test_ingest_metadata_carries_isolation_fields(settings, providers, scope_a):
    result = _ingest(settings, scope_a)
    for record in providers.index.records.values():
        meta = record["metadata"]
        assert meta["session_scope"] == scope_a
        assert meta["document_id"] == result.document_id
        assert meta["ingest_schema_version"] == settings.ingest_schema_version
        assert isinstance(meta["created_at"], int)


def test_upsert_disables_progress_output(settings, providers, scope_a):
    """show_progress defaults to True and would corrupt JSON logs via stdout."""
    _ingest(settings, scope_a)
    assert providers.index.show_progress_values
    assert all(v is False for v in providers.index.show_progress_values)


def test_upsert_batches_stay_within_limit(settings, providers, scope_a):
    _ingest(settings, scope_a)
    assert all(n <= 100 for n in providers.index.upsert_batch_sizes)


# --- Idempotency / reuse ------------------------------------------------------


def test_reupload_of_identical_bytes_is_free(settings, providers, scope_a):
    """Re-upload must cost zero embed calls, not just avoid duplicates."""
    first = _ingest(settings, scope_a)
    providers.cohere.embed_calls.clear()

    second = _ingest(settings, scope_a)
    assert second.reused is True
    assert second.document_id == first.document_id
    assert providers.cohere.embed_calls == []


def test_retry_creates_no_duplicate_chunks(settings, providers, scope_a):
    first = _ingest(settings, scope_a)
    count = len(providers.index.records)
    _ingest(settings, scope_a)
    assert len(providers.index.records) == count
    assert first.chunk_count == count


def test_same_bytes_different_scopes_do_not_share_records(
    settings, providers, scope_a, scope_b
):
    """Two visitors uploading the same PDF must not see each other's data."""
    a = _ingest(settings, scope_a)
    b = _ingest(settings, scope_b)
    assert a.document_id != b.document_id
    assert b.reused is False  # scope B did not inherit scope A's records

    scopes = {r["metadata"]["session_scope"] for r in providers.index.records.values()}
    assert scopes == {scope_a, scope_b}


# --- Transactionality --------------------------------------------------------


def test_embedding_failure_writes_nothing(settings, providers, scope_a):
    providers.cohere.embed_errors = [httpx.ConnectError("down")]
    with pytest.raises(ProviderError):
        _ingest(settings, scope_a)
    assert providers.index.records == {}


def test_embedding_failure_preserves_the_previous_document(
    settings, providers, scope_a
):
    """The core promise: a failed upload leaves the working document intact."""
    good = _ingest(settings, scope_a)
    before = dict(providers.index.records)

    providers.cohere.embed_errors = [httpx.ConnectError("down")]
    with pytest.raises(ProviderError):
        _ingest(settings, scope_a, pages=["Totally different content here."])

    assert providers.index.records == before
    answer = pipeline.answer_question(
        settings, question="What do widgets cost?",
        document_id=good.document_id, scope=scope_a,
    )
    assert answer.abstained is False


def test_upsert_failure_rolls_back_partial_writes(
    settings, providers, scope_a, monkeypatch
):
    """Second batch fails: the first batch must not survive as a partial doc.

    The batch size is lowered rather than generating hundreds of chunks -- the
    behaviour under test is "a later batch fails after an earlier one succeeded",
    not the chunk count that gets us there.
    """
    from rag_core import store as store_mod

    monkeypatch.setattr(store_mod, "UPSERT_BATCH", 2)

    calls = {"n": 0}
    original = providers.index.upsert

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second batch failed")
        return original(*args, **kwargs)

    providers.index.upsert = failing
    with pytest.raises(ProviderError):
        _ingest(settings, scope_a)
    providers.index.upsert = original

    assert calls["n"] >= 2, "test did not reach a second batch"
    assert providers.index.records == {}, "partial write was not rolled back"


def test_verification_survives_eventual_consistency(settings, providers, scope_a):
    """Records invisible for a few reads must still verify, not fail."""
    providers.index.visibility_delay = 2
    result = _ingest(settings, scope_a)
    assert result.chunk_count > 0


def test_verification_timeout_rolls_back(settings, providers, scope_a):
    """Never-visible records must be removed and the ingest reported failed."""
    original = providers.index.upsert

    def hide(*args, **kwargs):
        out = original(*args, **kwargs)
        for record in kwargs.get("vectors", []):
            providers.index.never_visible.add(record["id"])
        return out

    providers.index.upsert = hide
    with pytest.raises(IngestVerificationError):
        _ingest(settings, scope_a)
    providers.index.upsert = original

    assert providers.index.records == {}


def test_verification_uses_list_not_stats_or_sampling(
    settings, providers, scope_a
):
    """Only prefix listing can prove a specific document is fully stored."""
    _ingest(settings, scope_a)
    assert "list" in providers.index.calls
    assert "describe_index_stats" not in providers.index.calls


def test_rollback_failure_does_not_mask_the_original_error(
    settings, providers, scope_a
):
    providers.index.delete_errors = [RuntimeError("delete also failed")]
    original = providers.index.upsert

    def hide(*args, **kwargs):
        out = original(*args, **kwargs)
        for record in kwargs.get("vectors", []):
            providers.index.never_visible.add(record["id"])
        return out

    providers.index.upsert = hide
    # The verification error must surface, not the rollback error.
    with pytest.raises(IngestVerificationError):
        _ingest(settings, scope_a)
    providers.index.upsert = original


# --- Validation --------------------------------------------------------------


def test_non_pdf_rejected_without_provider_calls(settings, providers, scope_a):
    with pytest.raises(InvalidPdfError):
        pipeline.ingest_pdf(
            settings, data=b"not a pdf at all", filename="x.pdf", scope=scope_a
        )
    assert providers.cohere.embed_calls == []
    assert providers.index.records == {}


def test_oversized_upload_rejected_in_core(settings, providers, scope_a):
    from rag_core.errors import PayloadTooLargeError

    big = b"%PDF-" + b"x" * (settings.max_upload_bytes + 1)
    with pytest.raises(PayloadTooLargeError):
        pipeline.ingest_pdf(settings, data=big, filename="x.pdf", scope=scope_a)


# --- Ask ---------------------------------------------------------------------


def test_answer_uses_retrieved_evidence(settings, providers, scope_a):
    result = _ingest(settings, scope_a)
    answer = pipeline.answer_question(
        settings, question="What is the calibration constant?",
        document_id=result.document_id, scope=scope_a,
    )
    assert answer.abstained is False
    assert answer.answer
    assert answer.citations


def test_question_answerable_only_from_a_later_page(settings, providers, scope_a):
    """The decisive regression test for the single-vector implementation.

    The pre-audit code embedded the whole PDF as one vector, so content past the
    truncation point was unreachable. Here page 3's fact must be retrievable.
    """
    pages = [
        "Introduction. " + "filler content about nothing in particular. " * 30,
        "Methodology. " + "more filler content of no consequence. " * 30,
        "Results. The measured throughput was 8421 requests per second.",
    ]
    result = _ingest(settings, scope_a, pages=pages)
    answer = pipeline.answer_question(
        settings, question="What was the measured throughput?",
        document_id=result.document_id, scope=scope_a,
    )
    assert answer.abstained is False
    context = _last_chat_context(providers)
    assert "8421" in context, "later-page evidence was not retrieved"


def test_cross_page_synthesis_retrieves_both_pages(settings, providers, scope_a):
    pages = [
        "The alpha module consumes 30 watts under load.",
        "Unrelated discussion of packaging and logistics.",
        "The beta module consumes 45 watts under load.",
    ]
    result = _ingest(settings, scope_a, pages=pages)
    pipeline.answer_question(
        settings, question="How many watts do the alpha and beta modules consume?",
        document_id=result.document_id, scope=scope_a,
    )
    context = _last_chat_context(providers)
    assert "30 watts" in context and "45 watts" in context


def test_abstains_when_no_evidence_clears_the_threshold(
    settings, providers, scope_a, env
):
    """An impossible threshold must produce abstention, not invention."""
    env(RERANK_MIN_SCORE=0.99, MIN_EVIDENCE_CHUNKS=1)
    from rag_core.config import load_settings

    strict = load_settings()

    result = _ingest(strict, scope_a)
    answer = pipeline.answer_question(
        strict, question="What is the airspeed velocity of an unladen swallow?",
        document_id=result.document_id, scope=scope_a,
    )
    assert answer.abstained is True
    assert answer.citations == []
    assert providers.cohere.chat_calls == [], "must not pay to generate when abstaining"


def test_unknown_document_raises_404(settings, providers, scope_a):
    with pytest.raises(DocumentNotFoundError):
        pipeline.answer_question(
            settings, question="anything",
            document_id="v1:" + "0" * 32 + ":" + "1" * 32, scope=scope_a,
        )


def test_cannot_read_another_scopes_document(
    settings, providers, scope_a, scope_b
):
    """Cross-scope access is 404, so existence is not disclosed."""
    a = _ingest(settings, scope_a)
    with pytest.raises(DocumentNotFoundError):
        pipeline.answer_question(
            settings, question="What do widgets cost?",
            document_id=a.document_id, scope=scope_b,
        )


def test_document_removed_by_retention_reports_404(
    settings, providers, scope_a
):
    """A session open past RETENTION_HOURS gets a clean 'upload again'."""
    result = _ingest(settings, scope_a)
    providers.index.records.clear()
    with pytest.raises(DocumentNotFoundError):
        pipeline.answer_question(
            settings, question="anything",
            document_id=result.document_id, scope=scope_a,
        )


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_empty_question_rejected(settings, providers, scope_a, bad):
    result = _ingest(settings, scope_a)
    with pytest.raises(ValidationError):
        pipeline.answer_question(
            settings, question=bad, document_id=result.document_id, scope=scope_a
        )


def test_overlong_question_rejected(settings, providers, scope_a):
    result = _ingest(settings, scope_a)
    with pytest.raises(ValidationError, match="limited to"):
        pipeline.answer_question(
            settings, question="x" * (settings.max_question_chars + 1),
            document_id=result.document_id, scope=scope_a,
        )


# --- Retry policy ------------------------------------------------------------


def test_chat_is_not_retried_on_read_timeout(settings, providers, scope_a):
    """An accepted-but-slow request may already be billed; never retry it."""
    result = _ingest(settings, scope_a)
    providers.cohere.chat_errors = [
        httpx.ReadTimeout("too slow"), httpx.ReadTimeout("too slow")
    ]
    from rag_core.errors import ProviderTimeoutError

    with pytest.raises(ProviderTimeoutError):
        pipeline.answer_question(
            settings, question="What do widgets cost?",
            document_id=result.document_id, scope=scope_a,
        )
    assert len(providers.cohere.chat_calls) == 0


def test_chat_retries_once_on_connection_error(settings, providers, scope_a):
    """A connection that was never established is safe to retry exactly once."""
    result = _ingest(settings, scope_a)
    providers.cohere.chat_errors = [httpx.ConnectError("refused")]
    answer = pipeline.answer_question(
        settings, question="What do widgets cost?",
        document_id=result.document_id, scope=scope_a,
    )
    assert answer.abstained is False
    assert len(providers.cohere.chat_calls) == 1


def test_chat_gives_up_after_a_single_retry(settings, providers, scope_a):
    result = _ingest(settings, scope_a)
    providers.cohere.chat_errors = [
        httpx.ConnectError("refused"), httpx.ConnectError("refused")
    ]
    with pytest.raises(ProviderError):
        pipeline.answer_question(
            settings, question="What do widgets cost?",
            document_id=result.document_id, scope=scope_a,
        )
    assert len(providers.cohere.chat_calls) == 0


def test_upstream_rate_limit_maps_to_provider_error(settings, providers, scope_a):
    from rag_core.errors import ProviderRateLimitedError

    providers.cohere.embed_errors = [
        TooManyRequestsError(body="slow down"), TooManyRequestsError(body="slow down")
    ]
    with pytest.raises(ProviderRateLimitedError):
        _ingest(settings, scope_a)


# --- Deletion ----------------------------------------------------------------


def test_delete_document_removes_only_that_document(
    settings, providers, scope_a
):
    first = _ingest(settings, scope_a)
    second = _ingest(settings, scope_a, pages=["Entirely separate content."])
    assert first.document_id != second.document_id

    pipeline.delete_document(settings, document_id=first.document_id, scope=scope_a)
    remaining = {r["metadata"]["document_id"] for r in providers.index.records.values()}
    assert remaining == {second.document_id}


def test_delete_is_idempotent(settings, providers, scope_a):
    result = _ingest(settings, scope_a)
    pipeline.delete_document(settings, document_id=result.document_id, scope=scope_a)
    pipeline.delete_document(settings, document_id=result.document_id, scope=scope_a)
    assert providers.index.records == {}


def test_cannot_delete_another_scopes_document(
    settings, providers, scope_a, scope_b
):
    a = _ingest(settings, scope_a)
    with pytest.raises(DocumentNotFoundError):
        pipeline.delete_document(settings, document_id=a.document_id, scope=scope_b)
    assert providers.index.records, "scope B must not have deleted scope A's data"


def test_reset_removes_only_the_calling_scope(
    settings, providers, scope_a, scope_b
):
    _ingest(settings, scope_a)
    b = _ingest(settings, scope_b)

    pipeline.reset_scope(settings, scope=scope_a)
    scopes = {r["metadata"]["session_scope"] for r in providers.index.records.values()}
    assert scopes == {scope_b}
    assert any(
        r["metadata"]["document_id"] == b.document_id
        for r in providers.index.records.values()
    )
