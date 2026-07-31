"""The only public entry points into the RAG core.

Ingest is **transactional with immutable staging**. The previously active
document is never touched: a new document is written under its own deterministic
ids and verified before the caller is told it exists, and deleting the old one is
a separate call the frontend makes only after it has switched over. So any
failure -- bad PDF, embedding outage, partial write, verification timeout --
leaves the working document working.

The ordering below is not arbitrary. Extraction, chunking and sizing all happen
before the first embedding call because they are local and free; embedding
happens before the first write because a provider failure then leaves nothing to
clean up.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from rag_core import embeddings as embed_mod
from rag_core import generation as gen_mod
from rag_core import pdf as pdf_mod
from rag_core import rerank as rerank_mod
from rag_core import store as store_mod
from rag_core.config import Settings
from rag_core.errors import (
    DocumentNotFoundError,
    IngestVerificationError,
    ProviderError,
    ProviderTimeoutError,
    ValidationError,
)
from rag_core.generation import Answer, Evidence
from rag_core.ids import (
    file_hash as hash_bytes,
    make_document_id,
    make_vector_id,
    owns_document,
)
from rag_core.metadata import build_metadata, overhead_bytes

log = logging.getLogger(__name__)

ABSTENTION_TEXT = (
    "I couldn't find anything in this document that answers that. "
    "Try rephrasing, or ask about something the document actually covers."
)


@dataclass
class IngestResult:
    document_id: str
    file_hash: str
    page_count: int
    chunk_count: int
    reused: bool
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AskResult:
    answer: str
    citations: list[dict[str, Any]]
    abstained: bool
    metrics: dict[str, Any] = field(default_factory=dict)


# --- Ingest ------------------------------------------------------------------


def ingest_pdf(
    settings: Settings, *, data: bytes, filename: str, scope: str
) -> IngestResult:
    """Ingest a PDF and return its identity. Never deletes anything."""
    if len(data) > settings.max_upload_bytes:
        # Defence in depth: Flask's MAX_CONTENT_LENGTH should already have
        # rejected this, but the core must not trust its caller.
        from rag_core.errors import PayloadTooLargeError

        raise PayloadTooLargeError()

    started = time.monotonic()
    file_hash = hash_bytes(data)
    document_id = make_document_id(settings.ingest_schema_version, scope, file_hash)

    # 1. Validate and extract. Local only -- no provider calls, no cost.
    reader = pdf_mod.validate_and_read(data, max_pages=settings.max_pages)
    pages = pdf_mod.extract_pages(
        reader, max_extracted_chars=settings.max_extracted_chars
    )

    # 2. Chunk, then size-and-split until every record fits the byte ceiling.
    #    Indexes are assigned inside fit_chunks, once no further split can occur.
    raw_chunks = pdf_mod.chunk_pages(
        pages, chunk_chars=settings.chunk_chars, overlap=settings.chunk_overlap
    )
    created_at = int(time.time())
    probe = build_metadata(
        scope=scope,
        document_id=document_id,
        file_hash=file_hash,
        schema_version=settings.ingest_schema_version,
        created_at=created_at,
        page=9999,
        chunk_index=99999,
        filename=filename,
        text="",
    )
    chunks = pdf_mod.fit_chunks(
        raw_chunks,
        metadata_ceiling_bytes=settings.metadata_max_bytes,
        overhead_bytes=overhead_bytes(probe),
        max_chunks=settings.max_chunks,
    )
    expected_ids = [make_vector_id(document_id, c.index) for c in chunks]

    # 3. Already fully stored? Then this is a re-upload of identical bytes by the
    #    same scope, and there is nothing to do. Checked before embedding, so a
    #    repeat upload costs nothing.
    already = store_mod.list_document_ids(settings, document_id)
    if already == set(expected_ids):
        log.info(
            "ingest_reused",
            extra={"chunk_count": len(chunks), "page_count": len(pages)},
        )
        return IngestResult(
            document_id=document_id,
            file_hash=file_hash,
            page_count=len(pages),
            chunk_count=len(chunks),
            reused=True,
            metrics={"elapsed_ms": _ms(started), "embed_calls": 0},
        )

    # 4. Embed. Nothing has been written yet, so a failure here is clean.
    vectors = embed_mod.embed_documents(settings, [c.text for c in chunks])

    metadatas = [
        build_metadata(
            scope=scope,
            document_id=document_id,
            file_hash=file_hash,
            schema_version=settings.ingest_schema_version,
            created_at=created_at,
            page=chunk.page,
            chunk_index=chunk.index,
            filename=filename,
            text=chunk.text,
        )
        for chunk in chunks
    ]

    # 5. Write, then verify. The phases are deliberately separated so an upsert
    #    failure keeps its original provider classification, while only an actual
    #    visibility timeout becomes INGEST_VERIFY_TIMEOUT.
    try:
        store_mod.upsert_chunks(
            settings,
            vector_ids=expected_ids,
            embeddings=vectors,
            metadatas=metadatas,
        )
    except Exception as exc:
        _rollback(settings, expected_ids, reason=type(exc).__name__)
        raise

    try:
        store_mod.await_visible(settings, document_id, expected_ids)
    except ProviderTimeoutError as exc:
        _rollback(settings, expected_ids, reason=type(exc).__name__)
        raise IngestVerificationError(detail=exc.detail) from exc
    except Exception as exc:
        _rollback(settings, expected_ids, reason=type(exc).__name__)
        raise

    log.info(
        "ingest_complete",
        extra={
            "chunk_count": len(chunks),
            "page_count": len(pages),
            "elapsed_ms": _ms(started),
        },
    )
    return IngestResult(
        document_id=document_id,
        file_hash=file_hash,
        page_count=len(pages),
        chunk_count=len(chunks),
        reused=False,
        metrics={"elapsed_ms": _ms(started), "embed_calls": len(chunks)},
    )


def _rollback(settings: Settings, vector_ids: list[str], *, reason: str) -> None:
    """Remove a partially written document.

    Best effort by design: if this fails the orphan is logged and reclaimed by
    the retention cleanup. Raising here would mask the original error.
    """
    try:
        store_mod.delete_ids(settings, vector_ids)
        log.warning(
            "ingest_rolled_back", extra={"reason": reason, "ids": len(vector_ids)}
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.error(
            "ingest_rollback_failed",
            extra={"reason": reason, "rollback_error": str(exc), "ids": len(vector_ids)},
        )


# --- Ask ---------------------------------------------------------------------


def answer_question(
    settings: Settings, *, question: str, document_id: str, scope: str
) -> AskResult:
    """Answer ``question`` from ``document_id``, or abstain."""
    question = (question or "").strip()
    if not question:
        raise ValidationError("A question is required.")
    if len(question) > settings.max_question_chars:
        raise ValidationError(
            f"Questions are limited to {settings.max_question_chars} characters."
        )
    # Ownership is structural: the scope is inside the document id, so this is a
    # comparison rather than a lookup. 404 on failure, so the response cannot
    # confirm another session's document exists.
    if not owns_document(document_id, scope, settings.supported_schema_versions):
        raise DocumentNotFoundError()

    started = time.monotonic()
    query_vector = embed_mod.embed_query(settings, question)
    matches = store_mod.query(
        settings, scope=scope, document_id=document_id, vector=query_vector
    )

    if not matches:
        # No records at all: either never ingested, or retention cleanup removed
        # them while this browser session stayed open. Both are "upload again".
        raise DocumentNotFoundError()

    candidates = [
        m for m in matches if m["score"] >= settings.vector_min_score
    ]
    evidence, top_score, rerank_used = _select_evidence(
        settings, question, candidates
    )

    if len(evidence) < settings.min_evidence_chunks:
        log.info(
            "abstained",
            extra={
                "retrieved_k": len(matches),
                "reranked_n": len(evidence),
                "top_score": round(top_score, 4),
                "rerank_used": rerank_used,
                "elapsed_ms": _ms(started),
            },
        )
        return AskResult(
            answer=ABSTENTION_TEXT,
            citations=[],
            abstained=True,
            metrics={
                "retrieved_k": len(matches),
                "reranked_n": 0,
                "top_score": top_score,
                "elapsed_ms": _ms(started),
            },
        )

    generated: Answer = gen_mod.generate(
        settings, question=question, evidence=evidence
    )
    answer_text = (generated.text or "").strip()
    if not answer_text:
        return AskResult(
            answer=ABSTENTION_TEXT,
            citations=[],
            abstained=True,
            metrics={
                "retrieved_k": len(matches),
                "reranked_n": len(evidence),
                "top_score": top_score,
                "rerank_used": rerank_used,
                "elapsed_ms": _ms(started),
            },
        )

    return AskResult(
        answer=answer_text,
        citations=generated.citations,
        abstained=False,
        metrics={
            "retrieved_k": len(matches),
            "reranked_n": len(evidence),
            "top_score": top_score,
            "rerank_used": rerank_used,
            "elapsed_ms": _ms(started),
        },
    )


def _select_evidence(
    settings: Settings, question: str, candidates: list[dict[str, Any]]
) -> tuple[list[Evidence], float, bool]:
    """Narrow candidates to the evidence worth answering from.

    Three signals stay separate on purpose. The cosine score from the vector
    index and the cross-encoder relevance score are not the same quantity and are
    not comparable, so they get their own thresholds; the count of surviving
    chunks is the third. Collapsing them into one number is what makes abstention
    behave unpredictably.

    Candidate metadata is validated before reranking so malformed provider data
    cannot be sent onward to another provider or silently converted into fake
    evidence.
    """
    if not candidates:
        return [], 0.0, False

    validated = [
        _to_evidence(candidate, candidate.get("score"))
        for candidate in candidates
    ]

    if settings.enable_rerank:
        ranked = rerank_mod.rerank(
            settings,
            question,
            [item.text for item in validated],
        )

        evidence: list[Evidence] = []
        seen_positions: set[int] = set()
        scores: list[float] = []

        for raw_position, raw_score in ranked:
            position = _required_int(
                raw_position,
                field="rerank result index",
                minimum=0,
            )
            if position >= len(validated):
                raise ProviderError(
                    detail=(
                        "Cohere rerank returned an out-of-range candidate index: "
                        f"{position} for {len(validated)} candidates"
                    )
                )
            if position in seen_positions:
                raise ProviderError(
                    detail=(
                        "Cohere rerank returned a duplicate candidate index: "
                        f"{position}"
                    )
                )
            seen_positions.add(position)

            score = _finite_float(
                raw_score,
                field="rerank relevance_score",
            )
            scores.append(score)

            if score < settings.rerank_min_score:
                continue

            item = validated[position]
            evidence.append(
                Evidence(
                    chunk_index=item.chunk_index,
                    page=item.page,
                    text=item.text,
                    score=score,
                )
            )

        return evidence, max(scores, default=0.0), True

    # Without reranking, the cosine score is the only signal available, so the
    # rerank threshold cannot apply -- using it here would compare unlike scales.
    top = validated[: settings.rerank_top_n]
    best = max((item.score for item in validated), default=0.0)
    return top, best, False


def _to_evidence(match: dict[str, Any], score: Any) -> Evidence:
    """Convert one Pinecone match into validated generation evidence.

    Pinecone numeric metadata may be returned as integers or integral floats.
    Chunk index zero is valid and must never be collapsed into a missing-value
    sentinel by truthiness expressions such as ``value or -1``.
    """
    metadata = match.get("metadata")
    if not isinstance(metadata, dict):
        raise ProviderError(
            detail="Retrieved Pinecone match is missing metadata"
        )

    chunk_index = _required_int(
        metadata.get("chunk_index"),
        field="chunk_index metadata",
        minimum=0,
    )
    page = _required_int(
        metadata.get("page"),
        field="page metadata",
        minimum=1,
    )
    text = _required_text(
        metadata.get("text"),
        field="text metadata",
    )
    relevance = _finite_float(
        score,
        field="retrieval score",
    )

    return Evidence(
        chunk_index=chunk_index,
        page=page,
        text=text,
        score=relevance,
    )


def _required_int(value: Any, *, field: str, minimum: int) -> int:
    """Parse a required integer without truncating malformed numeric values."""
    if value is None or isinstance(value, bool):
        raise ProviderError(
            detail=f"Retrieved provider data has invalid {field}: {value!r}"
        )

    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ProviderError(
                detail=f"Retrieved provider data has invalid {field}: {value!r}"
            )

    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProviderError(
            detail=f"Retrieved provider data has invalid {field}: {value!r}"
        ) from exc

    # Prevent values such as "1.5" from being accepted through a future custom
    # numeric type whose int conversion silently truncates.
    if isinstance(value, str) and value.strip() != str(parsed):
        raise ProviderError(
            detail=f"Retrieved provider data has invalid {field}: {value!r}"
        )

    if parsed < minimum:
        raise ProviderError(
            detail=(
                f"Retrieved provider data has out-of-range {field}: "
                f"{parsed}; minimum is {minimum}"
            )
        )

    return parsed


def _required_text(value: Any, *, field: str) -> str:
    """Return non-empty provider text without altering the stored bytes."""
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(
            detail=f"Retrieved provider data has invalid {field}"
        )
    return value


def _finite_float(value: Any, *, field: str) -> float:
    """Parse a finite provider score and reject NaN/Infinity."""
    if value is None or isinstance(value, bool):
        raise ProviderError(
            detail=f"Retrieved provider data has invalid {field}: {value!r}"
        )

    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProviderError(
            detail=f"Retrieved provider data has invalid {field}: {value!r}"
        ) from exc

    if not math.isfinite(parsed):
        raise ProviderError(
            detail=f"Retrieved provider data has invalid {field}: {value!r}"
        )

    return parsed


# --- Deletion ----------------------------------------------------------------


def delete_document(settings: Settings, *, document_id: str, scope: str) -> None:
    """Delete one document owned by ``scope``.

    Accepted rather than confirmed: Pinecone reports no deleted count and
    deletion is eventually consistent.
    """
    if not owns_document(document_id, scope, settings.supported_schema_versions):
        raise DocumentNotFoundError()
    store_mod.delete_document(settings, scope=scope, document_id=document_id)


def reset_scope(settings: Settings, *, scope: str) -> None:
    """Delete every document owned by ``scope``."""
    store_mod.delete_scope(settings, scope=scope)


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)