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
    ProviderError,
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

    # 5. Write, then verify. On any failure roll back by exact id -- which is
    #    possible precisely because the ids are deterministic.
    try:
        store_mod.upsert_chunks(
            settings,
            vector_ids=expected_ids,
            embeddings=vectors,
            metadatas=metadatas,
        )
        store_mod.await_visible(settings, document_id, expected_ids)
    except Exception as exc:
        _rollback(settings, expected_ids, reason=type(exc).__name__)
        if isinstance(exc, ProviderError):
            from rag_core.errors import IngestVerificationError

            raise IngestVerificationError(detail=exc.detail) from exc
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
    return AskResult(
        answer=generated.text or ABSTENTION_TEXT,
        citations=generated.citations,
        abstained=not generated.text,
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
    """
    if not candidates:
        return [], 0.0, False

    if settings.enable_rerank:
        texts = [str(c["metadata"].get("text", "")) for c in candidates]
        ranked = rerank_mod.rerank(settings, question, texts)
        evidence: list[Evidence] = []
        best = 0.0
        for position, score in ranked:
            best = max(best, score)
            if score < settings.rerank_min_score:
                continue
            match = candidates[position]
            evidence.append(_to_evidence(match, score))
        return evidence, best, True

    # Without reranking, the cosine score is the only signal available, so the
    # rerank threshold cannot apply -- using it here would compare unlike scales.
    top = candidates[: settings.rerank_top_n]
    best = max((c["score"] for c in candidates), default=0.0)
    return [_to_evidence(m, m["score"]) for m in top], best, False


def _to_evidence(match: dict[str, Any], score: float) -> Evidence:
    meta = match.get("metadata", {}) or {}
    return Evidence(
        chunk_index=int(meta.get("chunk_index", -1) or -1),
        page=int(meta.get("page", 0) or 0),
        text=str(meta.get("text", "")),
        score=float(score),
    )


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
