"""HTTP routes. Thin by design: validate, call the pipeline, serialise.

No retrieval or generation logic lives here -- that is all in
:mod:`rag_core.pipeline`, which is testable without booting Flask.

Response discipline: no endpoint returns embeddings, provider payloads, raw
exception text, API keys, raw session ids, or full document text. The previous
``/upload`` returned the entire embedding vector to the browser (finding A8);
this one returns operational metadata only.
"""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, g, jsonify, request

from myapp.limits import ask_limits, mutation_limits, upload_limits
from rag_core import pipeline, store
from rag_core.config import Settings
from rag_core.errors import (
    AuthError,
    ProviderError,
    ReadinessError,
    ValidationError,
)
from rag_core.identity import (
    constant_time_equals,
    derive_scope,
    scope_ref,
    validate_session_id,
)

log = logging.getLogger(__name__)

routes = Blueprint("routes", __name__)

# /health must stay reachable without credentials so the platform can poll it.
# /ready is readable unauthenticated but returns only a boolean; detail requires
# the key.
_PUBLIC_PATHS = {"/health", "/ready"}


def _settings() -> Settings:
    return current_app.config["SETTINGS"]


def _authenticated() -> bool:
    provided = request.headers.get("X-API-Key", "")
    if not provided:
        return False
    return constant_time_equals(provided, _settings().backend_api_key)


@routes.before_request
def _authenticate():
    """Require the shared key on everything except the health endpoints.

    Also derives ``session_scope`` once per request so no route has to remember
    to, and so the raw session id stops travelling past this function.
    """
    if request.path in _PUBLIC_PATHS:
        return None

    if not _authenticated():
        raise AuthError()

    raw_session_id = validate_session_id(request.headers.get("X-Session-Id"))
    settings = _settings()
    g.session_scope = derive_scope(settings.session_scope_key, raw_session_id)
    # Only the short reference is retained for logging; the raw id is not stored.
    g.scope_ref = scope_ref(g.session_scope)
    return None


def _json_body() -> dict:
    if not request.is_json:
        raise ValidationError("Expected Content-Type: application/json.")
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValidationError("Expected a JSON object.")
    return body


def _require_str(body: dict, field: str, *, max_len: int) -> str:
    value = body.get(field)
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string.")
    value = value.strip()
    if not value:
        raise ValidationError(f"{field} is required.")
    if len(value) > max_len:
        raise ValidationError(f"{field} exceeds {max_len} characters.")
    return value


# --- Health / readiness ------------------------------------------------------


@routes.get("/health")
def health():
    """Liveness only. Makes no provider call.

    This is what Render's health check polls, so it must stay green while Cohere
    or Pinecone are down -- otherwise a provider blip would cause the platform to
    cycle a perfectly healthy container.
    """
    return jsonify({"status": "ok", "service": "rag-backend"}), 200


@routes.get("/ready")
def ready():
    """Readiness: configuration shape plus Pinecone compatibility.

    Deliberately makes **no billable Cohere call** -- a health probe that spends
    money per poll is a bill, not a check. Remote model availability is verified
    separately by ``scripts/smoke_models.py`` as a deployment gate.

    Detail is only returned to an authenticated caller; anyone else gets a bare
    boolean, so the endpoint cannot be used to enumerate our infrastructure.
    """
    settings = _settings()
    detailed = _authenticated()
    try:
        described = store.check_compatible(settings)
    except (ReadinessError, ProviderError) as exc:
        # ProviderError is caught here too, not just ReadinessError. Being unable
        # to *reach* Pinecone is a readiness failure (503), not a bad-gateway on
        # a user request (502) -- and letting it propagate to the generic handler
        # would return a full error envelope to unauthenticated callers,
        # leaking which failure mode we are in.
        if isinstance(exc, ReadinessError):
            reason = exc.message
        else:
            # A generic "upstream failed" is useless to an operator debugging a
            # 503. Name what we were doing; the index name is already in the
            # authenticated payload, and the provider text stays in the log.
            reason = (
                f"Could not reach Pinecone to verify index "
                f"{settings.pinecone_index!r}. Check PINECONE_API_KEY and "
                f"network egress."
            )
        log.error("not_ready", extra={"reason": reason, "detail": exc.detail})
        if not detailed:
            return jsonify({"ready": False}), 503
        return jsonify({"ready": False, "reason": reason}), 503

    if not detailed:
        return jsonify({"ready": True}), 200
    return (
        jsonify(
            {
                "ready": True,
                "index": described,
                "config": settings.public_summary(),
            }
        ),
        200,
    )


# --- Documents ---------------------------------------------------------------


@routes.post("/upload")
@upload_limits
def upload():
    """Ingest a PDF. Never deletes the caller's previous document.

    Returns the new document's identity so the frontend can switch to it, and
    accepts an ``upload_generation`` nonce which is echoed back unchanged. The
    frontend uses that plus ``file_hash`` to decide whether a response is still
    relevant, which is what stops a slow earlier upload from overwriting a newer
    selection when responses arrive out of order.
    """
    settings = _settings()

    if "file" not in request.files:
        raise ValidationError("No file was included in the request.")
    upload_file = request.files["file"]
    filename = (upload_file.filename or "").strip()
    if not filename:
        raise ValidationError("The uploaded file has no name.")
    if len(filename) > 255:
        raise ValidationError("The file name is too long.")

    try:
        generation = int(request.form.get("upload_generation", "0"))
    except ValueError:
        raise ValidationError("upload_generation must be an integer.") from None

    data = upload_file.read()

    result = pipeline.ingest_pdf(
        settings, data=data, filename=filename, scope=g.session_scope
    )

    log.info(
        "upload_ok",
        extra={
            "doc_ref": result.file_hash[:12],
            "page_count": result.page_count,
            "chunk_count": result.chunk_count,
            "reused": result.reused,
            "upload_generation": generation,
            **result.metrics,
        },
    )
    return (
        jsonify(
            {
                "document_id": result.document_id,
                "file_hash": result.file_hash,
                "page_count": result.page_count,
                "chunk_count": result.chunk_count,
                "reused": result.reused,
                "upload_generation": generation,
                "request_id": g.request_id,
            }
        ),
        200,
    )


@routes.post("/ask")
@ask_limits
def ask():
    """Answer a question from one document, or abstain."""
    settings = _settings()
    body = _json_body()
    question = _require_str(body, "question", max_len=settings.max_question_chars)
    document_id = _require_str(body, "document_id", max_len=128)

    result = pipeline.answer_question(
        settings, question=question, document_id=document_id, scope=g.session_scope
    )

    extra = {
        "abstained": result.abstained,
        "model_chat": settings.chat_model,
        "model_embed": settings.embed_model,
        **result.metrics,
    }
    if settings.log_questions:
        # Off by default. Enabled only for offline evaluation runs.
        extra["question"] = question
    log.info("ask_ok", extra=extra)

    return (
        jsonify(
            {
                "answer": result.answer,
                "citations": result.citations,
                "abstained": result.abstained,
                "request_id": g.request_id,
                "latency_ms": result.metrics.get("elapsed_ms"),
            }
        ),
        200,
    )


@routes.post("/documents/delete")
@mutation_limits
def delete_document():
    """Delete one of the caller's documents.

    POST rather than ``DELETE /documents/<id>`` because a document id contains
    ``:`` separators, which would need escaping in a path segment.

    Accepted, not confirmed: Pinecone returns no deleted-record count and its
    deletes are eventually consistent, so reporting a count here would be a
    fabrication.
    """
    body = _json_body()
    document_id = _require_str(body, "document_id", max_len=128)
    pipeline.delete_document(
        _settings(), document_id=document_id, scope=g.session_scope
    )
    log.info("document_delete_accepted")
    return (
        jsonify(
            {
                "accepted": True,
                "document_id": document_id,
                "request_id": g.request_id,
            }
        ),
        200,
    )


@routes.post("/reset")
@mutation_limits
def reset():
    """Delete every document owned by the calling session scope."""
    pipeline.reset_scope(_settings(), scope=g.session_scope)
    log.info("scope_reset_accepted")
    return (
        jsonify(
            {
                "accepted": True,
                "scope_ref": g.scope_ref,
                "request_id": g.request_id,
            }
        ),
        200,
    )
