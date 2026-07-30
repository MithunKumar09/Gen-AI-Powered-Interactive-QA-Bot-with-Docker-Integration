"""Typed domain errors.

Every error carries a stable machine-readable ``code`` and an HTTP ``status``.
The Flask layer maps these directly; it never invents its own statuses and never
forwards a raw provider exception string to a client (audit finding A7).
"""

from __future__ import annotations


class RagError(Exception):
    """Base for all domain errors.

    ``message`` is client-safe by construction: subclasses must not interpolate
    provider payloads, credentials, or document text into it. Anything sensitive
    belongs in ``detail``, which is logged but never serialised to a response.
    """

    code = "INTERNAL_ERROR"
    status = 500
    message = "An internal error occurred."

    def __init__(self, message: str | None = None, *, detail: str | None = None):
        if message is not None:
            self.message = message
        self.detail = detail
        super().__init__(self.message)


# --- Configuration -----------------------------------------------------------


class ConfigError(RagError):
    """Invalid local configuration. Raised at startup, never per-request.

    Distinct from ReadinessError: this is a malformed local value we can detect
    without touching any provider.
    """

    code = "CONFIG_INVALID"
    status = 500
    message = "Server configuration is invalid."


# --- Request validation ------------------------------------------------------


class ValidationError(RagError):
    code = "BAD_REQUEST"
    status = 400
    message = "The request was malformed."


class AuthError(RagError):
    code = "UNAUTHORIZED"
    status = 401
    message = "Missing or invalid API key."


class PayloadTooLargeError(RagError):
    code = "PAYLOAD_TOO_LARGE"
    status = 413
    message = "The uploaded file is too large."


class RateLimitedError(RagError):
    code = "RATE_LIMITED"
    status = 429
    message = "Too many requests. Please wait and try again."


# --- Documents ---------------------------------------------------------------


class DocumentNotFoundError(RagError):
    """Unknown document, or a document belonging to a different session scope.

    Deliberately 404 rather than 403 for the cross-scope case, so the response
    cannot be used to confirm that another session's document exists.
    """

    code = "UNKNOWN_DOCUMENT"
    status = 404
    message = "No such document for this session. Please upload again."


class InvalidPdfError(RagError):
    """The upload is not a usable PDF: malformed, encrypted, or image-only."""

    code = "INVALID_PDF"
    status = 422
    message = "The file could not be read as a text-bearing PDF."


class NoExtractableTextError(InvalidPdfError):
    code = "PDF_NO_EXTRACTABLE_TEXT"
    status = 422
    message = (
        "No selectable text was found in this PDF. Scanned or image-only "
        "documents are not supported."
    )


class ChunkBudgetExceededError(InvalidPdfError):
    """Over MAX_CHUNKS after metadata-driven splitting.

    A 422 rather than a silent truncation: dropping content would violate the
    grounding invariant that stored evidence and embeddings correspond exactly.
    """

    code = "CHUNK_BUDGET_EXCEEDED"
    status = 422
    message = "This document is too long for the demo. Please try a shorter one."


# --- Providers ---------------------------------------------------------------


class ProviderError(RagError):
    """Upstream provider failed in a way we could not classify more precisely."""

    code = "PROVIDER_ERROR"
    status = 502
    message = "An upstream service failed. Please try again."


class ProviderTimeoutError(ProviderError):
    """Upstream read timed out.

    For chat generation this is explicitly NOT retried: an ambiguous timeout may
    already have been billed and generated.
    """

    code = "PROVIDER_TIMEOUT"
    status = 504
    message = "An upstream service took too long. Please try again."


class ProviderRateLimitedError(ProviderError):
    code = "PROVIDER_RATE_LIMITED"
    status = 502
    message = "An upstream service is rate limiting us. Please try again shortly."


class IngestVerificationError(ProviderError):
    """Written vectors never became visible within the deadline.

    The partial write is rolled back by exact id before this is raised, so the
    previously active document remains usable.
    """

    code = "INGEST_VERIFY_TIMEOUT"
    status = 502
    message = "The document could not be confirmed as stored. Please try again."


class ReadinessError(RagError):
    """Infrastructure is present but incompatible with this configuration.

    Never self-heals: the application does not create or mutate infrastructure
    to resolve this (audit finding B9).
    """

    code = "NOT_READY"
    status = 503
    message = "The service is not ready."
