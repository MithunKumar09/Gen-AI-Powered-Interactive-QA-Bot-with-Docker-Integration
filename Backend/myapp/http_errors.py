"""Uniform error envelope.

Every failure -- domain error, Flask abort, or unexpected exception -- leaves as:

    {"error": {"code": "...", "message": "...", "request_id": "..."}}

The pre-audit routes returned ``{'status': 'error', 'message': str(e)}``, which
forwarded raw provider and parser exception text to the client (finding A7).
Here the full exception is logged with its request id and the caller gets a
generic message for that class of failure. The request id is the link between
the two, so a user can report "request 3f9a2b failed" and it is findable.
"""

from __future__ import annotations

import logging

from flask import Flask, g, jsonify
from werkzeug.exceptions import HTTPException

from rag_core.errors import RagError

log = logging.getLogger(__name__)

# Generic messages for Werkzeug's own aborts, so a raised HTTPException cannot
# leak framework internals either.
_HTTP_CODES = {
    400: ("BAD_REQUEST", "The request was malformed."),
    401: ("UNAUTHORIZED", "Missing or invalid API key."),
    404: ("NOT_FOUND", "Not found."),
    405: ("METHOD_NOT_ALLOWED", "Method not allowed."),
    413: ("PAYLOAD_TOO_LARGE", "The uploaded file is too large."),
    415: ("UNSUPPORTED_MEDIA_TYPE", "Unsupported content type."),
    429: ("RATE_LIMITED", "Too many requests. Please wait and try again."),
    500: ("INTERNAL_ERROR", "An internal error occurred."),
    503: ("NOT_READY", "The service is not ready."),
}


def _envelope(code: str, message: str, status: int):
    return (
        jsonify(
            {
                "error": {
                    "code": code,
                    "message": message,
                    "request_id": getattr(g, "request_id", None),
                }
            }
        ),
        status,
    )


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(RagError)
    def _domain(exc: RagError):
        # detail carries provider text; it is logged, never serialised.
        level = logging.ERROR if exc.status >= 500 else logging.WARNING
        log.log(
            level,
            "request_failed",
            extra={
                "error_code": exc.code,
                "status": exc.status,
                "reason": exc.detail,
            },
        )
        return _envelope(exc.code, exc.message, exc.status)

    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException):
        status = exc.code or 500
        code, message = _HTTP_CODES.get(status, ("HTTP_ERROR", "Request failed."))
        log.warning(
            "request_failed", extra={"error_code": code, "status": status}
        )
        return _envelope(code, message, status)

    @app.errorhandler(Exception)
    def _unexpected(exc: Exception):
        # Nothing about an unanticipated exception is safe to return, so the
        # traceback is logged and the caller gets a bare 500.
        log.exception(
            "unhandled_exception",
            extra={"error_code": "INTERNAL_ERROR", "status": 500},
        )
        return _envelope("INTERNAL_ERROR", "An internal error occurred.", 500)
