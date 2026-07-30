"""Structured JSON logging and request correlation.

Every line is one JSON object on stdout, which is what both ``docker logs`` and
Render's log viewer expect. The formatter has an explicit allow-list mindset:
fields arrive via ``extra=`` and anything sensitive is excluded by construction
rather than by remembering to redact it.

**Never logged:** API keys, ``SESSION_SCOPE_KEY``, raw session ids, document
text, embeddings, or question text (unless ``LOG_QUESTIONS=1``, which exists for
offline evaluation runs and is off in the demo).

Identity appears only as short derived references -- ``scope_ref`` (12 hex chars
of the HMAC scope) and ``doc_ref`` (12 hex chars of the file hash). Both are
enough to correlate a session's requests and neither is a usable handle to the
underlying records.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from typing import Any

from flask import g, has_request_context, request

# LogRecord attributes that are never emitted: either noise or already captured
# under a better name.
_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs",
    "message", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
}

# Belt-and-braces: if one of these ever reaches a log call via extra=, drop it.
_FORBIDDEN = {
    "cohere_api_key", "pinecone_api_key", "backend_api_key",
    "session_scope_key", "api_key", "authorization", "x_api_key",
    "session_id", "raw_session_id", "embedding", "embeddings",
    "text", "document_text", "password", "demo_password",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)
            ),
            "level": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
        }

        if has_request_context():
            payload["request_id"] = getattr(g, "request_id", None)
            payload["route"] = f"{request.method} {request.path}"
            scope_ref = getattr(g, "scope_ref", None)
            if scope_ref:
                payload["scope_ref"] = scope_ref

        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            if key.lower() in _FORBIDDEN:
                continue
            payload[key] = value

        if record.exc_info:
            # Traceback goes to the log, never to the client.
            payload["traceback"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str) -> None:
    """Send all logging to stdout as JSON, replacing any inherited handlers."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))

    # Gunicorn installs its own handlers; route them through ours so access and
    # error lines are JSON too rather than a second, differently-shaped format.
    for name in ("gunicorn.error", "gunicorn.access", "werkzeug"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    # These libraries log request bodies and headers at DEBUG, which would defeat
    # the exclusions above.
    for noisy in ("httpx", "httpcore", "urllib3", "pinecone", "cohere"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
