"""Session identity derivation.

The browser sends a raw session id. That value never reaches Pinecone and never
reaches the logs: it is used once to derive an HMAC ``session_scope``, then
discarded.

Why HMAC rather than a plain hash: a plain SHA-256 of a uuid4 would be just as
unguessable, but HMAC with a server-held key means an attacker who somehow
learns a raw session id still cannot compute the scope under which that
session's records are stored, and cannot enumerate scopes offline.

Rotating ``SESSION_SCOPE_KEY`` makes every existing scoped record unreachable.
They are not leaked -- nobody can derive their scope -- and the retention
cleanup removes them on the normal ``created_at`` cutoff.
"""

from __future__ import annotations

import hashlib
import hmac
import re

from rag_core.errors import ValidationError

# Accept the shapes a frontend would legitimately generate: uuid4 (with or
# without hyphens) and secrets.token_urlsafe / token_hex output. Anything else
# is rejected rather than normalised, so a malformed id cannot silently collapse
# two callers into one scope.
_SESSION_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{16,128}\Z")

# Length of the scope segment embedded in a document id. 32 hex characters is
# 128 bits of the HMAC -- far beyond collision risk for a demo, and it keeps the
# resulting id short enough to read in a log line.
SCOPE_SEGMENT_LEN = 32

# Length of the short reference used in logs. Long enough to correlate requests
# within a session, short enough that it is not a usable handle to the scope.
SCOPE_REF_LEN = 12


def validate_session_id(raw: str | None) -> str:
    """Validate the caller-supplied session id.

    Raises :class:`ValidationError` on anything that is not a plausible opaque
    token. The bound matters: an unbounded header would otherwise flow into
    HMAC input and log fields.
    """
    if not raw:
        raise ValidationError("Missing X-Session-Id header.")
    if not _SESSION_ID_RE.match(raw):
        raise ValidationError(
            "X-Session-Id must be 16-128 characters of [A-Za-z0-9_-]."
        )
    return raw


def derive_scope(session_scope_key: str, raw_session_id: str) -> str:
    """Derive the storage scope for a session. Returns a hex digest.

    The caller is expected to have validated ``raw_session_id`` already; this
    function does not log or retain its input.
    """
    return hmac.new(
        session_scope_key.encode("utf-8"),
        raw_session_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def scope_segment(scope: str) -> str:
    """The portion of a scope that is embedded in a document id."""
    return scope[:SCOPE_SEGMENT_LEN]


def scope_ref(scope: str) -> str:
    """Short, non-reversible correlation prefix for structured logs.

    This is what appears in log output -- never the raw session id, and never
    the full scope.
    """
    return scope[:SCOPE_REF_LEN]


def constant_time_equals(a: str, b: str) -> bool:
    """Timing-safe comparison for shared secrets and passphrases."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
