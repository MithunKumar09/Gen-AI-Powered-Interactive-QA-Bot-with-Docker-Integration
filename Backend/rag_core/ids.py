"""Deterministic document and vector identifiers.

    document_id = v{schema}:{scope[:32]}:{file_hash[:32]}
    vector_id   = {document_id}:{chunk_index:05d}

Three properties fall out of this, and the whole ingest design leans on them:

* **Idempotency.** The same bytes in the same scope always produce the same ids,
  so a retried upsert overwrites rather than duplicating, and re-uploading an
  identical file is a no-op instead of a second embedding bill.
* **Prefix addressability.** ``document_id`` is a strict prefix of every one of
  its vector ids, so ``index.list(prefix=...)`` enumerates exactly one document
  without needing to know its chunk count.
* **Ownership is structural.** The scope is *inside* the id, so verifying that a
  caller owns a document is a string comparison against their derived scope --
  no lookup, no registry, and no way to reach another session's records.

``:05d`` supports up to 100 000 chunks, comfortably above MAX_CHUNKS's own
ceiling of 1000 (invariant 3), and zero-padding keeps ids lexicographically
ordered which makes paginated listings easier to read.
"""

from __future__ import annotations

import hashlib
import hmac
import re

from rag_core.errors import ValidationError
from rag_core.identity import SCOPE_SEGMENT_LEN, scope_segment

FILE_HASH_SEGMENT_LEN = 32
CHUNK_INDEX_WIDTH = 5

_DOCUMENT_ID_RE = re.compile(
    r"\Av(?P<schema>\d{1,4})"
    rf":(?P<scope>[0-9a-f]{{{SCOPE_SEGMENT_LEN}}})"
    rf":(?P<file_hash>[0-9a-f]{{{FILE_HASH_SEGMENT_LEN}}})\Z"
)


def file_hash(data: bytes) -> str:
    """SHA-256 of the raw upload bytes, hex.

    Hashing the immutable bytes -- not the filename, size, or an upload handle --
    is what makes re-upload detection reliable across Streamlit reruns.
    """
    return hashlib.sha256(data).hexdigest()


def make_document_id(schema_version: int, scope: str, file_hash_hex: str) -> str:
    return (
        f"v{schema_version}"
        f":{scope_segment(scope)}"
        f":{file_hash_hex[:FILE_HASH_SEGMENT_LEN]}"
    )


def make_vector_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index:0{CHUNK_INDEX_WIDTH}d}"


def make_vector_ids(document_id: str, chunk_count: int) -> list[str]:
    return [make_vector_id(document_id, i) for i in range(chunk_count)]


def document_prefix(document_id: str) -> str:
    """Prefix for ``index.list(prefix=...)``.

    The trailing colon is essential. Without it, listing document
    ``v1:aaa...:bbb`` would also match a hypothetical ``v1:aaa...:bbbb...``,
    conflating two documents.
    """
    return f"{document_id}:"


def parse_document_id(document_id: str) -> dict[str, object]:
    """Validate the structure of a client-supplied document id.

    Strict by design: this value arrives in a request body, so a loose parse
    would let a caller probe the id space or inject filter values.
    """
    if not document_id or not isinstance(document_id, str):
        raise ValidationError("document_id is required.")
    if len(document_id) > 128:
        raise ValidationError("document_id is malformed.")
    match = _DOCUMENT_ID_RE.match(document_id)
    if not match:
        raise ValidationError("document_id is malformed.")
    return {
        "schema_version": int(match.group("schema")),
        "scope_segment": match.group("scope"),
        "file_hash": match.group("file_hash"),
    }


def owns_document(document_id: str, scope: str, supported_schemas: frozenset[int]) -> bool:
    """Whether ``scope`` owns ``document_id`` under a supported schema version.

    A caller failing this check is answered with 404 rather than 403, so the
    response cannot confirm that another session's document exists.
    """
    try:
        parsed = parse_document_id(document_id)
    except ValidationError:
        return False
    if parsed["schema_version"] not in supported_schemas:
        return False
    return hmac.compare_digest(
        scope_segment(scope), str(parsed["scope_segment"])
    )
