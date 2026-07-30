"""Pinecone record metadata: construction and byte-accurate measurement.

Pinecone's per-record limit is **40 960 bytes of filterable metadata**, measured
in bytes, not characters. Inferring size from ``len(text)`` is wrong for any
non-ASCII document: a page of CJK text is roughly three bytes per character, and
emoji four. So every record is serialised and measured for real before it is
allowed anywhere near an embedding call.

The ordering is the important part. Sizing happens *before* embedding, and an
over-ceiling chunk is **split**, never trimmed -- see the note on
:func:`fit_chunks` below.
"""

from __future__ import annotations

import json

# json.dumps with ensure_ascii=False emits real UTF-8 rather than \uXXXX
# escapes. That matters: escaping would inflate a CJK string roughly 2x and we
# would reject documents that actually fit.
_COMPACT = {"separators": (",", ":"), "ensure_ascii": False}


def build_metadata(
    *,
    scope: str,
    document_id: str,
    file_hash: str,
    schema_version: int,
    created_at: int,
    page: int,
    chunk_index: int,
    filename: str,
    text: str,
) -> dict[str, object]:
    """Assemble one record's metadata.

    ``created_at`` is a Unix epoch **integer** rather than an ISO string
    specifically so retention cleanup can express its cutoff as a numeric
    Pinecone filter (``{"created_at": {"$lt": cutoff}}``) in a single call.
    """
    return {
        "session_scope": scope,
        "document_id": document_id,
        "file_hash": file_hash,
        "ingest_schema_version": schema_version,
        "created_at": created_at,
        "page": page,
        "chunk_index": chunk_index,
        "filename": filename,
        "text": text,
    }


def measure_bytes(metadata: dict[str, object]) -> int:
    """UTF-8 byte length of ``metadata`` serialised as compact JSON."""
    return len(json.dumps(metadata, **_COMPACT).encode("utf-8"))


def fits(metadata: dict[str, object], ceiling_bytes: int) -> bool:
    return measure_bytes(metadata) <= ceiling_bytes


def overhead_bytes(metadata: dict[str, object]) -> int:
    """Bytes consumed by everything except the chunk text.

    Used to compute how much text budget a split actually has, instead of
    guessing at the scalar-field cost.
    """
    without_text = dict(metadata)
    without_text["text"] = ""
    return measure_bytes(without_text)
