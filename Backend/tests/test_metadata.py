"""Metadata sizing.

Pinecone's ceiling is measured in **bytes**. These tests pin down that we measure
bytes too, because inferring size from character count silently under-counts any
non-Latin document by 2-4x and would reject or accept the wrong files.
"""

from __future__ import annotations

import json

from rag_core.metadata import (
    build_metadata,
    fits,
    measure_bytes,
    overhead_bytes,
)


def _meta(text: str) -> dict:
    return build_metadata(
        scope="s" * 64,
        document_id="v1:" + "0" * 32 + ":" + "1" * 32,
        file_hash="f" * 64,
        schema_version=1,
        created_at=1_780_000_000,
        page=3,
        chunk_index=7,
        filename="report.pdf",
        text=text,
    )


def test_created_at_is_numeric_for_range_filtering():
    """Retention cleanup filters on `{"created_at": {"$lt": cutoff}}`."""
    assert isinstance(_meta("x")["created_at"], int)


def test_measurement_is_bytes_not_characters():
    ascii_meta = _meta("a" * 100)
    cjk_meta = _meta("漢" * 100)  # 3 bytes each in UTF-8
    assert measure_bytes(cjk_meta) > measure_bytes(ascii_meta)


def test_ascii_measurement_is_close_to_character_count():
    small, large = _meta("a" * 100), _meta("a" * 1100)
    assert measure_bytes(large) - measure_bytes(small) == 1000


def test_cjk_costs_three_bytes_per_character():
    base = measure_bytes(_meta(""))
    assert measure_bytes(_meta("漢" * 100)) - base == 300


def test_emoji_costs_four_bytes_per_codepoint():
    base = measure_bytes(_meta(""))
    assert measure_bytes(_meta("\U0001f600" * 50)) - base == 200


def test_combining_marks_counted_per_codepoint():
    base = measure_bytes(_meta(""))
    # "e" + combining acute: 1 byte + 2 bytes.
    assert measure_bytes(_meta("é" * 10)) - base == 30


def test_serialization_does_not_escape_non_ascii():
    """ensure_ascii would inflate CJK ~2x and wrongly reject documents that fit."""
    blob = json.dumps(_meta("漢字"), separators=(",", ":"), ensure_ascii=False)
    assert "漢" in blob
    assert "\\u6f22" not in blob


def test_overhead_excludes_the_text_field():
    with_text = measure_bytes(_meta("a" * 500))
    assert overhead_bytes(_meta("a" * 500)) == with_text - 500


def test_fits_respects_the_ceiling():
    assert fits(_meta("a" * 10), 1000) is True
    assert fits(_meta("a" * 10_000), 1000) is False


def test_shipped_defaults_leave_real_headroom(settings):
    """A worst-case 4-byte-per-char chunk must still fit the 32 KB ceiling."""
    worst = _meta("\U0001f600" * settings.chunk_chars)
    assert measure_bytes(worst) <= settings.metadata_max_bytes


def test_metadata_contains_no_raw_session_id():
    """Only the derived scope is stored, never the browser's session id."""
    meta = _meta("body text")
    assert "session_id" not in meta
    assert set(meta) == {
        "session_scope", "document_id", "file_hash", "ingest_schema_version",
        "created_at", "page", "chunk_index", "filename", "text",
    }
