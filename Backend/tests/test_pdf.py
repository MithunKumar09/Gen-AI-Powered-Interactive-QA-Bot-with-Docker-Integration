"""PDF validation, extraction, and chunking.

Covers the inputs the pre-audit implementation crashed on -- notably
``extract_text()`` returning ``None``, which made ``document_text += None`` raise
TypeError -- plus the page attribution that citations depend on.
"""

from __future__ import annotations

import pytest
from conftest import make_pdf

from rag_core.errors import (
    ChunkBudgetExceededError,
    InvalidPdfError,
    NoExtractableTextError,
)
from rag_core.metadata import build_metadata, overhead_bytes
from rag_core.pdf import (
    Chunk,
    Page,
    chunk_pages,
    extract_pages,
    fit_chunks,
    normalize_whitespace,
    validate_and_read,
)


def _probe_overhead() -> int:
    return overhead_bytes(
        build_metadata(
            scope="s" * 64, document_id="v1:" + "0" * 32 + ":" + "1" * 32,
            file_hash="f" * 64, schema_version=1, created_at=1_780_000_000,
            page=1, chunk_index=1, filename="f.pdf", text="",
        )
    )


# --- Validation --------------------------------------------------------------


def test_empty_file_rejected():
    with pytest.raises(InvalidPdfError, match="empty"):
        validate_and_read(b"", max_pages=40)


def test_non_pdf_bytes_rejected():
    """A .docx or PNG renamed to .pdf must fail cleanly, not in the parser."""
    with pytest.raises(InvalidPdfError, match="not a PDF"):
        validate_and_read(b"PK\x03\x04 this is a zip", max_pages=40)


def test_truncated_pdf_rejected():
    with pytest.raises(InvalidPdfError):
        validate_and_read(b"%PDF-1.4\ngarbage", max_pages=40)


def test_page_limit_enforced():
    pdf = make_pdf([f"Page {i} content here." for i in range(6)])
    with pytest.raises(InvalidPdfError, match="at most"):
        validate_and_read(pdf, max_pages=3)


def test_valid_pdf_accepted():
    reader = validate_and_read(make_pdf(["Hello world."]), max_pages=40)
    assert len(reader.pages) == 1


# --- Extraction --------------------------------------------------------------


def test_extraction_preserves_page_numbers():
    pdf = make_pdf(["Alpha content.", "Beta content.", "Gamma content."])
    pages = extract_pages(validate_and_read(pdf, max_pages=40),
                          max_extracted_chars=100_000)
    assert [p.number for p in pages] == [1, 2, 3]
    assert "Gamma" in pages[2].text


def test_image_only_pdf_raises_422(monkeypatch):
    """No selectable text: report it rather than embedding an empty string."""
    reader = validate_and_read(make_pdf(["text"]), max_pages=40)
    monkeypatch.setattr(type(reader.pages[0]), "extract_text",
                        lambda self, *a, **k: "")
    with pytest.raises(NoExtractableTextError):
        extract_pages(reader, max_extracted_chars=100_000)


def test_extract_text_returning_none_is_handled(monkeypatch):
    """The exact crash in the pre-audit code: `document_text += None`."""
    reader = validate_and_read(make_pdf(["text"]), max_pages=40)
    monkeypatch.setattr(type(reader.pages[0]), "extract_text",
                        lambda self, *a, **k: None)
    with pytest.raises(NoExtractableTextError):
        extract_pages(reader, max_extracted_chars=100_000)


def test_page_that_raises_is_skipped_not_fatal(monkeypatch):
    pdf = make_pdf(["Good page one.", "Good page two."])
    reader = validate_and_read(pdf, max_pages=40)
    calls = {"n": 0}
    original = type(reader.pages[0]).extract_text

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("damaged content stream")
        return original(self, *a, **k)

    monkeypatch.setattr(type(reader.pages[0]), "extract_text", flaky)
    pages = extract_pages(reader, max_extracted_chars=100_000)
    assert len(pages) == 1  # the good page survived


def test_character_budget_caps_extraction():
    pdf = make_pdf(["word " * 200, "word " * 200, "word " * 200])
    pages = extract_pages(validate_and_read(pdf, max_pages=40),
                          max_extracted_chars=300)
    assert sum(len(p.text) for p in pages) <= 300


# --- Normalization -----------------------------------------------------------


def test_whitespace_runs_collapsed():
    assert normalize_whitespace("a     b\t\tc") == "a b c"


def test_paragraph_breaks_preserved():
    assert normalize_whitespace("a\n\n\n\n\nb") == "a\n\nb"


def test_hyphenation_across_line_breaks_rejoined():
    assert normalize_whitespace("compre-\nhensive") == "comprehensive"


def test_crlf_normalized():
    assert normalize_whitespace("a\r\nb") == "a\nb"


# --- Chunking ----------------------------------------------------------------


def test_short_page_is_one_chunk():
    chunks = chunk_pages([Page(1, "Short text.")], chunk_chars=1200, overlap=200)
    assert len(chunks) == 1
    assert chunks[0].page == 1


def test_long_page_splits_into_multiple_chunks():
    chunks = chunk_pages([Page(1, "word " * 1000)], chunk_chars=500, overlap=100)
    assert len(chunks) > 1
    assert all(c.page == 1 for c in chunks)


def test_chunks_respect_the_size_bound():
    chunks = chunk_pages([Page(1, "word " * 2000)], chunk_chars=500, overlap=50)
    assert all(len(c.text) <= 500 for c in chunks)


def test_pages_are_chunked_independently_so_citations_stay_exact():
    pages = [Page(1, "alpha " * 200), Page(2, "beta " * 200)]
    chunks = chunk_pages(pages, chunk_chars=300, overlap=50)
    for chunk in chunks:
        # No chunk may mix content from two pages, or its page number is a lie.
        assert not ("alpha" in chunk.text and "beta" in chunk.text)


def test_overlap_actually_overlaps():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_pages([Page(1, text)], chunk_chars=400, overlap=150)
    assert len(chunks) >= 2
    first_tail = set(chunks[0].text.split()[-8:])
    second = set(chunks[1].text.split())
    assert first_tail & second


def test_repeated_headers_and_footers_do_not_merge_pages():
    header = "ACME CONFIDENTIAL REPORT 2026"
    pages = [Page(i, f"{header}\n\nUnique body {i}. " + "filler " * 40)
             for i in range(1, 4)]
    chunks = chunk_pages(pages, chunk_chars=1200, overlap=200)
    for i in range(1, 4):
        assert any(f"Unique body {i}" in c.text and c.page == i for c in chunks)


def test_empty_pages_raise_rather_than_producing_zero_chunks():
    with pytest.raises(NoExtractableTextError):
        chunk_pages([Page(1, "")], chunk_chars=1200, overlap=200)


# --- fit_chunks: indexing and the split-to-fit path -------------------------


def test_indexes_assigned_contiguously_from_zero():
    chunks = [Chunk("text", 1) for _ in range(5)]
    fitted = fit_chunks(chunks, metadata_ceiling_bytes=32_768,
                        overhead_bytes=_probe_overhead(), max_chunks=320)
    assert [c.index for c in fitted] == [0, 1, 2, 3, 4]


def test_no_split_needed_at_shipped_defaults(settings):
    chunks = [Chunk("a" * settings.chunk_chars, 1) for _ in range(3)]
    fitted = fit_chunks(chunks, metadata_ceiling_bytes=settings.metadata_max_bytes,
                        overhead_bytes=_probe_overhead(),
                        max_chunks=settings.max_chunks)
    assert len(fitted) == 3


def test_oversized_chunk_is_split_not_truncated():
    """The grounding invariant: evidence is divided, never shortened."""
    text = " ".join(f"word{i}" for i in range(400))
    original_words = text.split()
    overhead = _probe_overhead()
    fitted = fit_chunks([Chunk(text, 1)],
                        metadata_ceiling_bytes=overhead + 400,
                        overhead_bytes=overhead, max_chunks=320)
    assert len(fitted) > 1
    # Every original word must survive somewhere in the output.
    recovered = " ".join(c.text for c in fitted).split()
    assert set(original_words) == set(recovered)


def test_split_pieces_all_fit_the_budget():
    overhead = _probe_overhead()
    budget = 300
    fitted = fit_chunks([Chunk("word " * 500, 1)],
                        metadata_ceiling_bytes=overhead + budget,
                        overhead_bytes=overhead, max_chunks=320)
    assert all(len(c.text.encode("utf-8")) <= budget for c in fitted)


def test_split_preserves_page_attribution():
    overhead = _probe_overhead()
    fitted = fit_chunks([Chunk("word " * 500, 7)],
                        metadata_ceiling_bytes=overhead + 300,
                        overhead_bytes=overhead, max_chunks=320)
    assert all(c.page == 7 for c in fitted)


def test_split_never_breaks_a_codepoint():
    """Splitting on a raw byte boundary would corrupt multi-byte characters."""
    overhead = _probe_overhead()
    text = "\U0001f600" * 200  # 4 bytes each
    fitted = fit_chunks([Chunk(text, 1)],
                        metadata_ceiling_bytes=overhead + 100,
                        overhead_bytes=overhead, max_chunks=320)
    for chunk in fitted:
        chunk.text.encode("utf-8").decode("utf-8")  # must not raise
    assert "".join(c.text for c in fitted) == text


def test_chunk_budget_exceeded_raises_422_rather_than_dropping_content():
    with pytest.raises(ChunkBudgetExceededError):
        fit_chunks([Chunk("text", 1) for _ in range(50)],
                   metadata_ceiling_bytes=32_768,
                   overhead_bytes=_probe_overhead(), max_chunks=10)


def test_split_that_would_exceed_the_budget_raises():
    overhead = _probe_overhead()
    with pytest.raises(ChunkBudgetExceededError):
        fit_chunks([Chunk("word " * 2000, 1)],
                   metadata_ceiling_bytes=overhead + 100,
                   overhead_bytes=overhead, max_chunks=5)


def test_zero_text_budget_is_rejected():
    with pytest.raises(InvalidPdfError):
        fit_chunks([Chunk("x", 1)], metadata_ceiling_bytes=10,
                   overhead_bytes=100, max_chunks=10)
