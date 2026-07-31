"""PDF validation, text extraction, and page-attributed chunking.

Deliberately dependency-free beyond ``pypdf``: no tiktoken, no LangChain. A
character-based chunker with a fixed overlap is entirely adequate here, and it
keeps the ingest path free of a tokenizer whose vocabulary would have to stay in
sync with whichever embedding model is configured.

The chunker's contract, which the rest of the pipeline depends on:

* Every chunk carries the page it started on, so citations can name a page.
* Chunk text is finalised **before** embedding. Nothing downstream may alter it.
"""

from __future__ import annotations

import io
import logging
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag_core.errors import (
    ChunkBudgetExceededError,
    InvalidPdfError,
    NoExtractableTextError,
)

log = logging.getLogger(__name__)

# Every PDF begins with %PDF-. Checking this rejects a renamed .docx or an image
# before pypdf is handed the bytes, which turns a confusing parser traceback
# into a clean 422.
_PDF_MAGIC = b"%PDF-"

# Collapse runs of whitespace but preserve paragraph breaks: extracted PDF text
# is full of layout artefacts, and normalising them improves both embedding
# quality and how citations read.
_MULTI_SPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
# Hyphenation across a line break ("compre-\nhensive" -> "comprehensive").
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


class Page:
    """One extracted page."""

    __slots__ = ("number", "text")

    def __init__(self, number: int, text: str):
        self.number = number
        self.text = text


class Chunk:
    """A finalised chunk. ``text`` is immutable from here on."""

    __slots__ = ("text", "page", "index")

    def __init__(self, text: str, page: int, index: int = -1):
        self.text = text
        self.page = page
        self.index = index

    def with_index(self, index: int) -> "Chunk":
        return Chunk(self.text, self.page, index)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Chunk(index={self.index}, page={self.page}, chars={len(self.text)})"


def normalize_whitespace(text: str) -> str:
    """Tidy extracted text without changing its meaning."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def validate_and_read(data: bytes, *, max_pages: int) -> PdfReader:
    """Validate the upload and return a reader.

    Covers the cases the pre-audit implementation crashed on: empty files,
    non-PDF payloads, malformed structures, and encrypted documents.
    """
    if not data:
        raise InvalidPdfError("The uploaded file is empty.")
    if not data.startswith(_PDF_MAGIC):
        raise InvalidPdfError("The uploaded file is not a PDF.")

    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, ValueError, OSError) as exc:
        raise InvalidPdfError(detail=f"pypdf failed to open: {exc}") from exc

    if reader.is_encrypted:
        # pypdf can sometimes open with an empty password; try that once before
        # giving up, since plenty of PDFs are "encrypted" with no password at all.
        try:
            if reader.decrypt("") == 0:
                raise InvalidPdfError(
                    "This PDF is password protected and cannot be read."
                )
        except InvalidPdfError:
            raise
        except Exception as exc:  # pypdf raises assorted types here
            raise InvalidPdfError(
                "This PDF is password protected and cannot be read.",
                detail=str(exc),
            ) from exc

    try:
        page_count = len(reader.pages)
    except (PdfReadError, ValueError, OSError) as exc:
        raise InvalidPdfError(detail=f"unreadable page tree: {exc}") from exc

    if page_count == 0:
        raise InvalidPdfError("This PDF has no pages.")
    if page_count > max_pages:
        raise InvalidPdfError(
            f"This PDF has {page_count} pages; the demo accepts at most "
            f"{max_pages}."
        )
    return reader


def extract_pages(reader: PdfReader, *, max_extracted_chars: int) -> list[Page]:
    """Extract normalized per-page text.

    ``extract_text()`` returns ``None`` for some malformed pages and ``""`` for
    image-only ones. The pre-audit code did ``document_text += page.extract_text()``
    and raised ``TypeError`` on the former; both are handled as empty here, and a
    document that yields nothing at all is reported as 422 rather than embedded
    as an empty string.
    """
    pages: list[Page] = []
    total = 0
    truncated = False

    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text()
        except Exception as exc:  # pypdf raises broadly on damaged pages
            log.warning(
                "page_extract_failed", extra={"page": number, "reason": str(exc)}
            )
            continue
        if not raw:
            continue

        text = normalize_whitespace(raw)
        if not text:
            continue

        remaining = max_extracted_chars - total
        if remaining <= 0:
            truncated = True
            break
        if len(text) > remaining:
            # Cut on a whitespace boundary so we do not split mid-word. This
            # bounds total work; it is not the metadata ceiling, which is
            # handled by splitting rather than cutting.
            cut = text.rfind(" ", 0, remaining)
            text = text[: cut if cut > 0 else remaining]
            truncated = True

        total += len(text)
        pages.append(Page(number, text))
        if truncated:
            break

    if truncated:
        log.warning(
            "extraction_truncated",
            extra={"extracted_chars": total, "limit": max_extracted_chars},
        )
    if not pages:
        raise NoExtractableTextError()
    return pages


def _split_page(text: str, *, chunk_chars: int, overlap: int) -> list[str]:
    """Split one page into overlapping windows, preferring clean boundaries."""
    text = text.strip()
    # A blank page must yield no chunks at all. Returning [""] here would create
    # an empty chunk that then gets embedded as an empty string and stored as
    # meaningless evidence.
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    stride = chunk_chars - overlap
    out: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:
            # Prefer a paragraph break, then a sentence end, then any space.
            window = text[start:end]
            for pattern in ("\n\n", ". ", " "):
                cut = window.rfind(pattern)
                # Only accept a boundary in the last third, so we never produce
                # a pathologically short chunk chasing a nicer break.
                if cut > len(window) * 0.66:
                    end = start + cut + len(pattern)
                    break
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_pages(
    pages: list[Page], *, chunk_chars: int, overlap: int
) -> list[Chunk]:
    """Chunk each page independently so page attribution stays exact.

    Chunking across page boundaries would make citations ambiguous, which
    matters more here than squeezing out slightly larger chunks.
    """
    chunks: list[Chunk] = []
    for page in pages:
        for piece in _split_page(page.text, chunk_chars=chunk_chars, overlap=overlap):
            if piece.strip():
                chunks.append(Chunk(piece, page.number))
    if not chunks:
        raise NoExtractableTextError()
    return chunks


def fit_chunks(
    chunks: list[Chunk],
    *,
    metadata_ceiling_bytes: int,
    overhead_bytes: int,
    max_chunks: int,
) -> list[Chunk]:
    """Split any chunk whose metadata would exceed the byte ceiling, then index.

    **This is where the grounding invariant is enforced.** An over-ceiling chunk
    is split into smaller chunks, each stored and embedded in full. It is never
    trimmed: trimming would leave the embedding representing text that is absent
    from the stored metadata and from the generation context, so retrieval would
    match on evidence the model never sees and citations could point at text that
    was never embedded.

    Indexes are assigned only once every chunk fits, because a later split would
    renumber everything and invalidate any ids derived earlier.

    At the shipped defaults this splitting path is unreachable -- config
    invariant 7 guarantees ``CHUNK_CHARS * 4 + METADATA_OVERHEAD_BUDGET`` fits
    within the ceiling -- so reaching it is logged as an anomaly.
    """
    text_budget = metadata_ceiling_bytes - overhead_bytes
    if text_budget <= 0:
        raise InvalidPdfError(
            detail=(
                f"metadata overhead {overhead_bytes}B leaves no room under the "
                f"{metadata_ceiling_bytes}B ceiling"
            )
        )

    fitted: list[Chunk] = []
    queue = list(chunks)
    splits = 0

    while queue:
        chunk = queue.pop(0)
        encoded = len(chunk.text.encode("utf-8"))
        if encoded <= text_budget:
            fitted.append(chunk)
            continue

        splits += 1
        halves = _split_on_budget(chunk.text, text_budget)
        if len(halves) == 1:
            # A single codepoint cannot be divided further. Unreachable for any
            # realistic budget, but fail loudly rather than store a lie.
            raise InvalidPdfError(
                detail="indivisible chunk exceeds the metadata byte budget"
            )
        # Preserve order: the pieces of this chunk come before the rest.
        queue[:0] = [Chunk(h, chunk.page) for h in halves]

        if len(fitted) + len(queue) > max_chunks:
            raise ChunkBudgetExceededError()

    if splits:
        log.warning(
            "metadata_split_triggered",
            extra={"splits": splits, "text_budget_bytes": text_budget},
        )

    if len(fitted) > max_chunks:
        raise ChunkBudgetExceededError()

    return [chunk.with_index(i) for i, chunk in enumerate(fitted)]


def _split_on_budget(text: str, budget_bytes: int) -> list[str]:
    """Split ``text`` at the last safe boundary that fits in ``budget_bytes``."""
    encoded = text.encode("utf-8")
    if len(encoded) <= budget_bytes:
        return [text]

    # Walk back from the byte budget to a valid codepoint boundary.
    head_bytes = encoded[:budget_bytes]
    while head_bytes:
        try:
            head = head_bytes.decode("utf-8")
            break
        except UnicodeDecodeError:
            head_bytes = head_bytes[:-1]
    else:
        return [text]

    # Prefer a paragraph, then sentence, then word boundary inside that head.
    for pattern in ("\n\n", ". ", " "):
        cut = head.rfind(pattern)
        if cut > len(head) * 0.5:
            head = head[: cut + len(pattern)]
            break

    remainder = text[len(head) :]
    head, remainder = head.strip(), remainder.strip()
    if not head or not remainder:
        return [text.strip()]
    return [head, *_split_on_budget(remainder, budget_bytes)]
