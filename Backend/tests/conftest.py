"""Test fixtures and provider fakes.

The fakes stand in at the **SDK client** level rather than at the level of our own
functions. That is a deliberate choice: it means every test exercises the real
``rag_core.store``, ``rag_core.embeddings`` and ``rag_core.generation`` code,
including the response-shape handling that is easy to get wrong (``.embeddings.
float_``, ``list()`` pagination, keyword-only signatures). Patching our own
functions would test the fakes instead.

No test makes a network call.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_ENV = {
    "COHERE_API_KEY": "test-cohere-key",
    "PINECONE_API_KEY": "test-pinecone-key",
    "BACKEND_API_KEY": "test-backend-key",
    "SESSION_SCOPE_KEY": "test-scope-key",
    "COHERE_CHAT_MODEL": "command-a-03-2025",
    "LOG_LEVEL": "CRITICAL",
    "INGEST_VERIFY_TIMEOUT_S": "2",
    "INGEST_VERIFY_POLL_MS": "10",
}

SESSION_A = "a" * 32
SESSION_B = "b" * 32


# --- Cohere fake -------------------------------------------------------------


class _Embeddings:
    """Mirrors EmbedByTypeResponseEmbeddings: embeddings grouped by type."""

    def __init__(self, vectors: list[list[float]]):
        self.float_ = vectors
        self.int8 = None
        self.uint8 = None
        self.binary = None
        self.ubinary = None
        self.base64 = None


class _EmbedResponse:
    def __init__(self, vectors: list[list[float]]):
        self.embeddings = _Embeddings(vectors)


class _TextBlock:
    def __init__(self, text: str):
        self.text = text


class _Source:
    def __init__(self, doc_id: str):
        self.id = doc_id
        self.document = {"id": doc_id}


class _Citation:
    def __init__(self, text: str, doc_ids: list[str]):
        self.text = text
        self.sources = [_Source(d) for d in doc_ids]


class _Message:
    def __init__(self, text: str, citations: list[_Citation]):
        self.content = [_TextBlock(text)]
        self.citations = citations
        self.role = "assistant"


class _ChatResponse:
    def __init__(self, text: str, citations: list[_Citation]):
        self.message = _Message(text, citations)
        self.finish_reason = "COMPLETE"


class _RerankItem:
    def __init__(self, index: int, score: float):
        self.index = index
        self.relevance_score = score


class _RerankResponse:
    def __init__(self, results: list[_RerankItem]):
        self.results = results


class FakeCohere:
    """Deterministic stand-in for cohere.ClientV2."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.embed_calls: list[dict[str, Any]] = []
        self.chat_calls: list[dict[str, Any]] = []
        self.rerank_calls: list[dict[str, Any]] = []
        # Queues of exceptions to raise, one per call, for fault injection.
        self.embed_errors: list[Exception] = []
        self.chat_errors: list[Exception] = []
        self.rerank_errors: list[Exception] = []
        self.answer = "A grounded answer."
        self.cite_indices: list[int] | None = None
        self.return_wrong_dimension = False
        self.return_short_batch = False

    # Embeddings are a deterministic hash-derived unit-ish vector, so "similar"
    # text produces similar vectors without needing a real model.
    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode()).digest()
        dim = 3 if self.return_wrong_dimension else self.dimension
        raw = [digest[i % len(digest)] / 255.0 for i in range(dim)]
        # Bias the first components by word content so lexical overlap shows up
        # as vector similarity.
        for word in set(text.lower().split()):
            slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
            raw[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed(self, *, model, input_type, texts, output_dimension,
              embedding_types, truncate, request_options=None, **_):
        if self.embed_errors:
            raise self.embed_errors.pop(0)
        self.embed_calls.append(
            {
                "model": model,
                "input_type": input_type,
                "texts": list(texts),
                "count": len(texts),
                "output_dimension": output_dimension,
                "truncate": truncate,
            }
        )
        vectors = [self._vector(t) for t in texts]
        if self.return_short_batch:
            vectors = vectors[:-1] or []
        return _EmbedResponse(vectors)

    def chat(
        self,
        *,
        model,
        messages,
        documents,
        max_tokens=None,
        temperature=None,
        request_options=None,
        **_,
    ):
        if self.chat_errors:
            raise self.chat_errors.pop(0)

        # Mirror Cohere Chat V2's required document structure. This ensures
        # tests reject payloads that the real Cohere API would reject.
        for index, document in enumerate(documents):
            assert isinstance(document, dict), (
                f"documents[{index}] must be a mapping"
            )

            document_id = document.get("id")
            assert isinstance(document_id, str) and document_id.strip(), (
                f"documents[{index}].id must be a non-empty string"
            )

            data = document.get("data")
            assert isinstance(data, dict) and data, (
                f"documents[{index}].data is required"
            )

            text = data.get("text")
            assert isinstance(text, str) and text.strip(), (
                f"documents[{index}].data.text is required"
            )

            page = data.get("page")
            assert isinstance(page, str) and page.strip(), (
                f"documents[{index}].data.page is required"
            )

        self.chat_calls.append(
            {
                "model": model,
                "messages": messages,
                "documents": documents,
            }
        )

        ids = (
            [str(i) for i in self.cite_indices]
            if self.cite_indices is not None
            else [str(document["id"]) for document in documents[:1]]
        )

        citations = [_Citation("cited span", ids)] if ids else []
        return _ChatResponse(self.answer, citations)

    def rerank(self, *, model, query, documents, top_n, max_tokens_per_doc=None,
               request_options=None, **_):
        if self.rerank_errors:
            raise self.rerank_errors.pop(0)
        self.rerank_calls.append(
            {"model": model, "query": query, "count": len(documents)}
        )
        # Score by word overlap so relevance is meaningful in tests.
        q = set(query.lower().split())
        scored = []
        for i, doc in enumerate(documents):
            words = set(doc.lower().split())
            overlap = len(q & words) / (len(q) or 1)
            scored.append(_RerankItem(i, min(0.99, overlap)))
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        return _RerankResponse(scored[:top_n])


# --- Pinecone fake -----------------------------------------------------------


def _matches_filter(metadata: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    """Evaluate the subset of Pinecone filter operators this app uses."""
    if not flt:
        return True
    for field, condition in flt.items():
        value = metadata.get(field)
        if not isinstance(condition, dict):
            if value != condition:
                return False
            continue
        for op, operand in condition.items():
            if op == "$eq" and value != operand:
                return False
            if op == "$in" and value not in operand:
                return False
            if op == "$lt" and not (value is not None and value < operand):
                return False
            if op == "$gt" and not (value is not None and value > operand):
                return False
    return True


class _VectorId:
    """Minimal stand-in for a Pinecone listed-vector identifier."""

    def __init__(self, vector_id: str):
        self.id = vector_id


class _ListPage:
    """Minimal stand-in for a current Pinecone list response page."""

    def __init__(self, ids: list[str]):
        self.vectors = [_VectorId(vector_id) for vector_id in ids]


class FakeIndex:
    """In-memory Pinecone index with realistic semantics.

    Models two behaviours that matter and are easy to forget:

    * **eventual consistency** -- ``visibility_delay`` withholds newly upserted
      ids from ``list``/``query`` for N reads, which is what the ingest
      verification poll exists to survive;
    * **pagination** -- ``list`` yields ids in pages rather than one batch.
    """

    def __init__(self, dimension: int = 1024, page_size: int = 100):
        self.records: dict[str, dict[str, Any]] = {}
        self.dimension = dimension
        self.page_size = page_size
        self.calls: list[str] = []
        self.visibility_delay = 0
        self._pending: dict[str, int] = {}
        self.upsert_errors: list[Exception] = []
        self.delete_errors: list[Exception] = []
        self.query_errors: list[Exception] = []
        self.never_visible: set[str] = set()
        self.upsert_batch_sizes: list[int] = []
        self.show_progress_values: list[Any] = []

    def _tick(self) -> None:
        for key in list(self._pending):
            self._pending[key] -= 1
            if self._pending[key] <= 0:
                del self._pending[key]

    def _visible(self) -> dict[str, dict[str, Any]]:
        return {
            k: v
            for k, v in self.records.items()
            if k not in self._pending and k not in self.never_visible
        }

    def upsert(self, *, vectors, namespace="", show_progress=True, **_):
        if self.upsert_errors:
            raise self.upsert_errors.pop(0)
        self.calls.append("upsert")
        self.upsert_batch_sizes.append(len(vectors))
        self.show_progress_values.append(show_progress)
        for record in vectors:
            vid = record["id"]
            if len(record["values"]) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(record['values'])} does not match "
                    f"the index dimension {self.dimension}"
                )
            self.records[vid] = {
                "values": list(record["values"]),
                "metadata": dict(record.get("metadata") or {}),
                "namespace": namespace,
            }
            if self.visibility_delay:
                self._pending[vid] = self.visibility_delay
        return {"upserted_count": len(vectors)}

    def query(self, *, top_k, vector=None, namespace="", filter=None,
              include_metadata=False, include_values=False, **_):
        if self.query_errors:
            raise self.query_errors.pop(0)
        self.calls.append("query")
        self._tick()
        scored = []
        for vid, record in self._visible().items():
            if record["namespace"] != namespace:
                continue
            if not _matches_filter(record["metadata"], filter):
                continue
            scored.append((vid, _cosine(vector, record["values"]), record))
        scored.sort(key=lambda row: row[1], reverse=True)

        class _M:
            def __init__(self, vid, score, record):
                self.id = vid
                self.score = score
                self.metadata = record["metadata"] if include_metadata else {}

        class _R:
            def __init__(self, matches):
                self.matches = matches

        return _R([_M(*row) for row in scored[:top_k]])

    def fetch(self, *, ids, namespace="", **_):
        self.calls.append("fetch")
        self._tick()
        visible = self._visible()

        class _F:
            def __init__(self, vectors):
                self.vectors = vectors

        return _F({i: visible[i] for i in ids if i in visible})

    def list(self, *, prefix=None, limit=None, namespace="", **_):
        self.calls.append("list")
        self._tick()
        ids = sorted(
            vid
            for vid, record in self._visible().items()
            if record["namespace"] == namespace
            and (prefix is None or vid.startswith(prefix))
        )
        for start in range(0, len(ids), self.page_size):
            yield _ListPage(ids[start : start + self.page_size])

    def delete(self, *, ids=None, delete_all=False, filter=None, namespace="", **_):
        if self.delete_errors:
            raise self.delete_errors.pop(0)
        self.calls.append("delete")
        if delete_all:
            targets = [
                k for k, v in self.records.items() if v["namespace"] == namespace
            ]
        elif ids is not None:
            targets = [i for i in ids if i in self.records]
        elif filter is not None:
            targets = [
                k
                for k, v in self.records.items()
                if v["namespace"] == namespace
                and _matches_filter(v["metadata"], filter)
            ]
        else:
            targets = []
        for key in targets:
            self.records.pop(key, None)
            self._pending.pop(key, None)
        return None

    def describe_index_stats(self, *, filter=None, **_):
        return {"total_vector_count": len(self._visible())}


class _IndexModel:
    def __init__(self, name, dimension, metric, cloud, region):
        self.name = name
        self.dimension = dimension
        self.metric = metric
        self.spec = type("S", (), {"serverless": type(
            "SL", (), {"cloud": cloud, "region": region})()})()


class FakePinecone:
    def __init__(self, index: FakeIndex, *, dimension=1024, metric="cosine",
                 cloud="aws", region="us-east-1", exists=True):
        self._index = index
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.exists = exists

    def has_index(self, name):
        return self.exists

    def describe_index(self, name):
        return _IndexModel(name, self.dimension, self.metric, self.cloud, self.region)

    def Index(self, name="", host="", **_):
        return self._index


def _cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# --- PDF builder -------------------------------------------------------------


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal multi-page PDF with real extractable text.

    Hand-rolled rather than pulled from a fixture file so tests can place known
    content on a known page -- which is what the later-page-evidence test needs.
    """
    from pypdf import PdfWriter

    try:
        from reportlab.pdfgen import canvas  # noqa: F401

        has_reportlab = True
    except ImportError:
        has_reportlab = False

    if has_reportlab:  # pragma: no cover - only if reportlab is available
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        buf = io.BytesIO()
        pdf = canvas.Canvas(buf, pagesize=letter)
        for text in pages:
            y = 750
            for line in _wrap(text, 90):
                pdf.drawString(60, y, line)
                y -= 14
                if y < 60:
                    break
            pdf.showPage()
        pdf.save()
        return buf.getvalue()

    # Fallback: emit raw PDF content streams. Enough for pypdf to extract text.
    return _raw_pdf(pages)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _raw_pdf(pages: list[str]) -> bytes:
    def esc(s: str) -> str:
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    objects: list[bytes] = []
    page_ids = []
    # 1 = catalog, 2 = pages tree, then per page: page obj + content obj.
    next_id = 3
    contents = []
    for text in pages:
        lines = _wrap(text, 90)
        body = "BT /F1 11 Tf 60 750 Td 14 TL\n"
        for line in lines[:48]:
            body += f"({esc(line)}) Tj T*\n"
        body += "ET"
        contents.append(body)

    for i, body in enumerate(contents):
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_ids.append(page_id)
        objects.append(
            f"{page_id} 0 obj\n<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
            f"/BaseFont /Helvetica >> >> >> >>\nendobj\n".encode()
        )
        stream = body.encode("latin-1", "replace")
        objects.append(
            f"{content_id} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream\nendobj\n"
        )

    kids = " ".join(f"{p} 0 R" for p in page_ids)
    head = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>\n"
        f"endobj\n".encode(),
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj in head + objects:
        offsets.append(len(out))
        out += obj
    xref_at = len(out)
    total = len(offsets) + 1
    out += f"xref\n0 {total}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref_at}\n"
        f"%%EOF\n".encode()
    )
    return bytes(out)


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(
            ("COHERE_", "PINECONE_", "BACKEND_", "SESSION_", "RATE_LIMIT_",
             "MAX_", "CHUNK_", "TOP_K", "RERANK_", "ENABLE_", "VECTOR_",
             "MIN_", "METADATA_", "INGEST_", "RETENTION_", "TRUST_", "LOG_")
        ):
            monkeypatch.delenv(key, raising=False)
    for key, value in BASE_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def env(monkeypatch):
    def _set(**overrides):
        for key, value in overrides.items():
            monkeypatch.setenv(key, str(value))
    return _set


@pytest.fixture
def settings():
    from rag_core.config import load_settings
    return load_settings()


@pytest.fixture
def fake_index():
    return FakeIndex()


@pytest.fixture
def fake_cohere():
    return FakeCohere()


@pytest.fixture
def providers(monkeypatch, fake_index, fake_cohere):
    """Install the fakes into the client cache.

    Patching the cache -- rather than our own functions -- keeps the real
    store/embeddings/generation code in the path under test.
    """
    from rag_core import clients

    pc = FakePinecone(fake_index)
    clients.CLIENTS.reset()
    monkeypatch.setattr(clients.CLIENTS, "cohere", lambda s: fake_cohere)
    monkeypatch.setattr(clients.CLIENTS, "cohere_no_retry", lambda s: fake_cohere)
    monkeypatch.setattr(clients.CLIENTS, "pinecone", lambda s: pc)
    monkeypatch.setattr(clients.CLIENTS, "index", lambda s: fake_index)
    yield type("P", (), {"index": fake_index, "cohere": fake_cohere, "pinecone": pc})
    clients.CLIENTS.reset()


@pytest.fixture
def app(providers):
    """App with rate limiting installed but not enforcing.

    Ordering here is load-bearing. ``Limiter.init_app`` starts with
    ``if not self.enabled: return`` -- so if the limiter is disabled when
    ``create_app()`` runs, its ``before_request`` hook is never registered and
    limits stay silently off for that app *even if re-enabled afterwards*. Since
    the module-level limiter is shared across every test in the process, a
    fixture that left it disabled would quietly disarm the rate-limit tests.

    So: enable before the factory runs, disable after. Limits are exercised for
    real in test_limits.py.
    """
    from myapp import create_app
    from myapp.limits import limiter

    limiter.enabled = True
    application = create_app()
    application.config["TESTING"] = True
    limiter.reset()
    limiter.enabled = False
    yield application
    limiter.enabled = True


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_headers():
    return {
        "X-API-Key": BASE_ENV["BACKEND_API_KEY"],
        "X-Session-Id": SESSION_A,
    }


@pytest.fixture
def headers_b():
    return {
        "X-API-Key": BASE_ENV["BACKEND_API_KEY"],
        "X-Session-Id": SESSION_B,
    }


@pytest.fixture
def scope_a(settings):
    from rag_core.identity import derive_scope
    return derive_scope(settings.session_scope_key, SESSION_A)


@pytest.fixture
def scope_b(settings):
    from rag_core.identity import derive_scope
    return derive_scope(settings.session_scope_key, SESSION_B)


@pytest.fixture
def captured_logs(monkeypatch):
    """Capture every emitted log record for privacy assertions."""
    import logging

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    yield records
    root.removeHandler(handler)


def serialize_records(records) -> str:
    """Render captured records the way the JSON formatter would."""
    from myapp.observability import JsonFormatter

    formatter = JsonFormatter()
    return "\n".join(formatter.format(r) for r in records)


def upsert_payload_text(index: FakeIndex) -> str:
    """All metadata ever written, as JSON, for leak assertions."""
    return json.dumps(
        [r["metadata"] for r in index.records.values()], default=str
    )
