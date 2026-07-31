"""Grounded answer generation via the Cohere Chat API.

The pre-audit implementation concatenated retrieved text into one prompt string
and called ``co.generate(model="command-xlarge-nightly")``. That endpoint was
deprecated on 2025-09-15 and the model predates every currently listed Command
release.

The replacement passes retrieved chunks through ``documents=`` instead of
splicing them into the prompt. That is not cosmetic:

* the model is explicitly told what is evidence and what is instruction, so a
  document containing something that reads like an instruction cannot hijack the
  prompt;
* the response comes back with citations pointing at specific documents, so
  every claim can be traced to a page rather than trusted.

Retry policy here is deliberately narrower than for any other call. The client
used has SDK retries disabled, and a single retry is attempted only when we can
be confident generation never happened. A read timeout is never retried: the
request was accepted, only the response was lost, so retrying risks paying and
generating twice.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from cohere.core.request_options import RequestOptions

from rag_core.clients import (
    CLIENTS,
    is_retryable_connection_error,
    translate_cohere_error,
)
from rag_core.config import Settings

log = logging.getLogger(__name__)

_SYSTEM_PREAMBLE = (
    "You answer questions strictly from the supplied documents. "
    "If the documents do not contain the answer, say so plainly instead of "
    "guessing. Never use outside knowledge. Be concise and specific, and "
    "prefer quoting the document's own wording where it is clearer."
)

_MAX_ANSWER_TOKENS = 600

# Low but non-zero: near-deterministic for factual QA, without the degenerate
# repetition that temperature 0 can sometimes produce.
_TEMPERATURE = 0.2


@dataclass
class Evidence:
    """One retrieved chunk offered to the model as a Cohere document."""

    chunk_index: int
    page: int
    text: str
    score: float

    def as_document(self) -> dict[str, Any]:
        """Convert this evidence chunk to a Cohere Chat V2 document.

        Cohere Chat V2 requires all document fields except ``id`` to be nested
        inside ``data``. The stable outer id maps returned citations back to the
        corresponding evidence chunk.
        """
        return {
            "id": str(self.chunk_index),
            "data": {
                "text": self.text,
                "page": str(self.page),
            },
        }


@dataclass
class Answer:
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    abstained: bool = False


def generate(
    settings: Settings,
    *,
    question: str,
    evidence: list[Evidence],
) -> Answer:
    """Generate a grounded answer using only the supplied evidence."""

    documents = [item.as_document() for item in evidence]

    messages = [
        {
            "role": "system",
            "content": _SYSTEM_PREAMBLE,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    response = _chat_with_single_guarded_retry(
        settings,
        messages=messages,
        documents=documents,
    )

    text = _extract_text(response)
    citations = _extract_citations(response, evidence)

    return Answer(
        text=text,
        citations=citations,
        abstained=not bool(text),
    )


def _chat_with_single_guarded_retry(
    settings: Settings,
    *,
    messages: list[dict[str, str]],
    documents: list[dict[str, Any]],
) -> Any:
    """Call Cohere Chat with at most one safe connection-level retry."""

    client = CLIENTS.cohere_no_retry(settings)
    attempts = 0
    last_error: Exception | None = None

    while attempts < 2:
        attempts += 1

        try:
            return client.chat(
                model=settings.chat_model,
                messages=messages,
                documents=documents,
                max_tokens=_MAX_ANSWER_TOKENS,
                temperature=_TEMPERATURE,
                request_options=RequestOptions(
                    max_retries=0,
                    timeout_in_seconds=int(settings.provider_timeout_s),
                ),
            )

        except Exception as exc:
            last_error = exc

            if attempts >= 2 or not is_retryable_connection_error(exc):
                # Either the retry allowance is exhausted, or the failure is
                # ambiguous. Read timeouts must not be retried because Cohere
                # may already have accepted and billed the request.
                raise translate_cohere_error(exc) from exc

            delay = 0.5 + random.random() * 0.5

            log.warning(
                "chat_retrying",
                extra={
                    "reason": type(exc).__name__,
                    "delay_s": round(delay, 2),
                },
            )

            time.sleep(delay)

    # Defensive only: the loop either returns or raises.
    raise translate_cohere_error(
        last_error or RuntimeError("Cohere chat failed")
    )


def _extract_text(response: Any) -> str:
    """Extract assistant text from a Cohere V2 chat response.

    Supports both SDK model objects and mapping-shaped test doubles.
    """

    message = _get(response, "message")
    if message is None:
        return ""

    blocks = _get(message, "content", []) or []
    parts: list[str] = []

    for block in blocks:
        text = _get(block, "text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())

    return "\n".join(parts).strip()


def _extract_citations(
    response: Any,
    evidence: list[Evidence],
) -> list[dict[str, Any]]:
    """Map Cohere citations back to local evidence and page numbers.

    Cohere citations can expose their document source as SDK objects or
    mappings. Only short snippets are returned to the frontend.
    """

    message = _get(response, "message")
    raw_citations = _get(message, "citations", []) or []

    by_index = {
        item.chunk_index: item
        for item in evidence
    }

    seen: set[int] = set()
    output: list[dict[str, Any]] = []

    for citation in raw_citations:
        sources = _get(citation, "sources", []) or []

        for source in sources:
            identifier = _extract_source_document_id(source)

            try:
                chunk_index = int(str(identifier).split(":")[-1])
            except (TypeError, ValueError):
                continue

            if chunk_index in seen:
                continue

            item = by_index.get(chunk_index)
            if item is None:
                continue

            seen.add(chunk_index)

            citation_text = _get(citation, "text", "")
            snippet_source = (
                citation_text
                if isinstance(citation_text, str) and citation_text.strip()
                else item.text
            )

            output.append(
                {
                    "page": item.page,
                    "chunk_index": item.chunk_index,
                    "snippet": _snippet(snippet_source),
                }
            )

    # A model may generate grounded text without returning structured citations.
    # Preserve provenance by exposing the strongest retrieved evidence, clearly
    # marked as inferred rather than model-asserted.
    if not output and evidence:
        best = max(evidence, key=lambda item: item.score)

        output.append(
            {
                "page": best.page,
                "chunk_index": best.chunk_index,
                "snippet": _snippet(best.text),
                "inferred": True,
            }
        )

    return output


def _extract_source_document_id(source: Any) -> Any:
    """Extract a cited document id from Cohere source response variants."""

    document = _get(source, "document")

    if document is not None:
        identifier = _get(document, "id")
        if identifier is not None:
            return identifier

        # Some SDK/transport variants may expose the document value directly.
        if isinstance(document, (str, int)):
            return document

    return _get(source, "id")


def _get(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an SDK object or a mapping."""

    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(name, default)

    return getattr(value, name, default)


def _snippet(text: str, limit: int = 240) -> str:
    """Collapse whitespace and return a bounded citation snippet."""

    normalized = " ".join((text or "").split())

    if len(normalized) <= limit:
        return normalized

    return normalized[: limit - 1].rstrip() + "…"