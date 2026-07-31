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
* the response comes back with **citations** pointing at specific documents, so
  every claim can be traced to a page rather than trusted.

Retry policy here is deliberately narrower than for any other call. The client
used has SDK retries **disabled**, and a single retry is attempted only when we
can be confident generation never happened. A read timeout is never retried: the
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
# Low but non-zero: near-deterministic for a factual QA demo, without the
# degenerate repetition that temperature 0 sometimes produces.
_TEMPERATURE = 0.2


@dataclass
class Evidence:
    """One retrieved chunk offered to the model as a document."""

    chunk_index: int
    page: int
    text: str
    score: float

    def as_document(self) -> dict[str, str]:
        """Cohere document payload.

        ``page`` is included as data so the model can reference it, and the id is
        the chunk index so returned citations map straight back to our evidence
        list.
        """
        return {
            "id": str(self.chunk_index),
            "text": self.text,
            "page": str(self.page),
        }


@dataclass
class Answer:
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    abstained: bool = False


def generate(
    settings: Settings, *, question: str, evidence: list[Evidence]
) -> Answer:
    """Generate a grounded answer with citations."""
    documents = [item.as_document() for item in evidence]
    messages = [
        {"role": "system", "content": _SYSTEM_PREAMBLE},
        {"role": "user", "content": question},
    ]

    response = _chat_with_single_guarded_retry(
        settings, messages=messages, documents=documents
    )

    text = _extract_text(response)
    citations = _extract_citations(response, evidence)
    return Answer(text=text, citations=citations, abstained=False)


def _chat_with_single_guarded_retry(
    settings: Settings, *, messages: list[dict[str, str]], documents: list[dict[str, str]]
) -> Any:
    client = CLIENTS.cohere_no_retry(settings)
    attempts = 0
    last: Exception | None = None

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
            last = exc
            if attempts >= 2 or not is_retryable_connection_error(exc):
                # Either out of attempts, or the failure is ambiguous (e.g. a
                # read timeout) and a retry could double-bill.
                raise translate_cohere_error(exc) from exc
            # Short jittered backoff before the single permitted retry.
            delay = 0.5 + random.random() * 0.5
            log.warning(
                "chat_retrying",
                extra={"reason": f"{type(exc).__name__}", "delay_s": round(delay, 2)},
            )
            time.sleep(delay)

    raise translate_cohere_error(last or RuntimeError("chat failed"))


def _extract_text(response: Any) -> str:
    """Pull the assistant text out of a V2ChatResponse.

    The v2 response nests content as a list of typed blocks, so this walks them
    rather than assuming a single text block exists.
    """
    message = getattr(response, "message", None)
    if message is None:
        return ""
    blocks = getattr(message, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _extract_citations(response: Any, evidence: list[Evidence]) -> list[dict[str, Any]]:
    """Map Cohere citations onto our evidence, resolving page numbers.

    Cohere returns character spans plus the document ids they came from. We
    surface the page and a short snippet -- enough for a reader to verify the
    claim, without echoing whole chunks back to the client.
    """
    message = getattr(response, "message", None)
    raw = getattr(message, "citations", None) or []
    by_index = {item.chunk_index: item for item in evidence}
    seen: set[int] = set()
    out: list[dict[str, Any]] = []

    for citation in raw:
        for source in getattr(citation, "sources", None) or []:
            doc_id = getattr(source, "document", None)
            # Sources expose either a document mapping or a plain id, depending
            # on the citation kind.
            if isinstance(doc_id, dict):
                ident = doc_id.get("id")
            else:
                ident = getattr(source, "id", None) or doc_id
            try:
                chunk_index = int(str(ident).split(":")[-1])
            except (TypeError, ValueError):
                continue
            if chunk_index in seen:
                continue
            item = by_index.get(chunk_index)
            if item is None:
                continue
            seen.add(chunk_index)
            out.append(
                {
                    "page": item.page,
                    "chunk_index": item.chunk_index,
                    "snippet": _snippet(getattr(citation, "text", "") or item.text),
                }
            )

    # Fall back to the strongest evidence if the model returned no citations, so
    # the UI can always show provenance. Marked so it is not mistaken for a
    # model-asserted citation.
    if not out and evidence:
        best = max(evidence, key=lambda e: e.score)
        out.append(
            {
                "page": best.page,
                "chunk_index": best.chunk_index,
                "snippet": _snippet(best.text),
                "inferred": True,
            }
        )
    return out


def _snippet(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
