"""Cohere reranking.

Retrieve wide with the vector index, then narrow with a cross-encoder. This is
the cheapest available fix for the retrieval quality gap: the vector search is
recall-oriented and approximate, while rerank scores each candidate against the
actual question.

On scores: Cohere's documentation shows ``relevance_score`` values that look like
0-1, but it does not *guarantee* a normalised range. Nothing here assumes one --
the threshold is a calibrated setting, and the abstention decision lives in
:mod:`rag_core.pipeline`, not in this module.
"""

from __future__ import annotations

import logging

from cohere.core.request_options import RequestOptions

from rag_core.clients import CLIENTS, translate_cohere_error
from rag_core.config import Settings

log = logging.getLogger(__name__)

# rerank-v3.5 has a 4k-token context; the v4.0 models have 32k. Bounding
# per-document tokens keeps a long chunk from being silently dropped by the
# model's own truncation.
_MAX_TOKENS_PER_DOC = 1024


def rerank(
    settings: Settings, question: str, candidates: list[str]
) -> list[tuple[int, float]]:
    """Rerank ``candidates`` against ``question``.

    Returns ``(original_index, relevance_score)`` pairs, best first, truncated to
    ``RERANK_TOP_N``. Indices refer to positions in the input list so the caller
    keeps ownership of the candidate metadata.
    """
    if not candidates:
        return []

    client = CLIENTS.cohere(settings)
    try:
        response = client.rerank(
            model=settings.rerank_model,
            query=question,
            documents=candidates,
            top_n=min(settings.rerank_top_n, len(candidates)),
            max_tokens_per_doc=_MAX_TOKENS_PER_DOC,
            request_options=RequestOptions(
                timeout_in_seconds=int(settings.provider_timeout_s)
            ),
        )
    except Exception as exc:
        raise translate_cohere_error(exc) from exc

    return [(item.index, float(item.relevance_score)) for item in response.results]
