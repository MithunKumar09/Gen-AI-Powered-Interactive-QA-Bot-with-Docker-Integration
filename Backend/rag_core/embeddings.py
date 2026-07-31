"""Cohere embeddings.

Two things here are load-bearing.

**Asymmetric input_type.** Documents are embedded with
``input_type="search_document"`` and questions with ``"search_query"``. Cohere's
v3+ embedding models are trained for this asymmetry, and using it is the single
largest retrieval-quality gain in the migration away from the retired
``embed-english-v2.0``. Getting it backwards, or using one type for both,
quietly degrades every result.

**truncate="NONE".** The default would silently truncate over-length input,
which would break the grounding invariant: the stored and cited text would no
longer match what was actually embedded. Failing loudly instead means our own
chunk sizing is the only thing that decides chunk boundaries.
"""

from __future__ import annotations

import logging

from cohere.core.request_options import RequestOptions

from rag_core.clients import CLIENTS, translate_cohere_error
from rag_core.config import Settings

log = logging.getLogger(__name__)


def _embed(
    settings: Settings, texts: list[str], input_type: str
) -> list[list[float]]:
    client = CLIENTS.cohere(settings)
    out: list[list[float]] = []

    # Cohere accepts at most 96 texts per call.
    for start in range(0, len(texts), settings.max_embed_batch):
        batch = texts[start : start + settings.max_embed_batch]
        try:
            response = client.embed(
                model=settings.embed_model,
                input_type=input_type,
                texts=batch,
                output_dimension=settings.embed_dimension,
                embedding_types=["float"],
                truncate="NONE",
                request_options=RequestOptions(
                    timeout_in_seconds=int(settings.provider_timeout_s)
                ),
            )
        except Exception as exc:
            raise translate_cohere_error(exc) from exc

        # Note the trailing underscore: the v2 response groups embeddings by
        # type, so `.embeddings.float_` is the access path. The v1-era
        # `.embeddings[0]` does not exist on this response object.
        vectors = response.embeddings.float_
        if vectors is None:
            raise translate_cohere_error(
                RuntimeError("Cohere returned no float embeddings")
            )
        if len(vectors) != len(batch):
            raise translate_cohere_error(
                RuntimeError(
                    f"Cohere returned {len(vectors)} embeddings for "
                    f"{len(batch)} inputs"
                )
            )
        out.extend(list(v) for v in vectors)

    _assert_dimension(out, settings.embed_dimension)
    return out


def _assert_dimension(vectors: list[list[float]], expected: int) -> None:
    """Fail fast on a dimension mismatch.

    Pinecone would reject the upsert anyway, but its error does not say which
    side is wrong. Catching it here names the actual problem: the configured
    dimension and the index no longer agree.
    """
    for vector in vectors:
        if len(vector) != expected:
            raise translate_cohere_error(
                RuntimeError(
                    f"embedding dimension {len(vector)} does not match "
                    f"configured COHERE_EMBED_DIMENSION={expected}"
                )
            )


def embed_documents(settings: Settings, texts: list[str]) -> list[list[float]]:
    """Embed chunk texts for storage."""
    if not texts:
        return []
    return _embed(settings, texts, "search_document")


def embed_query(settings: Settings, question: str) -> list[float]:
    """Embed a single question for retrieval."""
    vectors = _embed(settings, [question], "search_query")
    return vectors[0]
