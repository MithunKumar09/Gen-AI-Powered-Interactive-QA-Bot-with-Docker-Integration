"""Pinecone access. All isolation is enforced here.

**One namespace, metadata isolation.** A namespace per browser session looks
tempting but Pinecone Starter allows only 100 namespaces per index, and nothing
reclaims an abandoned one -- 100 demo visitors would exhaust the index. So every
record lives in a single fixed namespace and is isolated by ``session_scope`` +
``document_id`` + ``ingest_schema_version`` metadata.

That makes forgetting a filter a data-leak bug rather than a mere mistake, so no
call site is allowed to build one: :func:`isolation_filter` is the only source,
and every read and scoped delete in this module routes through it.

**Operation choice is deliberate**, since Pinecone offers several ways to do
similar things:

===============================  ==========================================
Purpose                          Mechanism
===============================  ==========================================
Ingest verification              ``list(prefix=...)``, exact set equality
Rollback of a partial ingest     ``delete(ids=...)``  (exact, one request)
Orphan discovery                 ``list(prefix=...)``
Retention cleanup                ``delete(filter={"created_at": ...})``
User-initiated delete            ``delete(filter=isolation_filter)``
Reading values/metadata          ``fetch(ids=...)``  -- only when needed
===============================  ==========================================

Verification uses ``list`` rather than ``fetch``: it returns ids only (no vector
payload, so far less data), and comparing the observed *set* against the
expected set catches unexpected extras as well as missing records. It never uses
an unfiltered query, ``describe_index_stats``, or a sampled record -- none of
those can prove a specific document is completely stored.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Sequence

from rag_core.clients import CLIENTS
from rag_core.config import PINECONE_MAX_IDS_PER_REQUEST, Settings
from rag_core.errors import (
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ReadinessError,
)
from rag_core.ids import document_prefix

log = logging.getLogger(__name__)

# Pinecone accepts up to 1000 records or 2 MB per upsert. 100 keeps each request
# small enough that a transient failure loses little work.
UPSERT_BATCH = 100


def _translate(exc: Exception) -> ProviderError:
    """Map a Pinecone SDK exception onto a typed domain error.

    Matched on class name rather than by importing each type: pinecone exposes
    both ``*Error`` and legacy ``*Exception`` aliases and has reshuffled their
    module paths across majors, so name matching is the stable option here. The
    original message goes to ``detail`` (logged), never to the client.
    """
    name = type(exc).__name__
    detail = f"{name}: {exc}"

    if "Timeout" in name:
        return ProviderTimeoutError(detail=detail)
    if "RateLimit" in name:
        return ProviderRateLimitedError(detail=detail)
    if name in ("UnauthorizedError", "UnauthorizedException", "ForbiddenError",
                "ForbiddenException"):
        # Our key is wrong or lacks permission -- an operator problem, not the
        # visitor's. Still a 502 outward, but logged loudly.
        log.error("pinecone_auth_error", extra={"reason": detail})
        return ProviderError(detail=detail)
    return ProviderError(detail=detail)


def isolation_filter(
    *, scope: str, document_id: str | None, schema_versions: Iterable[int]
) -> dict[str, Any]:
    """The only place an isolation filter is constructed.

    ``document_id=None`` scopes to every document owned by ``scope`` -- used by
    ``/reset``, never by a read path.
    """
    flt: dict[str, Any] = {
        "session_scope": {"$eq": scope},
        "ingest_schema_version": {"$in": sorted(schema_versions)},
    }
    if document_id is not None:
        flt["document_id"] = {"$eq": document_id}
    return flt


# --- Writes ------------------------------------------------------------------


def upsert_chunks(
    settings: Settings,
    *,
    vector_ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadatas: Sequence[dict[str, Any]],
) -> None:
    """Upsert records in batches.

    ``show_progress=False`` is not cosmetic: the SDK defaults to ``True`` and
    writes a tqdm progress bar to stdout, which inside a Gunicorn worker
    interleaves with and corrupts the structured JSON log stream.
    """
    if not (len(vector_ids) == len(embeddings) == len(metadatas)):
        raise ProviderError(
            detail=(
                f"upsert arity mismatch: {len(vector_ids)} ids, "
                f"{len(embeddings)} vectors, {len(metadatas)} metadatas"
            )
        )

    index = CLIENTS.index(settings)
    records = [
        {"id": vid, "values": list(vec), "metadata": meta}
        for vid, vec, meta in zip(vector_ids, embeddings, metadatas)
    ]

    for start in range(0, len(records), UPSERT_BATCH):
        batch = records[start : start + UPSERT_BATCH]
        try:
            index.upsert(
                vectors=batch,
                namespace=settings.pinecone_namespace,
                show_progress=False,
            )
        except Exception as exc:
            raise _translate(exc) from exc


def delete_ids(settings: Settings, vector_ids: Sequence[str]) -> None:
    """Delete exact ids. Used for rollback of a partial ingest.

    Chunk counts are capped below Pinecone's 1000-id limit by config invariant 3,
    so this is normally a single request; the loop is a safety net rather than an
    expected path.
    """
    if not vector_ids:
        return
    index = CLIENTS.index(settings)
    for start in range(0, len(vector_ids), PINECONE_MAX_IDS_PER_REQUEST):
        batch = list(vector_ids[start : start + PINECONE_MAX_IDS_PER_REQUEST])
        try:
            index.delete(ids=batch, namespace=settings.pinecone_namespace)
        except Exception as exc:
            raise _translate(exc) from exc


def delete_document(settings: Settings, *, scope: str, document_id: str) -> None:
    """Delete one document owned by ``scope``.

    Pinecone reports no deleted-record count, and deletion is eventually
    consistent, so this returns nothing: the caller treats it as accepted, not
    confirmed.
    """
    index = CLIENTS.index(settings)
    try:
        index.delete(
            filter=isolation_filter(
                scope=scope,
                document_id=document_id,
                schema_versions=settings.supported_schema_versions,
            ),
            namespace=settings.pinecone_namespace,
        )
    except Exception as exc:
        raise _translate(exc) from exc


def delete_scope(settings: Settings, *, scope: str) -> None:
    """Delete every document owned by ``scope``. Backs ``/reset``."""
    index = CLIENTS.index(settings)
    try:
        index.delete(
            filter=isolation_filter(
                scope=scope,
                document_id=None,
                schema_versions=settings.supported_schema_versions,
            ),
            namespace=settings.pinecone_namespace,
        )
    except Exception as exc:
        raise _translate(exc) from exc


def delete_older_than(settings: Settings, cutoff_epoch: int) -> None:
    """Retention deletion by numeric ``created_at``. Used only by the cleanup script."""
    index = CLIENTS.index(settings)
    try:
        index.delete(
            filter={"created_at": {"$lt": cutoff_epoch}},
            namespace=settings.pinecone_namespace,
        )
    except Exception as exc:
        raise _translate(exc) from exc


# --- Reads -------------------------------------------------------------------


def list_document_ids(settings: Settings, document_id: str) -> set[str]:
    """Every vector id currently visible for ``document_id``.

    ``index.list`` returns an iterator that walks pagination internally. The
    generator is drained fully here -- a partial read would look like a failed
    ingest.
    """
    index = CLIENTS.index(settings)
    found: set[str] = set()
    try:
        for page in index.list(
            prefix=document_prefix(document_id),
            namespace=settings.pinecone_namespace,
        ):
            # Pages come back either as a list of ids or as an object exposing
            # `.ids`, depending on transport. Handle both rather than betting on
            # one shape.
            ids = getattr(page, "ids", page)
            if ids:
                found.update(str(i) for i in ids)
    except Exception as exc:
        raise _translate(exc) from exc
    return found


def await_visible(
    settings: Settings, document_id: str, expected_ids: Sequence[str]
) -> set[str]:
    """Poll until the visible id set equals ``expected_ids``, or time out.

    Pinecone writes are eventually consistent, so a read immediately after an
    upsert can legitimately come back short. Returning only on exact set
    equality means a missing record *and* an unexpected extra both fail.
    """
    expected = set(expected_ids)
    deadline = time.monotonic() + settings.ingest_verify_timeout_s
    interval = settings.ingest_verify_poll_ms / 1000.0
    observed: set[str] = set()

    while True:
        observed = list_document_ids(settings, document_id)
        if observed == expected:
            return observed
        if time.monotonic() >= deadline:
            raise ProviderTimeoutError(
                detail=(
                    f"ingest verification timed out: observed {len(observed)} of "
                    f"{len(expected)} expected ids"
                    + (
                        f", {len(observed - expected)} unexpected"
                        if observed - expected
                        else ""
                    )
                )
            )
        time.sleep(interval)


def query(
    settings: Settings,
    *,
    scope: str,
    document_id: str,
    vector: Sequence[float],
) -> list[dict[str, Any]]:
    """Similarity search within one document owned by ``scope``.

    Returns ``[{"id", "score", "metadata"}, ...]``. Values are not requested --
    we need the text, not the vectors.
    """
    index = CLIENTS.index(settings)
    try:
        response = index.query(
            top_k=settings.top_k,
            vector=list(vector),
            namespace=settings.pinecone_namespace,
            filter=isolation_filter(
                scope=scope,
                document_id=document_id,
                schema_versions=settings.supported_schema_versions,
            ),
            include_metadata=True,
            include_values=False,
        )
    except Exception as exc:
        raise _translate(exc) from exc

    matches = getattr(response, "matches", None)
    if matches is None and isinstance(response, dict):
        matches = response.get("matches", [])
    out: list[dict[str, Any]] = []
    for match in matches or []:
        out.append(
            {
                "id": str(_attr(match, "id", "")),
                "score": float(_attr(match, "score", 0.0) or 0.0),
                "metadata": dict(_attr(match, "metadata", {}) or {}),
            }
        )
    return out


def _attr(obj: Any, name: str, default: Any) -> Any:
    """Read ``name`` from an object or mapping."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# --- Readiness ---------------------------------------------------------------


def describe_index(settings: Settings) -> dict[str, Any]:
    """Index description for ``/ready``. Never creates or mutates anything."""
    pc = CLIENTS.pinecone(settings)
    try:
        if not pc.has_index(settings.pinecone_index):
            raise ReadinessError(
                f"Pinecone index {settings.pinecone_index!r} does not exist. "
                f"Create it with scripts/init_index.py."
            )
        model = pc.describe_index(settings.pinecone_index)
    except ReadinessError:
        raise
    except Exception as exc:
        raise _translate(exc) from exc

    spec = _attr(model, "spec", {}) or {}
    serverless = _attr(spec, "serverless", {}) or {}
    return {
        "name": _attr(model, "name", settings.pinecone_index),
        "dimension": _attr(model, "dimension", None),
        "metric": _attr(model, "metric", None),
        "cloud": _attr(serverless, "cloud", None),
        "region": _attr(serverless, "region", None),
    }


def check_compatible(settings: Settings) -> dict[str, Any]:
    """Assert the index matches this configuration.

    Raises :class:`ReadinessError` naming the exact mismatch. A dimension
    mismatch is the failure that would otherwise surface as an opaque Pinecone
    rejection on the first upload of the day.
    """
    described = describe_index(settings)

    dimension = described.get("dimension")
    if dimension is not None and int(dimension) != settings.embed_dimension:
        raise ReadinessError(
            f"Index {described['name']!r} has dimension {dimension} but "
            f"COHERE_EMBED_DIMENSION={settings.embed_dimension}. These must "
            f"match; vectors cannot be migrated between dimensions."
        )

    metric = (described.get("metric") or "").lower()
    if metric and metric != "cosine":
        raise ReadinessError(
            f"Index {described['name']!r} uses metric {metric!r}; this "
            f"application assumes cosine."
        )

    region = described.get("region")
    if region and region != settings.pinecone_region:
        raise ReadinessError(
            f"Index {described['name']!r} is in region {region!r} but "
            f"PINECONE_REGION={settings.pinecone_region!r}."
        )

    return described
