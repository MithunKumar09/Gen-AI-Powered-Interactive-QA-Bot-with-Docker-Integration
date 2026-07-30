"""Lazy provider clients and provider-error translation.

Two concerns live here so they cannot drift apart.

**Lazy construction.** The pre-audit code built its model object at *import*
time, which also created a Pinecone index as a side effect. A missing key or a
network blip therefore stopped the app from starting at all, with an opaque
traceback and no working ``/health`` (finding B9). Clients here are built on
first use and cached, so importing ``rag_core`` never touches the network.

**One retry layer, not two.** Both SDKs retry internally -- ``ClientV2`` takes
``max_retries``, ``Pinecone`` takes a ``RetryConfig``. Adding our own wrapper on
top would silently multiply attempts (3 x 3 = 9 calls for one logical request).
So the SDK is configured as the single retry layer for idempotent operations,
and the only hand-written retry in the codebase is the deliberately narrower one
for chat generation in :mod:`rag_core.generation`.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from cohere import ClientV2
from cohere.core.api_error import ApiError as CohereApiError
from cohere.errors import (
    BadRequestError,
    ForbiddenError,
    GatewayTimeoutError,
    InternalServerError,
    ServiceUnavailableError,
    TooManyRequestsError,
    UnauthorizedError,
    UnprocessableEntityError,
)
from pinecone import Pinecone, RetryConfig

from rag_core.config import Settings
from rag_core.errors import (
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
)

log = logging.getLogger(__name__)

# Cohere errors that mean "the request itself was wrong". Retrying cannot help,
# and surfacing them as 502 would wrongly blame the upstream.
_COHERE_PERMANENT = (
    BadRequestError,
    UnprocessableEntityError,
    UnauthorizedError,
    ForbiddenError,
)
_COHERE_TRANSIENT = (
    InternalServerError,
    ServiceUnavailableError,
    GatewayTimeoutError,
)


class _ClientCache:
    """Process-local client cache.

    One instance per worker. Gunicorn workers do not share it, which is correct:
    each SDK client owns its own connection pool.
    """

    def __init__(self) -> None:
        self._cohere: ClientV2 | None = None
        self._cohere_no_retry: ClientV2 | None = None
        self._pinecone: Pinecone | None = None
        self._index: Any = None

    def cohere(self, settings: Settings) -> ClientV2:
        """Client for idempotent calls (embed, rerank). SDK retries enabled."""
        if self._cohere is None:
            self._cohere = ClientV2(
                api_key=settings.cohere_api_key,
                timeout=settings.provider_timeout_s,
                max_retries=settings.provider_max_retries,
            )
        return self._cohere

    def cohere_no_retry(self, settings: Settings) -> ClientV2:
        """Client for chat generation. SDK retries disabled on purpose.

        An ambiguous chat timeout may already have been billed and generated, so
        the retry decision has to be made by us, per-error, rather than blindly
        by the transport. See :mod:`rag_core.generation`.
        """
        if self._cohere_no_retry is None:
            self._cohere_no_retry = ClientV2(
                api_key=settings.cohere_api_key,
                timeout=settings.provider_timeout_s,
                max_retries=0,
            )
        return self._cohere_no_retry

    def pinecone(self, settings: Settings) -> Pinecone:
        if self._pinecone is None:
            self._pinecone = Pinecone(
                api_key=settings.pinecone_api_key,
                timeout=settings.provider_timeout_s,
                retry_config=RetryConfig(
                    max_retries=settings.provider_max_retries,
                    backoff_factor=0.25,
                    max_wait=10.0,
                ),
            )
        return self._pinecone

    def index(self, settings: Settings) -> Any:
        if self._index is None:
            self._index = self.pinecone(settings).Index(name=settings.pinecone_index)
        return self._index

    def reset(self) -> None:
        """Drop cached clients. Used by tests; not called in normal operation."""
        self._cohere = None
        self._cohere_no_retry = None
        self._pinecone = None
        self._index = None


CLIENTS = _ClientCache()


def translate_cohere_error(exc: Exception) -> ProviderError:
    """Map a Cohere SDK exception onto a typed domain error.

    The original message goes into ``detail`` (logged) and never into the
    client-facing ``message`` (finding A7).
    """
    detail = f"{type(exc).__name__}: {exc}"

    if isinstance(exc, TooManyRequestsError):
        return ProviderRateLimitedError(detail=detail)
    if isinstance(exc, (httpx.ReadTimeout, httpx.TimeoutException, GatewayTimeoutError)):
        return ProviderTimeoutError(detail=detail)
    if isinstance(exc, _COHERE_PERMANENT):
        # A 4xx from Cohere on our own well-formed request means our
        # configuration is wrong (bad key, retired model, bad dimension). Still
        # a 502 to the caller -- it is not the visitor's fault -- but logged at
        # a level that makes the operator look.
        log.error("cohere_permanent_error", extra={"reason": detail})
        return ProviderError(detail=detail)
    if isinstance(exc, _COHERE_TRANSIENT):
        return ProviderError(detail=detail)
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return ProviderError(detail=detail)
    if isinstance(exc, CohereApiError):
        status = getattr(exc, "status_code", None)
        if status == 429:
            return ProviderRateLimitedError(detail=detail)
        if status == 504:
            return ProviderTimeoutError(detail=detail)
        return ProviderError(detail=detail)
    return ProviderError(detail=detail)


def is_retryable_connection_error(exc: Exception) -> bool:
    """Whether a chat failure is safe to retry exactly once.

    True only when we can be confident the request never reached the model, or
    that it was explicitly rejected before generation:

    * connect errors  -- the connection was never established;
    * 429 / 5xx       -- the service declined to serve the request.

    Deliberately **False** for read timeouts: the request was accepted, the
    response was simply not received in time, so a retry risks paying twice.
    """
    if isinstance(exc, (httpx.ReadTimeout,)):
        return False
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return True
    if isinstance(exc, TooManyRequestsError):
        return True
    if isinstance(exc, _COHERE_TRANSIENT):
        return True
    if isinstance(exc, CohereApiError):
        status = getattr(exc, "status_code", None)
        return status in (429, 500, 502, 503)
    return False
