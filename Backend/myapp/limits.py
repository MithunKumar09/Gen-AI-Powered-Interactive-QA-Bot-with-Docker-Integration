"""Rate limiting across three independent domains.

A single composite ``ip + session_id`` key is not a rate limit -- session ids are
free to mint, so one client can trivially create a fresh bucket per request. So
three separate limits apply to every protected route, and **all** of them are
enforced:

* **per trusted client IP** -- bounds total volume from one network origin, and
  is what actually stops session-id churn;
* **per derived session_scope** -- bounds one conversation, and is what stops a
  single session hopping IPs;
* **per application process** -- a coarse daily spend ceiling.

Those three are ``@limiter.limit`` decorators, and Flask-Limiter resolves
decorated limits **only when the view function is actually invoked** (see
``_manager.resolve_limits``, which skips them while ``in_middleware`` is true).
So a request rejected earlier -- bad API key, malformed session id -- never
reaches them and consumes no bucket. Left alone that would allow unlimited
``BACKEND_API_KEY`` guessing.

Hence a fourth limit: an **application limit**, which Flask-Limiter *does*
evaluate in middleware, before any ``before_request`` handler can reject the
request. It is a coarse per-IP ceiling on every request regardless of outcome --
generous enough to be invisible to a real user, tight enough that brute-forcing
a 32-byte key is hopeless.

The scope key derives the HMAC inside the key function rather than reading it off
``g``, so it does not depend on another hook having run first; a rate limit that
quietly stops applying is worse than none.

**The process-local ceiling is not a budget.** Storage is in-memory, so it is
per Gunicorn worker and resets on restart or redeploy: with 2 workers the real
ceiling is up to double the configured number. It is a best-effort guard against
runaway demo spend, not an enforceable account-wide cap. Exact limits would need
shared storage, which is out of scope. The same per-worker caveat applies to the
IP and scope limits.
"""

from __future__ import annotations

from flask import current_app, request
from flask_limiter import Limiter

from rag_core.identity import derive_scope, scope_ref

# Fallback used before app config exists (import time) and if config is missing.
_ALL_REQUESTS_FALLBACK = "120/minute;2000/hour"


def _remote_address() -> str:
    """Client address, honouring ProxyFix when it is installed.

    Reads ``request.remote_addr`` rather than a forwarded header directly: with
    TRUST_PROXY=0 no forwarded header is trusted at all, so a caller cannot spoof
    a fresh bucket per request.
    """
    return request.remote_addr or "unknown"


def _all_requests_limit() -> str:
    limits = current_app.config.get("RATE_LIMIT_ALL_REQUESTS_IP")
    return ";".join(limits) if limits else _ALL_REQUESTS_FALLBACK


def _health_is_exempt() -> bool:
    """The platform polls /health constantly; limiting it would flap deploys."""
    return request.path == "/health"


# Constructed without an app; bound in create_app via init_app. Storage is
# in-memory, so no network at import time.
limiter = Limiter(
    key_func=_remote_address,
    storage_uri="memory://",
    application_limits=[_all_requests_limit],
    application_limits_exempt_when=_health_is_exempt,
)


def ip_key() -> str:
    """Trusted client IP.

    With ProxyFix enabled (TRUST_PROXY=1 on Render) ``remote_addr`` is the real
    client; without it, the direct peer.
    """
    return f"ip:{_remote_address()}"


def scope_key() -> str:
    """Derived session scope, or the IP when no usable session id was sent.

    The IP fallback keeps a bucket attributable for requests that do reach the
    view without a usable session id. Requests rejected *before* the view are
    covered by the application limit instead, not by this.
    """
    raw = request.headers.get("X-Session-Id", "")
    key = current_app.config.get("SESSION_SCOPE_KEY")
    if not raw or not key or len(raw) > 128:
        return f"anon:{_remote_address()}"
    return f"scope:{scope_ref(derive_scope(key, raw))}"


def process_key() -> str:
    """Single shared bucket for this worker process."""
    return "process"


# Limit strings are resolved per request from app config, so a Settings value can
# reach a decorator that was evaluated at import time.
def _cfg(name: str):
    def resolve() -> str:
        return ";".join(current_app.config[name])

    return resolve


def ask_limits(view):
    """Apply the three /ask limits."""
    view = limiter.limit(_cfg("RATE_LIMIT_ASK_APP_PROCESS"), key_func=process_key)(view)
    view = limiter.limit(_cfg("RATE_LIMIT_ASK_SCOPE"), key_func=scope_key)(view)
    view = limiter.limit(_cfg("RATE_LIMIT_ASK_IP"), key_func=ip_key)(view)
    return view


def upload_limits(view):
    """Apply the three /upload limits."""
    view = limiter.limit(
        _cfg("RATE_LIMIT_UPLOAD_APP_PROCESS"), key_func=process_key
    )(view)
    view = limiter.limit(_cfg("RATE_LIMIT_UPLOAD_SCOPE"), key_func=scope_key)(view)
    view = limiter.limit(_cfg("RATE_LIMIT_UPLOAD_IP"), key_func=ip_key)(view)
    return view


def mutation_limits(view):
    """Looser limits for delete/reset: cheap operations, no provider spend."""
    view = limiter.limit("10/minute", key_func=scope_key)(view)
    view = limiter.limit("30/minute", key_func=ip_key)(view)
    return view
