"""Flask application factory.

Note the WSGI entry point is ``myapp:create_app()`` -- **with parentheses**.
Gunicorn treats a bare ``myapp:create_app`` as the WSGI callable itself and then
invokes the factory with ``(environ, start_response)``, which raises TypeError on
every request. That was the committed configuration before this change (finding
B3), and is the most likely reason someone switched the container to the Flask
development server to get it working.

``DEBUG`` is forced off here regardless of environment. The previous code set
``app.config['DEBUG'] = True`` unconditionally (finding A4); combined with the
dev-server entrypoint that exposed the Werkzeug interactive debugger, which is
remote code execution for anyone who can reach it.
"""

from __future__ import annotations

import logging
import time

from flask import Flask, g, request
from werkzeug.middleware.proxy_fix import ProxyFix

from myapp.http_errors import register_error_handlers
from myapp.limits import limiter
from myapp.observability import configure_logging, new_request_id
from myapp.routes import routes
from rag_core.config import load_settings

log = logging.getLogger(__name__)


def create_app() -> Flask:
    settings = load_settings()  # raises ConfigError with an actionable message
    configure_logging(settings.log_level)

    app = Flask(__name__)

    # Never debug, never a reloader, regardless of FLASK_ENV or FLASK_DEBUG.
    app.config["DEBUG"] = False
    app.config["TESTING"] = False
    app.config["PROPAGATE_EXCEPTIONS"] = False
    # Werkzeug rejects a larger body before it is buffered into memory, which is
    # what keeps a big upload from OOMing a small free-tier container (A6).
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes
    app.config["JSON_SORT_KEYS"] = False

    app.config["SETTINGS"] = settings
    # Needed by the rate-limit key functions, which run outside a request-scoped
    # settings object.
    app.config["SESSION_SCOPE_KEY"] = settings.session_scope_key
    for name in (
        "RATE_LIMIT_ALL_REQUESTS_IP",
        "RATE_LIMIT_ASK_IP",
        "RATE_LIMIT_ASK_SCOPE",
        "RATE_LIMIT_ASK_APP_PROCESS",
        "RATE_LIMIT_UPLOAD_IP",
        "RATE_LIMIT_UPLOAD_SCOPE",
        "RATE_LIMIT_UPLOAD_APP_PROCESS",
    ):
        app.config[name] = getattr(settings, name.lower())

    # Trust exactly one proxy hop, and only when told to. Render puts a single
    # proxy in front of the service; locally under Compose there is none, so
    # trusting forwarded headers there would let any caller spoof their IP and
    # defeat the per-IP rate limit.
    if settings.trust_proxy:
        app.wsgi_app = ProxyFix(
            app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=0
        )

    limiter.init_app(app)
    # Verify the limiter actually armed. Flask-Limiter's init_app begins with
    # `if not self.enabled: return`, so a disabled limiter silently skips
    # registering its middleware and every limit stops applying -- with no error.
    # On a publicly reachable demo that is worth failing loudly for.
    if "limiter" not in app.extensions:
        raise RuntimeError(
            "Flask-Limiter did not initialise, so rate limiting is inactive. "
            "This happens when limiter.enabled is False at create_app() time."
        )

    register_error_handlers(app)
    app.register_blueprint(routes)

    @app.before_request
    def _begin():
        g.request_id = new_request_id()
        g.started = time.monotonic()

    @app.after_request
    def _finish(response):
        # One access line per request, in the same JSON shape as everything else.
        # /health is excluded: the platform polls it constantly and it would bury
        # every other line.
        if request.path != "/health":
            log.info(
                "request",
                extra={
                    "status": response.status_code,
                    "latency_ms": int(
                        (time.monotonic() - getattr(g, "started", time.monotonic()))
                        * 1000
                    ),
                },
            )
        response.headers["X-Request-Id"] = getattr(g, "request_id", "")
        # This is a JSON API consumed by our own frontend; no reason for a
        # browser to sniff, frame, or cache it.
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        return response

    log.info("app_started", extra=settings.public_summary())
    return app
