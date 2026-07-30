"""Rate limiting across three independent domains.

The point of these tests is the bypass cases. A single composite
``ip + session_id`` key looks like a rate limit but is not one, because session
ids are free to mint. So the two directions are checked explicitly:

* rotating session ids from one IP must still hit the **IP** ceiling;
* one session moving across IPs must still hit the **scope** ceiling.
"""

from __future__ import annotations

import pytest
from conftest import BASE_ENV, FakeIndex, FakePinecone


@pytest.fixture
def limited_app(monkeypatch, providers, env):
    """App with limits enabled and small ceilings, for fast exhaustion."""
    env(
        RATE_LIMIT_ALL_REQUESTS_IP="10/minute",
        RATE_LIMIT_ASK_IP="3/minute",
        RATE_LIMIT_ASK_SCOPE="2/minute",
        RATE_LIMIT_ASK_APP_PROCESS="5/minute",
        RATE_LIMIT_UPLOAD_IP="2/minute",
        RATE_LIMIT_UPLOAD_SCOPE="1/minute",
        RATE_LIMIT_UPLOAD_APP_PROCESS="4/minute",
    )
    from myapp import create_app
    from myapp.limits import limiter

    # Must be enabled BEFORE create_app: Limiter.init_app returns early when
    # disabled, so the middleware hook would never be registered. See the
    # `app` fixture in conftest for the full explanation.
    limiter.enabled = True
    app = create_app()
    app.config["TESTING"] = True
    limiter.reset()
    yield app
    limiter.reset()


def _ask(client, *, session, ip="10.0.0.1"):
    return client.post(
        "/ask",
        headers={"X-API-Key": BASE_ENV["BACKEND_API_KEY"], "X-Session-Id": session},
        json={"question": "anything at all", "document_id": "v1:" + "0" * 32 + ":" + "1" * 32},
        environ_base={"REMOTE_ADDR": ip},
    )


def _statuses(responses):
    return [r.status_code for r in responses]


def test_scope_limit_triggers(limited_app):
    client = limited_app.test_client()
    session = "s" * 32
    results = [_ask(client, session=session) for _ in range(4)]
    assert 429 in _statuses(results)


def test_rotating_session_ids_still_hits_the_ip_ceiling(limited_app):
    """The bypass a composite key would allow: a fresh session per request."""
    client = limited_app.test_client()
    results = [
        _ask(client, session=f"{i:032d}", ip="10.0.0.9") for i in range(6)
    ]
    # Each request has a unique scope, so only the IP limit (3/minute) can stop it.
    assert 429 in _statuses(results), "minting session ids bypassed the IP limit"


def test_rotating_ips_still_hits_the_scope_ceiling(limited_app):
    """The mirror case: one session hopping IPs must still be bounded."""
    client = limited_app.test_client()
    session = "z" * 32
    results = [
        _ask(client, session=session, ip=f"10.0.1.{i}") for i in range(5)
    ]
    assert 429 in _statuses(results), "changing IP bypassed the scope limit"


def test_process_ceiling_triggers_across_distinct_ips_and_sessions(limited_app):
    """Both per-key limits are evaded; only the process ceiling can stop this."""
    client = limited_app.test_client()
    results = [
        _ask(client, session=f"{i:032d}", ip=f"10.0.2.{i}") for i in range(8)
    ]
    assert 429 in _statuses(results)


def test_rate_limited_response_uses_the_error_envelope(limited_app):
    client = limited_app.test_client()
    session = "q" * 32
    last = None
    for _ in range(6):
        last = _ask(client, session=session)
    assert last.status_code == 429
    body = last.get_json()
    assert body["error"]["code"] == "RATE_LIMITED"
    assert set(body["error"]) == {"code", "message", "request_id"}


def test_health_is_never_rate_limited(limited_app):
    """The platform polls /health constantly; limiting it would flap the deploy."""
    client = limited_app.test_client()
    assert all(client.get("/health").status_code == 200 for _ in range(30))


def test_forwarded_headers_ignored_when_proxy_is_untrusted(limited_app):
    """TRUST_PROXY=0 (the Compose default): X-Forwarded-For must not set identity.

    If it did, any caller could spoof a fresh IP bucket per request and the
    per-IP ceiling would be meaningless.
    """
    client = limited_app.test_client()
    results = []
    for i in range(6):
        results.append(
            client.post(
                "/ask",
                headers={
                    "X-API-Key": BASE_ENV["BACKEND_API_KEY"],
                    "X-Session-Id": f"{i:032d}",
                    "X-Forwarded-For": f"203.0.113.{i}",
                },
                json={"question": "anything",
                      "document_id": "v1:" + "0" * 32 + ":" + "1" * 32},
                environ_base={"REMOTE_ADDR": "10.0.3.1"},
            )
        )
    assert 429 in _statuses(results), "spoofed X-Forwarded-For created new buckets"


def test_proxy_fix_applied_only_when_trusted(env, providers):
    from werkzeug.middleware.proxy_fix import ProxyFix

    from myapp import create_app

    env(TRUST_PROXY="0")
    assert not isinstance(create_app().wsgi_app, ProxyFix)

    env(TRUST_PROXY="1")
    assert isinstance(create_app().wsgi_app, ProxyFix)


def test_requests_rejected_before_the_view_are_still_limited(limited_app):
    """Requests that never reach the view must still be bounded.

    Flask-Limiter resolves ``@limiter.limit`` decorators only when the view is
    invoked, so a request rejected in ``before_request`` consumes none of those
    buckets. The application limit -- evaluated in middleware -- is what covers
    this case.
    """
    client = limited_app.test_client()
    results = [
        client.post(
            "/ask",
            headers={"X-API-Key": BASE_ENV["BACKEND_API_KEY"]},  # no session id
            json={"question": "x", "document_id": "y"},
            environ_base={"REMOTE_ADDR": "10.0.4.1"},
        )
        for _ in range(12)
    ]
    assert 400 in _statuses(results)
    assert 429 in _statuses(results), "rejected requests escaped all limiting"


def test_failed_auth_attempts_are_rate_limited(limited_app):
    """Otherwise BACKEND_API_KEY could be brute-forced as fast as the network.

    401s never reach the view, so only the middleware application limit can
    bound them.
    """
    client = limited_app.test_client()
    results = [
        client.post(
            "/ask",
            headers={"X-API-Key": f"guess-{i}", "X-Session-Id": "s" * 32},
            json={"question": "x", "document_id": "y"},
            environ_base={"REMOTE_ADDR": "10.0.5.1"},
        )
        for i in range(12)
    ]
    assert 401 in _statuses(results)
    assert 429 in _statuses(results), "API key guessing was not rate limited"


def test_health_exempt_from_the_application_limit(limited_app):
    """/health must stay pollable even after the all-requests ceiling is hit."""
    client = limited_app.test_client()
    for i in range(12):
        client.post(
            "/ask",
            headers={"X-API-Key": "bad"},
            json={},
            environ_base={"REMOTE_ADDR": "10.0.6.1"},
        )
    assert client.get(
        "/health", environ_base={"REMOTE_ADDR": "10.0.6.1"}
    ).status_code == 200
