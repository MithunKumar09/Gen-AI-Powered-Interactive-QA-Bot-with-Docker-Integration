"""HTTP contract, auth, and response discipline.

Two themes: statuses are stable and meaningful, and nothing sensitive ever
appears in a response body.
"""

from __future__ import annotations

import io
import json

import httpx
import pytest
from conftest import BASE_ENV, SESSION_A, make_pdf, serialize_records

PAGES = ["Alpha widgets cost ten pounds.", "Beta gears weigh two kilograms.",
         "The calibration constant is 47 microfarads."]


def _upload(client, headers, pages=None, filename="doc.pdf", generation=1):
    return client.post(
        "/upload",
        headers=headers,
        data={
            "file": (io.BytesIO(make_pdf(pages or PAGES)), filename),
            "upload_generation": str(generation),
        },
        content_type="multipart/form-data",
    )


# --- Health / readiness ------------------------------------------------------


def test_health_is_public_and_makes_no_provider_call(client, providers):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
    assert providers.index.calls == []


def test_health_stays_green_when_providers_are_down(client, providers):
    """A provider blip must not make the platform cycle a healthy container."""
    providers.index.query_errors = [RuntimeError("pinecone down")]
    assert client.get("/health").status_code == 200


def test_ready_unauthenticated_returns_only_a_boolean(client):
    body = client.get("/ready").get_json()
    assert body == {"ready": True}
    assert "index" not in body and "config" not in body


def test_ready_authenticated_returns_detail(client, auth_headers):
    body = client.get("/ready", headers=auth_headers).get_json()
    assert body["ready"] is True
    assert body["index"]["dimension"] == 1024
    assert body["config"]["chat_model"] == "command-a-03-2025"


def test_ready_makes_no_cohere_call(client, auth_headers, providers):
    """A readiness probe that bills per poll is a bill, not a check."""
    client.get("/ready", headers=auth_headers)
    assert providers.cohere.embed_calls == []
    assert providers.cohere.chat_calls == []
    assert providers.cohere.rerank_calls == []


def test_ready_reports_dimension_mismatch(client, auth_headers, providers):
    providers.pinecone.dimension = 512
    response = client.get("/ready", headers=auth_headers)
    assert response.status_code == 503
    assert "dimension" in response.get_json()["reason"].lower()


def test_ready_hides_the_reason_from_unauthenticated_callers(client, providers):
    providers.pinecone.dimension = 512
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.get_json() == {"ready": False}


def test_ready_reports_missing_index(client, auth_headers, providers):
    providers.pinecone.exists = False
    response = client.get("/ready", headers=auth_headers)
    assert response.status_code == 503
    assert "does not exist" in response.get_json()["reason"]


def test_ready_reports_wrong_metric(client, auth_headers, providers):
    providers.pinecone.metric = "euclidean"
    assert client.get("/ready", headers=auth_headers).status_code == 503


def test_unreachable_pinecone_is_503_not_502(client, auth_headers, providers):
    """Being unable to reach Pinecone is a readiness failure, not a bad gateway.

    Regression: the provider exception escaped to the generic error handler and
    surfaced as 502 PROVIDER_ERROR.
    """
    def boom(name):
        raise RuntimeError("getaddrinfo failed")

    providers.pinecone.has_index = boom
    response = client.get("/ready", headers=auth_headers)
    assert response.status_code == 503
    assert response.get_json()["ready"] is False


def test_unreachable_pinecone_stays_generic_when_unauthenticated(
    client, providers
):
    """Regression: an anonymous caller received the full error envelope, which
    disclosed which failure mode the service was in."""
    def boom(name):
        raise RuntimeError("getaddrinfo failed")

    providers.pinecone.has_index = boom
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.get_json() == {"ready": False}


def test_ready_never_leaks_provider_detail_to_anonymous_callers(
    client, providers
):
    secret_host = "internal-pinecone-host.example"

    def boom(name):
        raise RuntimeError(f"cannot resolve {secret_host}")

    providers.pinecone.has_index = boom
    assert secret_host not in client.get("/ready").get_data(as_text=True)


# --- Auth --------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/upload", "/ask", "/documents/delete", "/reset"])
def test_protected_routes_require_a_key(client, path):
    response = client.post(path, json={})
    assert response.status_code == 401
    assert response.get_json()["error"]["code"] == "UNAUTHORIZED"


def test_wrong_key_rejected(client):
    response = client.post(
        "/ask",
        headers={"X-API-Key": "wrong", "X-Session-Id": SESSION_A},
        json={"question": "x", "document_id": "y"},
    )
    assert response.status_code == 401


def test_missing_session_id_rejected(client):
    response = client.post(
        "/ask",
        headers={"X-API-Key": BASE_ENV["BACKEND_API_KEY"]},
        json={"question": "x", "document_id": "y"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("bad", ["short", "x" * 200, "has space", "semi;colon"])
def test_malformed_session_id_rejected(client, bad):
    response = client.post(
        "/ask",
        headers={"X-API-Key": BASE_ENV["BACKEND_API_KEY"], "X-Session-Id": bad},
        json={"question": "x", "document_id": "y"},
    )
    assert response.status_code == 400


# --- Upload ------------------------------------------------------------------


def test_upload_returns_operational_metadata_only(client, auth_headers):
    body = _upload(client, auth_headers).get_json()
    assert set(body) == {
        "document_id", "file_hash", "page_count", "chunk_count",
        "reused", "upload_generation", "request_id",
    }


def test_upload_never_returns_an_embedding(client, auth_headers):
    """The pre-audit endpoint returned the whole embedding array (finding A8)."""
    blob = json.dumps(_upload(client, auth_headers).get_json())
    assert "embedding" not in blob.lower()
    assert "values" not in blob.lower()


def test_upload_does_not_return_document_text(client, auth_headers):
    blob = json.dumps(_upload(client, auth_headers).get_json())
    assert "calibration constant" not in blob


def test_upload_echoes_the_generation_nonce(client, auth_headers):
    body = _upload(client, auth_headers, generation=7).get_json()
    assert body["upload_generation"] == 7


def test_upload_without_a_file_is_400(client, auth_headers):
    response = client.post("/upload", headers=auth_headers, data={})
    assert response.status_code == 400


def test_upload_of_non_pdf_is_422(client, auth_headers):
    response = client.post(
        "/upload",
        headers=auth_headers,
        data={"file": (io.BytesIO(b"just text"), "fake.pdf")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "INVALID_PDF"


def test_upload_over_max_content_length_is_413(client, auth_headers, app):
    oversized = b"%PDF-" + b"x" * (app.config["MAX_CONTENT_LENGTH"] + 1024)
    response = client.post(
        "/upload",
        headers=auth_headers,
        data={"file": (io.BytesIO(oversized), "big.pdf")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_upload_with_blank_filename_is_400(client, auth_headers):
    response = client.post(
        "/upload",
        headers=auth_headers,
        data={"file": (io.BytesIO(make_pdf(["x"])), "")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


# --- Ask ---------------------------------------------------------------------


def test_ask_returns_answer_and_citations(client, auth_headers):
    document_id = _upload(client, auth_headers).get_json()["document_id"]
    response = client.post(
        "/ask",
        headers=auth_headers,
        json={"question": "What is the calibration constant?",
              "document_id": document_id},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {
        "answer", "citations", "abstained", "request_id", "latency_ms"
    }
    assert body["citations"][0]["page"] in (1, 2, 3)


def test_ask_requires_json_content_type(client, auth_headers):
    response = client.post(
        "/ask", headers=auth_headers, data="question=x",
        content_type="application/x-www-form-urlencoded",
    )
    assert response.status_code == 400


def test_ask_rejects_a_non_object_body(client, auth_headers):
    response = client.post("/ask", headers=auth_headers, json=["a", "list"])
    assert response.status_code == 400


@pytest.mark.parametrize("body", [
    {}, {"question": "x"}, {"document_id": "y"},
    {"question": "", "document_id": "y"}, {"question": 5, "document_id": "y"},
])
def test_ask_validates_the_body(client, auth_headers, body):
    assert client.post("/ask", headers=auth_headers, json=body).status_code in (400, 404)


def test_ask_rejects_an_overlong_question(client, auth_headers, settings):
    document_id = _upload(client, auth_headers).get_json()["document_id"]
    response = client.post(
        "/ask",
        headers=auth_headers,
        json={"question": "x" * (settings.max_question_chars + 1),
              "document_id": document_id},
    )
    assert response.status_code == 400


def test_ask_for_an_unknown_document_is_404(client, auth_headers):
    response = client.post(
        "/ask",
        headers=auth_headers,
        json={"question": "x", "document_id": "v1:" + "0" * 32 + ":" + "1" * 32},
    )
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "UNKNOWN_DOCUMENT"


def test_cross_scope_ask_is_404_not_403(client, auth_headers, headers_b):
    """403 would confirm the document exists; 404 discloses nothing."""
    document_id = _upload(client, auth_headers).get_json()["document_id"]
    response = client.post(
        "/ask",
        headers=headers_b,
        json={"question": "What do widgets cost?", "document_id": document_id},
    )
    assert response.status_code == 404


# --- Provider failures -------------------------------------------------------

def test_provider_connection_failure_is_502(client, auth_headers, providers):
    providers.cohere.embed_errors = [httpx.ConnectError("down")] * 4
    response = client.post(
        "/upload",
        headers=auth_headers,
        data={"file": (io.BytesIO(make_pdf(PAGES)), "d.pdf")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 502


def test_provider_timeout_is_504(client, auth_headers, providers):
    document_id = _upload(client, auth_headers).get_json()["document_id"]
    providers.cohere.chat_errors = [httpx.ReadTimeout("slow")]
    response = client.post(
        "/ask",
        headers=auth_headers,
        json={"question": "What do widgets cost?", "document_id": document_id},
    )
    assert response.status_code == 504
    assert response.get_json()["error"]["code"] == "PROVIDER_TIMEOUT"


def test_provider_error_text_is_not_leaked(client, auth_headers, providers):
    """The pre-audit code returned str(e) straight to the client (finding A7)."""
    secret = "postgres://user:hunter2@internal-host/db"
    providers.cohere.embed_errors = [RuntimeError(secret)] * 4
    response = client.post(
        "/upload",
        headers=auth_headers,
        data={"file": (io.BytesIO(make_pdf(PAGES)), "d.pdf")},
        content_type="multipart/form-data",
    )
    assert secret not in response.get_data(as_text=True)
    assert "hunter2" not in response.get_data(as_text=True)


# --- Deletion ----------------------------------------------------------------


def test_delete_returns_accepted_without_a_count(client, auth_headers):
    """Pinecone reports no deleted count, so claiming one would be fabricated."""
    document_id = _upload(client, auth_headers).get_json()["document_id"]
    body = client.post(
        "/documents/delete", headers=auth_headers, json={"document_id": document_id}
    ).get_json()
    assert body["accepted"] is True
    assert "deleted" not in body


def test_delete_is_idempotent_over_http(client, auth_headers):
    document_id = _upload(client, auth_headers).get_json()["document_id"]
    for _ in range(2):
        response = client.post(
            "/documents/delete", headers=auth_headers,
            json={"document_id": document_id},
        )
        assert response.status_code == 200


def test_cross_scope_delete_is_404(client, auth_headers, headers_b, providers):
    document_id = _upload(client, auth_headers).get_json()["document_id"]
    response = client.post(
        "/documents/delete", headers=headers_b, json={"document_id": document_id}
    )
    assert response.status_code == 404
    assert providers.index.records, "scope B must not have deleted scope A's data"


def test_reset_returns_accepted_with_a_scope_ref(client, auth_headers):
    _upload(client, auth_headers)
    body = client.post("/reset", headers=auth_headers).get_json()
    assert body["accepted"] is True
    assert len(body["scope_ref"]) == 12
    assert "deleted" not in body


def test_reset_does_not_expose_the_raw_session_id(client, auth_headers):
    body = client.post("/reset", headers=auth_headers).get_json()
    assert SESSION_A not in json.dumps(body)


# --- Response hygiene --------------------------------------------------------


def test_request_id_is_returned_in_a_header(client):
    assert client.get("/health").headers.get("X-Request-Id")


def test_security_headers_present(client, auth_headers):
    headers = _upload(client, auth_headers).headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Cache-Control"] == "no-store"


def test_unknown_route_returns_the_error_envelope(client):
    body = client.get("/does-not-exist").get_json()
    assert set(body["error"]) == {"code", "message", "request_id"}


def test_no_secret_appears_in_any_response(client, auth_headers, settings):
    document_id = _upload(client, auth_headers).get_json()["document_id"]
    blobs = [
        client.get("/health").get_data(as_text=True),
        client.get("/ready", headers=auth_headers).get_data(as_text=True),
        client.post("/ask", headers=auth_headers, json={
            "question": "What do widgets cost?", "document_id": document_id
        }).get_data(as_text=True),
    ]
    for blob in blobs:
        for secret in (settings.cohere_api_key, settings.pinecone_api_key,
                       settings.backend_api_key, settings.session_scope_key):
            assert secret not in blob


# --- Logging privacy ---------------------------------------------------------


def test_raw_session_id_never_appears_in_logs(
    client, auth_headers, captured_logs
):
    document_id = _upload(client, auth_headers).get_json()["document_id"]
    client.post("/ask", headers=auth_headers, json={
        "question": "What do widgets cost?", "document_id": document_id
    })
    assert SESSION_A not in serialize_records(captured_logs)


def test_question_text_not_logged_by_default(
    client, auth_headers, captured_logs
):
    document_id = _upload(client, auth_headers).get_json()["document_id"]
    client.post("/ask", headers=auth_headers, json={
        "question": "MY_SECRET_QUESTION_MARKER", "document_id": document_id
    })
    assert "MY_SECRET_QUESTION_MARKER" not in serialize_records(captured_logs)


def test_document_text_not_logged(client, auth_headers, captured_logs):
    _upload(client, auth_headers)
    assert "calibration constant" not in serialize_records(captured_logs)


def test_secrets_never_appear_in_logs(
    client, auth_headers, captured_logs, settings
):
    document_id = _upload(client, auth_headers).get_json()["document_id"]
    client.post("/ask", headers=auth_headers, json={
        "question": "anything", "document_id": document_id
    })
    blob = serialize_records(captured_logs)
    for secret in (settings.cohere_api_key, settings.pinecone_api_key,
                   settings.backend_api_key, settings.session_scope_key):
        assert secret not in blob
