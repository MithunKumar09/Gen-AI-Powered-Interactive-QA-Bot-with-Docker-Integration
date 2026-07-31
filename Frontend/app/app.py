"""Streamlit frontend for the RAG QA bot.

The central correctness concern here is Streamlit's execution model: the whole
script re-runs top to bottom on **every** widget interaction. The previous
version called ``requests.post(f"{backend_url}/upload", ...)`` at module level
whenever a file was selected, so submitting a question -- or touching any other
widget -- re-uploaded and re-embedded the entire PDF. Slow, and it spent real
Cohere credits on every interaction.

The fix is to make upload a function of *content*, not of script execution:
hash the immutable bytes once and only upload when that hash changes.

The second concern is out-of-order responses. Two uploads can be in flight after
a quick file swap, and the slower earlier one may land last. A response is
therefore only allowed to become the active document if it still matches both the
currently selected file hash **and** the current upload generation.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Connect timeout is short; read timeout is long because Render's free tier
# spins a sleeping service back up on the first request, which can take ~50s.
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 120
UPLOAD_READ_TIMEOUT = 180


def _backend_url() -> str:
    """Normalise BACKEND_URL.

    Accepts a full URL (local Compose: ``http://backend:5000``) or a bare
    hostname, which is what Render's ``fromService`` host property yields. The
    previous code interpolated the raw env var, so an unset variable produced
    requests to the literal URL ``None/upload``.
    """
    raw = (os.getenv("BACKEND_URL") or "").strip().rstrip("/")
    if not raw:
        st.error(
            "BACKEND_URL is not set. The app cannot reach its API. "
            "See DEPLOYMENT.md."
        )
        st.stop()
    if not raw.startswith(("http://", "https://")):
        # A bare host from Render is always TLS-terminated.
        raw = f"https://{raw}"
    return raw


BACKEND_URL = _backend_url()
API_KEY = os.getenv("BACKEND_API_KEY", "")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "")
# Non-secret opt-in for the public portfolio demo: publishes DEMO_PASSWORD on the
# login page so a visitor can get in without being sent it out of band. Parsed the
# same way as the backend's booleans (rag_core/config.py), which cannot be imported
# here -- the frontend image is built from the Frontend/ context alone.
PUBLIC_DEMO_MODE = os.getenv(
    "PUBLIC_DEMO_MODE", ""
).strip().lower() in {"1", "true", "yes", "on"}
# Never render an empty code block: with no passphrase configured the gate is
# skipped entirely (see `authed` below), so this is a second, independent guard.
SHOW_DEMO_CODE = PUBLIC_DEMO_MODE and bool(DEMO_PASSWORD)

st.set_page_config(
    page_title="Interactive QA Bot with RAG",
    page_icon="📄",
    layout="centered",
)


# --- Session state -----------------------------------------------------------


def _init_state() -> None:
    defaults = {
        # Opaque per-browser-session id. The backend never stores this: it
        # derives an HMAC scope from it and discards the original.
        "session_id": uuid.uuid4().hex,
        "authed": not DEMO_PASSWORD,  # no passphrase configured => open
        "uploaded_hash": None,
        "uploaded_filename": None,
        "active_document_id": None,
        "upload_status": "idle",  # idle | uploading | ready | failed
        "upload_error": None,
        "upload_generation": 0,
        "page_count": 0,
        "chunk_count": 0,
        "messages": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY, "X-Session-Id": st.session_state.session_id}


def _friendly_error(response: requests.Response) -> str:
    """Extract the API's error message, never the raw response body.

    Dumping ``response.text`` into the UI is how internal detail leaks into
    screenshots.
    """
    try:
        payload = response.json()
        error = payload.get("error", {})
        message = error.get("message")
        request_id = error.get("request_id")
        if message:
            return f"{message}" + (f" (ref {request_id})" if request_id else "")
    except ValueError:
        pass
    return f"The server returned an unexpected error (HTTP {response.status_code})."


# --- Passphrase gate ---------------------------------------------------------


def _gate() -> None:
    """Shared passphrase gate.

    When PUBLIC_DEMO_MODE is disabled, the passphrase provides a basic spend
    guard. When public demo mode is enabled, the code is intentionally displayed
    and the gate serves only as an explicit portfolio-demo entry step. It is not
    user authentication in either mode.
    """
    st.title("📄 Interactive QA Bot with RAG")
    st.caption("Ask questions about your own PDF, answered only from its contents.")
    with st.form("gate"):
        entered = st.text_input("Demo passphrase", type="password")
        if SHOW_DEMO_CODE:
            # Deliberately public: this code is displayed, so it is an entry step
            # rather than access control. Rendered only -- never auto-filled or
            # auto-submitted; the visitor still submits the form below.
            st.caption("Portfolio demo access code")
            st.code(DEMO_PASSWORD, language=None)
            st.caption(
                "This public code is provided so recruiters and hiring managers "
                "can test the demo."
            )
        if st.form_submit_button("Enter", use_container_width=True):
            if hmac.compare_digest(entered or "", DEMO_PASSWORD):
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("That passphrase is not correct.")
    st.stop()


if not st.session_state.authed:
    _gate()


# --- Upload ------------------------------------------------------------------


def _upload(data: bytes, filename: str, generation: int) -> None:
    """Upload one PDF and adopt it only if the response is still current."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/upload",
            headers=_headers(),
            files={"file": (filename, data, "application/pdf")},
            data={"upload_generation": str(generation)},
            timeout=(CONNECT_TIMEOUT, UPLOAD_READ_TIMEOUT),
        )
    except requests.exceptions.Timeout:
        st.session_state.upload_status = "failed"
        st.session_state.upload_error = (
            "The server took too long to process this document. Please retry."
        )
        return
    except requests.exceptions.RequestException:
        st.session_state.upload_status = "failed"
        st.session_state.upload_error = (
            "Could not reach the API. If this is a free-tier demo it may be "
            "waking up — wait a moment and retry."
        )
        return

    if response.status_code != 200:
        st.session_state.upload_status = "failed"
        st.session_state.upload_error = _friendly_error(response)
        return

    body = response.json()

    # Only adopt this response if it still describes what the user has selected
    # now. A slower earlier upload landing after a newer one must not become the
    # active document.
    current_hash = hashlib.sha256(data).hexdigest()
    if (
        body.get("file_hash") != current_hash
        or int(body.get("upload_generation", -1)) != st.session_state.upload_generation
    ):
        _delete_document(body.get("document_id"))
        return

    previous = st.session_state.active_document_id
    st.session_state.active_document_id = body["document_id"]
    st.session_state.page_count = body.get("page_count", 0)
    st.session_state.chunk_count = body.get("chunk_count", 0)
    st.session_state.upload_status = "ready"
    st.session_state.upload_error = None
    st.session_state.messages = []

    # Only now that the new document is active is the old one safe to remove.
    # Best effort: a failure leaves an orphan for the retention cleanup, which is
    # strictly better than deleting first and risking having nothing.
    if previous and previous != body["document_id"]:
        _delete_document(previous)


def _delete_document(document_id: str | None) -> None:
    if not document_id:
        return
    try:
        requests.post(
            f"{BACKEND_URL}/documents/delete",
            headers=_headers(),
            json={"document_id": document_id},
            timeout=(CONNECT_TIMEOUT, 30),
        )
    except requests.exceptions.RequestException:
        # Deliberately silent: this is housekeeping the user did not ask for and
        # cannot act on. TTL cleanup is the backstop.
        pass


st.title("📄 Interactive QA Bot with RAG")
st.caption("Ask questions about your own PDF, answered only from its contents.")

uploaded = st.file_uploader(
    "Upload a PDF", type="pdf", help="Text-based PDFs only — scans have no extractable text."
)

if uploaded is not None:
    # getvalue() returns the whole buffer without consuming it, so it is safe to
    # call on every rerun -- unlike read(), which would return b"" the second time.
    data = uploaded.getvalue()
    file_hash = hashlib.sha256(data).hexdigest()

    needs_upload = file_hash != st.session_state.uploaded_hash
    retry_available = (
        st.session_state.upload_status == "failed"
        and file_hash == st.session_state.uploaded_hash
    )

    if needs_upload:
        # New content: claim it, bump the generation, and upload exactly once.
        st.session_state.uploaded_hash = file_hash
        st.session_state.uploaded_filename = uploaded.name
        st.session_state.upload_generation += 1
        st.session_state.upload_status = "uploading"
        with st.spinner("Reading and indexing your document…"):
            _upload(data, uploaded.name, st.session_state.upload_generation)

    elif retry_available:
        st.error(st.session_state.upload_error or "The upload failed.")
        if st.button("Retry upload"):
            st.session_state.upload_generation += 1
            st.session_state.upload_status = "uploading"
            with st.spinner("Retrying…"):
                _upload(data, uploaded.name, st.session_state.upload_generation)
            st.rerun()

if st.session_state.upload_status == "ready":
    st.success(
        f"**{st.session_state.uploaded_filename}** indexed — "
        f"{st.session_state.page_count} pages, "
        f"{st.session_state.chunk_count} searchable sections."
    )
elif st.session_state.upload_status == "failed" and st.session_state.upload_error:
    st.error(st.session_state.upload_error)


# --- Ask ---------------------------------------------------------------------


def _ask(question: str) -> dict | None:
    try:
        response = requests.post(
            f"{BACKEND_URL}/ask",
            headers=_headers(),
            json={
                "question": question,
                "document_id": st.session_state.active_document_id,
            },
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
    except requests.exceptions.Timeout:
        return {"error": "The answer took too long to generate. Please try again."}
    except requests.exceptions.RequestException:
        return {"error": "Could not reach the API. Please try again in a moment."}

    if response.status_code == 404:
        # The document is gone -- most likely retention cleanup removed it while
        # this tab stayed open. Clear state so the user is prompted to re-upload
        # rather than being stuck asking about nothing.
        st.session_state.active_document_id = None
        st.session_state.uploaded_hash = None
        st.session_state.upload_status = "idle"
        return {
            "error": "This document is no longer available. Please upload it again."
        }
    if response.status_code == 429:
        return {"error": "Rate limit reached for this demo. Please wait a minute."}
    if response.status_code != 200:
        return {"error": _friendly_error(response)}
    return response.json()


if st.session_state.active_document_id:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for citation in message.get("citations", []):
                label = f"page {citation['page']}"
                if citation.get("inferred"):
                    label += " (closest match)"
                with st.expander(f"Source — {label}"):
                    st.caption(citation.get("snippet", ""))

    question = st.chat_input("Ask a question about this document")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document…"):
                result = _ask(question)

            if result is None or "error" in result:
                text = (result or {}).get("error", "Something went wrong.")
                st.error(text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": text}
                )
            else:
                st.markdown(result["answer"])
                citations = result.get("citations", [])
                for citation in citations:
                    label = f"page {citation['page']}"
                    if citation.get("inferred"):
                        label += " (closest match)"
                    with st.expander(f"Source — {label}"):
                        st.caption(citation.get("snippet", ""))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "citations": citations,
                    }
                )

elif st.session_state.upload_status != "failed":
    st.info("Upload a text-based PDF to begin.")


with st.sidebar:
    st.subheader("Session")
    st.caption(
        "Your document is isolated to this browser session and is removed "
        "automatically after a period of inactivity."
    )
    if st.session_state.active_document_id:
        if st.button("Clear my document", use_container_width=True):
            try:
                requests.post(
                    f"{BACKEND_URL}/reset",
                    headers=_headers(),
                    timeout=(CONNECT_TIMEOUT, 30),
                )
            except requests.exceptions.RequestException:
                pass
            for key in (
                "uploaded_hash", "uploaded_filename", "active_document_id",
                "upload_error",
            ):
                st.session_state[key] = None
            st.session_state.upload_status = "idle"
            st.session_state.messages = []
            st.rerun()
    st.divider()
    st.caption(
        "Answers are grounded in the uploaded document only. If the document "
        "does not contain an answer, the bot says so rather than guessing."
    )
