"""Startup configuration invariants.

Each invariant exists to prevent a failure that would otherwise be invisible at
runtime -- most importantly silently dropping document content, or a chunk budget
that cannot hold the character budget.
"""

from __future__ import annotations

import pytest

from rag_core.config import load_settings
from rag_core.errors import ConfigError


def test_shipped_defaults_are_valid(settings):
    assert settings.chunk_chars == 1200
    assert settings.chunk_overlap == 200
    assert settings.max_chunks == 320
    assert settings.max_extracted_chars == 300_000
    assert settings.embed_model == "embed-v4.0"
    assert settings.embed_dimension == 1024


def test_default_chunk_capacity_holds(settings):
    """Invariant 4 must hold for the values we actually ship."""
    capacity = settings.max_chunks * settings.chunk_stride + settings.chunk_overlap
    assert settings.max_extracted_chars <= capacity


def test_default_metadata_worst_case_fits(settings):
    """Invariant 7: the split path must be unreachable at shipped defaults."""
    worst = settings.chunk_chars * 4 + settings.metadata_overhead_budget
    assert worst <= settings.metadata_max_bytes


@pytest.mark.parametrize("missing", [
    "COHERE_API_KEY", "PINECONE_API_KEY", "BACKEND_API_KEY",
    "SESSION_SCOPE_KEY", "COHERE_CHAT_MODEL",
])
def test_missing_secret_fails_at_startup(monkeypatch, missing):
    monkeypatch.delenv(missing, raising=False)
    with pytest.raises(ConfigError, match=missing):
        load_settings()


def test_invariant_1_overlap_below_chunk_size(env):
    env(CHUNK_OVERLAP=1200)
    with pytest.raises(ConfigError, match="Invariant 1"):
        load_settings()


def test_invariant_2_top_k_at_least_top_n(env):
    env(TOP_K=3, RERANK_TOP_N=5)
    with pytest.raises(ConfigError, match="Invariant 2"):
        load_settings()


def test_invariant_3_max_chunks_within_pinecone_id_limit(env):
    """Above 1000 ids, rollback would need multiple delete requests."""
    env(MAX_CHUNKS=1001)
    with pytest.raises(ConfigError, match="Invariant 3"):
        load_settings()


def test_invariant_4_catches_the_original_inconsistency(env):
    """300k chars against a 400-chunk cap: the contradiction this guards against.

    600_000 chars at a 1000-char stride needs ~600 chunks, so a 400-chunk budget
    would reject a legal document only *after* paying to embed it.
    """
    env(MAX_EXTRACTED_CHARS=600_000, MAX_CHUNKS=400)
    with pytest.raises(ConfigError, match="Invariant 4"):
        load_settings()


def test_invariant_5_metadata_below_pinecone_hard_limit(env):
    env(METADATA_MAX_BYTES=40_960)
    with pytest.raises(ConfigError, match="Invariant 5"):
        load_settings()


def test_invariant_7_worst_case_utf8_must_fit(env):
    env(METADATA_MAX_BYTES=4000)
    with pytest.raises(ConfigError, match="Invariant 7"):
        load_settings()


def test_retired_embed_model_rejected(env):
    """embed-english-v2.0 was retired 2026-04-04 and must not be configurable."""
    env(COHERE_EMBED_MODEL="embed-english-v2.0")
    with pytest.raises(ConfigError, match="retired"):
        load_settings()


def test_retired_chat_model_rejected(env):
    """command-xlarge-nightly predates every current Command release."""
    env(COHERE_CHAT_MODEL="command-xlarge-nightly")
    with pytest.raises(ConfigError, match="not supported"):
        load_settings()


def test_dimension_must_be_valid_for_the_embed_model(env):
    """4096 was the old v2 dimension; embed-v4.0 does not offer it."""
    env(COHERE_EMBED_DIMENSION=4096)
    with pytest.raises(ConfigError, match="not valid for"):
        load_settings()


def test_unsupported_rerank_model_rejected(env):
    env(COHERE_RERANK_MODEL="rerank-english-v2.0", ENABLE_RERANK="true")
    with pytest.raises(ConfigError, match="not supported"):
        load_settings()


def test_rerank_model_unchecked_when_disabled(env):
    """A stale rerank model must not block startup if reranking is off."""
    env(COHERE_RERANK_MODEL="rerank-english-v2.0", ENABLE_RERANK="false")
    assert load_settings().enable_rerank is False


def test_public_summary_contains_no_secrets(settings):
    blob = repr(settings.public_summary())
    for secret in (
        settings.cohere_api_key,
        settings.pinecone_api_key,
        settings.backend_api_key,
        settings.session_scope_key,
    ):
        assert secret not in blob


def test_multi_limit_strings_parse(env):
    env(RATE_LIMIT_ASK_IP="20/minute;120/hour")
    assert load_settings().rate_limit_ask_ip == ("20/minute", "120/hour")


def test_env_example_is_a_valid_configuration(monkeypatch):
    """.env.example must actually work, not merely describe.

    Guards against documentation drift: a new required setting, a renamed
    variable, or a default that violates an invariant would all be caught here
    rather than by the first person who copies the file.
    """
    from pathlib import Path

    from dotenv import dotenv_values

    example = Path(__file__).resolve().parents[2] / ".env.example"
    assert example.exists(), "expected .env.example at the repository root"

    for key, value in dotenv_values(example).items():
        if value:
            monkeypatch.setenv(key, value)

    settings = load_settings()  # must not raise
    assert settings.embed_dimension in (256, 512, 1024, 1536)


def test_env_example_documents_every_setting():
    """Every variable config reads should appear in .env.example."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    config_src = (root / "Backend" / "rag_core" / "config.py").read_text(
        encoding="utf-8"
    )
    documented = set(
        re.findall(
            r"^([A-Z][A-Z_0-9]+)=",
            (root / ".env.example").read_text(encoding="utf-8"),
            re.M,
        )
    )
    # Names passed to the env readers, including line-wrapped calls.
    read = set(
        re.findall(
            r"_(?:raw|req_str|int|float|bool|limits)\(\s*\"([A-Z][A-Z_0-9]+)\"",
            config_src,
        )
    )
    assert read, "regex found no env reads -- update this test"
    assert not (read - documented), (
        f"undocumented settings: {sorted(read - documented)}"
    )


def test_env_example_contains_no_real_looking_secrets():
    """The example must never carry a usable credential.

    The previously leaked keys are checked explicitly, since .env.example was
    created in the same change that removed them.
    """
    from pathlib import Path

    text = (
        Path(__file__).resolve().parents[2] / ".env.example"
    ).read_text(encoding="utf-8")
    for leaked in ("acb4f1cc-ab8c", "byLThFeN1N4TbwTGEItse"):
        assert leaked not in text
