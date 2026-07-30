"""Environment-driven settings with fail-fast validation.

Two rules govern this module:

1. Nothing here touches the network or constructs a provider client. Importing
   ``rag_core`` must never require reachable Cohere or Pinecone (finding B9).
2. Invalid configuration fails at startup with a specific message, never as a
   confusing per-request 500.

The seven invariants at the bottom are the ones that actually matter; several
encode constraints that are not locally obvious (Pinecone's 40 KB metadata
ceiling, its 1000-id-per-request cap, and the interaction between chunk sizing
and the chunk budget).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from rag_core.errors import ConfigError

# Pinecone hard limits, from the Starter-plan quota documentation. Referenced by
# the invariants below so the reason for each bound is traceable.
PINECONE_METADATA_LIMIT_BYTES = 40_960
PINECONE_MAX_IDS_PER_REQUEST = 1_000
# Cohere accepts at most 96 texts per embed call.
COHERE_MAX_EMBED_BATCH = 96

_TRUE = {"1", "true", "yes", "on"}


def _raw(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _req_str(name: str) -> str:
    value = _raw(name)
    if not value:
        raise ConfigError(f"Required environment variable {name} is not set.")
    return value


def _int(name: str, default: int) -> int:
    value = _raw(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {value!r}.") from exc


def _float(name: str, default: float) -> float:
    value = _raw(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {value!r}.") from exc


def _bool(name: str, default: bool) -> bool:
    value = _raw(name)
    if value is None:
        return default
    return value.lower() in _TRUE


def _limits(name: str, default: str) -> tuple[str, ...]:
    """Parse a ``10/minute;60/hour`` style multi-limit string.

    Flask-Limiter accepts several limits for one rule; expressing them as a
    single env var keeps the deployment surface small.
    """
    raw = _raw(name, default) or ""
    parts = tuple(p.strip() for p in raw.split(";") if p.strip())
    if not parts:
        raise ConfigError(f"{name} must contain at least one rate limit.")
    return parts


# Model identifiers accepted for each role. Explicit allow-lists rather than a
# free-form string, so a typo or a model retired upstream is a startup failure
# instead of a runtime 502 mid-demo. Cohere retired embed-english-v2.0 on
# 2026-04-04 and deprecated /v1/generate on 2025-09-15; neither can appear here.
SUPPORTED_CHAT_MODELS = frozenset(
    {
        "command-a-03-2025",
        "command-a-plus-05-2026",
        "command-r-08-2024",
        "command-r-plus-08-2024",
    }
)
SUPPORTED_EMBED_MODELS = frozenset(
    {
        "embed-v4.0",
        "embed-english-v3.0",
        "embed-multilingual-v3.0",
    }
)
SUPPORTED_RERANK_MODELS = frozenset(
    {
        "rerank-v3.5",
        "rerank-v4.0-fast",
        "rerank-v4.0-pro",
    }
)
# embed-v4.0 is the only model here with a selectable output dimension.
EMBED_MODEL_DIMENSIONS: dict[str, frozenset[int]] = {
    "embed-v4.0": frozenset({256, 512, 1024, 1536}),
    "embed-english-v3.0": frozenset({1024}),
    "embed-multilingual-v3.0": frozenset({1024}),
}


@dataclass(frozen=True)
class Settings:
    """Immutable, validated configuration."""

    # --- Secrets (never logged, never serialised) ---
    cohere_api_key: str
    pinecone_api_key: str
    backend_api_key: str
    session_scope_key: str

    # --- Pinecone ---
    pinecone_index: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str

    # --- Cohere models ---
    chat_model: str
    embed_model: str
    embed_dimension: int
    rerank_model: str
    enable_rerank: bool

    # --- Ingest schema ---
    ingest_schema_version: int
    supported_schema_versions: frozenset[int]

    # --- Retrieval ---
    top_k: int
    rerank_top_n: int
    vector_min_score: float
    rerank_min_score: float
    min_evidence_chunks: int

    # --- Chunking / ingest limits ---
    chunk_chars: int
    chunk_overlap: int
    max_chunks: int
    max_extracted_chars: int
    max_pages: int
    max_upload_bytes: int
    max_question_chars: int
    metadata_max_bytes: int
    metadata_overhead_budget: int

    # --- Ingest verification ---
    ingest_verify_timeout_s: float
    ingest_verify_poll_ms: int

    # --- Retention ---
    retention_hours: int

    # --- Rate limits ---
    rate_limit_all_requests_ip: tuple[str, ...]
    rate_limit_ask_ip: tuple[str, ...]
    rate_limit_ask_scope: tuple[str, ...]
    rate_limit_ask_app_process: tuple[str, ...]
    rate_limit_upload_ip: tuple[str, ...]
    rate_limit_upload_scope: tuple[str, ...]
    rate_limit_upload_app_process: tuple[str, ...]

    # --- Runtime ---
    trust_proxy: bool
    log_level: str
    log_questions: bool

    # --- Provider timeouts / retries ---
    provider_timeout_s: float
    provider_max_retries: int

    max_embed_batch: int = field(default=COHERE_MAX_EMBED_BATCH)

    # ------------------------------------------------------------------ #

    @property
    def chunk_stride(self) -> int:
        return self.chunk_chars - self.chunk_overlap

    def public_summary(self) -> dict[str, object]:
        """Non-secret configuration, safe to log at startup and expose on /ready.

        Deliberately excludes every key. Model identifiers are included because
        knowing which model answered is operationally essential.
        """
        return {
            "index": self.pinecone_index,
            "namespace": self.pinecone_namespace,
            "cloud": self.pinecone_cloud,
            "region": self.pinecone_region,
            "chat_model": self.chat_model,
            "embed_model": self.embed_model,
            "embed_dimension": self.embed_dimension,
            "rerank_model": self.rerank_model if self.enable_rerank else None,
            "rerank_enabled": self.enable_rerank,
            "schema_version": self.ingest_schema_version,
            "top_k": self.top_k,
            "rerank_top_n": self.rerank_top_n,
        }


def _validate(s: Settings) -> None:
    """Enforce the seven startup invariants.

    Every failure names the offending variables and the arithmetic, because the
    failure mode these prevent -- silently dropping document content -- is
    invisible at runtime.
    """
    # 1. Overlap must be a real overlap, and stride must be positive or chunking
    #    would never advance.
    if not (0 <= s.chunk_overlap < s.chunk_chars):
        raise ConfigError(
            f"Invariant 1 violated: require 0 <= CHUNK_OVERLAP < CHUNK_CHARS, "
            f"got CHUNK_OVERLAP={s.chunk_overlap}, CHUNK_CHARS={s.chunk_chars}."
        )

    # 2. Reranking narrows a candidate set; it cannot widen one.
    if not (s.top_k >= s.rerank_top_n >= 1):
        raise ConfigError(
            f"Invariant 2 violated: require TOP_K >= RERANK_TOP_N >= 1, "
            f"got TOP_K={s.top_k}, RERANK_TOP_N={s.rerank_top_n}."
        )

    # 3. Exact-id rollback and prefix verification must each stay a single
    #    Pinecone request, which caps ids at 1000 per call.
    if not (1 <= s.max_chunks <= PINECONE_MAX_IDS_PER_REQUEST):
        raise ConfigError(
            f"Invariant 3 violated: require 1 <= MAX_CHUNKS <= "
            f"{PINECONE_MAX_IDS_PER_REQUEST} so rollback stays one delete "
            f"request, got MAX_CHUNKS={s.max_chunks}."
        )

    # 4. The chunk budget must be able to hold the character budget, or a
    #    legal-sized document would be rejected only after paying to embed it.
    capacity = s.max_chunks * s.chunk_stride + s.chunk_overlap
    if s.max_extracted_chars > capacity:
        raise ConfigError(
            f"Invariant 4 violated: MAX_EXTRACTED_CHARS ({s.max_extracted_chars}) "
            f"exceeds the chunk capacity MAX_CHUNKS * (CHUNK_CHARS - "
            f"CHUNK_OVERLAP) + CHUNK_OVERLAP = {s.max_chunks} * {s.chunk_stride} "
            f"+ {s.chunk_overlap} = {capacity}. Lower MAX_EXTRACTED_CHARS or "
            f"raise MAX_CHUNKS."
        )

    # 5. Stay strictly inside Pinecone's per-record metadata ceiling.
    if not (0 < s.metadata_max_bytes < PINECONE_METADATA_LIMIT_BYTES):
        raise ConfigError(
            f"Invariant 5 violated: require 0 < METADATA_MAX_BYTES < "
            f"{PINECONE_METADATA_LIMIT_BYTES} (Pinecone's hard limit), got "
            f"METADATA_MAX_BYTES={s.metadata_max_bytes}."
        )

    # 6. Model identifiers must be ones we actually support. Checked here so a
    #    retired or mistyped model is a startup failure, not a mid-demo 502.
    if s.chat_model not in SUPPORTED_CHAT_MODELS:
        raise ConfigError(
            f"COHERE_CHAT_MODEL={s.chat_model!r} is not supported. Choose one "
            f"of: {', '.join(sorted(SUPPORTED_CHAT_MODELS))}."
        )
    if s.embed_model not in SUPPORTED_EMBED_MODELS:
        raise ConfigError(
            f"COHERE_EMBED_MODEL={s.embed_model!r} is not supported. Choose one "
            f"of: {', '.join(sorted(SUPPORTED_EMBED_MODELS))}. Note that "
            f"embed-english-v2.0 was retired on 2026-04-04."
        )
    allowed_dims = EMBED_MODEL_DIMENSIONS.get(s.embed_model, frozenset())
    if s.embed_dimension not in allowed_dims:
        raise ConfigError(
            f"COHERE_EMBED_DIMENSION={s.embed_dimension} is not valid for "
            f"{s.embed_model}. Allowed: {sorted(allowed_dims)}. This value must "
            f"also equal the dimension of Pinecone index {s.pinecone_index!r}; "
            f"/ready asserts that separately."
        )
    if s.enable_rerank and s.rerank_model not in SUPPORTED_RERANK_MODELS:
        raise ConfigError(
            f"COHERE_RERANK_MODEL={s.rerank_model!r} is not supported. Choose "
            f"one of: {', '.join(sorted(SUPPORTED_RERANK_MODELS))}, or set "
            f"ENABLE_RERANK=false."
        )

    # 7. Guarantee the metadata split path is unreachable at these settings: a
    #    chunk of CHUNK_CHARS codepoints cannot exceed 4 bytes each in UTF-8.
    #    This is what keeps invariant 4's arithmetic sound, because splitting
    #    would produce more chunks than CHUNK_CHARS alone predicts.
    worst_case = s.chunk_chars * 4 + s.metadata_overhead_budget
    if worst_case > s.metadata_max_bytes:
        raise ConfigError(
            f"Invariant 7 violated: worst-case chunk metadata "
            f"CHUNK_CHARS * 4 + METADATA_OVERHEAD_BUDGET = {s.chunk_chars} * 4 "
            f"+ {s.metadata_overhead_budget} = {worst_case} bytes exceeds "
            f"METADATA_MAX_BYTES={s.metadata_max_bytes}. Lower CHUNK_CHARS or "
            f"raise METADATA_MAX_BYTES (max "
            f"{PINECONE_METADATA_LIMIT_BYTES - 1})."
        )

    # Remaining sanity bounds.
    if s.min_evidence_chunks < 1:
        raise ConfigError("MIN_EVIDENCE_CHUNKS must be at least 1.")
    if s.max_pages < 1:
        raise ConfigError("MAX_PAGES must be at least 1.")
    if s.max_upload_bytes < 1024:
        raise ConfigError("MAX_UPLOAD_MB must be at least 1 MB for a PDF demo.")
    if s.max_question_chars < 1:
        raise ConfigError("MAX_QUESTION_CHARS must be positive.")
    if s.retention_hours < 1:
        raise ConfigError("RETENTION_HOURS must be at least 1.")
    if s.ingest_verify_timeout_s <= 0:
        raise ConfigError("INGEST_VERIFY_TIMEOUT_S must be positive.")
    if s.ingest_schema_version not in s.supported_schema_versions:
        raise ConfigError(
            f"INGEST_SCHEMA_VERSION={s.ingest_schema_version} is not in the "
            f"supported set {sorted(s.supported_schema_versions)}."
        )


def load_settings() -> Settings:
    """Build and validate settings from the environment.

    Raises :class:`ConfigError` with an actionable message on any problem. Call
    once at application startup.
    """
    current_schema = _int("INGEST_SCHEMA_VERSION", 1)

    settings = Settings(
        cohere_api_key=_req_str("COHERE_API_KEY"),
        pinecone_api_key=_req_str("PINECONE_API_KEY"),
        backend_api_key=_req_str("BACKEND_API_KEY"),
        session_scope_key=_req_str("SESSION_SCOPE_KEY"),
        pinecone_index=_raw("PINECONE_INDEX", "qa-bot-v1"),
        pinecone_namespace=_raw("PINECONE_NAMESPACE", "demo-v1"),
        pinecone_cloud=_raw("PINECONE_CLOUD", "aws"),
        pinecone_region=_raw("PINECONE_REGION", "us-east-1"),
        chat_model=_req_str("COHERE_CHAT_MODEL"),
        embed_model=_raw("COHERE_EMBED_MODEL", "embed-v4.0"),
        embed_dimension=_int("COHERE_EMBED_DIMENSION", 1024),
        rerank_model=_raw("COHERE_RERANK_MODEL", "rerank-v3.5"),
        enable_rerank=_bool("ENABLE_RERANK", True),
        ingest_schema_version=current_schema,
        # Older schema versions stay readable so a version bump does not orphan
        # live documents mid-demo; cleanup retires them on the normal cutoff.
        supported_schema_versions=frozenset(range(1, current_schema + 1)),
        top_k=_int("TOP_K", 20),
        rerank_top_n=_int("RERANK_TOP_N", 5),
        vector_min_score=_float("VECTOR_MIN_SCORE", 0.0),
        rerank_min_score=_float("RERANK_MIN_SCORE", 0.20),
        min_evidence_chunks=_int("MIN_EVIDENCE_CHUNKS", 1),
        chunk_chars=_int("CHUNK_CHARS", 1200),
        chunk_overlap=_int("CHUNK_OVERLAP", 200),
        max_chunks=_int("MAX_CHUNKS", 320),
        max_extracted_chars=_int("MAX_EXTRACTED_CHARS", 300_000),
        max_pages=_int("MAX_PAGES", 40),
        max_upload_bytes=_int("MAX_UPLOAD_MB", 10) * 1024 * 1024,
        max_question_chars=_int("MAX_QUESTION_CHARS", 800),
        metadata_max_bytes=_int("METADATA_MAX_BYTES", 32_768),
        metadata_overhead_budget=_int("METADATA_OVERHEAD_BUDGET", 2_048),
        ingest_verify_timeout_s=_float("INGEST_VERIFY_TIMEOUT_S", 20.0),
        ingest_verify_poll_ms=_int("INGEST_VERIFY_POLL_MS", 500),
        retention_hours=_int("RETENTION_HOURS", 24),
        # Evaluated in Flask-Limiter middleware, so it counts EVERY request --
        # including ones rejected at auth or validation, which the per-route
        # limits never see. This is what bounds BACKEND_API_KEY guessing.
        rate_limit_all_requests_ip=_limits(
            "RATE_LIMIT_ALL_REQUESTS_IP", "120/minute;2000/hour"
        ),
        rate_limit_ask_ip=_limits("RATE_LIMIT_ASK_IP", "20/minute;120/hour"),
        rate_limit_ask_scope=_limits("RATE_LIMIT_ASK_SCOPE", "10/minute;60/hour"),
        rate_limit_ask_app_process=_limits("RATE_LIMIT_ASK_APP_PROCESS", "500/day"),
        rate_limit_upload_ip=_limits("RATE_LIMIT_UPLOAD_IP", "5/minute;20/hour"),
        rate_limit_upload_scope=_limits(
            "RATE_LIMIT_UPLOAD_SCOPE", "3/minute;10/hour"
        ),
        rate_limit_upload_app_process=_limits(
            "RATE_LIMIT_UPLOAD_APP_PROCESS", "100/day"
        ),
        trust_proxy=_bool("TRUST_PROXY", False),
        log_level=(_raw("LOG_LEVEL", "INFO") or "INFO").upper(),
        log_questions=_bool("LOG_QUESTIONS", False),
        provider_timeout_s=_float("PROVIDER_TIMEOUT_S", 30.0),
        provider_max_retries=_int("PROVIDER_MAX_RETRIES", 2),
    )
    _validate(settings)
    return settings
