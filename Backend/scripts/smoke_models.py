"""Verify the configured Cohere models actually respond. A deployment gate.

This exists because ``/ready`` deliberately makes no Cohere call: a readiness
probe that bills per poll is a bill, not a check. But "is the configured chat
model still served?" is a real question that only a live call can answer -- and
Cohere retires models on a published schedule, which is precisely how this
project ended up on ``embed-english-v2.0`` (retired 2026-04-04) and
``command-xlarge-nightly``.

So model availability is checked here, once, explicitly, before a deploy is
accepted. Each call is bounded and minimal: a two-word embed, a one-token chat, a
two-document rerank. The cost is negligible; the information is not.

Usage::

    python scripts/smoke_models.py
    python scripts/smoke_models.py --timeout 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from rag_core import embeddings, generation, rerank  # noqa: E402
from rag_core.config import load_settings  # noqa: E402
from rag_core.errors import ConfigError, RagError  # noqa: E402
from rag_core.generation import Evidence  # noqa: E402


def _check(label: str, fn) -> tuple[bool, str, int]:
    started = time.monotonic()
    try:
        detail = fn()
        elapsed = int((time.monotonic() - started) * 1000)
        return True, detail, elapsed
    except RagError as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        # exc.detail carries the provider message; this is an operator tool, so
        # showing it here is correct -- unlike in an HTTP response.
        return False, exc.detail or exc.message, elapsed
    except Exception as exc:  # pragma: no cover
        elapsed = int((time.monotonic() - started) * 1000)
        return False, f"{type(exc).__name__}: {exc}", elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=None,
                        help="Per-call timeout in seconds.")
    args = parser.parse_args()

    load_dotenv()
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.timeout:
        from dataclasses import replace
        settings = replace(settings, provider_timeout_s=args.timeout)

    print("Configured models:")
    print(f"  chat    {settings.chat_model}")
    print(f"  embed   {settings.embed_model} (dim {settings.embed_dimension})")
    print(f"  rerank  {settings.rerank_model}"
          f"{'' if settings.enable_rerank else '  (disabled)'}")
    print()

    checks: list[tuple[str, object]] = [
        (
            f"embed[{settings.embed_model}]",
            lambda: _describe_embed(settings),
        ),
        (
            f"chat[{settings.chat_model}]",
            lambda: _describe_chat(settings),
        ),
    ]
    if settings.enable_rerank:
        checks.append((
            f"rerank[{settings.rerank_model}]",
            lambda: _describe_rerank(settings),
        ))

    failures = 0
    for label, fn in checks:
        ok, detail, elapsed = _check(label, fn)
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {label:38} {elapsed:>6}ms  {detail}")
        if not ok:
            failures += 1

    print()
    if failures:
        print(f"{failures} model check(s) failed. Do not deploy.", file=sys.stderr)
        return 1
    print("All configured models responded.")
    return 0


def _describe_embed(settings) -> str:
    vectors = embeddings.embed_documents(settings, ["smoke test document"])
    got = len(vectors[0])
    if got != settings.embed_dimension:
        raise RagError(
            f"returned dimension {got}, expected {settings.embed_dimension}"
        )
    return f"dimension {got}"


def _describe_chat(settings) -> str:
    answer = generation.generate(
        settings,
        question="Reply with the single word: ok",
        evidence=[Evidence(chunk_index=0, page=1,
                           text="The agreed acknowledgement word is ok.",
                           score=1.0)],
    )
    text = (answer.text or "").strip()
    if not text:
        raise RagError("empty completion")
    return f"{len(text)} chars, {len(answer.citations)} citation(s)"


def _describe_rerank(settings) -> str:
    ranked = rerank.rerank(
        settings,
        "what colour is the sky",
        ["The sky is blue.", "Bananas are a fruit."],
    )
    if not ranked:
        raise RagError("no results returned")
    return f"top score {ranked[0][1]:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
