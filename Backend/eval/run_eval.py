"""Retrieval and abstention evaluation.

Measures the things that actually determine whether this demo is trustworthy:
can it find evidence that is not on page one, does it cite the right page, and
does it refuse to answer when the document does not contain the answer.

That last one matters most. A fabricated answer is worse than no answer, because
the user has no way to tell the difference.

Two modes, and the distinction is enforced rather than merely documented:

* default -- calibration. Tune freely, sweep thresholds.
* ``--holdout`` -- single-use gate. Refuses to run if a report already exists for
  this fixture, because a second run against a fixture whose result you have
  already seen is no longer a holdout.

Requires real credentials and spends real Cohere tokens. Correctness is covered
by ``tests/`` with fakes and no network; this measures quality.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from rag_core import pipeline  # noqa: E402
from rag_core.config import load_settings  # noqa: E402
from rag_core.errors import ConfigError, RagError  # noqa: E402
from rag_core.identity import derive_scope  # noqa: E402

REPORTS = Path(__file__).parent / "reports"

# Anything not answerable must be refused. These are the kinds where answering is
# a failure, not a success.
UNANSWERABLE_KINDS = {"unanswerable", "misleading"}


@dataclass
class CaseResult:
    case_id: str
    kind: str
    answerable: bool
    abstained: bool
    pages_cited: list[int] = field(default_factory=list)
    top_score: float = 0.0
    retrieved_k: int = 0
    latency_ms: int = 0
    error: str | None = None
    answer_preview: str = ""
    keywords_hit: list[str] = field(default_factory=list)
    keywords_missed: list[str] = field(default_factory=list)
    min_page_ok: bool | None = None

    @property
    def correct_abstention(self) -> bool:
        return self.abstained and not self.answerable

    @property
    def unsupported_answer(self) -> bool:
        """Answered when it should have refused. The number that matters most."""
        return (not self.abstained) and (not self.answerable)

    @property
    def missed_answer(self) -> bool:
        """Refused when the document does contain the answer."""
        return self.abstained and self.answerable


def config_hash(settings) -> str:
    """Stable fingerprint of the settings that affect quality.

    Recorded with every report so a result can always be tied back to the exact
    configuration that produced it -- which is what makes a frozen-then-measured
    workflow verifiable rather than a claim.
    """
    material = json.dumps(
        {
            "chat_model": settings.chat_model,
            "embed_model": settings.embed_model,
            "embed_dimension": settings.embed_dimension,
            "rerank_model": settings.rerank_model,
            "enable_rerank": settings.enable_rerank,
            "top_k": settings.top_k,
            "rerank_top_n": settings.rerank_top_n,
            "vector_min_score": settings.vector_min_score,
            "rerank_min_score": settings.rerank_min_score,
            "min_evidence_chunks": settings.min_evidence_chunks,
            "chunk_chars": settings.chunk_chars,
            "chunk_overlap": settings.chunk_overlap,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def run_fixture(settings, fixture: dict, fixture_dir: Path) -> list[CaseResult]:
    scope = derive_scope(
        settings.session_scope_key, f"eval-{config_hash(settings)}-{int(time.time())}"
    )

    documents: dict[str, str] = {}
    for spec in fixture["documents"]:
        path = (fixture_dir / spec["path"]).resolve()
        if not path.exists():
            print(f"  ! missing document: {path}", file=sys.stderr)
            continue
        print(f"  ingesting {spec['id']} ({path.name}) …", end="", flush=True)
        result = pipeline.ingest_pdf(
            settings, data=path.read_bytes(), filename=path.name, scope=scope
        )
        documents[spec["id"]] = result.document_id
        print(f" {result.page_count}p / {result.chunk_count} chunks"
              f"{' (reused)' if result.reused else ''}")

    results: list[CaseResult] = []
    for case in fixture["cases"]:
        document_id = documents.get(case["document"])
        if not document_id:
            continue
        results.append(_run_case(settings, case, document_id, scope))

    # Leave nothing behind: eval documents would otherwise sit in the index until
    # the retention cutoff.
    try:
        pipeline.reset_scope(settings, scope=scope)
    except RagError:
        pass
    return results


def _run_case(settings, case: dict, document_id: str, scope: str) -> CaseResult:
    kind = case.get("kind", "direct")
    answerable = bool(case.get("answerable", True))
    started = time.monotonic()
    try:
        answer = pipeline.answer_question(
            settings,
            question=case["question"],
            document_id=document_id,
            scope=scope,
        )
    except RagError as exc:
        return CaseResult(
            case_id=case["id"], kind=kind, answerable=answerable, abstained=False,
            latency_ms=int((time.monotonic() - started) * 1000),
            error=exc.code,
        )

    pages = [c["page"] for c in answer.citations]
    text_lower = (answer.answer or "").lower()
    expected = [k.lower() for k in case.get("expect_keywords", [])]
    hit = [k for k in expected if k in text_lower]
    missed = [k for k in expected if k not in text_lower]

    min_page_ok = None
    if case.get("min_page") is not None and pages:
        min_page_ok = max(pages) >= int(case["min_page"])

    return CaseResult(
        case_id=case["id"],
        kind=kind,
        answerable=answerable,
        abstained=answer.abstained,
        pages_cited=pages,
        top_score=float(answer.metrics.get("top_score", 0.0) or 0.0),
        retrieved_k=int(answer.metrics.get("retrieved_k", 0) or 0),
        latency_ms=int(answer.metrics.get("elapsed_ms", 0) or 0),
        answer_preview=(answer.answer or "")[:160],
        keywords_hit=hit,
        keywords_missed=missed,
        min_page_ok=min_page_ok,
    )


def summarize(results: list[CaseResult]) -> dict:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    answered_correctly = [r for r in answerable if not r.abstained and not r.error]
    correct_abstentions = [r for r in unanswerable if r.abstained]
    unsupported = [r for r in results if r.unsupported_answer]
    missed = [r for r in results if r.missed_answer]

    all_abstentions = [r for r in results if r.abstained]
    abstention_precision = (
        len(correct_abstentions) / len(all_abstentions) if all_abstentions else None
    )
    abstention_recall = (
        len(correct_abstentions) / len(unanswerable) if unanswerable else None
    )

    cited = [r for r in answered_correctly if r.pages_cited]
    page_checks = [r for r in results if r.min_page_ok is not None]
    keyword_cases = [r for r in results if r.keywords_hit or r.keywords_missed]

    latencies = [r.latency_ms for r in results if r.latency_ms]

    return {
        "cases": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "answer_rate_on_answerable": _ratio(len(answered_correctly), len(answerable)),
        "citation_presence": _ratio(len(cited), len(answered_correctly)),
        "later_page_evidence_ok": _ratio(
            len([r for r in page_checks if r.min_page_ok]), len(page_checks)
        ),
        "keyword_coverage": _ratio(
            sum(len(r.keywords_hit) for r in keyword_cases),
            sum(len(r.keywords_hit) + len(r.keywords_missed) for r in keyword_cases),
        ),
        "abstention_precision": abstention_precision,
        "abstention_recall": abstention_recall,
        "unsupported_answer_rate": _ratio(len(unsupported), len(unanswerable)),
        "missed_answer_rate": _ratio(len(missed), len(answerable)),
        "errors": len([r for r in results if r.error]),
        "latency_ms_median": int(statistics.median(latencies)) if latencies else None,
        "latency_ms_p95": (
            int(sorted(latencies)[int(len(latencies) * 0.95) - 1])
            if len(latencies) >= 2 else (latencies[0] if latencies else None)
        ),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(numerator / denominator, 3)


def print_report(results: list[CaseResult], summary: dict) -> None:
    print()
    print(f"{'case':22} {'kind':13} {'expect':11} {'got':11} {'pages':10} {'ms':>6}")
    print("-" * 82)
    for r in results:
        expect = "abstain" if not r.answerable else "answer"
        got = "ERROR" if r.error else ("abstained" if r.abstained else "answered")
        flag = ""
        if r.unsupported_answer:
            flag = "  <-- FABRICATED"
        elif r.missed_answer:
            flag = "  <-- missed"
        elif r.min_page_ok is False:
            flag = "  <-- shallow"
        pages = ",".join(str(p) for p in r.pages_cited[:4]) or "-"
        print(f"{r.case_id:22} {r.kind:13} {expect:11} {got:11} {pages:10} "
              f"{r.latency_ms:>6}{flag}")

    print()
    print("summary")
    print("-" * 82)
    for key, value in summary.items():
        print(f"  {key:32} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--holdout", action="store_true",
                        help="Single-use gate mode. Refuses to re-run.")
    parser.add_argument("--sweep", default=None,
                        help="VAR=v1,v2,v3 -- re-run for each value (calibration only).")
    parser.add_argument("--force", action="store_true",
                        help="Override the holdout single-use guard. Read eval/README.md first.")
    args = parser.parse_args()

    load_dotenv()
    fixture_path = Path(args.fixture).resolve()
    if not fixture_path.exists():
        print(f"No such fixture: {fixture_path}", file=sys.stderr)
        return 2
    fixture = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))

    if args.holdout and args.sweep:
        print(
            "Refusing to sweep a holdout fixture. Sweeping IS tuning, and tuning "
            "on the holdout is exactly what it exists to prevent.",
            file=sys.stderr,
        )
        return 2

    REPORTS.mkdir(exist_ok=True)

    sweep_values: list[tuple[str, str]] = []
    if args.sweep:
        name, _, raw = args.sweep.partition("=")
        sweep_values = [(name.strip(), v.strip()) for v in raw.split(",") if v.strip()]
    if not sweep_values:
        sweep_values = [("", "")]

    exit_code = 0
    for name, value in sweep_values:
        if name:
            os.environ[name] = value
        try:
            settings = load_settings()
        except ConfigError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2

        digest = config_hash(settings)
        stem = fixture_path.stem
        report_path = REPORTS / f"{stem}_{digest}.json"

        if args.holdout and report_path.exists() and not args.force:
            print(
                f"REFUSING TO RUN.\n\n"
                f"A holdout report already exists for this configuration:\n"
                f"  {report_path}\n\n"
                f"A holdout is single-use. Once you have seen its result, running\n"
                f"it again -- especially after changing a threshold -- measures how\n"
                f"hard you tuned, not how well the system works.\n\n"
                f"If the previous run failed: fold these cases into\n"
                f"calibration_fixture.yaml and write a new holdout_v2.yaml.\n"
                f"See eval/README.md.",
                file=sys.stderr,
            )
            return 3

        label = f"{name}={value}" if name else "current configuration"
        print()
        print("=" * 82)
        print(f"fixture      {fixture_path.name}")
        print(f"mode         {'HOLDOUT (single use)' if args.holdout else 'calibration'}")
        print(f"config_hash  {digest}   ({label})")
        print(f"chat={settings.chat_model}  embed={settings.embed_model}"
              f"/{settings.embed_dimension}  rerank="
              f"{settings.rerank_model if settings.enable_rerank else 'off'}")
        print(f"thresholds   vector>={settings.vector_min_score} "
              f"rerank>={settings.rerank_min_score} "
              f"min_chunks={settings.min_evidence_chunks}")
        print("=" * 82)

        results = run_fixture(settings, fixture, fixture_path.parent)
        if not results:
            print("No cases ran -- check the document paths.", file=sys.stderr)
            return 2

        summary = summarize(results)
        print_report(results, summary)

        report_path.write_text(
            json.dumps(
                {
                    "fixture": fixture_path.name,
                    "holdout_id": fixture.get("holdout_id"),
                    "mode": "holdout" if args.holdout else "calibration",
                    "config_hash": digest,
                    "config": settings.public_summary(),
                    "thresholds": {
                        "vector_min_score": settings.vector_min_score,
                        "rerank_min_score": settings.rerank_min_score,
                        "min_evidence_chunks": settings.min_evidence_chunks,
                        "chunk_chars": settings.chunk_chars,
                        "chunk_overlap": settings.chunk_overlap,
                    },
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "summary": summary,
                    "cases": [vars(r) for r in results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nreport -> {report_path}")

        if args.holdout:
            fabricated = summary["unsupported_answer_rate"]
            if fabricated is not None and fabricated > 0.0:
                print(
                    f"\nHOLDOUT FAILED: unsupported_answer_rate="
                    f"{fabricated} (must be 0.0).\n"
                    f"This fixture is now retired -- see eval/README.md.",
                    file=sys.stderr,
                )
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
