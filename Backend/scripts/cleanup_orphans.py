"""Retention cleanup. Manual, dry-run by default.

**This script does not run itself.** There is no scheduler in this deployment, so
storage is not self-bounding -- it stays bounded only as far as this is actually
run. See DEPLOYMENT.md "Cleanup cadence" for the recommended process:

1. dry-run before the first public deploy, to sanity-check the criteria;
2. an explicit confirmed run after each demo period, or weekly;
3. a second dry-run afterwards to verify eligible records are gone.

Eligibility is limited to what is **observable from stored metadata**:

* ``created_at`` older than the cutoff;
* an ``ingest_schema_version`` this build no longer supports;
* explicitly supplied document ids.

It deliberately makes no attempt to spare "the document a user is currently
looking at". Streamlit session memory is not visible from here, and no persistent
registry exists, so that cannot be known. The consequence is accepted and handled
gracefully: a session left open past the retention window gets a 404 on its next
question and the UI prompts a fresh upload.

Usage::

    python scripts/cleanup_orphans.py                        # dry run
    python scripts/cleanup_orphans.py --delete --yes-i-am-sure
    python scripts/cleanup_orphans.py --document-id <id> --delete --yes-i-am-sure
    python scripts/cleanup_orphans.py --hours 1              # tighter cutoff
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from rag_core import store  # noqa: E402
from rag_core.clients import CLIENTS  # noqa: E402
from rag_core.config import load_settings  # noqa: E402
from rag_core.errors import ConfigError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=None,
                        help="Override RETENTION_HOURS for this run.")
    parser.add_argument("--document-id", action="append", default=[],
                        help="Target a specific document id. Repeatable.")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete. Requires --yes-i-am-sure.")
    parser.add_argument("--yes-i-am-sure", action="store_true",
                        help="Confirm a destructive operation.")
    args = parser.parse_args()

    load_dotenv()
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    hours = args.hours if args.hours is not None else settings.retention_hours
    cutoff = int(time.time()) - hours * 3600
    request_id = f"cleanup-{int(time.time())}"

    print(f"request_id       {request_id}")
    print(f"index            {settings.pinecone_index}")
    print(f"namespace        {settings.pinecone_namespace}")
    print(f"retention        {hours}h")
    print(f"cutoff (epoch)   {cutoff}")
    print(f"schema supported {sorted(settings.supported_schema_versions)}")
    print(f"mode             {'DELETE' if args.delete else 'DRY RUN'}")
    print()

    index = CLIENTS.index(settings)

    if args.document_id:
        return _delete_explicit(settings, args, index, request_id)

    # Survey the namespace, since delete-by-filter reports no counts and we want
    # to know what we are about to remove before removing it.
    survey = _survey(index, settings, cutoff)
    _report(survey)

    eligible = survey["eligible_ids"]
    if not eligible:
        print("Nothing eligible for cleanup.")
        return 0

    if not args.delete:
        print(f"\nDRY RUN: {len(eligible)} vector(s) across "
              f"{len(survey['eligible_docs'])} document(s) would be deleted.")
        print("Re-run with --delete --yes-i-am-sure to proceed.")
        return 0

    if not args.yes_i_am_sure:
        print("Refusing to delete without --yes-i-am-sure.", file=sys.stderr)
        return 2

    print(f"\nDeleting {len(survey['eligible_docs'])} document(s) older than "
          f"{hours}h …")
    store.delete_older_than(settings, cutoff)
    for version in survey["unsupported_versions"]:
        _delete_unsupported_version(settings, index, version)

    print("Delete accepted (Pinecone deletions are eventually consistent and "
          "report no count).")
    print("\nRe-run this script without --delete to verify the records are gone.")
    return 0


def _survey(index, settings, cutoff: int) -> dict:
    """Enumerate the namespace and classify every record."""
    total = 0
    eligible_ids: list[str] = []
    eligible_docs: set[str] = set()
    live_docs: set[str] = set()
    unsupported_versions: Counter = Counter()
    reasons: Counter = Counter()

    for page in index.list(namespace=settings.pinecone_namespace):
        ids = getattr(page, "ids", page) or []
        ids = [str(i) for i in ids]
        if not ids:
            continue
        fetched = index.fetch(ids=ids, namespace=settings.pinecone_namespace)
        vectors = getattr(fetched, "vectors", None) or {}
        for vid, record in vectors.items():
            total += 1
            meta = _metadata(record)
            document_id = str(meta.get("document_id", "?"))
            created_at = meta.get("created_at")
            version = meta.get("ingest_schema_version")

            if version is not None and int(version) not in settings.supported_schema_versions:
                eligible_ids.append(vid)
                eligible_docs.add(document_id)
                unsupported_versions[int(version)] += 1
                reasons["unsupported_schema"] += 1
            elif created_at is not None and int(created_at) < cutoff:
                eligible_ids.append(vid)
                eligible_docs.add(document_id)
                reasons["older_than_cutoff"] += 1
            elif created_at is None:
                # No timestamp: pre-dates this schema. Treat as eligible, since
                # it cannot be aged out any other way.
                eligible_ids.append(vid)
                eligible_docs.add(document_id)
                reasons["missing_created_at"] += 1
            else:
                live_docs.add(document_id)

    return {
        "total": total,
        "eligible_ids": eligible_ids,
        "eligible_docs": eligible_docs,
        "live_docs": live_docs,
        "unsupported_versions": sorted(unsupported_versions),
        "reasons": reasons,
    }


def _report(survey: dict) -> None:
    print(f"vectors in namespace   {survey['total']}")
    print(f"documents retained     {len(survey['live_docs'])}")
    print(f"documents eligible     {len(survey['eligible_docs'])}")
    print(f"vectors eligible       {len(survey['eligible_ids'])}")
    if survey["reasons"]:
        print("matched criteria:")
        for reason, count in survey["reasons"].most_common():
            print(f"  {reason:22} {count}")
    if survey["eligible_docs"]:
        print("affected document prefixes:")
        for document_id in sorted(survey["eligible_docs"])[:20]:
            print(f"  {document_id}")
        if len(survey["eligible_docs"]) > 20:
            print(f"  … and {len(survey['eligible_docs']) - 20} more")


def _delete_explicit(settings, args, index, request_id: str) -> int:
    print(f"explicit targets: {len(args.document_id)}")
    for document_id in args.document_id:
        ids = sorted(store.list_document_ids(settings, document_id))
        print(f"  {document_id}  ({len(ids)} vectors)")
        if not args.delete:
            continue
        if not args.yes_i_am_sure:
            print("Refusing to delete without --yes-i-am-sure.", file=sys.stderr)
            return 2
        if ids:
            store.delete_ids(settings, ids)
            print("    deleted by exact id")
    if not args.delete:
        print("\nDRY RUN. Re-run with --delete --yes-i-am-sure to proceed.")
    return 0


def _delete_unsupported_version(settings, index, version: int) -> None:
    index.delete(
        filter={"ingest_schema_version": {"$eq": version}},
        namespace=settings.pinecone_namespace,
    )
    print(f"  removed records with schema version {version}")


def _metadata(record) -> dict:
    if isinstance(record, dict):
        return record.get("metadata", {}) or {}
    return getattr(record, "metadata", {}) or {}


if __name__ == "__main__":
    raise SystemExit(main())
