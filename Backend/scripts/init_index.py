"""Create the Pinecone index. Idempotent, and destructive only on demand.

Replaces the previous ``manage_index.py``, which (a) carried a live API key
hard-coded in tracked source and (b) called ``delete_index`` unconditionally on
every run, with no confirmation. Running it once wiped the index.

Usage::

    python scripts/init_index.py                     # create if missing
    python scripts/init_index.py --describe          # inspect only
    python scripts/init_index.py --delete --yes-i-am-sure

Deletion needs both flags. One flag is a typo; two is a decision.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402
from pinecone import Pinecone, ServerlessSpec  # noqa: E402

from rag_core.config import load_settings  # noqa: E402
from rag_core.errors import ConfigError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true",
                        help="Print the index description and exit.")
    parser.add_argument("--delete", action="store_true",
                        help="Delete the index. Requires --yes-i-am-sure.")
    parser.add_argument("--yes-i-am-sure", action="store_true",
                        help="Confirm a destructive operation.")
    args = parser.parse_args()

    load_dotenv()
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    pc = Pinecone(api_key=settings.pinecone_api_key)
    name = settings.pinecone_index
    exists = pc.has_index(name)

    if args.describe:
        if not exists:
            print(f"Index {name!r} does not exist.")
            return 1
        _print_description(pc, name, settings)
        return 0

    if args.delete:
        if not args.yes_i_am_sure:
            print(
                f"Refusing to delete index {name!r} without --yes-i-am-sure.\n"
                f"This permanently destroys every stored vector.",
                file=sys.stderr,
            )
            return 2
        if not exists:
            print(f"Index {name!r} does not exist; nothing to delete.")
            return 0
        print(f"Deleting index {name!r} …")
        pc.delete_index(name)
        print("Deleted.")
        return 0

    if exists:
        print(f"Index {name!r} already exists.")
        _print_description(pc, name, settings)
        # Report a mismatch rather than silently recreating: recreating would
        # destroy live data, and the dimension of an index cannot be changed.
        described = pc.describe_index(name)
        dimension = getattr(described, "dimension", None)
        if dimension is not None and int(dimension) != settings.embed_dimension:
            print(
                f"\nMISMATCH: index dimension {dimension} != "
                f"COHERE_EMBED_DIMENSION {settings.embed_dimension}.\n"
                f"Vectors cannot be migrated between dimensions. Either set\n"
                f"COHERE_EMBED_DIMENSION={dimension}, or choose a new\n"
                f"PINECONE_INDEX name and create it fresh.",
                file=sys.stderr,
            )
            return 1
        return 0

    print(
        f"Creating serverless index {name!r} "
        f"(dim={settings.embed_dimension}, metric=cosine, "
        f"{settings.pinecone_cloud}/{settings.pinecone_region}) …"
    )
    pc.create_index(
        name=name,
        dimension=settings.embed_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        ),
    )

    # Creation is asynchronous; wait so a following smoke test does not race it.
    for _ in range(60):
        if pc.has_index(name):
            described = pc.describe_index(name)
            if getattr(getattr(described, "status", None), "ready", False):
                break
        time.sleep(2)

    print("Created.")
    _print_description(pc, name, settings)
    return 0


def _print_description(pc: Pinecone, name: str, settings) -> None:
    described = pc.describe_index(name)
    spec = getattr(described, "spec", None)
    serverless = getattr(spec, "serverless", None)
    print(f"  name       {getattr(described, 'name', name)}")
    print(f"  dimension  {getattr(described, 'dimension', '?')}")
    print(f"  metric     {getattr(described, 'metric', '?')}")
    print(f"  cloud      {getattr(serverless, 'cloud', '?')}")
    print(f"  region     {getattr(serverless, 'region', '?')}")
    print(f"  namespace  {settings.pinecone_namespace} (single, fixed)")


if __name__ == "__main__":
    raise SystemExit(main())
