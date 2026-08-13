#!/usr/bin/env python3
"""Build one explicit ACCAD G1 dynamic-feasibility split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from accad_g1.dynamic_retime import (
    DEFAULT_OUTPUT_ROOT,
    build_dynamic_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and derive exactly one promoted G1 split. There is no implicit "
            "all-splits mode; sealed test NPZs are opened only with --split test."
        )
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
        help="Explicit split to open and process.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional promoted v27 manifest for the selected split.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Pipeline-local output root (default: work/dynamic).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace derived outputs only; immutable v27 inputs are never overwritten.",
    )
    parser.add_argument(
        "--skip-gate3-revalidation",
        action="store_true",
        help="Debug only: publish an explicitly unvalidated derived manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_dynamic_split(
        split=args.split,
        manifest_path=args.manifest,
        output_root=args.output_root,
        force=args.force,
        revalidate=not args.skip_gate3_revalidation,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
