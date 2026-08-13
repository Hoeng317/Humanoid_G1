#!/usr/bin/env python3
"""Publish training manifests from a completed ACCAD→G1 batch snapshot."""

from __future__ import annotations

import argparse
import json

from accad_g1.gate3_manifests import (
    DEFAULT_BATCH_SUMMARY,
    DEFAULT_ITEMS_JSONL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFERENCE_ROOT,
    build_gate3_manifests,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-jsonl", default=str(DEFAULT_ITEMS_JSONL))
    parser.add_argument("--batch-summary", default=str(DEFAULT_BATCH_SUMMARY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Debug only: promote completed records from an intentionally partial batch.",
    )
    parser.add_argument(
        "--trust-batch-reports",
        action="store_true",
        help="Debug only: skip fresh independent Gate-3 validation of every archive.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_gate3_manifests(
        items_jsonl=args.items_jsonl,
        batch_summary=args.batch_summary,
        output_dir=args.output_dir,
        reference_root=args.reference_root,
        require_complete_batch=not args.allow_partial,
        revalidate=not args.trust_batch_reports,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
