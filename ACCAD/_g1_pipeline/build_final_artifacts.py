#!/usr/bin/env python3
"""Publish the final checksummed ACCAD-to-G1 experiment snapshot."""

from __future__ import annotations

import argparse
import json

from accad_g1.final_artifacts import FinalArtifactError, build_final_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-run",
        required=True,
        help="Final training run directory, or its run_summary.json, below ACCAD.",
    )
    parser.add_argument(
        "--validation-report",
        required=True,
        help="Validation suite_evaluation_summary.json below ACCAD.",
    )
    parser.add_argument(
        "--test-report",
        required=True,
        help="Test suite_evaluation_summary.json below ACCAD.",
    )
    parser.add_argument(
        "--representative-report",
        required=True,
        help="Held-out representative evaluation_summary.json with its MP4 below ACCAD.",
    )
    parser.add_argument(
        "--professor-report",
        required=True,
        help="Completed professor-facing Markdown report below ACCAD.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = build_final_artifacts(
            training_run=args.training_run,
            validation_report=args.validation_report,
            test_report=args.test_report,
            representative_evaluation_report=args.representative_report,
            professor_report=args.professor_report,
        )
    except FinalArtifactError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
