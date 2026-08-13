#!/usr/bin/env python3
"""Record a terminal Gate-5 validation failure without opening the test split."""

from __future__ import annotations

import argparse
import json

from accad_g1.gate5_outcome import Gate5OutcomeError, build_gate5_terminal_outcome
from accad_g1.paths import ACCAD_ROOT, PIPELINE_ROOT, DEFAULT_WORK_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-run",
        required=True,
        help="Completed production run directory or its run_summary.json below ACCAD.",
    )
    parser.add_argument(
        "--validation-report",
        required=True,
        help="Completed official validation suite_evaluation_summary.json below ACCAD.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output; must be work/reports/gate5_terminal_outcome.json by default.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = build_gate5_terminal_outcome(
            training_run=args.training_run,
            validation_report=args.validation_report,
            output=args.output,
            accad_root=ACCAD_ROOT,
            pipeline_root=PIPELINE_ROOT,
            work_root=DEFAULT_WORK_ROOT,
        )
    except Gate5OutcomeError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
