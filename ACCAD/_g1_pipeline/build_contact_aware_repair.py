#!/usr/bin/env python3
"""Quarantined experimental G1 v30 contact-aware repair entry point.

The offline optimizer did not improve the frozen validation witness and its
prototype validation-unlock chain is not suitable for production evidence.
Keep the implementation for research audit, but fail closed here so it cannot
publish train/validation/test artifacts or consume the sealed test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from accad_g1.contact_aware_repair import DEFAULT_OUTPUT_ROOT


PRODUCTION_DISABLED_REASON = (
    "G1 v30 contact-aware repair is quarantined: the frozen validation motion "
    "showed identity/no-improvement, 24/103 train motions were outside the "
    "fixed seed trust region, and the experimental validation witness chain "
    "has not been certified for production publication. The authoritative "
    "pipeline remains v27/v28; no v30 split, receipt, or test artifact is built."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair exactly one v28 split with contact-locked G1 sequence IK. "
            "Train freezes the algorithm; validation reuses it unchanged. Test "
            "requires an explicit successful-validation witness and is consumed once."
        )
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--v27-manifest", type=Path, default=None)
    parser.add_argument("--v28-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace train/validation v30 outputs; immutable inputs and test cannot be forced.",
    )
    parser.add_argument(
        "--validation-pass-witness",
        type=Path,
        default=None,
        help="Required only for test: successful g1_validation_report.json.",
    )
    parser.add_argument(
        "--validation-pass-witness-sha256",
        default=None,
        help="Required only for test: exact SHA-256 of the validation witness.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = {
        "schema_version": "accad_g1_contact_repair_quarantine_v1",
        "status": "production_disabled",
        "requested_split": args.split,
        "reason": PRODUCTION_DISABLED_REASON,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
