#!/usr/bin/env python3
"""Audit one explicit ACCAD v27/v28 split with the frozen v29 contact gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from accad_g1.contact_feasibility import (
    DEFAULT_OUTPUT_ROOT,
    build_contact_feasibility_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exactly one v27/v28 split for centroidal-ZMP and preview-capture "
            "feasibility. Audit-only is the default. There is no implicit all-splits "
            "mode, and test data are opened only by an explicit --split test."
        )
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
        help="The only split whose manifests and NPZ files may be opened.",
    )
    parser.add_argument(
        "--v27-manifest",
        type=Path,
        default=None,
        help="Optional Gate-3 v27 manifest for the selected split.",
    )
    parser.add_argument(
        "--v28-manifest",
        type=Path,
        default=None,
        help="Optional dynamic v28 manifest for the selected split.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Pipeline-local audit output directory (default: work/contact_v29).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace audit JSON only; source v27/v28 archives are never modified.",
    )
    parser.add_argument(
        "--publish-manifest",
        action="store_true",
        help=(
            "Opt in to publishing a training-compatible eligible-only manifest. "
            "Without this flag, only policy/plan/report audit JSON is written."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_contact_feasibility_split(
        split=args.split,
        v27_manifest_path=args.v27_manifest,
        v28_manifest_path=args.v28_manifest,
        output_root=args.output_root,
        force=args.force,
        publish_manifest=args.publish_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
