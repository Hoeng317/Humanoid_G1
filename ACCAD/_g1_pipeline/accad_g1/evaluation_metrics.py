"""CPU-only metrics and acceptance contract for tracking-suite reports."""

from __future__ import annotations

from typing import Any, Mapping


SUITE_ACCEPTANCE_SCHEMA_VERSION = "accad_g1_suite_acceptance_v1"
DEFAULT_SUITE_THRESHOLDS = {
    "minimum_macro_completion_ratio": 0.90,
    "maximum_macro_joint_mae_rad": 0.20,
    "maximum_macro_root_position_error_m": 0.20,
    "maximum_macro_tracking_body_position_error_m": 0.15,
    "minimum_pooled_contact_f1": 0.80,
    "maximum_weighted_raw_action_clip_fraction": 0.05,
    "maximum_nonfinite_terminations": 0,
    "maximum_joint_limit_terminations": 0,
}


def contact_precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Return JSON-safe binary contact scores from non-negative counts."""

    values = {"tp": tp, "fp": fp, "fn": fn}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return {
        "true_positive_frames": tp,
        "false_positive_frames": fp,
        "false_negative_frames": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def build_suite_acceptance(
    aggregate: Mapping[str, Any],
    *,
    manifest_promotion_verified: bool,
    thresholds: Mapping[str, float | int] = DEFAULT_SUITE_THRESHOLDS,
) -> dict[str, Any]:
    """Evaluate the professor-report acceptance contract from suite aggregates.

    Every check stores the exact JSON path of its input. Missing or non-finite
    values fail closed instead of being silently omitted.
    """

    required_thresholds = set(DEFAULT_SUITE_THRESHOLDS)
    if set(thresholds) != required_thresholds:
        missing = sorted(required_thresholds - set(thresholds))
        extra = sorted(set(thresholds) - required_thresholds)
        raise ValueError(f"acceptance threshold keys differ; missing={missing}, extra={extra}")
    if not isinstance(manifest_promotion_verified, bool):
        raise ValueError("manifest_promotion_verified must be boolean")

    macro = aggregate.get("macro_motion_metrics", {})
    weighted = aggregate.get("weighted_frame_or_element_metrics", {})
    contact = aggregate.get("aggregate_foot_contact", {}).get("overall", {})
    terminations = aggregate.get("termination_counts", {})
    motion_count = _finite_number(aggregate.get("motion_count"))
    success_count = _finite_number(aggregate.get("successful_motion_end_episodes"))

    def minimum_check(path: str, value: Any, threshold_name: str) -> dict[str, Any]:
        numeric = _finite_number(value)
        threshold = float(thresholds[threshold_name])
        return {
            "json_path": path,
            "comparison": ">=",
            "threshold": threshold,
            "value": numeric,
            "passed": bool(numeric is not None and numeric >= threshold),
        }

    def maximum_check(path: str, value: Any, threshold_name: str) -> dict[str, Any]:
        numeric = _finite_number(value)
        threshold = float(thresholds[threshold_name])
        return {
            "json_path": path,
            "comparison": "<=",
            "threshold": threshold,
            "value": numeric,
            "passed": bool(numeric is not None and numeric <= threshold),
        }

    checks = {
        "checksum_verified_promoted_manifest": {
            "json_path": "motion_input.manifest_promotion_verified",
            "comparison": "==",
            "expected": True,
            "value": manifest_promotion_verified,
            "passed": manifest_promotion_verified,
        },
        "all_first_episodes_successful": {
            "json_paths": [
                "aggregate.successful_motion_end_episodes",
                "aggregate.motion_count",
            ],
            "comparison": "==",
            "expected": motion_count,
            "value": success_count,
            "passed": bool(
                motion_count is not None
                and motion_count > 0
                and success_count == motion_count
            ),
        },
        "macro_completion_ratio": minimum_check(
            "aggregate.completion_ratio_mean",
            aggregate.get("completion_ratio_mean"),
            "minimum_macro_completion_ratio",
        ),
        "macro_joint_mae_rad": maximum_check(
            "aggregate.macro_motion_metrics.mean_absolute_joint_error_rad",
            macro.get("mean_absolute_joint_error_rad"),
            "maximum_macro_joint_mae_rad",
        ),
        "macro_root_position_error_m": maximum_check(
            "aggregate.macro_motion_metrics.mean_root_position_error_m",
            macro.get("mean_root_position_error_m"),
            "maximum_macro_root_position_error_m",
        ),
        "macro_tracking_body_position_error_m": maximum_check(
            "aggregate.macro_motion_metrics.mean_tracking_body_position_error_m",
            macro.get("mean_tracking_body_position_error_m"),
            "maximum_macro_tracking_body_position_error_m",
        ),
        "pooled_contact_f1": minimum_check(
            "aggregate.aggregate_foot_contact.overall.f1",
            contact.get("f1"),
            "minimum_pooled_contact_f1",
        ),
        "weighted_raw_action_clip_fraction": maximum_check(
            "aggregate.weighted_frame_or_element_metrics.raw_action_clip_fraction",
            weighted.get("raw_action_clip_fraction"),
            "maximum_weighted_raw_action_clip_fraction",
        ),
        "nonfinite_terminations": maximum_check(
            "aggregate.termination_counts.nonfinite",
            terminations.get("nonfinite"),
            "maximum_nonfinite_terminations",
        ),
        "joint_limit_terminations": maximum_check(
            "aggregate.termination_counts.joint_limits",
            terminations.get("joint_limits"),
            "maximum_joint_limit_terminations",
        ),
    }
    return {
        "schema_version": SUITE_ACCEPTANCE_SCHEMA_VERSION,
        "thresholds": dict(thresholds),
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }
