from __future__ import annotations

from pathlib import Path
import sys

import pytest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from accad_g1.evaluation_metrics import (  # noqa: E402
    build_suite_acceptance,
    contact_precision_recall_f1,
)


def _passing_aggregate() -> dict:
    return {
        "motion_count": 10,
        "successful_motion_end_episodes": 10,
        "completion_ratio_mean": 0.95,
        "macro_motion_metrics": {
            "mean_absolute_joint_error_rad": 0.19,
            "mean_root_position_error_m": 0.19,
            "mean_tracking_body_position_error_m": 0.14,
        },
        "aggregate_foot_contact": {"overall": {"f1": 0.81}},
        "weighted_frame_or_element_metrics": {"raw_action_clip_fraction": 0.04},
        "termination_counts": {"nonfinite": 0, "joint_limits": 0},
    }


def test_contact_scores_include_pooled_f1_and_reject_invalid_counts() -> None:
    scores = contact_precision_recall_f1(8, 1, 1)
    assert scores["precision"] == pytest.approx(8 / 9)
    assert scores["recall"] == pytest.approx(8 / 9)
    assert scores["f1"] == pytest.approx(8 / 9)
    assert contact_precision_recall_f1(0, 0, 0)["f1"] is None
    with pytest.raises(ValueError, match="non-negative"):
        contact_precision_recall_f1(-1, 0, 0)


def test_suite_acceptance_exposes_exact_paths_and_passes_all_thresholds() -> None:
    acceptance = build_suite_acceptance(
        _passing_aggregate(), manifest_promotion_verified=True
    )
    assert acceptance["passed"] is True
    assert (
        acceptance["checks"]["pooled_contact_f1"]["json_path"]
        == "aggregate.aggregate_foot_contact.overall.f1"
    )
    assert (
        acceptance["checks"]["weighted_raw_action_clip_fraction"]["json_path"]
        == "aggregate.weighted_frame_or_element_metrics.raw_action_clip_fraction"
    )


@pytest.mark.parametrize(
    ("section", "key", "value", "check"),
    [
        (None, "successful_motion_end_episodes", 9, "all_first_episodes_successful"),
        (None, "completion_ratio_mean", 0.89, "macro_completion_ratio"),
        ("macro_motion_metrics", "mean_absolute_joint_error_rad", 0.21, "macro_joint_mae_rad"),
        ("macro_motion_metrics", "mean_root_position_error_m", 0.21, "macro_root_position_error_m"),
        (
            "macro_motion_metrics",
            "mean_tracking_body_position_error_m",
            0.16,
            "macro_tracking_body_position_error_m",
        ),
        ("aggregate_foot_contact", "overall", {"f1": 0.79}, "pooled_contact_f1"),
        (
            "weighted_frame_or_element_metrics",
            "raw_action_clip_fraction",
            0.06,
            "weighted_raw_action_clip_fraction",
        ),
        ("termination_counts", "nonfinite", 1, "nonfinite_terminations"),
        ("termination_counts", "joint_limits", 1, "joint_limit_terminations"),
    ],
)
def test_each_failed_threshold_fails_closed(
    section: str | None, key: str, value: object, check: str
) -> None:
    aggregate = _passing_aggregate()
    target = aggregate if section is None else aggregate[section]
    target[key] = value
    acceptance = build_suite_acceptance(
        aggregate, manifest_promotion_verified=True
    )
    assert acceptance["checks"][check]["passed"] is False
    assert acceptance["passed"] is False


def test_missing_metric_fails_closed() -> None:
    aggregate = _passing_aggregate()
    del aggregate["macro_motion_metrics"]["mean_root_position_error_m"]
    acceptance = build_suite_acceptance(
        aggregate, manifest_promotion_verified=True
    )
    assert acceptance["checks"]["macro_root_position_error_m"]["value"] is None
    assert acceptance["passed"] is False


def test_direct_motion_without_promoted_manifest_is_not_finalizable() -> None:
    acceptance = build_suite_acceptance(
        _passing_aggregate(), manifest_promotion_verified=False
    )
    check = acceptance["checks"]["checksum_verified_promoted_manifest"]
    assert check["json_path"] == "motion_input.manifest_promotion_verified"
    assert check["passed"] is False
    assert acceptance["passed"] is False
