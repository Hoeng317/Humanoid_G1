from __future__ import annotations

from pathlib import Path

import pytest

from accad_g1.cli import build_parser
from accad_g1.tensorboard_dashboard import (
    DashboardSection,
    _extract_retarget_validation_metrics,
    _load_selected_training_series,
    _markdown_table,
    _statistics,
    _validated_dashboard_output,
    _write_section,
)


def test_statistics_are_finite_and_use_expected_percentiles() -> None:
    result = _statistics([1.0, 2.0, 3.0, 4.0], label="sample")
    assert result["count"] == 4
    assert result["minimum"] == 1.0
    assert result["median"] == 2.5
    assert result["maximum"] == 4.0
    assert result["p95"] == pytest.approx(3.85)

    with pytest.raises(ValueError, match="finite"):
        _statistics([1.0, float("nan")], label="bad")


def test_markdown_table_escapes_cells_and_checks_width() -> None:
    table = _markdown_table(["name", "value"], [["a|b", "line1\nline2"]])
    assert "a\\|b" in table
    assert "line1<br>line2" in table
    with pytest.raises(ValueError, match="row length"):
        _markdown_table(["one", "two"], [[1]])


def test_dashboard_output_is_confined_to_generated_tensorboard_root(
    pipeline_tmp_path: Path,
) -> None:
    work_root = pipeline_tmp_path / "work"
    allowed = work_root / "tensorboard" / "overview"
    assert _validated_dashboard_output(allowed, work_root=work_root) == allowed.resolve()

    with pytest.raises(ValueError, match="below"):
        _validated_dashboard_output(
            work_root / "reports" / "not-a-dashboard", work_root=work_root
        )
    with pytest.raises(ValueError, match="child"):
        _validated_dashboard_output(work_root / "tensorboard", work_root=work_root)


def test_retarget_metric_extraction_reduces_sides_and_channels() -> None:
    def side(penetration: float, offset: float) -> dict:
        return {
            "maximum_penetration_m": penetration,
            "stance_channels": {
                part: {
                    "maximum_consecutive_stance_point_step_xy_m": offset + index,
                    "maximum_stance_run_drift_xy_m": offset + index + 0.1,
                    "p95_consecutive_stance_point_speed_xy_m_s": offset
                    + index
                    + 0.2,
                }
                for index, part in enumerate(("heel", "toe"))
            },
        }

    validation = {
        "metrics": {
            "joint_limits": {
                "velocity": {"maximum_limit_ratio": 0.4},
                "acceleration": {
                    "p95_abs_rad_s2": 12.0,
                    "maximum_abs_rad_s2": 30.0,
                },
                "hard": {"minimum_margin_rad": 0.05},
            },
            "pinocchio_fk": {
                "maximum_body_position_error_m": 1.0e-7,
                "maximum_body_orientation_error_rad": 2.0e-7,
            },
            "foot_contact_geometry": {
                "left": side(0.001, 0.01),
                "right": side(0.002, 0.02),
            },
            "action_reachability": {
                "reference_residual": {"full_clip_box_violation_fraction": 0.1}
            },
        }
    }
    metrics = _extract_retarget_validation_metrics(validation)
    assert metrics["max_foot_penetration_m"] == 0.002
    assert metrics["max_stance_step_m"] == 1.02
    assert metrics["max_stance_drift_m"] == pytest.approx(1.12)
    assert metrics["p95_stance_speed_mps"] == pytest.approx(1.22)
    assert metrics["max_joint_velocity_limit_ratio"] == 0.4


def test_section_writer_creates_scalar_histogram_series_and_text(
    pipeline_tmp_path: Path,
) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_dir = pipeline_tmp_path / "writer"
    section = DashboardSection(
        run_name="synthetic",
        scalars={"static/value": 3.0},
        histograms={"distribution/value": [1.0, 2.0, 3.0]},
        series={"curve/value": [(2, 4.0, None), (3, 5.0, None)]},
        texts={"00_read_me": "# synthetic"},
    )
    _write_section(section, log_dir)

    accumulator = EventAccumulator(
        str(log_dir), size_guidance={"scalars": 0, "histograms": 0, "tensors": 0}
    )
    accumulator.Reload()
    tags = accumulator.Tags()
    assert {"static/value", "curve/value"}.issubset(tags["scalars"])
    assert "distribution/value" in tags["histograms"]
    assert "00_read_me/text_summary" in tags["tensors"]
    assert [event.step for event in accumulator.Scalars("curve/value")] == [2, 3]


def test_selected_training_series_reads_only_required_curves(
    pipeline_tmp_path: Path,
) -> None:
    from torch.utils.tensorboard import SummaryWriter

    run_dir = pipeline_tmp_path / "selected-run"
    required = (
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Metrics/motion/joint_position_error",
        "Metrics/motion/root_position_error",
        "Metrics/motion/tracking_body_position_error",
        "Episode_Termination/motion_end",
        "Episode_Termination/fall",
        "Episode_Termination/tracking_divergence",
    )
    writer = SummaryWriter(str(run_dir))
    for tag_index, tag in enumerate(required):
        writer.add_scalar(tag, float(tag_index), 7)
        writer.add_scalar(tag, float(tag_index + 1), 8)
    writer.add_scalar("Loss/not_selected", 999.0, 8)
    writer.close()

    series, event_files = _load_selected_training_series(run_dir)
    assert len(series) == len(required)
    assert event_files
    assert "training/mean_reward" in series
    assert [point[0] for point in series["training/mean_reward"]] == [7, 8]
    assert all("not_selected" not in tag for tag in series)


def test_cli_exposes_independent_tensorboard_command() -> None:
    args = build_parser().parse_args(["tensorboard", "--force"])
    assert args.command == "tensorboard"
    assert args.force is True
    assert args.output is None
