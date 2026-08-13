"""Build a compact TensorBoard dashboard from existing ACCAD/G1 artifacts.

This module is deliberately a read-only post-processor.  It never regenerates
motions, opens the sealed dynamic/TEST branch, trains a policy, or changes a
gate result.  The only files it writes are TensorBoard event files and one
small provenance manifest below ``work/tensorboard``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .paths import PIPELINE_ROOT, assert_output_is_local


SCHEMA_VERSION = "accad_g1_tensorboard_dashboard_v1"
DEFAULT_DASHBOARD_NAME = "accad_g1_overview"


@dataclass
class DashboardSection:
    """One intentionally small TensorBoard run."""

    run_name: str
    scalars: dict[str, float] = field(default_factory=dict)
    histograms: dict[str, list[float]] = field(default_factory=dict)
    series: dict[str, list[tuple[int, float, float | None]]] = field(
        default_factory=dict
    )
    texts: dict[str, str] = field(default_factory=dict)
    sources: set[Path] = field(default_factory=set)
    summary: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required dashboard source is missing: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"dashboard source must contain a JSON object: {path}")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be numeric; received {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite; received {result!r}")
    return result


def _finite_values(values: Iterable[Any], *, label: str) -> list[float]:
    return [_finite_float(value, label=label) for value in values]


def _statistics(values: Iterable[Any], *, label: str) -> dict[str, float | int]:
    array = np.asarray(_finite_values(values, label=label), dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{label} cannot be empty")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(np.max(array)),
    }


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    escaped_headers = [_escape_markdown(value) for value in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join("---" for _ in escaped_headers) + " |",
    ]
    for row in rows:
        values = [_escape_markdown(value) for value in row]
        if len(values) != len(escaped_headers):
            raise ValueError("Markdown table row length does not match its header")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _resolve_artifact(value: str | Path, *, pipeline_root: Path) -> Path:
    """Resolve the two relative path conventions present in stored reports."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == pipeline_root.name:
        return (pipeline_root.parent / candidate).resolve()
    return (pipeline_root / candidate).resolve()


def _relative_source(path: Path, *, pipeline_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(pipeline_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_dashboard_output(output_dir: Path, *, work_root: Path) -> Path:
    output = assert_output_is_local(output_dir)
    tensorboard_root = (work_root / "tensorboard").resolve()
    try:
        relative = output.relative_to(tensorboard_root)
    except ValueError as exc:
        raise ValueError(
            f"TensorBoard dashboard must stay below {tensorboard_root}; got {output}"
        ) from exc
    if not relative.parts:
        raise ValueError("dashboard output must be a child of work/tensorboard")
    return output


def _build_data_section(*, pipeline_root: Path, work_root: Path) -> DashboardSection:
    manifest_root = work_root / "manifests"
    gate1_path = manifest_root / "gate1_summary.json"
    inventory_path = manifest_root / "accad_inventory.json"
    quality_path = manifest_root / "accad_quality.json"
    split_paths = {
        split: manifest_root / f"accad_{split}.json"
        for split in ("train", "validation", "test")
    }
    gate1 = _read_json(gate1_path)
    inventory = _read_json(inventory_path)
    quality = _read_json(quality_path)
    splits = {name: _read_json(path) for name, path in split_paths.items()}

    section = DashboardSection(run_name="01_data")
    section.sources.update({gate1_path, inventory_path, quality_path, *split_paths.values()})

    counts = gate1.get("counts", {})
    for name in ("stagei_files", "stageii_files", "eligible_files", "excluded_files"):
        section.scalars[f"files/{name}"] = _finite_float(
            counts[name], label=f"gate1.counts.{name}"
        )
    eligible_count = section.scalars["files/eligible_files"]
    stageii_count = section.scalars["files/stageii_files"]
    section.scalars["files/eligible_fraction"] = eligible_count / stageii_count
    section.scalars["gate/passed"] = float(bool(gate1.get("passed", False)))

    eligible_motions: list[dict[str, Any]] = []
    split_rows: list[list[Any]] = []
    for split, document in splits.items():
        motions = document.get("motions", [])
        if not isinstance(motions, list):
            raise ValueError(f"{split} motion manifest has an invalid motions field")
        eligible_motions.extend(value for value in motions if isinstance(value, dict))
        for metric in ("num_motions", "num_frames", "duration_sec"):
            section.scalars[f"splits/{split}/{metric}"] = _finite_float(
                document[metric], label=f"{split}.{metric}"
            )
        split_rows.append(
            [
                split,
                int(document["num_motions"]),
                int(document["num_frames"]),
                f"{float(document['duration_sec']):.2f}",
                ", ".join(document.get("subjects", [])),
            ]
        )

    if len(eligible_motions) != int(eligible_count):
        raise ValueError(
            "eligible split manifests do not match Gate-1 accounting: "
            f"{len(eligible_motions)} != {int(eligible_count)}"
        )
    durations = _finite_values(
        (motion["duration_sec"] for motion in eligible_motions), label="clip duration"
    )
    frames = _finite_values(
        (motion["num_frames"] for motion in eligible_motions), label="clip frame count"
    )
    fps = _finite_values(
        (motion["fps"] for motion in eligible_motions), label="source FPS"
    )
    section.histograms.update(
        {
            "distribution/clip_duration_seconds": durations,
            "distribution/source_frame_count": frames,
            "distribution/source_fps": fps,
        }
    )
    section.scalars["corpus/total_duration_seconds"] = float(sum(durations))
    section.scalars["corpus/total_source_frames"] = float(sum(frames))

    quality_motions = [
        motion
        for motion in quality.get("motions", [])
        if isinstance(motion, dict) and motion.get("valid") is True
    ]
    quality_metrics = [motion.get("quality_metrics", {}) for motion in quality_motions]
    quality_fields = {
        "max_translation_speed_mps": "quality/max_translation_speed_mps",
        "max_translation_step_m": "quality/max_translation_step_m",
        "max_rotation_step_rad": "quality/max_rotation_step_rad",
        "duplicate_transition_fraction": "quality/duplicate_transition_fraction",
    }
    for key, tag in quality_fields.items():
        section.histograms[tag] = _finite_values(
            (metrics[key] for metrics in quality_metrics), label=f"quality.{key}"
        )
    section.scalars["quality/motions_with_flags"] = _finite_float(
        quality.get("motions_with_flags", 0), label="quality.motions_with_flags"
    )
    for name, value in quality.get("flag_counts", {}).items():
        section.scalars[f"quality/flag_count/{name}"] = _finite_float(
            value, label=f"quality.flag_counts.{name}"
        )
    for name in (
        "max_translation_speed_mps",
        "max_translation_step_m",
        "max_rotation_step_rad",
        "max_duplicate_fraction",
    ):
        if name in quality.get("thresholds", {}):
            section.scalars[f"quality_threshold/{name}"] = _finite_float(
                quality["thresholds"][name], label=f"quality.thresholds.{name}"
            )

    section.texts["00_read_me"] = (
        "# 1. ACCAD 데이터 감사\n\n"
        "이 run은 원본 Stage-II 파일의 개수, split, 길이/FPS 및 기본 품질을 보여줍니다. "
        "`eligible=249`는 구조적으로 사용할 수 있고 checksum 중복이 제거됐다는 뜻이며, "
        "G1이 물리적으로 수행할 수 있다는 뜻은 아닙니다. 품질 flag는 현재 설정에서 "
        "정보성 경고이며 자동 제외 조건이 아닙니다."
    )
    section.texts["01_split_overview"] = _markdown_table(
        ["split", "motions", "source frames", "duration (s)", "subjects"], split_rows
    )

    category_rows = []
    category_split_counts = gate1.get("category_split_counts", {})
    for category in sorted(category_split_counts):
        values = category_split_counts[category]
        category_rows.append(
            [
                category,
                values.get("train", 0),
                values.get("validation", 0),
                values.get("test", 0),
                sum(int(values.get(split, 0)) for split in ("train", "validation", "test")),
            ]
        )
    section.texts["02_category_distribution"] = _markdown_table(
        ["category", "train", "validation", "test", "total"], category_rows
    )

    split_subjects = gate1.get("split_subjects", {})
    section.texts["03_subject_split"] = _markdown_table(
        ["split", "subjects"],
        [
            [split, ", ".join(split_subjects.get(split, []))]
            for split in ("train", "validation", "test")
        ],
    ) + "\n\n" + str(gate1.get("split_limitation_note", ""))

    example = next(
        (
            motion
            for motion in inventory.get("motions", [])
            if isinstance(motion, dict) and motion.get("valid") is True
        ),
        None,
    )
    if example is None:
        raise ValueError("inventory does not contain an eligible example motion")
    shape_rows = []
    for key, shape in sorted(example.get("array_shapes", {}).items()):
        rendered = list(shape)
        if rendered and rendered[0] == example.get("num_frames") and key != "betas":
            rendered[0] = "T"
        shape_rows.append(
            [key, str(rendered), example.get("array_dtypes", {}).get(key, "unknown")]
        )
    section.texts["04_stageii_dimensions"] = (
        "`T`는 모션마다 달라지는 시간 프레임 수입니다.\n\n"
        + _markdown_table(["array", "shape", "dtype"], shape_rows)
    )

    checks = gate1.get("checks", {})
    section.texts["05_integrity_checks"] = _markdown_table(
        ["check", "result"],
        [[name, "PASS" if value else "FAIL"] for name, value in sorted(checks.items())],
    )
    section.summary = {
        "passed": bool(gate1.get("passed", False)),
        "counts": counts,
        "duration_seconds": _statistics(durations, label="clip duration"),
        "source_fps": _statistics(fps, label="source FPS"),
        "quality_flag_counts": quality.get("flag_counts", {}),
    }
    return section


def _extract_retarget_validation_metrics(validation: dict[str, Any]) -> dict[str, float]:
    metrics = validation.get("metrics", {})
    limits = metrics.get("joint_limits", {})
    foot_geometry = metrics.get("foot_contact_geometry", {})

    penetrations: list[float] = []
    stance_steps: list[float] = []
    stance_drifts: list[float] = []
    stance_speeds: list[float] = []
    for side in ("left", "right"):
        side_metrics = foot_geometry.get(side, {})
        penetrations.append(
            _finite_float(
                side_metrics["maximum_penetration_m"],
                label=f"{side}.maximum_penetration_m",
            )
        )
        for part in ("heel", "toe"):
            channel = side_metrics.get("stance_channels", {}).get(part, {})
            stance_steps.append(
                _finite_float(
                    channel["maximum_consecutive_stance_point_step_xy_m"],
                    label=f"{side}.{part}.stance_step",
                )
            )
            stance_drifts.append(
                _finite_float(
                    channel["maximum_stance_run_drift_xy_m"],
                    label=f"{side}.{part}.stance_drift",
                )
            )
            stance_speeds.append(
                _finite_float(
                    channel["p95_consecutive_stance_point_speed_xy_m_s"],
                    label=f"{side}.{part}.stance_speed",
                )
            )

    return {
        "max_joint_velocity_limit_ratio": _finite_float(
            limits["velocity"]["maximum_limit_ratio"], label="maximum velocity ratio"
        ),
        "p95_joint_acceleration_rad_s2": _finite_float(
            limits["acceleration"]["p95_abs_rad_s2"], label="p95 acceleration"
        ),
        "max_joint_acceleration_rad_s2": _finite_float(
            limits["acceleration"]["maximum_abs_rad_s2"], label="maximum acceleration"
        ),
        "minimum_hard_limit_margin_rad": _finite_float(
            limits["hard"]["minimum_margin_rad"], label="hard-limit margin"
        ),
        "max_fk_position_error_m": _finite_float(
            metrics["pinocchio_fk"]["maximum_body_position_error_m"],
            label="FK position error",
        ),
        "max_fk_orientation_error_rad": _finite_float(
            metrics["pinocchio_fk"]["maximum_body_orientation_error_rad"],
            label="FK orientation error",
        ),
        "max_foot_penetration_m": max(penetrations),
        "max_stance_step_m": max(stance_steps),
        "max_stance_drift_m": max(stance_drifts),
        "p95_stance_speed_mps": max(stance_speeds),
        "residual_box_violation_fraction": _finite_float(
            metrics["action_reachability"]["reference_residual"][
                "full_clip_box_violation_fraction"
            ],
            label="residual box violation fraction",
        ),
    }


def _build_retarget_section(*, pipeline_root: Path, work_root: Path) -> DashboardSection:
    manifest_root = work_root / "manifests"
    build_path = manifest_root / "g1_manifest_build.json"
    build = _read_json(build_path)
    section = DashboardSection(run_name="02_retargeting")
    section.sources.add(build_path)

    item_reports = build.get("item_reports", {})
    if not isinstance(item_reports, dict) or not item_reports:
        raise ValueError("g1 manifest build does not enumerate item reports")
    per_motion: list[dict[str, float]] = []
    kinematic_passes = 0
    first_thresholds: dict[str, Any] | None = None
    for item_id, record in sorted(item_reports.items()):
        if not isinstance(record, dict) or "path" not in record:
            raise ValueError(f"invalid item report record: {item_id}")
        report_path = _resolve_artifact(record["path"], pipeline_root=pipeline_root)
        item = _read_json(report_path)
        section.sources.add(report_path)
        validation = item.get("stages", {}).get("g1", {}).get("validation")
        if not isinstance(validation, dict):
            raise ValueError(f"missing G1 validation in {report_path}")
        kinematic_passes += int(bool(validation.get("gate3_kinematic_complete", False)))
        per_motion.append(_extract_retarget_validation_metrics(validation))
        if first_thresholds is None:
            first_thresholds = validation.get("thresholds", {})

    source_records = int(build.get("counts", {}).get("source_records", 0))
    promoted_total = int(build.get("counts", {}).get("promoted_total", 0))
    if len(per_motion) != source_records:
        raise ValueError(
            f"retarget item report count mismatch: {len(per_motion)} != {source_records}"
        )
    section.scalars.update(
        {
            "coverage/source_motions": float(source_records),
            "coverage/promoted_motions": float(promoted_total),
            "coverage/promoted_fraction": promoted_total / max(source_records, 1),
            "coverage/kinematic_pass_fraction": kinematic_passes
            / max(len(per_motion), 1),
            "corpus/frames_50hz": _finite_float(
                build["num_frames"], label="g1_manifest_build.num_frames"
            ),
            "corpus/duration_seconds": _finite_float(
                build["duration_sec"], label="g1_manifest_build.duration_sec"
            ),
        }
    )

    histogram_tags = {
        "max_joint_velocity_limit_ratio": "distribution/max_joint_velocity_limit_ratio",
        "p95_joint_acceleration_rad_s2": "distribution/p95_joint_acceleration_rad_s2",
        "max_joint_acceleration_rad_s2": "distribution/max_joint_acceleration_rad_s2",
        "minimum_hard_limit_margin_rad": "distribution/minimum_hard_limit_margin_rad",
        "max_fk_position_error_m": "distribution/max_pinocchio_fk_position_error_m",
        "max_fk_orientation_error_rad": "distribution/max_pinocchio_fk_orientation_error_rad",
        "max_foot_penetration_m": "distribution/max_foot_penetration_m",
        "max_stance_step_m": "distribution/max_stance_step_m",
        "max_stance_drift_m": "distribution/max_stance_drift_m",
        "residual_box_violation_fraction": "distribution/residual_box_violation_fraction",
    }
    metric_summaries: dict[str, Any] = {}
    for key, tag in histogram_tags.items():
        values = [motion[key] for motion in per_motion]
        section.histograms[tag] = values
        stats = _statistics(values, label=f"retarget.{key}")
        metric_summaries[key] = stats
        section.scalars[f"dataset_worst/{key}"] = float(
            stats["minimum"] if key == "minimum_hard_limit_margin_rad" else stats["maximum"]
        )

    if first_thresholds:
        threshold_map = {
            "fk_position_m": "threshold/fk_position_error_m",
            "fk_orientation_rad": "threshold/fk_orientation_error_rad",
            "foot_penetration_m": "threshold/foot_penetration_m",
            "stance_support_step_xy_m": "threshold/stance_support_step_m",
            "stance_run_drift_xy_m": "threshold/stance_run_drift_m",
        }
        for key, tag in threshold_map.items():
            section.scalars[tag] = _finite_float(
                first_thresholds[key], label=f"retarget.thresholds.{key}"
            )

    train_manifest_path = manifest_root / "g1_train.json"
    train_manifest = _read_json(train_manifest_path)
    section.sources.add(train_manifest_path)
    b3_record = next(
        (
            motion
            for motion in train_manifest.get("motions", [])
            if isinstance(motion, dict) and motion.get("motion_name") == "B3_-_walk1"
        ),
        None,
    )
    if b3_record is None:
        raise ValueError("canonical B3 reference is missing from g1_train.json")
    b3_path = work_root / "g1" / b3_record["g1_file"]
    if _sha256_file(b3_path) != b3_record["g1_checksum_sha256"]:
        raise ValueError("canonical B3 reference checksum does not match its manifest")
    section.sources.add(b3_path)
    with np.load(b3_path, allow_pickle=False) as archive:
        joint_names = [str(value) for value in archive["joint_names"].tolist()]
        joint_pos = np.asarray(archive["joint_pos"], dtype=np.float64)
        root_pos = np.asarray(archive["root_pos_w"], dtype=np.float64)
        root_velocity = np.asarray(archive["root_lin_vel_w"], dtype=np.float64)
        frame_count = joint_pos.shape[0]
        representative_joints = (
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        )
        for name in representative_joints:
            index = joint_names.index(name)
            section.series[f"representative_b3/joint_position_rad/{name}"] = [
                (frame, float(joint_pos[frame, index]), None)
                for frame in range(frame_count)
            ]
        section.series["representative_b3/root_height_m"] = [
            (frame, float(root_pos[frame, 2]), None) for frame in range(frame_count)
        ]
        section.series["representative_b3/root_speed_mps"] = [
            (frame, float(np.linalg.norm(root_velocity[frame])), None)
            for frame in range(frame_count)
        ]
        for key in ("left_foot_contact", "right_foot_contact"):
            values = np.asarray(archive[key], dtype=np.float64)
            section.series[f"representative_b3/{key}"] = [
                (frame, float(values[frame]), None) for frame in range(frame_count)
            ]

    promoted_by_split = build.get("counts", {}).get("promoted_by_split", {})
    section.texts["00_read_me"] = (
        "# 2. G1 리타겟팅 결과\n\n"
        "이 run의 249/249 PASS는 **G1 archive의 관절 제한, 접촉 기하, 독립 "
        "Pinocchio FK 및 파일 계약이 일관적**이라는 뜻입니다. FK 오차는 저장된 G1 "
        "reference와 독립 FK의 일치도이며 인간 동작과 G1 동작의 의미적 유사도 오차가 "
        "아닙니다. 현재 canonical 산출물에는 human target↔G1 semantic fidelity 지표가 "
        "저장돼 있지 않습니다. 또한 이 결과는 Isaac 동역학 수행 성공을 보장하지 않습니다.\n\n"
        "대표 B3 곡선은 현재 manifest checksum을 통과한 train reference만 읽습니다. "
        "x축은 50 Hz frame이고, TEST의 수치 NPZ는 이 대시보드가 열지 않습니다."
    )
    section.texts["01_coverage"] = _markdown_table(
        ["scope", "motions"],
        [
            ["source", source_records],
            ["promoted total", promoted_total],
            ["train", promoted_by_split.get("train", 0)],
            ["validation", promoted_by_split.get("validation", 0)],
            ["test", promoted_by_split.get("test", 0)],
        ],
    )
    section.texts["02_metric_meaning"] = _markdown_table(
        ["metric", "interpretation"],
        [
            ["velocity ratio", "관절 속도 한계 대비 clip별 최댓값"],
            ["acceleration", "권위 있는 G1 가속도 한계가 없어 report-only"],
            ["hard-limit margin", "관절 hard limit까지 남은 최소 각도"],
            ["Pinocchio FK error", "G1 archive 내부 기구학 일관성"],
            ["foot penetration", "기하학적 지면 관통"],
            ["stance step/drift", "stance 동안 발 지지점 이동"],
            ["residual box violation", "near-limit 상태에서 허용 residual이 좁아지는 비율; report-only"],
        ],
    )
    section.texts["03_contract"] = _markdown_table(
        ["field", "value"],
        [
            ["retarget version", build.get("retarget_version")],
            ["G1 DoF", 29],
            ["reference/control frequency", "50 Hz"],
            ["coordinate", "right-handed, x-forward, y-left, z-up"],
            ["semantic fidelity metric", "UNAVAILABLE in current canonical artifacts"],
            ["dynamic guarantee", "NO — separate Isaac physics/policy evaluation required"],
        ],
    )
    section.summary = {
        "status": build.get("status"),
        "source_records": source_records,
        "promoted_total": promoted_total,
        "kinematic_passes": kinematic_passes,
        "num_frames": build.get("num_frames"),
        "duration_sec": build.get("duration_sec"),
        "retarget_version": build.get("retarget_version"),
        "metrics": metric_summaries,
        "semantic_fidelity_metric": "unavailable",
        "test_numeric_npz_opened": False,
    }
    return section


def _load_selected_training_series(
    run_dir: Path,
) -> tuple[dict[str, list[tuple[int, float, float | None]]], list[Path]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:  # pragma: no cover - exercised only in broken envs
        raise RuntimeError(
            "TensorBoard is unavailable. Run this command with ./_isaac_sim/python.sh."
        ) from exc

    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event file found in {run_dir}")
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    selected_tags = {
        "Train/mean_reward": "training/mean_reward",
        "Train/mean_episode_length": "training/mean_episode_length",
        "Metrics/motion/joint_position_error": "training/joint_position_error",
        "Metrics/motion/root_position_error": "training/root_position_error",
        "Metrics/motion/tracking_body_position_error": "training/body_position_error",
        "Episode_Termination/motion_end": "training/termination_motion_end",
        "Episode_Termination/fall": "training/termination_fall",
        "Episode_Termination/tracking_divergence": "training/termination_tracking_divergence",
    }
    missing = sorted(set(selected_tags) - available)
    if missing:
        raise ValueError(f"selected training log is missing required tags: {missing}")

    output: dict[str, list[tuple[int, float, float | None]]] = {}
    for source_tag, destination_tag in selected_tags.items():
        latest_by_step: dict[int, Any] = {}
        for event in accumulator.Scalars(source_tag):
            value = _finite_float(event.value, label=f"TensorBoard {source_tag}")
            previous = latest_by_step.get(int(event.step))
            if previous is None or float(event.wall_time) >= float(previous.wall_time):
                latest_by_step[int(event.step)] = event
                # Validate before retaining the event object.
                _ = value
        output[destination_tag] = [
            (
                step,
                _finite_float(event.value, label=f"TensorBoard {source_tag}"),
                float(event.wall_time),
            )
            for step, event in sorted(latest_by_step.items())
        ]
    return output, event_files


def _build_physics_policy_section(
    *, pipeline_root: Path, work_root: Path
) -> DashboardSection:
    isaac_path = work_root / "reports" / "B3_-_walk1_isaac_kinematic.json"
    dynamic_train_path = work_root / "dynamic" / "g1_train_report.json"
    dynamic_validation_path = work_root / "dynamic" / "g1_validation_report.json"
    contact_path = work_root / "contact_v29" / "g1_train_report.json"
    outcome_path = work_root / "reports" / "v7_train_selection_outcome.json"

    isaac = _read_json(isaac_path)
    dynamic_train = _read_json(dynamic_train_path)
    dynamic_validation = _read_json(dynamic_validation_path)
    contact = _read_json(contact_path)
    outcome = _read_json(outcome_path)
    section = DashboardSection(run_name="03_physics_policy")
    section.sources.update(
        {isaac_path, dynamic_train_path, dynamic_validation_path, contact_path, outcome_path}
    )

    isaac_input = isaac.get("input", {})
    isaac_metrics = isaac.get("metrics", {})
    total_frames = _finite_float(isaac_input["total_frames"], label="Isaac total frames")
    validated_frames = _finite_float(
        isaac_input["validated_frames"], label="Isaac validated frames"
    )
    section.scalars.update(
        {
            "isaac_ingestion/passed": float(bool(isaac.get("passed", False))),
            "isaac_ingestion/validated_frames": validated_frames,
            "isaac_ingestion/frame_coverage": validated_frames / total_frames,
            "isaac_ingestion/body_position_rmse_m": _finite_float(
                isaac_metrics["body_link_position_error_m"]["rmse"],
                label="Isaac body position RMSE",
            ),
            "isaac_ingestion/body_orientation_rmse_rad": _finite_float(
                isaac_metrics["body_link_orientation_error_rad"]["rmse"],
                label="Isaac body orientation RMSE",
            ),
        }
    )

    dynamic_summaries = {}
    for split, report in (
        ("train", dynamic_train),
        ("validation", dynamic_validation),
    ):
        counts = report.get("counts", {})
        source = _finite_float(counts["source"], label=f"dynamic {split} source")
        derived = _finite_float(counts["derived"], label=f"dynamic {split} derived")
        quarantined = _finite_float(
            counts["final_quarantined"], label=f"dynamic {split} quarantined"
        )
        section.scalars[f"dynamic_filter/{split}/source"] = source
        section.scalars[f"dynamic_filter/{split}/derived"] = derived
        section.scalars[f"dynamic_filter/{split}/retention_fraction"] = derived / source
        section.scalars[f"dynamic_filter/{split}/quarantined"] = quarantined
        dynamic_summaries[split] = {
            "source": int(source),
            "derived": int(derived),
            "quarantined": int(quarantined),
        }
    contact_counts = contact.get("counts", {})
    section.scalars["contact_audit/source_v28"] = _finite_float(
        contact_counts["source_v28"], label="contact source_v28"
    )
    section.scalars["contact_audit/eligible"] = _finite_float(
        contact_counts["eligible"], label="contact eligible"
    )

    selected = outcome.get("selected_candidate", {})
    selected_iteration = int(selected["checkpoint_iteration"])
    candidate = next(
        (
            value
            for value in outcome.get("candidates", [])
            if isinstance(value, dict)
            and Path(value.get("checkpoint", "")).stem == f"model_{selected_iteration}"
        ),
        None,
    )
    if candidate is None:
        raise ValueError("cannot resolve selected v7 candidate evaluation")
    suite_path = _resolve_artifact(candidate["evaluation"], pipeline_root=pipeline_root)
    suite = _read_json(suite_path)
    section.sources.add(suite_path)

    run_dir = _resolve_artifact(
        outcome.get("training", {}).get("run", ""), pipeline_root=pipeline_root
    )
    series, event_files = _load_selected_training_series(run_dir)
    section.series.update(series)
    section.sources.update(event_files)

    aggregate = suite.get("aggregate", {})
    macro = aggregate.get("macro_motion_metrics", {})
    weighted = aggregate.get("weighted_frame_or_element_metrics", {})
    contact_metrics = aggregate.get("aggregate_foot_contact", {}).get("overall", {})
    motion_count = _finite_float(aggregate["motion_count"], label="policy motion count")
    success_count = _finite_float(
        aggregate["successful_motion_end_episodes"], label="policy success count"
    )
    acceptance_values = {
        "successful_motions": success_count,
        "success_fraction": success_count / motion_count,
        "completion_ratio": _finite_float(
            aggregate["completion_ratio_mean"], label="completion ratio"
        ),
        "joint_mae_rad": _finite_float(
            macro["mean_absolute_joint_error_rad"], label="joint MAE"
        ),
        "root_position_error_m": _finite_float(
            macro["mean_root_position_error_m"], label="root position error"
        ),
        "body_position_error_m": _finite_float(
            macro["mean_tracking_body_position_error_m"], label="body position error"
        ),
        "contact_f1": _finite_float(contact_metrics["f1"], label="contact F1"),
        "raw_action_clip_fraction": _finite_float(
            weighted["raw_action_clip_fraction"], label="raw action clip fraction"
        ),
    }
    for key, value in acceptance_values.items():
        section.scalars[f"policy_acceptance/value/{key}"] = value
    section.scalars["policy_acceptance/motion_count"] = motion_count
    section.scalars["policy_acceptance/passed"] = float(
        bool(suite.get("acceptance", {}).get("passed", False))
    )
    for name in ("motion_end", "fall", "tracking_divergence"):
        value = aggregate.get("termination_counts", {})[name]
        section.scalars[f"policy_termination/{name}"] = _finite_float(
            value, label=f"termination count {name}"
        )

    checks = suite.get("acceptance", {}).get("checks", {})
    acceptance_rows = []
    for name, check in checks.items():
        expected = check.get("expected", check.get("threshold", ""))
        acceptance_rows.append(
            [
                name,
                f"{float(check['value']):.6g}",
                f"{check.get('comparison', '')} {expected}",
                "PASS" if check.get("passed") else "FAIL",
            ]
        )

    section.texts["00_read_me"] = (
        "# 3. Isaac·동역학 필터·정책\n\n"
        "Isaac ingestion RMSE는 state-write/FK 계약 정확도이며 물리적으로 걷는다는 "
        "뜻이 아닙니다. v28의 103/215는 속도·가속도·flight 사전 필터 통과 수이지 "
        "Isaac 물리 성공 수가 아닙니다. v29 contact 결과는 `audit-only` proxy입니다.\n\n"
        "정책 수치는 **train split, deterministic frame-0 first episode** 결과입니다. "
        "validation/TEST 일반화 결과가 아닙니다. MAE는 조기 종료 전 구간만 포함할 수 "
        "있으므로 success와 completion을 항상 함께 봐야 합니다. 종료 신호는 같은 step에 "
        "중복될 수 있어 fall과 divergence 합이 실패 episode 수와 같지 않을 수 있습니다."
    )
    section.texts["01_stage_overview"] = _markdown_table(
        ["stage", "result", "meaning"],
        [
            ["Isaac kinematic ingestion", f"{int(validated_frames)}/{int(total_frames)}", "format/FK contract"],
            ["v28 train dynamic filter", f"{dynamic_summaries['train']['derived']}/{dynamic_summaries['train']['source']}", "prefilter only"],
            ["v29 contact audit", f"{contact_counts.get('eligible', 0)}/{contact_counts.get('source_v28', 0)}", "audit-only proxy"],
            ["v7 train policy", f"{int(success_count)}/{int(motion_count)}", "acceptance FAIL"],
            ["validation", "not run for v7", "train gate failed"],
            ["TEST", "sealed", "not opened by dashboard"],
        ],
    )
    section.texts["02_policy_acceptance"] = _markdown_table(
        ["check", "value", "requirement", "result"], acceptance_rows
    )
    section.texts["03_training_log"] = _markdown_table(
        ["field", "value"],
        [
            ["selected checkpoint", f"model_{selected_iteration}.pt"],
            ["run", _relative_source(run_dir, pipeline_root=pipeline_root)],
            ["event files", len(event_files)],
            ["curve policy", "selected continuation run only; earlier run is not merged"],
            ["actor / critic / action", "293 / 505 / 29"],
            ["physics / control", "200 Hz / 50 Hz"],
        ],
    )
    section.texts["04_dynamic_filter_actions"] = _markdown_table(
        ["v28 train action", "motions"],
        [
            [name, count]
            for name, count in dynamic_train.get("counts", {})
            .get("final_actions", {})
            .items()
        ],
    )
    section.summary = {
        "isaac_ingestion": {
            "passed": bool(isaac.get("passed", False)),
            "validated_frames": int(validated_frames),
            "total_frames": int(total_frames),
            "body_position_rmse_m": section.scalars[
                "isaac_ingestion/body_position_rmse_m"
            ],
            "body_orientation_rmse_rad": section.scalars[
                "isaac_ingestion/body_orientation_rmse_rad"
            ],
        },
        "dynamic_filter": dynamic_summaries,
        "contact_audit": {
            "mode": contact.get("mode"),
            "source_v28": contact_counts.get("source_v28"),
            "eligible": contact_counts.get("eligible"),
            "quarantined": contact_counts.get("quarantined"),
        },
        "policy": {
            "selected_checkpoint_iteration": selected_iteration,
            "train_acceptance_passed": bool(
                suite.get("acceptance", {}).get("passed", False)
            ),
            **acceptance_values,
        },
        "validation_evaluated": False,
        "test_opened": False,
    }
    return section


def _write_section(section: DashboardSection, log_dir: Path) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:  # pragma: no cover - exercised only in broken envs
        raise RuntimeError(
            "TensorBoard SummaryWriter is unavailable. Run with ./_isaac_sim/python.sh."
        ) from exc

    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=1)
    try:
        for tag, value in sorted(section.scalars.items()):
            writer.add_scalar(tag, _finite_float(value, label=tag), global_step=0)
        for tag, values in sorted(section.histograms.items()):
            array = np.asarray(_finite_values(values, label=tag), dtype=np.float32)
            if array.size:
                writer.add_histogram(tag, array, global_step=0)
        for tag, points in sorted(section.series.items()):
            for step, value, walltime in points:
                writer.add_scalar(
                    tag,
                    _finite_float(value, label=tag),
                    global_step=int(step),
                    walltime=walltime,
                )
        for tag, markdown in sorted(section.texts.items()):
            writer.add_text(tag, markdown, global_step=0)
        writer.flush()
    finally:
        writer.close()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def export_tensorboard_dashboard(
    *,
    work_root: Path,
    output_dir: Path | None = None,
    force: bool = False,
    pipeline_root: Path = PIPELINE_ROOT,
) -> dict[str, Any]:
    """Create the three-run dashboard and return its concise manifest."""

    work_root = work_root.resolve()
    destination = _validated_dashboard_output(
        output_dir or work_root / "tensorboard" / DEFAULT_DASHBOARD_NAME,
        work_root=work_root,
    )
    if destination.exists() and not force:
        raise FileExistsError(
            f"dashboard already exists: {destination}; pass --force to rebuild only this "
            "generated dashboard"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    ).resolve()
    try:
        sections = [
            _build_data_section(pipeline_root=pipeline_root, work_root=work_root),
            _build_retarget_section(pipeline_root=pipeline_root, work_root=work_root),
            _build_physics_policy_section(
                pipeline_root=pipeline_root, work_root=work_root
            ),
        ]
        for section in sections:
            _write_section(section, temporary / section.run_name)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(destination),
            "launch_command": (
                "./_isaac_sim/python.sh -m tensorboard.main --logdir "
                f"{destination} --port 6006"
            ),
            "sections": {
                section.run_name: {
                    "scalar_tags": sorted(section.scalars),
                    "histogram_tags": sorted(section.histograms),
                    "series_tags": sorted(section.series),
                    "text_tags": sorted(section.texts),
                    "sources": sorted(
                        _relative_source(path, pipeline_root=pipeline_root)
                        for path in section.sources
                    ),
                    "summary": section.summary,
                }
                for section in sections
            },
            "contracts": {
                "read_only_source_processing": True,
                "writes_only_below_work_tensorboard": True,
                "retarget_semantic_fidelity_available": False,
                "test_numeric_npz_opened": False,
                "test_dynamic_or_policy_branch_opened": False,
                "training_curves": "selected v7 continuation run only; no cross-run merge",
            },
        }
        _write_json_atomic(temporary / "dashboard_manifest.json", manifest)
        if destination.exists():
            backup = destination.parent / f".{destination.name}.previous-{os.getpid()}"
            if backup.exists():
                raise FileExistsError(f"stale dashboard backup blocks replacement: {backup}")
            os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
            except BaseException:
                os.replace(backup, destination)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
