"""Fail-closed dynamic audit and offline 50 Hz retiming for promoted G1 clips.

This stage is deliberately downstream of the immutable v27 Gate-3 manifests.
It never overwrites those archives.  A policy frozen while processing the
training split is applied unchanged to validation and, only after an explicit
``split=test`` request, to the sealed test split.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from .g1_kinematics import (
    TorchG1Kinematics,
    load_g1_contract,
    matrix_to_quaternion_wxyz,
    policy_to_sdk,
)
from .human import _angular_velocity_world, _linear_velocity, _save_npz_atomic
from .math3d import resample_quaternion_slerp
from .paths import PIPELINE_ROOT, assert_output_is_local
from .retarget import (
    FOOT_COLLISION_RADIUS_M,
    _refine_output_stance_channels,
    load_correspondence,
)


POLICY_SCHEMA_VERSION = "accad_g1_dynamic_scale_policy_v1"
PLAN_SCHEMA_VERSION = "accad_g1_dynamic_scale_plan_v1"
REPORT_SCHEMA_VERSION = "accad_g1_dynamic_report_v1"
MANIFEST_SCHEMA_VERSION = "accad_g1_dynamic_manifest_v1"
ARCHIVE_DERIVATION_SCHEMA_VERSION = "accad_g1_dynamic_archive_derivation_v1"
DYNAMIC_RETARGET_VERSION = (
    "g1-sequence-retarget-v28-dynamic-feasibility-uniform-retime-v1"
)

DEFAULT_INPUT_MANIFESTS = {
    "train": PIPELINE_ROOT / "work" / "manifests" / "g1_train.json",
    "validation": PIPELINE_ROOT / "work" / "manifests" / "g1_validation.json",
    "test": PIPELINE_ROOT / "work" / "manifests" / "g1_test.json",
}
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "work" / "dynamic"

# Frozen train-derived prefilter.  The acceleration cap is intentionally
# labelled as a prefilter: the URDF has no authoritative acceleration limit.
POLICY_ALGORITHM: dict[str, Any] = {
    "schema_version": "accad_g1_dynamic_scale_formula_v1",
    "fps": 50.0,
    "joint_velocity_limit_fraction": 0.8,
    "root_linear_speed_cap_mps": 2.0,
    "root_angular_speed_cap_radps": 4.0,
    # Conservative proxy: allow one URDF velocity-limit change over 200 ms.
    # This is a deterministic prefilter only; effort/contact validation remains
    # the authoritative dynamic test because the URDF has no acceleration cap.
    "joint_acceleration_cap_velocity_multiple_per_second": 5.0,
    "scale_quantum": 0.05,
    "uniform_retime_maximum_consecutive_flight_frames_exclusive": 3,
    "derivatives": {
        "segment_joint_velocity": "(q[t+1]-q[t])*fps",
        "segment_root_linear_speed": "norm(p[t+1]-p[t])*fps",
        "segment_root_angular_speed": (
            "2*acos(abs(dot(unit_quat[t],unit_quat[t+1])))*fps"
        ),
        "segment_joint_acceleration": "diff(segment_joint_velocity)*fps",
    },
    "scale": (
        "ceil_quantum(max(1, max_abs_qdot/(0.8*vlim), max_root_v/2, "
        "max_root_w/4, sqrt(max_abs_qacc/(5*vlim))))"
    ),
    "boolean_resampling": "nearest_source_frame",
    "continuous_resampling": "linear_position_and_shortest_path_quaternion_slerp",
    "flight_policy": "quarantine_if_scale_gt_1_and_longest_flight_run_ge_3",
    "acceleration_gate": "prefilter_only_no_authoritative_urdf_acceleration_limit",
    "post_fk_headroom": {
        "maximum_attempts": 8,
        "multiplier": 1.02,
        "minimum_output_interval_growth": 1,
        "flight_fail_closed": (
            "quarantine_if_an_additional_stretch_is_required_and_the_current_"
            "longest_flight_run_is_at_least_3_frames"
        ),
    },
}

CONTACT_FIELDS = (
    "left_heel_contact",
    "left_toe_contact",
    "right_heel_contact",
    "right_toe_contact",
)
FOOT_CONTACT_FIELDS = ("left_foot_contact", "right_foot_contact")
STANCE_FIELDS = (
    "left_heel_stance",
    "left_toe_stance",
    "right_heel_stance",
    "right_toe_stance",
)
KNOWN_FRAME_FIELDS = {
    "time",
    "joint_pos",
    "joint_pos_sdk",
    "joint_vel",
    "joint_acc",
    "root_pos_w",
    "root_quat_wxyz",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "body_pos_w",
    "body_quat_wxyz",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "foot_sole_position_w",
    "foot_support_position_w",
    "contact_point_position_w",
    *FOOT_CONTACT_FIELDS,
    *CONTACT_FIELDS,
    *STANCE_FIELDS,
}


class DynamicRetimeError(RuntimeError):
    """Raised when a dynamic derivation cannot be published safely."""


class DynamicRetimeQuarantine(DynamicRetimeError):
    """Raised when a derivation must be promoted to a recorded quarantine."""

    def __init__(self, reason: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.evidence = dict(evidence)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_checksum(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    output = assert_output_is_local(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", dir=output.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                document,
                temporary,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _safe_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as exc:
        raise DynamicRetimeError(f"cannot safely load G1 archive {path}: {exc}") from exc
    objects = sorted(name for name, value in arrays.items() if value.dtype.hasobject)
    if objects:
        raise DynamicRetimeError(f"object arrays are forbidden in {path}: {objects}")
    return arrays


def _scalar(arrays: Mapping[str, np.ndarray], name: str) -> Any:
    try:
        value = arrays[name]
    except KeyError as exc:
        raise DynamicRetimeError(f"archive is missing scalar {name!r}") from exc
    if value.shape != () or value.dtype.hasobject:
        raise DynamicRetimeError(f"{name} must be a safe scalar; got {value.shape}/{value.dtype}")
    return value.item()


def _longest_true_run(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("run mask must be one-dimensional")
    padded = np.pad(values.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return int(np.max(stops - starts)) if starts.size else 0


def _true_run_count(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    return int(np.count_nonzero(np.diff(np.pad(values.astype(np.int8), (1, 0))) == 1))


def _quantize_scale(value: float, quantum: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("scale must be finite and positive")
    if not math.isfinite(quantum) or quantum <= 0.0:
        raise ValueError("scale quantum must be finite and positive")
    steps = math.ceil((value - 1.0e-12) / quantum)
    return float(round(max(1.0, steps * quantum), 10))


def audit_dynamic_feasibility(
    arrays: Mapping[str, np.ndarray],
    *,
    algorithm: Mapping[str, Any] = POLICY_ALGORITHM,
) -> dict[str, Any]:
    """Compute the frozen segment-derivative prefilter for one G1 archive."""

    fps = float(_scalar(arrays, "fps"))
    expected_fps = float(algorithm["fps"])
    q = np.asarray(arrays["joint_pos"], dtype=np.float64)
    root_position = np.asarray(arrays["root_pos_w"], dtype=np.float64)
    root_quaternion = np.asarray(arrays["root_quat_wxyz"], dtype=np.float64)
    if fps != expected_fps or q.ndim != 2 or q.shape[1] != 29 or len(q) < 3:
        raise DynamicRetimeError("dynamic audit requires a canonical >=3-frame 50 Hz G1 archive")
    if root_position.shape != (len(q), 3) or root_quaternion.shape != (len(q), 4):
        raise DynamicRetimeError("root arrays do not match the G1 frame count")
    if not all(name in arrays for name in FOOT_CONTACT_FIELDS):
        raise DynamicRetimeError("foot contact fields are required for fail-closed retiming")
    finite = np.isfinite(q).all() and np.isfinite(root_position).all() and np.isfinite(root_quaternion).all()
    if not finite:
        raise DynamicRetimeError("dynamic audit inputs contain NaN or Inf")

    contract = load_g1_contract()
    segment_joint_velocity = np.diff(q, axis=0) * fps
    segment_joint_acceleration = np.diff(segment_joint_velocity, axis=0) * fps
    segment_root_speed = np.linalg.norm(np.diff(root_position, axis=0) * fps, axis=-1)
    quaternion_norm = np.linalg.norm(root_quaternion, axis=-1, keepdims=True)
    if np.any(quaternion_norm <= 1.0e-12):
        raise DynamicRetimeError("root quaternion contains a zero-length value")
    unit_quaternion = root_quaternion / quaternion_norm
    adjacent_dot = np.abs(np.sum(unit_quaternion[:-1] * unit_quaternion[1:], axis=-1))
    segment_root_angular_speed = 2.0 * np.arccos(np.clip(adjacent_dot, 0.0, 1.0)) * fps

    velocity_fraction = float(algorithm["joint_velocity_limit_fraction"])
    acceleration_multiple = float(
        algorithm["joint_acceleration_cap_velocity_multiple_per_second"]
    )
    factors = {
        "joint_velocity": float(
            np.max(np.abs(segment_joint_velocity) / (velocity_fraction * contract.velocity))
        ),
        "root_linear_speed": float(
            np.max(segment_root_speed) / float(algorithm["root_linear_speed_cap_mps"])
        ),
        "root_angular_speed": float(
            np.max(segment_root_angular_speed)
            / float(algorithm["root_angular_speed_cap_radps"])
        ),
        "joint_acceleration": float(
            np.sqrt(
                np.max(
                    np.abs(segment_joint_acceleration)
                    / (acceleration_multiple * contract.velocity)
                )
            )
        ),
    }
    raw_scale = max(1.0, *factors.values())
    scale = _quantize_scale(raw_scale, float(algorithm["scale_quantum"]))
    left_contact = np.asarray(arrays["left_foot_contact"], dtype=bool)
    right_contact = np.asarray(arrays["right_foot_contact"], dtype=bool)
    if left_contact.shape != (len(q),) or right_contact.shape != (len(q),):
        raise DynamicRetimeError("foot contact arrays do not match the G1 frame count")
    flight = ~(left_contact | right_contact)
    longest_flight = _longest_true_run(flight)
    flight_limit = int(
        algorithm["uniform_retime_maximum_consecutive_flight_frames_exclusive"]
    )
    if scale == 1.0:
        action = "identity"
        reason = "all frozen dynamic prefilter factors are at or below one"
    elif longest_flight >= flight_limit:
        action = "quarantine"
        reason = (
            f"uniform slowdown forbidden: longest flight run {longest_flight} >= {flight_limit} frames"
        )
    else:
        action = "uniform_retime"
        reason = "scale exceeds one and no prohibited flight run is present"

    def percentiles(values: np.ndarray) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(
                ("p50", "p95", "p99", "maximum"),
                np.percentile(np.abs(values), (50.0, 95.0, 99.0, 100.0)),
                strict=True,
            )
        }

    return {
        "schema_version": "accad_g1_dynamic_clip_audit_v1",
        "frame_count": int(len(q)),
        "fps": fps,
        "duration_sec": float((len(q) - 1) / fps),
        "factors": factors,
        "raw_scale": float(raw_scale),
        "scale_pre": scale,
        "playback_speed_equivalent": float(1.0 / scale),
        "action": action,
        "reason": reason,
        "flight": {
            "frame_count": int(np.count_nonzero(flight)),
            "frame_fraction": float(np.mean(flight)),
            "longest_consecutive_frames": longest_flight,
            "starts_in_flight": bool(flight[0]),
            "ends_in_flight": bool(flight[-1]),
        },
        "segment_metrics": {
            "joint_velocity_abs_radps": percentiles(segment_joint_velocity),
            "joint_acceleration_abs_radps2": percentiles(segment_joint_acceleration),
            "root_linear_speed_mps": percentiles(segment_root_speed),
            "root_angular_speed_radps": percentiles(segment_root_angular_speed),
            "maximum_joint_velocity_urdf_ratio": float(
                np.max(np.abs(segment_joint_velocity) / contract.velocity)
            ),
            "urdf_velocity_violation_count": int(
                np.count_nonzero(np.abs(segment_joint_velocity) > contract.velocity)
            ),
        },
    }


def _linear_resample(values: np.ndarray, source_indices: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    lower = np.floor(source_indices).astype(np.int64)
    upper = np.minimum(lower + 1, len(array) - 1)
    alpha = source_indices - lower
    reshape = (len(alpha),) + (1,) * (array.ndim - 1)
    return (1.0 - alpha.reshape(reshape)) * array[lower] + alpha.reshape(reshape) * array[upper]


def _nearest_resample(values: np.ndarray, source_indices: np.ndarray) -> np.ndarray:
    indices = np.floor(source_indices + 0.5).astype(np.int64)
    indices = np.clip(indices, 0, len(values) - 1)
    return np.asarray(values)[indices].copy()


def _retime_grid(frame_count: int, requested_scale: float, fps: float) -> tuple[np.ndarray, float]:
    if frame_count < 3 or requested_scale < 1.0 or not math.isfinite(requested_scale):
        raise ValueError("retime requires >=3 frames and a finite scale >=1")
    intervals = int(math.ceil((frame_count - 1) * requested_scale - 1.0e-12))
    intervals = max(frame_count - 1, intervals)
    realized_scale = intervals / (frame_count - 1)
    source_indices = np.arange(intervals + 1, dtype=np.float64) / realized_scale
    source_indices[-1] = frame_count - 1
    return source_indices, float(realized_scale)


def _fk_payload(
    joint_position: np.ndarray,
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    body_names: tuple[str, ...],
    *,
    ground_margin_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    kinematics = TorchG1Kinematics(device="cpu", dtype=torch.float64)

    def solve(position: np.ndarray) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        with torch.inference_mode():
            transforms = kinematics.forward(
                torch.as_tensor(joint_position, dtype=torch.float64),
                torch.as_tensor(position, dtype=torch.float64),
                torch.as_tensor(root_quaternion, dtype=torch.float64),
            )
            support = kinematics.foot_support_positions(transforms)
        return transforms, support

    transforms, support = solve(root_position)
    required_shift = np.maximum(
        FOOT_COLLISION_RADIUS_M
        + float(ground_margin_m)
        - np.min(support[..., 2].cpu().numpy(), axis=(1, 2)),
        0.0,
    )
    projected_root = np.asarray(root_position, dtype=np.float64).copy()
    projected_root[:, 2] += required_shift
    transforms, support = solve(projected_root)
    with torch.inference_mode():
        _, body_position, body_rotation = kinematics.stacked(transforms, body_names)
        sole_position = kinematics.sole_positions(transforms)
    body_rotation_np = body_rotation.cpu().numpy()
    body_quaternion = matrix_to_quaternion_wxyz(body_rotation_np)
    support_np64 = support.cpu().numpy()
    contact_points = np.stack(
        (
            support_np64[:, 0, :2].mean(axis=1),
            support_np64[:, 0, 2:].mean(axis=1),
            support_np64[:, 1, :2].mean(axis=1),
            support_np64[:, 1, 2:].mean(axis=1),
        ),
        axis=1,
    )
    return {
        "root_pos_w": projected_root.astype(np.float32),
        "body_pos_w": body_position.cpu().numpy().astype(np.float32),
        "body_quat_wxyz": body_quaternion,
        "foot_sole_position_w": sole_position.cpu().numpy().astype(np.float32),
        "foot_support_position_w": support_np64.reshape(len(projected_root), 8, 3).astype(
            np.float32
        ),
        "contact_point_position_w": contact_points.astype(np.float32),
    }, {
        "maximum_z_shift_m": float(np.max(required_shift)),
        "mean_z_shift_m": float(np.mean(required_shift)),
        "ground_margin_m": float(ground_margin_m),
    }


def derive_retimed_payload(
    arrays: Mapping[str, np.ndarray],
    *,
    requested_scale: float,
    source_archive_checksum_sha256: str,
    source_manifest_checksum_sha256: str,
    split: str,
    initial_scale_pre: float | None = None,
    source_action: str | None = None,
    require_post_feasible: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create a self-consistent v28 payload without writing it."""

    frame_count = int(_scalar(arrays, "frame_count"))
    fps = float(_scalar(arrays, "fps"))
    if fps != 50.0:
        raise DynamicRetimeError(f"offline dynamic derivation requires 50 Hz; got {fps}")
    unknown_frame_fields = sorted(
        name
        for name, value in arrays.items()
        if value.ndim >= 1 and value.shape[0] == frame_count and name not in KNOWN_FRAME_FIELDS
    )
    if unknown_frame_fields:
        raise DynamicRetimeError(
            f"unrecognized frame-aligned arrays cannot be retimed safely: {unknown_frame_fields}"
        )
    initial_scale = float(requested_scale if initial_scale_pre is None else initial_scale_pre)
    if (
        not math.isfinite(initial_scale)
        or initial_scale < 1.0
        or requested_scale + 1.0e-12 < initial_scale
    ):
        raise DynamicRetimeError(
            "initial_scale_pre must be finite, >=1, and no larger than requested_scale"
        )
    source_audit = audit_dynamic_feasibility(arrays)
    source_action_value = str(source_action or source_audit["action"])

    # Branch A is a true identity derivation: every frame-aligned numeric and
    # boolean array remains byte-for-byte equal after loading.  Only scalar
    # version/provenance fields are added or updated.  In particular, FK ground
    # projection cannot accidentally turn an S=1 source into a slowed flight
    # clip and thereby bypass the frozen flight policy.
    if requested_scale == 1.0:
        if source_audit["scale_pre"] != 1.0:
            raise DynamicRetimeError(
                "identity derivation requested for a source that requires dynamic scaling"
            )
        payload = {name: np.asarray(value).copy() for name, value in arrays.items()}
        payload.update(
            {
                "retarget_version": np.asarray(DYNAMIC_RETARGET_VERSION),
                "solver_name": np.asarray(
                    "torch_global_sequence_adam+dynamic_identity_numeric_copy"
                ),
                "dynamic_derivation_schema_version": np.asarray(
                    ARCHIVE_DERIVATION_SCHEMA_VERSION
                ),
                "dynamic_source_g1_checksum_sha256": np.asarray(
                    source_archive_checksum_sha256
                ),
                "dynamic_source_manifest_checksum_sha256": np.asarray(
                    source_manifest_checksum_sha256
                ),
                "dynamic_split": np.asarray(split),
                "dynamic_source_action": np.asarray(source_action_value),
                "dynamic_derivation_mode": np.asarray("identity_numeric_copy"),
                "dynamic_initial_scale_pre": np.asarray(initial_scale, dtype=np.float64),
                "dynamic_requested_scale": np.asarray(1.0, dtype=np.float64),
                "dynamic_realized_scale": np.asarray(1.0, dtype=np.float64),
                "dynamic_boolean_resampling": np.asarray("identity_copy"),
                "dynamic_continuous_resampling": np.asarray("identity_copy"),
                "gate3_complete": np.asarray(False),
                "gate3_status": np.asarray(
                    "dynamic v28 identity derivative generated; independent Gate-3 validation pending"
                ),
            }
        )
        post_audit = audit_dynamic_feasibility(payload)
        if post_audit["scale_pre"] != 1.0:
            raise DynamicRetimeError("identity derivative changed the frozen dynamic audit")
        duration = float((frame_count - 1) / fps)
        return payload, {
            "requested_scale": 1.0,
            "realized_scale": 1.0,
            "initial_scale_pre": initial_scale,
            "source_frame_count": frame_count,
            "output_frame_count": frame_count,
            "source_duration_sec": duration,
            "output_duration_sec": duration,
            "derivation_mode": "identity_numeric_copy",
            "ground_projection": {
                "mode": "preserved_source_arrays_without_reprojection",
                "maximum_z_shift_m": float(
                    _scalar(arrays, "ground_projection_maximum_z_shift_m")
                    if "ground_projection_maximum_z_shift_m" in arrays
                    else 0.0
                ),
            },
            "post_audit": post_audit,
        }

    source_indices, realized_scale = _retime_grid(frame_count, requested_scale, fps)
    source_times = np.arange(frame_count, dtype=np.float64) / fps
    query_times = source_indices / fps
    output_count = len(source_indices)

    joint_position = _linear_resample(arrays["joint_pos"], source_indices).astype(np.float32)
    root_position = _linear_resample(arrays["root_pos_w"], source_indices).astype(np.float32)
    root_quaternion = resample_quaternion_slerp(
        arrays["root_quat_wxyz"], source_times, query_times
    ).astype(np.float32)
    body_names = tuple(str(name) for name in arrays["body_names"].tolist())
    ground_margin = float(
        _scalar(arrays, "ground_projection_ground_margin_m")
        if "ground_projection_ground_margin_m" in arrays
        else 0.0005
    )
    fk, projection = _fk_payload(
        joint_position,
        root_position,
        root_quaternion,
        body_names,
        ground_margin_m=ground_margin,
    )
    root_position = fk["root_pos_w"]

    contacts = {
        name: _nearest_resample(arrays[name], source_indices).astype(bool)
        for name in (*FOOT_CONTACT_FIELDS, *CONTACT_FIELDS)
        if name in arrays
    }
    if all(name in contacts for name in CONTACT_FIELDS):
        contacts["left_foot_contact"] = contacts["left_heel_contact"] | contacts["left_toe_contact"]
        contacts["right_foot_contact"] = contacts["right_heel_contact"] | contacts["right_toe_contact"]

    stance_updates: dict[str, np.ndarray] = {}
    if all(name in arrays for name in STANCE_FIELDS):
        preliminary = np.stack(
            [_nearest_resample(arrays[name], source_indices) for name in STANCE_FIELDS], axis=1
        ).astype(bool)
        if all(name in contacts for name in CONTACT_FIELDS):
            preliminary &= np.stack([contacts[name] for name in CONTACT_FIELDS], axis=1)
        correspondence = load_correspondence()
        refined, runs = _refine_output_stance_channels(
            np.asarray(fk["contact_point_position_w"], dtype=np.float64),
            preliminary,
            fps=fps,
            solver=correspondence["solver"],
        )
        stance_updates = {
            name: refined[:, index] for index, name in enumerate(STANCE_FIELDS)
        }
        preliminary_count = int(np.count_nonzero(preliminary))
        output_count_stance = int(np.count_nonzero(refined))
        stance_metadata = {
            "preliminary_stance_frame_count": preliminary_count,
            "output_stance_frame_count": output_count_stance,
            "output_stance_retention_fraction": (
                float(output_count_stance / preliminary_count) if preliminary_count else 1.0
            ),
            "output_stance_run_count": len(runs),
            "contact_lock_run_count": sum(_true_run_count(refined[:, i]) for i in range(4)),
        }
    else:
        stance_metadata = {}

    contract = load_g1_contract()
    body_position = fk["body_pos_w"]
    body_quaternion = fk["body_quat_wxyz"]
    payload = {name: np.asarray(value).copy() for name, value in arrays.items()}
    payload.update(
        {
            "retarget_version": np.asarray(DYNAMIC_RETARGET_VERSION),
            "time": np.arange(output_count, dtype=np.float64) / fps,
            "frame_count": np.asarray(output_count, dtype=np.int64),
            "joint_pos": joint_position,
            "joint_pos_sdk": policy_to_sdk(joint_position, contract),
            "joint_vel": _linear_velocity(joint_position, fps),
            "joint_acc": _linear_velocity(_linear_velocity(joint_position, fps), fps),
            "root_pos_w": root_position,
            "root_quat_wxyz": root_quaternion,
            "root_lin_vel_w": _linear_velocity(root_position, fps),
            "root_ang_vel_w": _angular_velocity_world(root_quaternion, fps),
            "body_pos_w": body_position,
            "body_quat_wxyz": body_quaternion,
            "body_lin_vel_w": _linear_velocity(body_position, fps),
            "body_ang_vel_w": _angular_velocity_world(body_quaternion, fps),
            "foot_sole_position_w": fk["foot_sole_position_w"],
            "foot_support_position_w": fk["foot_support_position_w"],
            "contact_point_position_w": fk["contact_point_position_w"],
            **contacts,
            **stance_updates,
            "solver_name": np.asarray("torch_global_sequence_adam+offline_dynamic_retime"),
            "ground_projection_maximum_z_shift_m": np.asarray(
                projection["maximum_z_shift_m"], dtype=np.float32
            ),
            "ground_projection_ground_margin_m": np.asarray(
                projection["ground_margin_m"], dtype=np.float32
            ),
            "dynamic_derivation_schema_version": np.asarray(
                ARCHIVE_DERIVATION_SCHEMA_VERSION
            ),
            "dynamic_source_g1_checksum_sha256": np.asarray(
                source_archive_checksum_sha256
            ),
            "dynamic_source_manifest_checksum_sha256": np.asarray(
                source_manifest_checksum_sha256
            ),
            "dynamic_split": np.asarray(split),
            "dynamic_source_action": np.asarray(source_action_value),
            "dynamic_derivation_mode": np.asarray("uniform_retime_recompute_fk"),
            "dynamic_initial_scale_pre": np.asarray(initial_scale, dtype=np.float64),
            "dynamic_requested_scale": np.asarray(requested_scale, dtype=np.float64),
            "dynamic_realized_scale": np.asarray(realized_scale, dtype=np.float64),
            "dynamic_boolean_resampling": np.asarray("nearest_source_frame"),
            "dynamic_continuous_resampling": np.asarray(
                "linear_position_and_shortest_path_quaternion_slerp"
            ),
            "gate3_complete": np.asarray(False),
            "gate3_status": np.asarray(
                "dynamic v28 basis generated; independent Gate-3 validation pending"
            ),
        }
    )
    for name, value in stance_metadata.items():
        dtype = np.float32 if "fraction" in name else np.int64
        payload[name] = np.asarray(value, dtype=dtype)

    post_audit = audit_dynamic_feasibility(payload)
    if require_post_feasible and post_audit["scale_pre"] != 1.0:
        raise DynamicRetimeError(
            "retimed output still violates the frozen dynamic prefilter: "
            f"post scale={post_audit['scale_pre']}, factors={post_audit['factors']}"
        )
    derivation = {
        "requested_scale": float(requested_scale),
        "realized_scale": realized_scale,
        "initial_scale_pre": initial_scale,
        "source_frame_count": frame_count,
        "output_frame_count": output_count,
        "source_duration_sec": float((frame_count - 1) / fps),
        "output_duration_sec": float((output_count - 1) / fps),
        "ground_projection": projection,
        "derivation_mode": "uniform_retime_recompute_fk",
        "post_audit": post_audit,
    }
    return payload, derivation


def derive_retimed_archive(
    source_path: str | Path,
    output_path: str | Path,
    *,
    requested_scale: float,
    source_manifest_checksum_sha256: str,
    split: str,
    frozen_source_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve()
    output = assert_output_is_local(output_path)
    if source == output:
        raise DynamicRetimeError("dynamic output must never overwrite its v27 source")
    source_checksum = _sha256(source)
    arrays = _safe_npz(source)
    computed_source_audit = audit_dynamic_feasibility(arrays)
    if frozen_source_audit is not None:
        witness = {
            "action": frozen_source_audit.get("action"),
            "scale_pre": float(frozen_source_audit.get("scale_pre", math.nan)),
            "longest_flight": int(
                frozen_source_audit.get("flight", {}).get(
                    "longest_consecutive_frames", -1
                )
            ),
        }
        recomputed = {
            "action": computed_source_audit["action"],
            "scale_pre": float(computed_source_audit["scale_pre"]),
            "longest_flight": int(
                computed_source_audit["flight"]["longest_consecutive_frames"]
            ),
        }
        if witness != recomputed:
            raise DynamicRetimeError(
                f"frozen source audit does not match recomputation: {witness} != {recomputed}"
            )
        if computed_source_audit["action"] == "quarantine":
            raise DynamicRetimeError("a source-flight quarantine must not enter derivation")
        if not math.isclose(
            float(requested_scale),
            float(computed_source_audit["scale_pre"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise DynamicRetimeError(
                "requested scale does not equal the frozen source S_pre"
            )

    initial_scale = float(requested_scale)
    current_requested_scale = initial_scale
    algorithm = POLICY_ALGORITHM
    post_policy = algorithm["post_fk_headroom"]
    maximum_attempts = int(post_policy["maximum_attempts"])
    headroom_multiplier = float(post_policy["multiplier"])
    flight_limit = int(
        algorithm["uniform_retime_maximum_consecutive_flight_frames_exclusive"]
    )
    attempts: list[dict[str, Any]] = []
    payload: dict[str, np.ndarray] | None = None
    derivation: dict[str, Any] | None = None
    for attempt_index in range(1, maximum_attempts + 1):
        payload, derivation = derive_retimed_payload(
            arrays,
            requested_scale=current_requested_scale,
            initial_scale_pre=initial_scale,
            source_action=str(computed_source_audit["action"]),
            source_archive_checksum_sha256=source_checksum,
            source_manifest_checksum_sha256=source_manifest_checksum_sha256,
            split=split,
            require_post_feasible=False,
        )
        post_audit = derivation["post_audit"]
        attempt = {
            "attempt_index": attempt_index,
            "requested_scale": float(current_requested_scale),
            "realized_scale": float(derivation["realized_scale"]),
            "output_frame_count": int(derivation["output_frame_count"]),
            "post_raw_scale": float(post_audit["raw_scale"]),
            "post_scale_pre": float(post_audit["scale_pre"]),
            "post_factors": post_audit["factors"],
            "post_longest_flight_frames": int(
                post_audit["flight"]["longest_consecutive_frames"]
            ),
        }
        attempts.append(attempt)
        if post_audit["scale_pre"] == 1.0:
            break
        if post_audit["flight"]["longest_consecutive_frames"] >= flight_limit:
            raise DynamicRetimeQuarantine(
                "additional post-FK stretch is required while the derived clip has "
                f"a flight run >= {flight_limit} frames",
                {
                    "source_action": computed_source_audit["action"],
                    "source_scale_pre": computed_source_audit["scale_pre"],
                    "source_longest_flight_frames": computed_source_audit["flight"][
                        "longest_consecutive_frames"
                    ],
                    "failed_attempts": attempts,
                    "final_action": "quarantine_post_fk_headroom_flight",
                },
            )
        if attempt_index == maximum_attempts:
            raise DynamicRetimeError(
                "post-FK dynamic headroom did not converge after "
                f"{maximum_attempts} attempts: {attempts}"
            )

        # The raw residual factor predicts the extra slowdown.  A fixed 2%
        # guard and at least one additional 50 Hz output interval make every
        # retry strictly monotonic and deterministic.
        minimum_scale_for_interval_growth = math.nextafter(
            float(derivation["realized_scale"]), math.inf
        )
        proposed_scale = max(
            current_requested_scale
            * max(1.0, float(post_audit["raw_scale"]))
            * headroom_multiplier,
            minimum_scale_for_interval_growth,
        )
        next_requested_scale = _quantize_scale(
            proposed_scale, float(algorithm["scale_quantum"])
        )
        if next_requested_scale <= current_requested_scale:
            next_requested_scale = _quantize_scale(
                current_requested_scale + float(algorithm["scale_quantum"]),
                float(algorithm["scale_quantum"]),
            )
        current_requested_scale = next_requested_scale

    if payload is None or derivation is None or derivation["post_audit"]["scale_pre"] != 1.0:
        raise DynamicRetimeError("internal error: no feasible dynamic payload was produced")
    requested_monotonic = all(
        later["requested_scale"] > earlier["requested_scale"]
        for earlier, later in zip(attempts, attempts[1:])
    )
    frame_count_monotonic = all(
        later["output_frame_count"] > earlier["output_frame_count"]
        for earlier, later in zip(attempts, attempts[1:])
    )
    if not requested_monotonic or not frame_count_monotonic:
        raise DynamicRetimeError(f"non-monotonic headroom retry witness: {attempts}")
    if initial_scale == 1.0:
        final_action = "identity_numeric_copy"
    elif len(attempts) == 1:
        final_action = "uniform_retime"
    else:
        final_action = "uniform_retime_with_post_fk_headroom"
    payload["dynamic_final_action"] = np.asarray(final_action)
    payload["dynamic_headroom_attempt_count"] = np.asarray(
        len(attempts), dtype=np.int64
    )
    _save_npz_atomic(output, payload)
    return {
        **derivation,
        "source_action": computed_source_audit["action"],
        "source_scale_pre": float(computed_source_audit["scale_pre"]),
        "final_action": final_action,
        "effective_requested_scale": float(current_requested_scale),
        "headroom_applied": len(attempts) > 1,
        "headroom_policy": post_policy,
        "headroom_attempts": attempts,
        "monotonic_witness": {
            "requested_scale_strictly_increasing": requested_monotonic,
            "output_frame_count_strictly_increasing": frame_count_monotonic,
        },
        "path": str(output),
        "bytes": output.stat().st_size,
        "checksum_sha256": _sha256(output),
        "retarget_version": DYNAMIC_RETARGET_VERSION,
    }


def _load_manifest(path: Path, split: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DynamicRetimeError(f"cannot read source manifest {path}: {exc}") from exc
    if document.get("schema_version") != "accad_g1_gate3_manifest_v1":
        raise DynamicRetimeError("dynamic stage requires an immutable promoted Gate-3 manifest")
    if document.get("split") != split or document.get("kind") != f"split:{split}":
        raise DynamicRetimeError(f"manifest split does not equal explicit --split {split!r}")
    motions = document.get("motions")
    if not isinstance(motions, list) or int(document.get("num_motions", -1)) != len(motions):
        raise DynamicRetimeError("source manifest motion count is invalid")
    return document


def _policy_document(train_manifest: Path, train_document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "status": "frozen_from_train",
        "created_at_utc": _utc_now(),
        "algorithm": POLICY_ALGORITHM,
        "algorithm_checksum_sha256": _canonical_checksum(POLICY_ALGORITHM),
        "train_source_manifest": {
            "path": train_manifest.relative_to(PIPELINE_ROOT).as_posix(),
            "checksum_sha256": _sha256(train_manifest),
            "motion_set_checksum_sha256": train_document.get(
                "motion_set_checksum_sha256"
            ),
            "num_motions": int(train_document["num_motions"]),
        },
        "split_policy": {
            "train": "allowed_and_freezes_policy",
            "validation": "allowed_only_with_existing_frozen_train_policy",
            "test": "sealed_until_explicit_split_test_invocation",
        },
    }


def _load_or_freeze_policy(
    *,
    split: str,
    manifest_path: Path,
    manifest_document: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    policy_path = output_root / "scale_policy.json"
    if split == "train":
        expected = _policy_document(manifest_path, manifest_document)
        if policy_path.exists():
            existing = json.loads(policy_path.read_text(encoding="utf-8"))
            comparable_existing = dict(existing)
            comparable_expected = dict(expected)
            comparable_existing.pop("created_at_utc", None)
            comparable_expected.pop("created_at_utc", None)
            if comparable_existing != comparable_expected:
                raise DynamicRetimeError(
                    "existing frozen train policy differs from the requested train manifest/algorithm"
                )
            return existing, policy_path
        _atomic_json(policy_path, expected)
        return expected, policy_path
    if not policy_path.is_file():
        raise DynamicRetimeError(
            f"{split} dynamic processing requires work/dynamic/scale_policy.json frozen by train"
        )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        policy.get("schema_version") != POLICY_SCHEMA_VERSION
        or policy.get("status") != "frozen_from_train"
        or policy.get("algorithm") != POLICY_ALGORITHM
        or policy.get("algorithm_checksum_sha256") != _canonical_checksum(POLICY_ALGORITHM)
    ):
        raise DynamicRetimeError("frozen scale policy is missing, altered, or unsupported")
    return policy, policy_path


def build_dynamic_split(
    *,
    split: str,
    manifest_path: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
    revalidate: bool = True,
) -> dict[str, Any]:
    """Audit and publish one explicit split; never scans another split."""

    if split not in DEFAULT_INPUT_MANIFESTS:
        raise ValueError("split must be exactly train, validation, or test")
    source_manifest = Path(
        manifest_path if manifest_path is not None else DEFAULT_INPUT_MANIFESTS[split]
    ).expanduser().resolve()
    dynamic_root = assert_output_is_local(output_root)
    source_document = _load_manifest(source_manifest, split)
    policy, policy_path = _load_or_freeze_policy(
        split=split,
        manifest_path=source_manifest,
        manifest_document=source_document,
        output_root=dynamic_root,
    )
    plan_path = dynamic_root / f"g1_{split}_scale_plan.json"
    report_path = dynamic_root / f"g1_{split}_report.json"
    manifest_output_path = dynamic_root / f"g1_{split}.json"
    existing_publications = [path for path in (plan_path, report_path, manifest_output_path) if path.exists()]
    if existing_publications and not force:
        raise DynamicRetimeError(
            "dynamic split outputs already exist; use --force only to replace derived outputs: "
            + ", ".join(str(path) for path in existing_publications)
        )

    source_manifest_checksum = _sha256(source_manifest)
    source_reference_root = (
        PIPELINE_ROOT / str(source_document["reference_root"])
    ).resolve()
    derived_reference_root = dynamic_root / "g1"
    records: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []
    validation_failures: list[dict[str, Any]] = []
    started = _utc_now()

    for source_record in source_document["motions"]:
        relative = Path(str(source_record["g1_file"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise DynamicRetimeError(f"unsafe source G1 path in manifest: {relative}")
        source_path = (source_reference_root / relative).resolve()
        try:
            source_path.relative_to(source_reference_root)
        except ValueError as exc:
            raise DynamicRetimeError(f"source G1 path escapes reference root: {relative}") from exc
        if not source_path.is_file():
            raise DynamicRetimeError(f"source G1 archive is missing: {source_path}")
        actual_checksum = _sha256(source_path)
        if actual_checksum != source_record.get("g1_checksum_sha256"):
            raise DynamicRetimeError(f"source G1 checksum mismatch: {relative}")
        if source_path.stat().st_size != int(source_record.get("g1_size_bytes", -1)):
            raise DynamicRetimeError(f"source G1 size mismatch: {relative}")

        arrays = _safe_npz(source_path)
        audit = audit_dynamic_feasibility(arrays, algorithm=policy["algorithm"])
        item = {
            "source_g1_file": relative.as_posix(),
            "source_g1_checksum_sha256": actual_checksum,
            "source_frame_count": int(source_record["frame_count"]),
            "source_duration_sec": float(source_record["duration_sec"]),
            "audit": audit,
            "source_action": audit["action"],
            "status": "quarantined" if audit["action"] == "quarantine" else "deriving",
        }
        if audit["action"] == "quarantine":
            item["quarantine_reason"] = audit["reason"]
            item["final_action"] = "quarantine_source_flight_policy"
            records.append(item)
            continue

        output_path = assert_output_is_local(derived_reference_root / relative)
        if output_path.exists() and not force:
            raise DynamicRetimeError(f"derived archive already exists: {output_path}")
        try:
            derivation = derive_retimed_archive(
                source_path,
                output_path,
                requested_scale=float(audit["scale_pre"]),
                source_manifest_checksum_sha256=source_manifest_checksum,
                split=split,
                frozen_source_audit=audit,
            )
        except DynamicRetimeQuarantine as exc:
            item.update(
                {
                    "status": "quarantined_after_derivation_audit",
                    "final_action": "quarantine_post_fk_headroom_flight",
                    "quarantine_reason": exc.reason,
                    "quarantine_evidence": exc.evidence,
                }
            )
            records.append(item)
            continue
        validation: dict[str, Any]
        if revalidate:
            from .g1_validation import validate_g1_archive

            human_path = (PIPELINE_ROOT / str(source_record["human_file"])).resolve()
            validation = validate_g1_archive(output_path, expected_source_path=human_path)
            if validation.get("gate3_kinematic_complete") is not True:
                failure = {
                    "g1_file": relative.as_posix(),
                    "required_failures": validation.get("required_failures"),
                    "errors": validation.get("errors"),
                }
                validation_failures.append(failure)
                raise DynamicRetimeError(
                    f"derived archive failed independent Gate-3 validation: {failure}"
                )
        else:
            validation = {"gate3_kinematic_complete": None, "skipped": True}
        item.update(
            {
                "status": "derived_and_validated" if revalidate else "derived_unvalidated",
                "final_action": derivation["final_action"],
                "derived_g1_file": relative.as_posix(),
                "derivation": derivation,
                "gate3_kinematic_complete": validation.get("gate3_kinematic_complete"),
            }
        )
        records.append(item)
        promoted.append(
            {
                **source_record,
                "source_v27_g1_file": relative.as_posix(),
                "source_v27_g1_checksum_sha256": actual_checksum,
                "g1_file": relative.as_posix(),
                "g1_checksum_sha256": derivation["checksum_sha256"],
                "g1_size_bytes": derivation["bytes"],
                "retarget_version": DYNAMIC_RETARGET_VERSION,
                "fps": 50.0,
                "frame_count": derivation["output_frame_count"],
                "duration_sec": derivation["output_duration_sec"],
                "dynamic_source_action": audit["action"],
                "dynamic_action": derivation["final_action"],
                "dynamic_scale_pre": audit["scale_pre"],
                "dynamic_effective_requested_scale": derivation[
                    "effective_requested_scale"
                ],
                "dynamic_realized_scale": derivation["realized_scale"],
                "dynamic_policy_checksum_sha256": _sha256(policy_path),
                "gate3_kinematic_complete": bool(revalidate),
            }
        )

    actions = {
        name: sum(record["audit"]["action"] == name for record in records)
        for name in ("identity", "uniform_retime", "quarantine")
    }
    final_actions = {
        name: sum(record.get("final_action") == name for record in records)
        for name in sorted({str(record.get("final_action")) for record in records})
    }
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "completed",
        "split": split,
        "created_at_utc": _utc_now(),
        "source_manifest": {
            "path": source_manifest.relative_to(PIPELINE_ROOT).as_posix(),
            "checksum_sha256": source_manifest_checksum,
            "motion_set_checksum_sha256": source_document.get("motion_set_checksum_sha256"),
        },
        "frozen_policy": {
            "path": policy_path.relative_to(PIPELINE_ROOT).as_posix(),
            "checksum_sha256": _sha256(policy_path),
            "algorithm_checksum_sha256": policy["algorithm_checksum_sha256"],
        },
        "counts": {
            "source": len(records),
            **actions,
            "derived": len(promoted),
            "final_quarantined": len(records) - len(promoted),
            "source_actions": actions,
            "final_actions": final_actions,
        },
        "records": records,
    }
    _atomic_json(plan_path, plan)

    promoted.sort(key=lambda record: record["g1_file"])
    motion_set_checksum = hashlib.sha256(
        json.dumps(
            [[split, record["g1_file"], record["g1_checksum_sha256"]] for record in promoted],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    output_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "ACCAD",
        "kind": f"split:{split}",
        "split": split,
        "gate": "gate3_kinematic_complete+dynamic_prefilter_v1",
        "reference_root": derived_reference_root.relative_to(PIPELINE_ROOT).as_posix(),
        "retarget_version": DYNAMIC_RETARGET_VERSION,
        "motion_set_checksum_sha256": motion_set_checksum,
        "source_motion_set_checksum_sha256": source_document.get("motion_set_checksum_sha256"),
        "source_manifest_checksum_sha256": source_manifest_checksum,
        "dynamic_policy_checksum_sha256": _sha256(policy_path),
        "scale_plan_checksum_sha256": _sha256(plan_path),
        "num_source_motions": len(records),
        "num_motions": len(promoted),
        "num_source_policy_quarantined": actions["quarantine"],
        "num_quarantined": len(records) - len(promoted),
        "num_frames": sum(int(record["frame_count"]) for record in promoted),
        "duration_sec": sum(float(record["duration_sec"]) for record in promoted),
        "motions": promoted,
    }
    _atomic_json(manifest_output_path, output_manifest)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "completed" if not validation_failures else "failed",
        "split": split,
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "counts": plan["counts"],
        "validation_failures": validation_failures,
        "sealed_test_contract": (
            "test NPZ numerical contents are opened only by an explicit split=test invocation"
        ),
        "artifacts": {
            "policy": {
                "path": policy_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": _sha256(policy_path),
            },
            "plan": {
                "path": plan_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": _sha256(plan_path),
            },
            "manifest": {
                "path": manifest_output_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": _sha256(manifest_output_path),
            },
            "derived_reference_root": derived_reference_root.relative_to(PIPELINE_ROOT).as_posix(),
        },
    }
    _atomic_json(report_path, report)
    return report


__all__ = [
    "ARCHIVE_DERIVATION_SCHEMA_VERSION",
    "DEFAULT_INPUT_MANIFESTS",
    "DEFAULT_OUTPUT_ROOT",
    "DYNAMIC_RETARGET_VERSION",
    "DynamicRetimeError",
    "DynamicRetimeQuarantine",
    "POLICY_ALGORITHM",
    "audit_dynamic_feasibility",
    "build_dynamic_split",
    "derive_retimed_archive",
    "derive_retimed_payload",
]
