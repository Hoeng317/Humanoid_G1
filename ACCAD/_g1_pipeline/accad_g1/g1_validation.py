"""Independent numeric validation for retargeted Unitree-G1 references.

This module intentionally does not use :mod:`accad_g1.g1_kinematics` for
forward kinematics.  The retargeter is implemented with a local Torch FK, so a
validator that reused it could reproduce the same implementation error.  Here
the authoritative YAML/URDF are parsed again and every stored body transform is
checked with Pinocchio's free-flyer model.

The public archive convention is policy-order joints and ``wxyz``
quaternions.  Pinocchio instead expects SDK/URDF-order joints and an ``xyzw``
free-flyer quaternion.  Both conversions are performed explicitly below.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .paths import ACCAD_ROOT, PIPELINE_ROOT, PROJECT_ROOT


G1_CONTRACT_PATH = PROJECT_ROOT / "configs" / "robot" / "g1_29dof.yaml"
G1_URDF_PATH = (
    PROJECT_ROOT
    / "third_party"
    / "unitree_ros"
    / "robots"
    / "g1_description"
    / "g1_29dof_rev_1_0.urdf"
)
DEFAULT_CORRESPONDENCE_PATH = PIPELINE_ROOT / "correspondence.yaml"

# Audited from the current Isaac URDF conversion.  Order is deliberately not
# prescribed: an archive carries body_names and the validator resolves every
# Pinocchio frame by name.  The set and count, however, are part of the G1
# reference schema.
EXPECTED_BODY_NAMES = frozenset(
    {
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "left_rubber_hand",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
        "right_rubber_hand",
    }
)

EXPECTED_TRACKING_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

EXPECTED_FOOT_SUPPORT_NAMES = (
    "left_rear_outer",
    "left_rear_inner",
    "left_front_outer",
    "left_front_inner",
    "right_rear_inner",
    "right_rear_outer",
    "right_front_inner",
    "right_front_outer",
)

SOFT_JOINT_POSITION_FACTOR = 0.9

DEFAULT_THRESHOLDS: dict[str, float] = {
    "time_seconds": 1.0e-12,
    "quaternion_norm": 1.0e-5,
    "quaternion_minimum_adjacent_dot": -1.0e-6,
    "policy_sdk_position_rad": 1.0e-7,
    "stored_velocity": 2.0e-4,
    # The second-order one-sided derivative used at the first/last frame has
    # four times the L1 coefficient norm of the centered interior derivative.
    # Keep the interior cross-backend check strict, while allowing that known
    # float32-Torch versus float64-Pinocchio endpoint amplification only at the
    # two boundary frames.  Serialized-position self-consistency is checked at
    # every frame below, so this does not hide a corrupt endpoint velocity.
    "pinocchio_body_velocity_endpoint": 8.0e-4,
    "hard_limit_rad": 1.0e-6,
    "soft_limit_rad": 1.0e-6,
    "velocity_limit_rad_s": 1.0e-4,
    "fk_position_m": 2.0e-5,
    # Float32 quaternions converted through rotation matrices produce errors
    # around 5e-4 rad even when the underlying FK agrees to float precision.
    "fk_orientation_rad": 1.0e-3,
    "stored_sole_position_m": 2.0e-5,
    "stored_foot_radius_m": 1.0e-7,
    "foot_penetration_m": 2.0e-3,
    # Gate-2 accepts semantic contact-point speeds up to 0.60 m/s.  At the
    # fixed 50 Hz control rate this is exactly 0.012 m per step; longer-term
    # slip remains independently bounded by stance_run_drift_xy_m.
    "stance_support_step_xy_m": 1.2e-2,
    # Bounded one-second stance anchors may accumulate a few centimetres of
    # morphology/IK residual.  This remains a strict total-path bound and is
    # paired with the 0.012 m single-step check above.
    "stance_run_drift_xy_m": 3.5e-2,
    "metadata_float": 1.0e-6,
}

DETAILED_CONTACT_KEYS = {
    "foot_support_names",
    "foot_support_position_w",
    "contact_point_names",
    "contact_point_position_w",
    "left_heel_contact",
    "left_toe_contact",
    "right_heel_contact",
    "right_toe_contact",
    "contact_lock_run_count",
}

# A stance label is deliberately distinct from a geometric contact label.  A
# foot can remain close enough to the floor to be classified as contact while
# sliding during crawl or another non-support phase.  Contact therefore gates
# active penetration metrics; stance alone gates support-point slip metrics.
# These fields are optional for backwards compatibility, but partial presence
# is never accepted because it would make the left/right or heel/toe semantics
# ambiguous.
STANCE_KEYS = (
    "left_heel_stance",
    "left_toe_stance",
    "right_heel_stance",
    "right_toe_stance",
)


REQUIRED_KEYS = {
    "schema_version",
    "retarget_version",
    "fps",
    "time",
    "frame_count",
    "coordinate_system",
    "quaternion_order",
    "position_unit",
    "angle_unit",
    "joint_order",
    "joint_names",
    "sdk_joint_names",
    "policy_to_sdk",
    "joint_pos",
    "joint_pos_sdk",
    "joint_vel",
    "joint_acc",
    "root_pos_w",
    "root_quat_wxyz",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "body_names",
    "body_pos_w",
    "body_quat_wxyz",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "tracking_body_names",
    "foot_sole_names",
    "foot_sole_position_w",
    "foot_collision_radius_m",
    "left_foot_contact",
    "right_foot_contact",
    "morphology_scale",
    "human_leg_height_m",
    "g1_leg_height_m",
    "source_file",
    "source_checksum_sha256",
    "source_human_schema",
    "source_stageii_file",
    "source_stageii_checksum_sha256",
    "g1_contract_path",
    "g1_contract_checksum_sha256",
    "g1_urdf_path",
    "g1_urdf_checksum_sha256",
    "correspondence_path",
    "correspondence_checksum_sha256",
    "solver_name",
    "solver_iterations",
    "tracking_action_mode",
    "tracking_residual_scale",
    "tracking_action_clip",
    "tracking_normalized_delta_limit",
    "gate3_complete",
    "gate3_status",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load all arrays without permitting Python object deserialization."""

    arrays: dict[str, np.ndarray] = {}
    unreadable: dict[str, str] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            try:
                arrays[name] = archive[name]
            except ValueError as exc:
                unreadable[name] = str(exc)
    return arrays, unreadable


def _scalar(arrays: dict[str, np.ndarray], name: str) -> Any:
    value = arrays[name]
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{name} must be a safe scalar; got {value.shape}/{value.dtype}")
    return value.item()


def _gradient(values: np.ndarray, fps: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 2:
        raise ValueError("velocity input must contain at least two frames")
    edge_order = 2 if array.shape[0] >= 3 else 1
    return np.gradient(array, 1.0 / fps, axis=0, edge_order=edge_order)


def _continuous_unit_quaternion(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-12):
        raise ValueError("zero-length quaternion")
    result /= norm
    for frame in range(1, len(result)):
        flip = np.sum(result[frame - 1] * result[frame], axis=-1, keepdims=True) < 0.0
        result[frame] = np.where(flip, -result[frame], result[frame])
    return result


def _quaternion_to_matrix_wxyz(quaternions: np.ndarray) -> np.ndarray:
    q = _continuous_unit_quaternion(quaternions)
    shape = q.shape[:-1]
    xyzw = q.reshape(-1, 4)[:, [1, 2, 3, 0]]
    return Rotation.from_quat(xyzw).as_matrix().reshape(shape + (3, 3))


def _angular_velocity_world(quaternions: np.ndarray, fps: float) -> np.ndarray:
    rotations = _quaternion_to_matrix_wxyz(quaternions)
    relative = rotations[1:] @ np.swapaxes(rotations[:-1], -1, -2)
    interval = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
    interval = interval.reshape(relative.shape[:-2] + (3,)) * fps
    output = np.empty(rotations.shape[:-2] + (3,), dtype=np.float64)
    output[0] = interval[0]
    output[-1] = interval[-1]
    if len(output) > 2:
        output[1:-1] = 0.5 * (interval[:-1] + interval[1:])
    return output


def _rotation_error_rad(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(expected, -1, -2) @ actual
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.arccos(cosine)


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_contract() -> dict[str, Any]:
    payload = yaml.safe_load(G1_CONTRACT_PATH.read_text(encoding="utf-8"))
    joints = sorted(payload["joints"], key=lambda item: int(item["policy_index"]))
    policy_names = tuple(str(item["name"]) for item in joints)
    policy_to_sdk = np.asarray([int(item["sdk_index"]) for item in joints], dtype=np.int64)
    if len(joints) != 29 or sorted(policy_to_sdk.tolist()) != list(range(29)):
        raise ValueError("authoritative G1 contract is not a 29-joint permutation")
    sdk_names = [""] * 29
    for name, sdk_index in zip(policy_names, policy_to_sdk, strict=True):
        sdk_names[int(sdk_index)] = name

    def field(name: str) -> np.ndarray:
        return np.asarray([float(item[name]) for item in joints], dtype=np.float64)

    return {
        "variant": str(payload["robot"]["variant"]),
        "policy_names": policy_names,
        "sdk_names": tuple(sdk_names),
        "policy_to_sdk": policy_to_sdk,
        "lower": field("lower"),
        "upper": field("upper"),
        "velocity": field("velocity"),
        "default": field("default"),
        "action_scale": float(payload["control"]["action_scale"]),
        "physics_dt": float(payload["timing"]["physics_dt"]),
        "decimation": int(payload["timing"]["decimation"]),
    }


def _parse_foot_support_spheres() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = ET.parse(G1_URDF_PATH).getroot()
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for link_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        link = next(
            (element for element in root.findall("link") if element.attrib.get("name") == link_name),
            None,
        )
        if link is None:
            raise ValueError(f"URDF has no link {link_name}")
        centers: list[np.ndarray] = []
        radii: list[float] = []
        for collision in link.findall("collision"):
            sphere = collision.find("./geometry/sphere")
            if sphere is None:
                continue
            origin = collision.find("origin")
            xyz_text = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            centers.append(np.asarray([float(value) for value in xyz_text.split()], dtype=np.float64))
            radii.append(float(sphere.attrib["radius"]))
        if len(centers) != 4:
            raise ValueError(f"{link_name} must expose exactly four support spheres; got {len(centers)}")
        result[link_name] = (np.stack(centers), np.asarray(radii, dtype=np.float64))
    return result


def _pinocchio_fk(
    root_position: np.ndarray,
    root_quaternion_wxyz: np.ndarray,
    joint_position_sdk: np.ndarray,
    body_names: tuple[str, ...],
    expected_sdk_names: tuple[str, ...],
) -> dict[str, Any]:
    try:
        import pinocchio as pin
    except ImportError as exc:  # pragma: no cover - exercised only outside Isaac Python
        raise RuntimeError(
            "Pinocchio is required; run validation with _isaac_sim/python.sh"
        ) from exc

    model = pin.buildModelFromUrdf(str(G1_URDF_PATH), pin.JointModelFreeFlyer())
    model_joint_names = tuple(str(name) for name in model.names[2:])
    if model_joint_names != expected_sdk_names:
        raise ValueError(
            "Pinocchio/URDF joint order differs from authoritative SDK order: "
            f"{model_joint_names}"
        )
    missing = [name for name in body_names if not model.existFrame(name)]
    if missing:
        raise ValueError(f"Pinocchio model is missing body frames: {missing}")
    frame_ids = [model.getFrameId(name) for name in body_names]
    data = model.createData()
    frames = len(root_position)
    body_position = np.empty((frames, len(body_names), 3), dtype=np.float64)
    body_rotation = np.empty((frames, len(body_names), 3, 3), dtype=np.float64)
    body_quaternion = np.empty((frames, len(body_names), 4), dtype=np.float64)
    q_pin = np.empty(model.nq, dtype=np.float64)
    for frame in range(frames):
        q_pin[:3] = root_position[frame]
        # Archive wxyz -> Pinocchio free-flyer xyzw.
        q_pin[3:7] = root_quaternion_wxyz[frame, [1, 2, 3, 0]]
        q_pin[7:] = joint_position_sdk[frame]
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacements(model, data)
        for body_index, frame_id in enumerate(frame_ids):
            placement = data.oMf[frame_id]
            body_position[frame, body_index] = placement.translation
            body_rotation[frame, body_index] = placement.rotation
            xyzw = np.asarray(pin.Quaternion(placement.rotation).coeffs(), dtype=np.float64)
            body_quaternion[frame, body_index] = xyzw[[3, 0, 1, 2]]
    body_quaternion = _continuous_unit_quaternion(body_quaternion)
    return {
        "pinocchio_version": str(pin.__version__),
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_joint_names": model_joint_names,
        "body_position": body_position,
        "body_rotation": body_rotation,
        "body_quaternion": body_quaternion,
    }


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(values, dtype=bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def validate_g1_archive(
    archive_path: str | Path,
    *,
    expected_source_path: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Independently validate one G1 reference archive.

    A successful result sets ``gate3_kinematic_complete``.  It deliberately
    leaves ``gate3_complete`` false: name-resolved replay inside a live Isaac
    articulation is a separate required gate and cannot be inferred from an
    offline NPZ/FK check.
    """

    path = Path(archive_path).expanduser().resolve()
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        unknown = sorted(set(thresholds) - set(limits))
        if unknown:
            raise ValueError(f"unknown Gate-3 thresholds: {unknown}")
        limits.update({name: float(value) for name, value in thresholds.items()})
    checks: dict[str, list[dict[str, Any]]] = {
        "schema": [],
        "provenance": [],
        "contract": [],
        "numeric": [],
        "limits": [],
        "pinocchio_fk": [],
        "contact_geometry": [],
        "action_contract": [],
    }
    metrics: dict[str, Any] = {}
    warnings: list[str] = []

    def check(
        group: str,
        name: str,
        passed: bool,
        *,
        required: bool = True,
        **details: Any,
    ) -> None:
        checks[group].append(
            {"name": name, "passed": bool(passed), "required": bool(required), **details}
        )

    def finish(message: str | None = None) -> dict[str, Any]:
        statuses = {
            group: all(item["passed"] for item in items if item["required"])
            for group, items in checks.items()
        }
        required_failures = [
            f"{group}.{item['name']}"
            for group, items in checks.items()
            for item in items
            if item["required"] and not item["passed"]
        ]
        kinematic_complete = bool(statuses) and all(statuses.values())
        if message and message not in warnings:
            warnings.append(message)
        if "Isaac name-resolved replay has not been executed by this offline validator." not in warnings:
            warnings.append(
                "Isaac name-resolved replay has not been executed by this offline validator."
            )
        return {
            "ok": kinematic_complete,
            "gate3_kinematic_complete": kinematic_complete,
            "gate3_complete": False,
            "gate3_status": (
                "kinematic validation passed; Isaac replay pending"
                if kinematic_complete
                else "kinematic validation failed; Isaac replay not authoritative"
            ),
            "isaac_replay": {
                "required": True,
                "status": "pending",
                "validated_by_this_report": False,
            },
            "archive_path": str(path),
            "archive_checksum_sha256": _sha256(path) if path.is_file() else None,
            "thresholds": limits,
            "statuses": statuses,
            "required_failures": required_failures,
            "metrics": metrics,
            "warnings": warnings,
            "checks": checks,
        }

    try:
        arrays, unreadable = _safe_load_npz(path)
    except (OSError, ValueError) as exc:
        check("schema", "safe_no_pickle_load", False, error=str(exc))
        return finish("Archive could not be opened with allow_pickle=False.")

    missing = sorted(REQUIRED_KEYS.difference(arrays))
    check("schema", "required_keys", not missing, missing=missing)
    check("schema", "safe_no_pickle_load", not unreadable, unreadable=unreadable)
    object_fields = sorted(name for name, value in arrays.items() if value.dtype.hasobject)
    check("schema", "no_object_dtype", not object_fields, fields=object_fields)
    finite_failures = sorted(
        name
        for name, value in arrays.items()
        if value.dtype.kind in {"f", "i", "u", "c"} and not np.isfinite(value).all()
    )
    check("schema", "all_numeric_finite", not finite_failures, fields=finite_failures)
    numeric_fields = {
        "fps",
        "time",
        "frame_count",
        "policy_to_sdk",
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
        "foot_collision_radius_m",
        "morphology_scale",
        "human_leg_height_m",
        "g1_leg_height_m",
        "solver_iterations",
        "tracking_residual_scale",
        "tracking_action_clip",
        "tracking_normalized_delta_limit",
    }
    numeric_fields.update(DETAILED_CONTACT_KEYS.intersection(arrays))
    numeric_fields.update(set(STANCE_KEYS).intersection(arrays))
    # Names and boolean contact/stance labels are handled by stricter checks.
    numeric_fields.difference_update(
        {
            "foot_support_names",
            "contact_point_names",
            "left_heel_contact",
            "left_toe_contact",
            "right_heel_contact",
            "right_toe_contact",
            *STANCE_KEYS,
        }
    )
    wrong_numeric_dtype = sorted(
        name
        for name in numeric_fields
        if name in arrays and arrays[name].dtype.kind not in {"f", "i", "u"}
    )
    check(
        "schema",
        "numeric_dtypes",
        not wrong_numeric_dtype,
        fields=wrong_numeric_dtype,
    )
    present_contact_detail = sorted(DETAILED_CONTACT_KEYS.intersection(arrays))
    detailed_contact = len(present_contact_detail) == len(DETAILED_CONTACT_KEYS)
    contact_detail_complete = detailed_contact or not present_contact_detail
    check(
        "schema",
        "detailed_contact_schema_is_complete_or_legacy",
        contact_detail_complete,
        mode="heel_toe" if detailed_contact else "legacy_foot_contact",
        present=present_contact_detail,
        missing=sorted(DETAILED_CONTACT_KEYS.difference(arrays)),
    )
    present_stance = [name for name in STANCE_KEYS if name in arrays]
    explicit_stance = len(present_stance) == len(STANCE_KEYS)
    stance_schema_complete = explicit_stance or not present_stance
    check(
        "schema",
        "stance_schema_is_complete_or_contact_fallback",
        stance_schema_complete,
        mode="explicit_heel_toe_stance" if explicit_stance else "contact_fallback",
        present=present_stance,
        missing=[name for name in STANCE_KEYS if name not in arrays],
    )
    if (
        missing
        or unreadable
        or object_fields
        or finite_failures
        or wrong_numeric_dtype
        or not contact_detail_complete
        or not stance_schema_complete
    ):
        return finish("Schema is incomplete or contains unsafe arrays.")

    scalar_names = {
        "schema_version",
        "retarget_version",
        "fps",
        "frame_count",
        "coordinate_system",
        "quaternion_order",
        "position_unit",
        "angle_unit",
        "joint_order",
        "foot_collision_radius_m",
        "morphology_scale",
        "human_leg_height_m",
        "g1_leg_height_m",
        "source_file",
        "source_checksum_sha256",
        "source_human_schema",
        "source_stageii_file",
        "source_stageii_checksum_sha256",
        "g1_contract_path",
        "g1_contract_checksum_sha256",
        "g1_urdf_path",
        "g1_urdf_checksum_sha256",
        "correspondence_path",
        "correspondence_checksum_sha256",
        "solver_name",
        "solver_iterations",
        "tracking_action_mode",
        "tracking_residual_scale",
        "tracking_action_clip",
        "tracking_normalized_delta_limit",
        "gate3_complete",
        "gate3_status",
    }
    if detailed_contact:
        scalar_names.add("contact_lock_run_count")
    non_scalars = {
        name: list(arrays[name].shape)
        for name in sorted(scalar_names)
        if arrays[name].shape != ()
    }
    check("schema", "metadata_scalars", not non_scalars, fields=non_scalars)
    if non_scalars:
        return finish("Scalar metadata contract failed.")

    frame_count = int(_scalar(arrays, "frame_count"))
    fps = float(_scalar(arrays, "fps"))
    body_names = tuple(str(value) for value in arrays["body_names"].tolist())
    tracking_names = tuple(str(value) for value in arrays["tracking_body_names"].tolist())
    expected_shapes = {
        "time": (frame_count,),
        "joint_names": (29,),
        "sdk_joint_names": (29,),
        "policy_to_sdk": (29,),
        "joint_pos": (frame_count, 29),
        "joint_pos_sdk": (frame_count, 29),
        "joint_vel": (frame_count, 29),
        "joint_acc": (frame_count, 29),
        "root_pos_w": (frame_count, 3),
        "root_quat_wxyz": (frame_count, 4),
        "root_lin_vel_w": (frame_count, 3),
        "root_ang_vel_w": (frame_count, 3),
        "body_names": (32,),
        "body_pos_w": (frame_count, 32, 3),
        "body_quat_wxyz": (frame_count, 32, 4),
        "body_lin_vel_w": (frame_count, 32, 3),
        "body_ang_vel_w": (frame_count, 32, 3),
        "tracking_body_names": (14,),
        "foot_sole_names": (2,),
        "foot_sole_position_w": (frame_count, 2, 3),
        "left_foot_contact": (frame_count,),
        "right_foot_contact": (frame_count,),
    }
    if detailed_contact:
        expected_shapes.update(
            {
                "foot_support_names": (8,),
                "foot_support_position_w": (frame_count, 8, 3),
                "contact_point_names": (4,),
                "contact_point_position_w": (frame_count, 4, 3),
                "left_heel_contact": (frame_count,),
                "left_toe_contact": (frame_count,),
                "right_heel_contact": (frame_count,),
                "right_toe_contact": (frame_count,),
            }
        )
    if explicit_stance:
        expected_shapes.update({name: (frame_count,) for name in STANCE_KEYS})
    shape_failures = {
        name: {"expected": list(shape), "actual": list(arrays[name].shape)}
        for name, shape in expected_shapes.items()
        if arrays[name].shape != shape
    }
    check("schema", "canonical_shapes", not shape_failures, failures=shape_failures)
    bool_fields = ["left_foot_contact", "right_foot_contact", "gate3_complete"]
    if detailed_contact:
        bool_fields.extend(
            (
                "left_heel_contact",
                "left_toe_contact",
                "right_heel_contact",
                "right_toe_contact",
            )
        )
    if explicit_stance:
        bool_fields.extend(STANCE_KEYS)
    wrong_bool = sorted(name for name in bool_fields if arrays[name].dtype != np.dtype(bool))
    check("schema", "boolean_dtype", not wrong_bool, fields=wrong_bool)
    if shape_failures or frame_count < 3 or not np.isfinite(fps) or fps <= 0.0:
        check(
            "schema",
            "minimum_frames_and_fps",
            frame_count >= 3 and np.isfinite(fps) and fps > 0.0,
            frame_count=frame_count,
            fps=fps,
        )
        return finish("Canonical tensor shapes, frame count, or fps failed.")
    check("schema", "minimum_frames_and_fps", True, frame_count=frame_count, fps=fps)

    fixed_metadata = {
        "schema_version": "accad_g1_reference_v1",
        "coordinate_system": "right-handed_x-forward_y-left_z-up",
        "quaternion_order": "wxyz",
        "position_unit": "meter",
        "angle_unit": "radian",
        "joint_order": "policy_index",
        "source_human_schema": "accad_human_reconstruction_v1",
    }
    metadata_mismatch = {
        name: {"expected": expected, "actual": str(_scalar(arrays, name))}
        for name, expected in fixed_metadata.items()
        if str(_scalar(arrays, name)) != expected
    }
    check("schema", "fixed_metadata", not metadata_mismatch, mismatches=metadata_mismatch)
    retarget_version = str(_scalar(arrays, "retarget_version"))
    check(
        "schema",
        "retarget_version_namespace",
        retarget_version.startswith("g1-sequence-retarget-v"),
        value=retarget_version,
    )

    time = np.asarray(arrays["time"], dtype=np.float64)
    expected_time = np.arange(frame_count, dtype=np.float64) / fps
    time_error = float(np.max(np.abs(time - expected_time)))
    check(
        "schema",
        "time_grid",
        fps == 50.0
        and np.all(np.diff(time) > 0.0)
        and time_error <= limits["time_seconds"],
        fps=fps,
        maximum_error_seconds=time_error,
        tolerance_seconds=limits["time_seconds"],
    )

    names_ok = (
        len(body_names) == 32
        and len(set(body_names)) == 32
        and set(body_names) == EXPECTED_BODY_NAMES
        and tracking_names == EXPECTED_TRACKING_BODY_NAMES
        and set(tracking_names).issubset(body_names)
        and tuple(str(value) for value in arrays["foot_sole_names"].tolist())
        == ("left_sole", "right_sole")
    )
    if detailed_contact:
        names_ok = (
            names_ok
            and tuple(str(value) for value in arrays["foot_support_names"].tolist())
            == EXPECTED_FOOT_SUPPORT_NAMES
            and tuple(str(value) for value in arrays["contact_point_names"].tolist())
            == ("left_heel", "left_toe", "right_heel", "right_toe")
        )
    check(
        "schema",
        "body_name_contract",
        names_ok,
        body_count=len(body_names),
        missing=sorted(EXPECTED_BODY_NAMES.difference(body_names)),
        extra=sorted(set(body_names).difference(EXPECTED_BODY_NAMES)),
    )
    if not names_ok:
        return finish("Body, tracking-body, or foot-frame names violate the G1 schema.")

    contract = _load_contract()
    policy_names = tuple(str(value) for value in arrays["joint_names"].tolist())
    sdk_names = tuple(str(value) for value in arrays["sdk_joint_names"].tolist())
    stored_policy_to_sdk = np.asarray(arrays["policy_to_sdk"], dtype=np.int64)
    contract_ok = (
        contract["variant"] == "g1_29dof_rev_1_0"
        and policy_names == contract["policy_names"]
        and sdk_names == contract["sdk_names"]
        and np.array_equal(stored_policy_to_sdk, contract["policy_to_sdk"])
        and abs(contract["physics_dt"] * contract["decimation"] - 1.0 / fps)
        <= limits["time_seconds"]
    )
    check(
        "contract",
        "authoritative_joint_order_and_timing",
        contract_ok,
        variant=contract["variant"],
        control_dt=contract["physics_dt"] * contract["decimation"],
    )

    joint_position = np.asarray(arrays["joint_pos"], dtype=np.float64)
    stored_sdk_position = np.asarray(arrays["joint_pos_sdk"], dtype=np.float64)
    independent_sdk_position = np.empty_like(joint_position)
    independent_sdk_position[:, contract["policy_to_sdk"]] = joint_position
    sdk_error = float(np.max(np.abs(stored_sdk_position - independent_sdk_position)))
    check(
        "contract",
        "policy_sdk_consistency",
        sdk_error <= limits["policy_sdk_position_rad"],
        maximum_error_rad=sdk_error,
        tolerance_rad=limits["policy_sdk_position_rad"],
    )

    # Provenance is resolved independently.  Contract and URDF paths must be
    # the canonical parent-project files; source/correspondence may be custom
    # paths but their recorded hashes must still match their bytes.
    source_path = _resolve_path(str(_scalar(arrays, "source_file")), PIPELINE_ROOT)
    stageii_path = _resolve_path(str(_scalar(arrays, "source_stageii_file")), ACCAD_ROOT)
    stored_contract_path = _resolve_path(str(_scalar(arrays, "g1_contract_path")), PIPELINE_ROOT)
    stored_urdf_path = _resolve_path(str(_scalar(arrays, "g1_urdf_path")), PIPELINE_ROOT)
    correspondence_path = _resolve_path(
        str(_scalar(arrays, "correspondence_path")), PIPELINE_ROOT
    )
    provenance_items = {
        "source_human": (
            source_path,
            str(_scalar(arrays, "source_checksum_sha256")),
        ),
        "source_stageii": (
            stageii_path,
            str(_scalar(arrays, "source_stageii_checksum_sha256")),
        ),
        "g1_contract": (
            G1_CONTRACT_PATH.resolve(),
            str(_scalar(arrays, "g1_contract_checksum_sha256")),
        ),
        "g1_urdf": (
            G1_URDF_PATH.resolve(),
            str(_scalar(arrays, "g1_urdf_checksum_sha256")),
        ),
        "correspondence": (
            correspondence_path,
            str(_scalar(arrays, "correspondence_checksum_sha256")),
        ),
    }
    for name, (provenance_path, expected_checksum) in provenance_items.items():
        actual_checksum = _sha256(provenance_path) if provenance_path.is_file() else None
        check(
            "provenance",
            f"{name}_sha256",
            actual_checksum == expected_checksum,
            path=str(provenance_path),
            expected=expected_checksum,
            actual=actual_checksum,
        )
    check(
        "provenance",
        "canonical_contract_and_urdf_paths",
        stored_contract_path == G1_CONTRACT_PATH.resolve()
        and stored_urdf_path == G1_URDF_PATH.resolve(),
        stored_contract_path=str(stored_contract_path),
        canonical_contract_path=str(G1_CONTRACT_PATH.resolve()),
        stored_urdf_path=str(stored_urdf_path),
        canonical_urdf_path=str(G1_URDF_PATH.resolve()),
    )
    if expected_source_path is not None:
        expected_source = Path(expected_source_path).expanduser().resolve()
        check(
            "provenance",
            "expected_source_path",
            source_path == expected_source,
            recorded=str(source_path),
            expected=str(expected_source),
        )
    try:
        source_arrays, source_unreadable = _safe_load_npz(source_path)
        source_schema = (
            str(_scalar(source_arrays, "schema_version"))
            if "schema_version" in source_arrays
            else None
        )
    except (OSError, ValueError):
        source_unreadable = {"archive": "unreadable"}
        source_schema = None
    check(
        "provenance",
        "source_human_schema",
        not source_unreadable
        and source_schema == str(_scalar(arrays, "source_human_schema")),
        recorded=str(_scalar(arrays, "source_human_schema")),
        actual=source_schema,
    )

    root_quaternion = np.asarray(arrays["root_quat_wxyz"], dtype=np.float64)
    body_quaternion = np.asarray(arrays["body_quat_wxyz"], dtype=np.float64)
    norm_errors = {
        "root": float(np.max(np.abs(np.linalg.norm(root_quaternion, axis=-1) - 1.0))),
        "body": float(np.max(np.abs(np.linalg.norm(body_quaternion, axis=-1) - 1.0))),
    }
    adjacent_dots = {
        "root": float(np.min(np.sum(root_quaternion[1:] * root_quaternion[:-1], axis=-1))),
        "body": float(np.min(np.sum(body_quaternion[1:] * body_quaternion[:-1], axis=-1))),
    }
    quaternions_valid = min(
        np.min(np.linalg.norm(root_quaternion, axis=-1)),
        np.min(np.linalg.norm(body_quaternion, axis=-1)),
    ) > 1.0e-12
    check(
        "numeric",
        "quaternion_norm",
        quaternions_valid
        and max(norm_errors.values()) <= limits["quaternion_norm"],
        maximum_errors=norm_errors,
        tolerance=limits["quaternion_norm"],
    )
    check(
        "numeric",
        "quaternion_sign_continuity",
        min(adjacent_dots.values()) >= limits["quaternion_minimum_adjacent_dot"],
        minimum_adjacent_dot=adjacent_dots,
        tolerance=limits["quaternion_minimum_adjacent_dot"],
    )
    if not quaternions_valid:
        return finish("Quaternion data cannot be used for FK or velocity validation.")

    recomputed = {
        "joint_vel": _gradient(joint_position, fps),
        "joint_acc": _gradient(_gradient(joint_position, fps), fps),
        "root_lin_vel_w": _gradient(np.asarray(arrays["root_pos_w"], dtype=np.float64), fps),
        "root_ang_vel_w": _angular_velocity_world(root_quaternion, fps),
    }
    velocity_errors: dict[str, float] = {}
    for name in ("joint_vel", "joint_acc", "root_lin_vel_w", "root_ang_vel_w"):
        error = float(np.max(np.abs(np.asarray(arrays[name], dtype=np.float64) - recomputed[name])))
        velocity_errors[name] = error
        check(
            "numeric",
            f"recomputed_{name}",
            error <= limits["stored_velocity"],
            maximum_absolute_error=error,
            tolerance=limits["stored_velocity"],
        )

    lower = contract["lower"]
    upper = contract["upper"]
    midpoint = 0.5 * (lower + upper)
    soft_half_range = 0.5 * SOFT_JOINT_POSITION_FACTOR * (upper - lower)
    soft_lower = midpoint - soft_half_range
    soft_upper = midpoint + soft_half_range

    def position_limit_metrics(limit_lower: np.ndarray, limit_upper: np.ndarray) -> dict[str, Any]:
        lower_excess = np.maximum(limit_lower - joint_position, 0.0)
        upper_excess = np.maximum(joint_position - limit_upper, 0.0)
        violation = (lower_excess > limits["hard_limit_rad"]) | (
            upper_excess > limits["hard_limit_rad"]
        )
        return {
            "violation_count": int(np.count_nonzero(violation)),
            "violation_fraction": float(np.mean(violation)),
            "maximum_excess_rad": float(max(np.max(lower_excess), np.max(upper_excess))),
            "minimum_margin_rad": float(
                np.min(np.minimum(joint_position - limit_lower, limit_upper - joint_position))
            ),
            "per_joint_violation_count": {
                name: int(np.count_nonzero(violation[:, index]))
                for index, name in enumerate(policy_names)
            },
        }

    hard_metrics = position_limit_metrics(lower, upper)
    # Soft tolerance has its own threshold even though both defaults are 1e-6.
    soft_lower_excess = np.maximum(soft_lower - joint_position, 0.0)
    soft_upper_excess = np.maximum(joint_position - soft_upper, 0.0)
    soft_violation = (soft_lower_excess > limits["soft_limit_rad"]) | (
        soft_upper_excess > limits["soft_limit_rad"]
    )
    soft_metrics = {
        "factor": SOFT_JOINT_POSITION_FACTOR,
        "violation_count": int(np.count_nonzero(soft_violation)),
        "violation_fraction": float(np.mean(soft_violation)),
        "maximum_excess_rad": float(max(np.max(soft_lower_excess), np.max(soft_upper_excess))),
        "minimum_margin_rad": float(
            np.min(np.minimum(joint_position - soft_lower, soft_upper - joint_position))
        ),
        "per_joint_violation_count": {
            name: int(np.count_nonzero(soft_violation[:, index]))
            for index, name in enumerate(policy_names)
        },
    }
    velocity_ratio = np.abs(recomputed["joint_vel"]) / contract["velocity"]
    velocity_violation = (
        np.abs(recomputed["joint_vel"])
        > contract["velocity"] + limits["velocity_limit_rad_s"]
    )
    velocity_metrics = {
        "violation_count": int(np.count_nonzero(velocity_violation)),
        "violation_fraction": float(np.mean(velocity_violation)),
        "maximum_abs_rad_s": float(np.max(np.abs(recomputed["joint_vel"]))),
        "maximum_limit_ratio": float(np.max(velocity_ratio)),
        "per_joint_max_abs_rad_s": {
            name: float(np.max(np.abs(recomputed["joint_vel"][:, index])))
            for index, name in enumerate(policy_names)
        },
    }
    acceleration_abs = np.abs(recomputed["joint_acc"])
    acceleration_metrics = {
        "contract_limit_available": False,
        "gate_policy": "report_only_no_authoritative_g1_acceleration_limit",
        "maximum_abs_rad_s2": float(np.max(acceleration_abs)),
        "p95_abs_rad_s2": float(np.percentile(acceleration_abs, 95.0)),
        "p99_abs_rad_s2": float(np.percentile(acceleration_abs, 99.0)),
        "rms_rad_s2": float(np.sqrt(np.mean(np.square(recomputed["joint_acc"])))),
        "per_joint_max_abs_rad_s2": {
            name: float(np.max(acceleration_abs[:, index]))
            for index, name in enumerate(policy_names)
        },
    }
    metrics["joint_limits"] = {
        "hard": hard_metrics,
        "soft": soft_metrics,
        "velocity": velocity_metrics,
        "acceleration": acceleration_metrics,
    }
    check(
        "limits",
        "hard_joint_position_limits",
        hard_metrics["violation_count"] == 0,
        **hard_metrics,
        tolerance_rad=limits["hard_limit_rad"],
    )
    check(
        "limits",
        "soft_joint_position_limits",
        soft_metrics["violation_count"] == 0,
        **soft_metrics,
        tolerance_rad=limits["soft_limit_rad"],
    )
    check(
        "limits",
        "joint_velocity_limits",
        velocity_metrics["violation_count"] == 0,
        **velocity_metrics,
        tolerance_rad_s=limits["velocity_limit_rad_s"],
    )
    check(
        "limits",
        "joint_acceleration_metrics",
        True,
        required=False,
        **acceleration_metrics,
    )
    warnings.append(
        "The authoritative G1 contract has no joint-acceleration limits; acceleration is report-only."
    )

    try:
        fk = _pinocchio_fk(
            np.asarray(arrays["root_pos_w"], dtype=np.float64),
            _continuous_unit_quaternion(root_quaternion),
            independent_sdk_position,
            body_names,
            contract["sdk_names"],
        )
    except (RuntimeError, ValueError) as exc:
        check("pinocchio_fk", "pinocchio_model_and_fk", False, error=str(exc))
        return finish("Independent Pinocchio FK could not be evaluated.")
    check(
        "pinocchio_fk",
        "pinocchio_model_and_fk",
        fk["model_nq"] == 36 and fk["model_nv"] == 35,
        pinocchio_version=fk["pinocchio_version"],
        nq=fk["model_nq"],
        nv=fk["model_nv"],
    )
    stored_body_position = np.asarray(arrays["body_pos_w"], dtype=np.float64)
    stored_body_rotation = _quaternion_to_matrix_wxyz(body_quaternion)
    fk_position_error = np.linalg.norm(
        stored_body_position - fk["body_position"], axis=-1
    )
    fk_rotation_error = _rotation_error_rad(fk["body_rotation"], stored_body_rotation)
    maximum_fk_position_error = float(np.max(fk_position_error))
    maximum_fk_rotation_error = float(np.max(fk_rotation_error))
    check(
        "pinocchio_fk",
        "stored_body_position_matches_pinocchio",
        maximum_fk_position_error <= limits["fk_position_m"],
        maximum_error_m=maximum_fk_position_error,
        tolerance_m=limits["fk_position_m"],
        worst_body=body_names[int(np.unravel_index(np.argmax(fk_position_error), fk_position_error.shape)[1])],
    )
    check(
        "pinocchio_fk",
        "stored_body_orientation_matches_pinocchio",
        maximum_fk_rotation_error <= limits["fk_orientation_rad"],
        maximum_error_rad=maximum_fk_rotation_error,
        tolerance_rad=limits["fk_orientation_rad"],
        worst_body=body_names[int(np.unravel_index(np.argmax(fk_rotation_error), fk_rotation_error.shape)[1])],
    )
    metrics["pinocchio_fk"] = {
        "version": fk["pinocchio_version"],
        "maximum_body_position_error_m": maximum_fk_position_error,
        "maximum_body_orientation_error_rad": maximum_fk_rotation_error,
    }

    stored_body_linear_velocity = np.asarray(
        arrays["body_lin_vel_w"], dtype=np.float64
    )
    serialized_body_linear_velocity = _gradient(stored_body_position, fps)
    serialized_body_linear_error = float(
        np.max(np.abs(stored_body_linear_velocity - serialized_body_linear_velocity))
    )
    check(
        "numeric",
        "recomputed_body_lin_vel_w_from_serialized_body_position",
        serialized_body_linear_error <= limits["stored_velocity"],
        maximum_absolute_error=serialized_body_linear_error,
        tolerance=limits["stored_velocity"],
        scope="all_frames",
    )

    independent_body_velocity = {
        "body_lin_vel_w": _gradient(fk["body_position"], fps),
        "body_ang_vel_w": _angular_velocity_world(fk["body_quaternion"], fps),
    }
    body_linear_cross_validation: dict[str, float] = {
        "serialized_position_all_frames_maximum_absolute_error": (
            serialized_body_linear_error
        )
    }
    for name, expected in independent_body_velocity.items():
        difference = np.abs(np.asarray(arrays[name], dtype=np.float64) - expected)
        error = float(np.max(difference))
        velocity_errors[name] = error
        if name == "body_lin_vel_w":
            # Keep the historical check name for report consumers, but make
            # its centered/interior scope explicit.  np.gradient(edge_order=2)
            # uses one-sided endpoint coefficients whose L1 norm is 4x larger;
            # those two frames therefore receive a separately reported bound.
            interior_error = float(np.max(difference[1:-1]))
            endpoint_error = float(np.max(difference[[0, -1]]))
            check(
                "numeric",
                "recomputed_body_lin_vel_w_from_pinocchio",
                interior_error <= limits["stored_velocity"],
                maximum_absolute_error=interior_error,
                tolerance=limits["stored_velocity"],
                scope="interior_frames_1_through_n_minus_2",
            )
            check(
                "numeric",
                "recomputed_body_lin_vel_w_endpoint_from_pinocchio",
                endpoint_error <= limits["pinocchio_body_velocity_endpoint"],
                maximum_absolute_error=endpoint_error,
                tolerance=limits["pinocchio_body_velocity_endpoint"],
                scope="endpoint_frames_0_and_n_minus_1",
            )
            body_linear_cross_validation.update(
                {
                    "pinocchio_all_frames_maximum_absolute_error": error,
                    "pinocchio_interior_maximum_absolute_error": interior_error,
                    "pinocchio_endpoint_maximum_absolute_error": endpoint_error,
                }
            )
            continue
        check(
            "numeric",
            f"recomputed_{name}_from_pinocchio",
            error <= limits["stored_velocity"],
            maximum_absolute_error=error,
            tolerance=limits["stored_velocity"],
            scope="all_frames",
        )
    metrics["stored_velocity_maximum_errors"] = velocity_errors
    metrics["body_linear_velocity_cross_validation"] = body_linear_cross_validation

    support_spheres = _parse_foot_support_spheres()
    foot_metrics: dict[str, Any] = {}
    stored_sole = np.asarray(arrays["foot_sole_position_w"], dtype=np.float64)
    full_contacts = (
        np.asarray(arrays["left_foot_contact"], dtype=bool),
        np.asarray(arrays["right_foot_contact"], dtype=bool),
    )
    heel_toe_specs = (("heel", (0, 1)), ("toe", (2, 3)))
    sole_specs = (("sole", (0, 1, 2, 3)),)
    if detailed_contact:
        stored_support = np.asarray(
            arrays["foot_support_position_w"], dtype=np.float64
        ).reshape(frame_count, 2, 4, 3)
        stored_contact_points = np.asarray(
            arrays["contact_point_position_w"], dtype=np.float64
        )
        contact_channels = (
            np.asarray(arrays["left_heel_contact"], dtype=bool),
            np.asarray(arrays["left_toe_contact"], dtype=bool),
            np.asarray(arrays["right_heel_contact"], dtype=bool),
            np.asarray(arrays["right_toe_contact"], dtype=bool),
        )
        composite_contacts_ok = (
            np.array_equal(full_contacts[0], contact_channels[0] | contact_channels[1])
            and np.array_equal(full_contacts[1], contact_channels[2] | contact_channels[3])
        )
        contact_channel_specs = heel_toe_specs
        contact_mode = "heel_toe"
    else:
        stored_support = None
        stored_contact_points = None
        contact_channels = full_contacts
        composite_contacts_ok = True
        contact_channel_specs = sole_specs
        contact_mode = "legacy_foot_contact"
        warnings.append(
            "Legacy foot-level contacts detected; stance slip uses the sole-center fallback."
        )
    check(
        "contact_geometry",
        "heel_toe_composite_contact_labels",
        composite_contacts_ok,
    )
    if explicit_stance:
        stance_channels = tuple(
            np.asarray(arrays[name], dtype=bool) for name in STANCE_KEYS
        )
        stance_channel_specs = heel_toe_specs
        stance_source = "explicit_heel_toe_stance"
    else:
        stance_channels = contact_channels
        stance_channel_specs = contact_channel_specs
        stance_source = (
            "heel_toe_contact_fallback"
            if detailed_contact
            else "foot_contact_sole_fallback"
        )
        warnings.append(
            "Explicit heel/toe stance channels are absent; stance slip uses contact-label "
            "fallback for legacy compatibility."
        )

    if explicit_stance:
        if detailed_contact:
            stance_contact_references = contact_channels
        else:
            stance_contact_references = (
                full_contacts[0],
                full_contacts[0],
                full_contacts[1],
                full_contacts[1],
            )
        stance_without_contact = {
            name: int(np.count_nonzero(stance & ~contact))
            for name, stance, contact in zip(
                STANCE_KEYS,
                stance_channels,
                stance_contact_references,
                strict=True,
            )
        }
        check(
            "contact_geometry",
            "stance_is_subset_of_contact",
            not any(stance_without_contact.values()),
            required=False,
            counts=stance_without_contact,
        )
    else:
        stance_without_contact = {name: 0 for name in STANCE_KEYS}

    metrics["contact_stance_semantics"] = {
        "contact_source": contact_mode,
        "contact_usage": "active support-sphere penetration only",
        "stance_source": stance_source,
        "stance_usage": "consecutive support-point step and stance-run drift",
        "explicit_stance_fields_present": explicit_stance,
        "contact_fallback_used_for_stance": not explicit_stance,
        "stance_without_contact_frame_count_report_only": stance_without_contact,
    }
    maximum_sole_error = 0.0
    maximum_support_error = 0.0
    maximum_contact_point_error = 0.0
    maximum_penetration = 0.0
    maximum_active_contact_penetration = 0.0
    maximum_stance_step = 0.0
    maximum_stance_run_drift = 0.0
    all_radii: list[float] = []
    for side, link_name in enumerate(
        ("left_ankle_roll_link", "right_ankle_roll_link")
    ):
        local_centers, radii = support_spheres[link_name]
        all_radii.extend(radii.tolist())
        body_index = body_names.index(link_name)
        rotations = fk["body_rotation"][:, body_index]
        translations = fk["body_position"][:, body_index]
        world_centers = translations[:, None, :] + np.einsum(
            "tij,pj->tpi", rotations, local_centers
        )
        sole_center = np.mean(world_centers, axis=1)
        expected_contact_points = np.stack(
            (np.mean(world_centers[:, :2], axis=1), np.mean(world_centers[:, 2:], axis=1)),
            axis=1,
        )
        sole_error = np.linalg.norm(sole_center - stored_sole[:, side], axis=-1)
        support_error = (
            np.linalg.norm(world_centers - stored_support[:, side], axis=-1)
            if stored_support is not None
            else np.zeros((frame_count, 4), dtype=np.float64)
        )
        stored_channel_slice = slice(2 * side, 2 * side + 2)
        contact_point_error = (
            np.linalg.norm(
                expected_contact_points
                - stored_contact_points[:, stored_channel_slice],
                axis=-1,
            )
            if stored_contact_points is not None
            else np.zeros((frame_count, 2), dtype=np.float64)
        )
        surface_height = world_centers[..., 2] - radii[None, :]
        penetration = np.maximum(-surface_height, 0.0)
        side_name = "left" if side == 0 else "right"
        contact_metrics: dict[str, Any] = {}
        for local_channel, (contact_name, point_indices) in enumerate(
            contact_channel_specs
        ):
            contact_index = 2 * side + local_channel if detailed_contact else side
            contact = contact_channels[contact_index]
            selected_penetration = penetration[:, point_indices]
            channel_active_penetration = (
                float(np.max(selected_penetration[contact]))
                if np.any(contact)
                else 0.0
            )
            contact_metrics[contact_name] = {
                "contact_frame_count": int(np.count_nonzero(contact)),
                "maximum_active_support_sphere_penetration_m": (
                    channel_active_penetration
                ),
            }
            maximum_active_contact_penetration = max(
                maximum_active_contact_penetration,
                channel_active_penetration,
            )

        stance_metrics: dict[str, Any] = {}
        for local_channel, (stance_name, point_indices) in enumerate(
            stance_channel_specs
        ):
            stance_index = (
                2 * side + local_channel
                if stance_channel_specs == heel_toe_specs
                else side
            )
            stance = stance_channels[stance_index]
            selected_points = world_centers[:, point_indices, :]
            consecutive = stance[1:] & stance[:-1]
            support_corner_step = np.linalg.norm(
                selected_points[1:, :, :2] - selected_points[:-1, :, :2], axis=-1
            )
            selected_corner_steps = support_corner_step[consecutive]
            center_xy = np.mean(selected_points, axis=1)[:, :2]
            representative_step = np.linalg.norm(
                center_xy[1:] - center_xy[:-1], axis=-1
            )
            selected_steps = representative_step[consecutive]
            channel_maximum_step = (
                float(np.max(selected_steps)) if selected_steps.size else 0.0
            )
            run_records: list[dict[str, Any]] = []
            channel_maximum_run_drift = 0.0
            for start, stop in _true_runs(stance):
                if stop - start < 2:
                    continue
                anchor_distance = np.linalg.norm(
                    center_xy[start:stop] - center_xy[start], axis=-1
                )
                path_length = float(
                    np.sum(
                        np.linalg.norm(
                            np.diff(center_xy[start:stop], axis=0), axis=-1
                        )
                    )
                )
                run_drift = float(np.max(anchor_distance))
                channel_maximum_run_drift = max(
                    channel_maximum_run_drift, run_drift
                )
                run_records.append(
                    {
                        "start_frame": int(start),
                        "stop_frame_exclusive": int(stop),
                        "frames": int(stop - start),
                        "maximum_anchor_drift_xy_m": run_drift,
                        "path_length_xy_m": path_length,
                    }
                )
            stance_metrics[stance_name] = {
                "label_source": stance_source,
                "stance_frame_count": int(np.count_nonzero(stance)),
                "consecutive_stance_pair_count": int(np.count_nonzero(consecutive)),
                "maximum_consecutive_stance_point_step_xy_m": channel_maximum_step,
                "mean_consecutive_stance_point_step_xy_m": (
                    float(np.mean(selected_steps)) if selected_steps.size else 0.0
                ),
                "p95_consecutive_stance_point_speed_xy_m_s": (
                    float(np.percentile(selected_steps * fps, 95.0))
                    if selected_steps.size
                    else 0.0
                ),
                "maximum_consecutive_stance_point_speed_xy_m_s": (
                    float(np.max(selected_steps) * fps) if selected_steps.size else 0.0
                ),
                "mean_consecutive_stance_point_speed_xy_m_s": (
                    float(np.mean(selected_steps) * fps) if selected_steps.size else 0.0
                ),
                "maximum_support_corner_step_xy_m_report_only": (
                    float(np.max(selected_corner_steps))
                    if selected_corner_steps.size
                    else 0.0
                ),
                "maximum_stance_run_drift_xy_m": channel_maximum_run_drift,
                "stance_runs": run_records,
            }
            maximum_stance_step = max(maximum_stance_step, channel_maximum_step)
            maximum_stance_run_drift = max(
                maximum_stance_run_drift, channel_maximum_run_drift
            )
        side_metrics = {
            "support_sphere_local_centers_m": local_centers.tolist(),
            "support_sphere_radii_m": radii.tolist(),
            "maximum_stored_sole_center_error_m": float(np.max(sole_error)),
            "maximum_stored_support_position_error_m": float(np.max(support_error)),
            "maximum_stored_contact_point_error_m": float(
                np.max(contact_point_error)
            ),
            "maximum_penetration_m": float(np.max(penetration)),
            "contact_channels": contact_metrics,
            "stance_channels": stance_metrics,
        }
        foot_metrics[side_name] = side_metrics
        maximum_sole_error = max(
            maximum_sole_error, side_metrics["maximum_stored_sole_center_error_m"]
        )
        maximum_support_error = max(
            maximum_support_error,
            side_metrics["maximum_stored_support_position_error_m"],
        )
        maximum_contact_point_error = max(
            maximum_contact_point_error,
            side_metrics["maximum_stored_contact_point_error_m"],
        )
        maximum_penetration = max(
            maximum_penetration, side_metrics["maximum_penetration_m"]
        )
    unique_radii = np.unique(np.asarray(all_radii, dtype=np.float64))
    stored_radius = float(_scalar(arrays, "foot_collision_radius_m"))
    radius_ok = (
        len(unique_radii) == 1
        and abs(stored_radius - float(unique_radii[0]))
        <= limits["stored_foot_radius_m"]
    )
    check(
        "contact_geometry",
        "four_urdf_support_spheres_per_foot_and_radius",
        radius_ok,
        urdf_unique_radii_m=unique_radii.tolist(),
        stored_radius_m=stored_radius,
        tolerance_m=limits["stored_foot_radius_m"],
    )
    check(
        "contact_geometry",
        "stored_sole_center_matches_support_sphere_mean",
        maximum_sole_error <= limits["stored_sole_position_m"],
        maximum_error_m=maximum_sole_error,
        tolerance_m=limits["stored_sole_position_m"],
    )
    check(
        "contact_geometry",
        "stored_support_and_heel_toe_positions_match_urdf_fk",
        (not detailed_contact)
        or max(maximum_support_error, maximum_contact_point_error)
        <= limits["stored_sole_position_m"],
        required=detailed_contact,
        mode="heel_toe" if detailed_contact else "legacy_sole_fallback",
        maximum_support_error_m=maximum_support_error,
        maximum_contact_point_error_m=maximum_contact_point_error,
        tolerance_m=limits["stored_sole_position_m"],
    )
    check(
        "contact_geometry",
        "ground_penetration",
        maximum_penetration <= limits["foot_penetration_m"],
        maximum_penetration_m=maximum_penetration,
        maximum_active_contact_penetration_m=maximum_active_contact_penetration,
        tolerance_m=limits["foot_penetration_m"],
    )
    check(
        "contact_geometry",
        "active_contact_penetration",
        maximum_active_contact_penetration <= limits["foot_penetration_m"],
        maximum_active_contact_penetration_m=maximum_active_contact_penetration,
        tolerance_m=limits["foot_penetration_m"],
        label_source=contact_mode,
    )
    check(
        "contact_geometry",
        "consecutive_contact_point_step",
        maximum_stance_step <= limits["stance_support_step_xy_m"],
        maximum_step_xy_m=maximum_stance_step,
        tolerance_m=limits["stance_support_step_xy_m"],
        label_source=stance_source,
    )
    check(
        "contact_geometry",
        "stance_run_anchor_drift",
        maximum_stance_run_drift <= limits["stance_run_drift_xy_m"],
        maximum_drift_xy_m=maximum_stance_run_drift,
        tolerance_m=limits["stance_run_drift_xy_m"],
        label_source=stance_source,
    )
    metrics["foot_contact_geometry"] = foot_metrics

    action_scale = contract["action_scale"]
    legacy_required_action = (joint_position - contract["default"]) / action_scale
    legacy_over_clip = np.abs(legacy_required_action) > 1.0 + limits["metadata_float"]
    legacy_metrics = {
        "meaning": "q_target = q_default + 0.25 * action",
        "clip": 1.0,
        "maximum_abs_required_action": float(np.max(np.abs(legacy_required_action))),
        "over_clip_count": int(np.count_nonzero(legacy_over_clip)),
        "over_clip_fraction": float(np.mean(legacy_over_clip)),
        "per_joint_over_clip_fraction": {
            name: float(np.mean(legacy_over_clip[:, index]))
            for index, name in enumerate(policy_names)
        },
    }
    check(
        "action_contract",
        "legacy_default_offset_reachability",
        legacy_metrics["over_clip_count"] == 0,
        required=False,
        **legacy_metrics,
    )
    if legacy_metrics["over_clip_count"]:
        warnings.append(
            "The reference is not fully reachable through the legacy default+0.25 action; "
            "this does not invalidate reference-residual tracking."
        )

    action_mode = str(_scalar(arrays, "tracking_action_mode"))
    residual_scale = float(_scalar(arrays, "tracking_residual_scale"))
    residual_clip = float(_scalar(arrays, "tracking_action_clip"))
    delta_limit = float(_scalar(arrays, "tracking_normalized_delta_limit"))
    reference_metadata_ok = (
        action_mode == "reference_residual"
        and abs(residual_scale - action_scale) <= limits["metadata_float"]
        and abs(residual_clip - 1.0) <= limits["metadata_float"]
        and abs(delta_limit - 0.15) <= limits["metadata_float"]
    )
    check(
        "action_contract",
        "reference_residual_metadata",
        reference_metadata_ok,
        mode=action_mode,
        residual_scale=residual_scale,
        clip=residual_clip,
        normalized_delta_limit=delta_limit,
    )
    full_lower = joint_position - residual_scale * residual_clip
    full_upper = joint_position + residual_scale * residual_clip
    full_box_violation = (full_lower < lower) | (full_upper > upper)
    if residual_scale > 1.0e-12:
        safe_low = np.maximum(
            -residual_clip, (lower - joint_position) / residual_scale
        )
        safe_high = np.minimum(
            residual_clip, (upper - joint_position) / residual_scale
        )
    else:
        safe_low = np.full_like(joint_position, np.nan)
        safe_high = np.full_like(joint_position, np.nan)
    residual_metrics = {
        "meaning": "q_target = q_reference + residual_scale * action",
        "zero_residual_reproduces_reference": True,
        "reference_hard_limit_feasible": hard_metrics["violation_count"] == 0,
        "full_clip_box_violation_count": int(np.count_nonzero(full_box_violation)),
        "full_clip_box_violation_fraction": float(np.mean(full_box_violation)),
        "minimum_safe_action_low": (
            float(np.min(safe_low)) if residual_scale > 1.0e-12 else None
        ),
        "maximum_safe_action_low": (
            float(np.max(safe_low)) if residual_scale > 1.0e-12 else None
        ),
        "minimum_safe_action_high": (
            float(np.min(safe_high)) if residual_scale > 1.0e-12 else None
        ),
        "maximum_safe_action_high": (
            float(np.max(safe_high)) if residual_scale > 1.0e-12 else None
        ),
        "normalized_delta_limit_applies_to_residual_only": True,
    }
    check(
        "action_contract",
        "reference_residual_zero_action_feasibility",
        reference_metadata_ok and hard_metrics["violation_count"] == 0,
        **residual_metrics,
    )
    check(
        "action_contract",
        "full_residual_clip_box_within_hard_limits",
        residual_metrics["full_clip_box_violation_count"] == 0,
        required=False,
        **residual_metrics,
    )
    if residual_metrics["full_clip_box_violation_count"]:
        warnings.append(
            "Some reference frames require state-dependent residual clipping near joint limits."
        )
    metrics["action_reachability"] = {
        "legacy_default_offset": legacy_metrics,
        "reference_residual": residual_metrics,
    }

    stored_gate_complete = bool(_scalar(arrays, "gate3_complete"))
    check(
        "schema",
        "generator_did_not_self_certify_gate3",
        not stored_gate_complete,
        stored_gate3_complete=stored_gate_complete,
    )
    return finish()


__all__ = [
    "DEFAULT_THRESHOLDS",
    "EXPECTED_BODY_NAMES",
    "EXPECTED_TRACKING_BODY_NAMES",
    "validate_g1_archive",
]
