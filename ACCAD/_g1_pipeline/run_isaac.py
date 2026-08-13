#!/usr/bin/env python3
"""Independent Isaac Sim/Lab validation for an ACCAD-retargeted G1 archive.

This entry point deliberately lives outside the parent project's task and training
scripts.  It reads the parent G1 asset and joint contract, but confines every report
and optional video to ``humanoid_G1/ACCAD``.

Examples
--------

.. code-block:: bash

    cd /home/hoeng/IsaacLab/humanoid_G1
    TERM=xterm ../isaaclab.sh -p ACCAD/_g1_pipeline/run_isaac.py \
      kinematic --headless \
      --input ACCAD/_g1_pipeline/work/g1/Female1Walking_c3d/B3_-_walk1_g1_50hz.npz \
      --output ACCAD/_g1_pipeline/work/reports/B3_isaac_kinematic.json

    TERM=xterm ../isaaclab.sh -p ACCAD/_g1_pipeline/run_isaac.py \
      pd-replay --headless \
      --input ACCAD/_g1_pipeline/work/g1/Female1Walking_c3d/B3_-_walk1_g1_50hz.npz \
      --output ACCAD/_g1_pipeline/work/reports/B3_isaac_pd_replay.json

``kinematic`` writes the complete root-link and joint state at every reference
frame, performs Isaac forward kinematics, and compares Isaac's result against the
archive.  It is a morphology/order/convention check, not a dynamics claim.

``pd-replay`` writes the root-link state only once, interpolates the 50 Hz joint
reference to the 200 Hz physics clock, and drives the project's implicit PD
actuators.  The floating root is never teleported after initialization.  A full,
finite diagnostic exits normally even when its dynamics verdict is negative.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parent
ACCAD_ROOT = PIPELINE_ROOT.parent
PROJECT_ROOT = ACCAD_ROOT.parent
PROJECT_SOURCE = PROJECT_ROOT / "source"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))


def _path_below_accad(value: str | Path, *, must_exist: bool, suffix: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    try:
        path.relative_to(ACCAD_ROOT.resolve())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"path must remain below {ACCAD_ROOT}; got {path}"
        ) from exc
    if must_exist and not path.is_file():
        raise argparse.ArgumentTypeError(f"input file does not exist: {path}")
    if path.suffix.lower() != suffix:
        raise argparse.ArgumentTypeError(f"path must end in {suffix}: {path}")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a 50 Hz ACCAD→G1 NPZ in the project's Isaac runtime."
    )
    parser.add_argument("mode", choices=("kinematic", "pd-replay"))
    parser.add_argument("--input", required=True, help="G1 reference NPZ below ACCAD")
    parser.add_argument("--output", required=True, help="JSON report below ACCAD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Debug-only prefix length; 0 validates the complete clip.",
    )
    parser.add_argument("--video", action="store_true", help="Write a 50 Hz MP4")
    parser.add_argument("--video-path", default=None, help="MP4 below ACCAD")
    parser.add_argument("--contact-force-threshold", type=float, default=1.0)
    parser.add_argument("--geometric-contact-height", type=float, default=0.02)
    parser.add_argument("--geometric-contact-speed", type=float, default=0.35)
    parser.add_argument("--fall-height", type=float, default=0.20)
    parser.add_argument("--fall-tilt", type=float, default=0.80)

    # Kinematic acceptance thresholds.  These cover numerical disagreement
    # between the independent archive FK and Isaac's imported articulation.
    parser.add_argument("--kinematic-joint-tol", type=float, default=1.0e-5)
    parser.add_argument("--kinematic-root-pos-tol", type=float, default=2.0e-5)
    parser.add_argument("--kinematic-root-angle-tol", type=float, default=2.0e-4)
    parser.add_argument("--kinematic-body-pos-tol", type=float, default=3.0e-4)
    parser.add_argument("--kinematic-body-angle-tol", type=float, default=3.0e-3)
    parser.add_argument("--kinematic-sole-pos-tol", type=float, default=3.0e-4)
    parser.add_argument("--max-sole-penetration", type=float, default=0.01)
    parser.add_argument("--kinematic-min-contact-f1", type=float, default=0.75)
    parser.add_argument("--kinematic-max-stance-slip-p95", type=float, default=0.35)

    # Open-loop PD thresholds are intentionally generous.  A fall makes the
    # dynamics verdict negative but is not an execution/runtime error.
    parser.add_argument("--pd-max-joint-rmse", type=float, default=0.35)
    parser.add_argument("--pd-max-joint-error", type=float, default=1.00)
    parser.add_argument("--pd-max-root-position-rmse", type=float, default=0.75)
    parser.add_argument("--pd-min-contact-f1", type=float, default=0.40)
    parser.add_argument("--pd-max-stance-slip-p95", type=float, default=0.75)
    return parser


# AppLauncher must be constructed before importing Gym, Torch, or any other
# Isaac Lab module.  Parsing and path validation above use only the stdlib.
from isaaclab.app import AppLauncher  # noqa: E402


PARSER = _build_parser()
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
try:
    INPUT_PATH = _path_below_accad(ARGS.input, must_exist=True, suffix=".npz")
    OUTPUT_PATH = _path_below_accad(ARGS.output, must_exist=False, suffix=".json")
    if ARGS.video_path is not None and not ARGS.video:
        PARSER.error("--video-path requires --video")
    VIDEO_PATH = None
    if ARGS.video:
        default_video = OUTPUT_PATH.with_suffix(".mp4")
        VIDEO_PATH = _path_below_accad(
            ARGS.video_path or default_video, must_exist=False, suffix=".mp4"
        )
        ARGS.enable_cameras = True
except argparse.ArgumentTypeError as exc:
    PARSER.error(str(exc))

APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


# Isaac imports are intentionally below AppLauncher construction.
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import humanoid_g1.tasks  # noqa: F401, E402
from humanoid_g1.assets.g1_asset import G1_29DOF_CFG  # noqa: E402
from humanoid_g1.assets.joint_contract import load_joint_contract  # noqa: E402
from humanoid_g1.utils.paths import CONTRACT_PATH, URDF_PATH  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


SCHEMA_VERSION = "accad_g1_isaac_validation_v3"
ARCHIVE_SCHEMA = "accad_g1_reference_v1"
TASK_NAME = "Humanoid-G1-29DoF-Velocity-Flat-v0"
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_SOLE_OFFSET = np.asarray([0.035, 0.0, -0.03], dtype=np.float64)
FOOT_SUPPORT_OFFSETS = np.asarray(
    [
        [-0.05, 0.025, -0.03],
        [-0.05, -0.025, -0.03],
        [0.12, 0.030, -0.03],
        [0.12, -0.030, -0.03],
    ],
    dtype=np.float64,
)
FOOT_SUPPORT_NAMES = (
    "left_rear_outer",
    "left_rear_inner",
    "left_front_outer",
    "left_front_inner",
    "right_rear_inner",
    "right_rear_outer",
    "right_front_inner",
    "right_front_outer",
)
CONTACT_POINT_NAMES = ("left_heel", "left_toe", "right_heel", "right_toe")
CONTACT_CHANNEL_KEYS = (
    "left_heel_contact",
    "left_toe_contact",
    "right_heel_contact",
    "right_toe_contact",
)
STANCE_CHANNEL_KEYS = (
    "left_heel_stance",
    "left_toe_stance",
    "right_heel_stance",
    "right_toe_stance",
)


class ValidationError(RuntimeError):
    """Raised for a malformed or incompatible archive/runtime contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                _json_value(payload), stream, ensure_ascii=False, indent=2, sort_keys=True
            )
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _scalar(arrays: dict[str, np.ndarray], key: str) -> Any:
    if key not in arrays or arrays[key].ndim != 0:
        raise ValidationError(f"{key} must be a scalar")
    return arrays[key].item()


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    required = {
        "schema_version",
        "fps",
        "time",
        "frame_count",
        "coordinate_system",
        "quaternion_order",
        "joint_order",
        "joint_names",
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_quat_wxyz",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "body_names",
        "body_pos_w",
        "body_quat_wxyz",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "foot_sole_position_w",
        "foot_collision_radius_m",
        "left_foot_contact",
        "right_foot_contact",
        "g1_contract_checksum_sha256",
        "g1_urdf_checksum_sha256",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValidationError(f"archive is missing required arrays: {missing}")
    if str(_scalar(arrays, "schema_version")) != ARCHIVE_SCHEMA:
        raise ValidationError(
            f"expected schema {ARCHIVE_SCHEMA}, got {_scalar(arrays, 'schema_version')}"
        )
    if float(_scalar(arrays, "fps")) != 50.0:
        raise ValidationError(f"reference must be 50 Hz, got {_scalar(arrays, 'fps')}")
    if str(_scalar(arrays, "coordinate_system")) != "right-handed_x-forward_y-left_z-up":
        raise ValidationError("archive coordinate system is not G1 X-forward/Y-left/Z-up")
    if str(_scalar(arrays, "quaternion_order")) != "wxyz":
        raise ValidationError("archive quaternions must use wxyz order")
    if str(_scalar(arrays, "joint_order")) != "policy_index":
        raise ValidationError("archive joint_pos must use policy_index order")

    frames = int(_scalar(arrays, "frame_count"))
    joint_names = [str(name) for name in arrays["joint_names"].tolist()]
    body_names = [str(name) for name in arrays["body_names"].tolist()]
    expected_shapes = {
        "time": (frames,),
        "joint_pos": (frames, 29),
        "joint_vel": (frames, 29),
        "root_pos_w": (frames, 3),
        "root_quat_wxyz": (frames, 4),
        "root_lin_vel_w": (frames, 3),
        "root_ang_vel_w": (frames, 3),
        "body_pos_w": (frames, len(body_names), 3),
        "body_quat_wxyz": (frames, len(body_names), 4),
        "body_lin_vel_w": (frames, len(body_names), 3),
        "body_ang_vel_w": (frames, len(body_names), 3),
        "foot_sole_position_w": (frames, 2, 3),
        "left_foot_contact": (frames,),
        "right_foot_contact": (frames,),
    }
    bad_shapes = {
        key: {"expected": shape, "actual": arrays[key].shape}
        for key, shape in expected_shapes.items()
        if arrays[key].shape != shape
    }
    if bad_shapes:
        raise ValidationError(f"archive shape mismatch: {bad_shapes}")
    if len(joint_names) != 29 or len(set(joint_names)) != 29:
        raise ValidationError("joint_names must contain 29 unique names")
    if len(body_names) == 0 or len(set(body_names)) != len(body_names):
        raise ValidationError("body_names must be non-empty and unique")
    if not set(FOOT_BODY_NAMES).issubset(body_names) or "pelvis" not in body_names:
        raise ValidationError("archive body_names must contain pelvis and both ankle-roll links")

    # Gate-3 v3 archives expose independent heel/toe contact channels and all
    # eight URDF support spheres.  Old Gate-3 archives remain readable only when
    # the whole extension is absent; a partially written v3 archive is unsafe.
    contact_extension = {
        "foot_support_names",
        "foot_support_position_w",
        "contact_point_names",
        "contact_point_position_w",
        *CONTACT_CHANNEL_KEYS,
    }
    present_extension = contact_extension.intersection(arrays)
    if present_extension and present_extension != contact_extension:
        missing_extension = sorted(contact_extension - present_extension)
        raise ValidationError(
            "archive contains a partial heel/toe contact extension; missing "
            f"{missing_extension}"
        )
    if present_extension:
        extension_shapes = {
            "foot_support_names": (8,),
            "foot_support_position_w": (frames, 8, 3),
            "contact_point_names": (4,),
            "contact_point_position_w": (frames, 4, 3),
            **{key: (frames,) for key in CONTACT_CHANNEL_KEYS},
        }
        bad_extension_shapes = {
            key: {"expected": shape, "actual": arrays[key].shape}
            for key, shape in extension_shapes.items()
            if arrays[key].shape != shape
        }
        if bad_extension_shapes:
            raise ValidationError(
                f"archive heel/toe contact shape mismatch: {bad_extension_shapes}"
            )
        support_names = tuple(
            str(name) for name in arrays["foot_support_names"].tolist()
        )
        contact_names = tuple(
            str(name) for name in arrays["contact_point_names"].tolist()
        )
        if support_names != FOOT_SUPPORT_NAMES:
            raise ValidationError(
                "archive foot_support_names do not match the G1 URDF support order"
            )
        if contact_names != CONTACT_POINT_NAMES:
            raise ValidationError(
                "archive contact_point_names must be left/right heel/toe order"
            )
        wrong_contact_dtype = [
            key for key in CONTACT_CHANNEL_KEYS if arrays[key].dtype != np.dtype(bool)
        ]
        if wrong_contact_dtype:
            raise ValidationError(
                f"heel/toe contact channels must be boolean: {wrong_contact_dtype}"
            )
        channel_contact = np.stack(
            [np.asarray(arrays[key], dtype=bool) for key in CONTACT_CHANNEL_KEYS],
            axis=1,
        )
        if not np.array_equal(
            arrays["left_foot_contact"],
            channel_contact[:, 0] | channel_contact[:, 1],
        ):
            raise ValidationError("left_foot_contact is not heel|toe")
        if not np.array_equal(
            arrays["right_foot_contact"],
            channel_contact[:, 2] | channel_contact[:, 3],
        ):
            raise ValidationError("right_foot_contact is not heel|toe")

    present_stance = set(STANCE_CHANNEL_KEYS).intersection(arrays)
    if present_stance and present_stance != set(STANCE_CHANNEL_KEYS):
        raise ValidationError(
            "archive contains partial heel/toe stance channels; missing "
            f"{sorted(set(STANCE_CHANNEL_KEYS) - present_stance)}"
        )
    if present_stance and not present_extension:
        raise ValidationError(
            "heel/toe stance channels require the v3 contact-point extension"
        )
    if present_stance:
        bad_stance_shapes = {
            key: {"expected": (frames,), "actual": arrays[key].shape}
            for key in STANCE_CHANNEL_KEYS
            if arrays[key].shape != (frames,)
        }
        if bad_stance_shapes:
            raise ValidationError(
                f"archive heel/toe stance shape mismatch: {bad_stance_shapes}"
            )
        wrong_stance_dtype = [
            key for key in STANCE_CHANNEL_KEYS if arrays[key].dtype != np.dtype(bool)
        ]
        if wrong_stance_dtype:
            raise ValidationError(
                f"heel/toe stance channels must be boolean: {wrong_stance_dtype}"
            )

    numeric_keys = [
        key
        for key, value in arrays.items()
        if value.dtype.kind in "fiu" and value.size > 0
    ]
    non_finite = [key for key in numeric_keys if not np.isfinite(arrays[key]).all()]
    if non_finite:
        raise ValidationError(f"archive contains NaN/Inf: {non_finite}")
    times = np.asarray(arrays["time"], dtype=np.float64)
    if frames < 2 or not np.all(np.diff(times) > 0.0):
        raise ValidationError("time must contain at least two strictly increasing samples")
    if float(np.max(np.abs(np.diff(times) - 0.02))) > 1.0e-8:
        raise ValidationError("time is not uniformly sampled at 50 Hz")
    for key in ("root_quat_wxyz", "body_quat_wxyz"):
        norm_error = float(
            np.max(np.abs(np.linalg.norm(arrays[key], axis=-1) - 1.0))
        )
        if norm_error > 1.0e-4:
            raise ValidationError(f"{key} quaternion norm error is {norm_error}")
    return arrays


def _exact_patterns(names: list[str] | tuple[str, ...]) -> list[str]:
    return [f"^{re.escape(name)}$" for name in names]


def _cpu(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _quat_angle_error(reference: np.ndarray, actual: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    act = np.asarray(actual, dtype=np.float64)
    ref /= np.maximum(np.linalg.norm(ref, axis=-1, keepdims=True), 1.0e-12)
    act /= np.maximum(np.linalg.norm(act, axis=-1, keepdims=True), 1.0e-12)
    dot = np.abs(np.sum(ref * act, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _quat_apply(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    vec = np.asarray(vector, dtype=np.float64)
    vec = np.broadcast_to(vec, quat.shape[:-1] + (3,))
    xyz = quat[..., 1:]
    uv = 2.0 * np.cross(xyz, vec)
    return vec + quat[..., :1] * uv + np.cross(xyz, uv)


def _sole_positions(
    body_pos: np.ndarray, body_quat: np.ndarray, body_names: list[str]
) -> np.ndarray:
    indices = [body_names.index(name) for name in FOOT_BODY_NAMES]
    foot_pos = body_pos[:, indices]
    foot_quat = body_quat[:, indices]
    return foot_pos + _quat_apply(foot_quat, FOOT_SOLE_OFFSET)


def _support_positions(
    body_pos: np.ndarray, body_quat: np.ndarray, body_names: list[str]
) -> np.ndarray:
    """Return the eight URDF foot collision-sphere centers as ``[T,2,4,3]``."""

    indices = [body_names.index(name) for name in FOOT_BODY_NAMES]
    foot_pos = np.asarray(body_pos[:, indices], dtype=np.float64)
    foot_quat = np.asarray(body_quat[:, indices], dtype=np.float64)
    frames = len(foot_pos)
    expanded_quat = np.broadcast_to(
        foot_quat[:, :, None, :], (frames, 2, 4, 4)
    )
    local_offsets = np.broadcast_to(
        FOOT_SUPPORT_OFFSETS[None, None, :, :], (frames, 2, 4, 3)
    )
    return foot_pos[:, :, None, :] + _quat_apply(expanded_quat, local_offsets)


def _representative_contact_points(support: np.ndarray) -> np.ndarray:
    """Average each heel/toe support-sphere pair into four representative points."""

    points = np.asarray(support, dtype=np.float64)
    return np.stack(
        (
            points[:, 0, :2].mean(axis=1),
            points[:, 0, 2:].mean(axis=1),
            points[:, 1, :2].mean(axis=1),
            points[:, 1, 2:].mean(axis=1),
        ),
        axis=1,
    )


def _contact_layout(
    arrays: dict[str, np.ndarray], frame_count: int
) -> tuple[np.ndarray, tuple[str, ...], str, tuple[str, ...]]:
    """Prefer the v3 heel/toe channels; fall back only for complete legacy files."""

    if all(key in arrays for key in CONTACT_CHANNEL_KEYS):
        contacts = np.stack(
            [np.asarray(arrays[key][:frame_count], dtype=bool) for key in CONTACT_CHANNEL_KEYS],
            axis=1,
        )
        return contacts, CONTACT_POINT_NAMES, "heel_toe_v3", CONTACT_CHANNEL_KEYS
    contacts = np.stack(
        (
            arrays["left_foot_contact"][:frame_count],
            arrays["right_foot_contact"][:frame_count],
        ),
        axis=1,
    ).astype(bool)
    legacy_keys = ("left_foot_contact", "right_foot_contact")
    return contacts, ("left_sole", "right_sole"), "legacy_sole_center", legacy_keys


def _stance_layout(
    arrays: dict[str, np.ndarray],
    frame_count: int,
    contact_labels: np.ndarray,
    contact_keys: tuple[str, ...],
) -> tuple[np.ndarray, str, tuple[str, ...]]:
    """Return support-lock labels, preferring explicit stance over contact fallback."""

    if all(key in arrays for key in STANCE_CHANNEL_KEYS):
        stance = np.stack(
            [
                np.asarray(arrays[key][:frame_count], dtype=bool)
                for key in STANCE_CHANNEL_KEYS
            ],
            axis=1,
        )
        if stance.shape != contact_labels.shape:
            raise ValidationError(
                "heel/toe stance channels do not match the active contact layout: "
                f"stance={stance.shape}, contact={contact_labels.shape}"
            )
        return stance, "explicit_heel_toe_stance", STANCE_CHANNEL_KEYS
    return (
        np.asarray(contact_labels, dtype=bool).copy(),
        "contact_fallback_no_explicit_stance",
        contact_keys,
    )


def _label_semantics(
    contact_layout: str,
    contact_keys: tuple[str, ...],
    stance_source: str,
    stance_keys: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "contact": {
            "source": contact_layout,
            "keys": list(contact_keys),
            "meaning": "physical/geometric contact labels",
            "used_for": [
                "geometric contact F1",
                "force-contact F1 (left/right composite)",
                "active support-sphere penetration",
            ],
        },
        "stance": {
            "source": stance_source,
            "keys": list(stance_keys),
            "meaning": "support-lock labels",
            "used_for": [
                "stance slip",
                "consecutive support-point step",
                "stance-run anchor drift",
            ],
            "uses_contact_fallback": stance_source.startswith("contact_fallback"),
        },
        "separation_rule": (
            "Explicit stance labels never replace contact labels for F1 or penetration."
        ),
    }


def _linear_velocity(position: np.ndarray, fps: float) -> np.ndarray:
    edge_order = 2 if len(position) >= 3 else 1
    return np.gradient(
        np.asarray(position, dtype=np.float64),
        1.0 / fps,
        axis=0,
        edge_order=edge_order,
    )


def _error_stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "rmse": float(np.sqrt(np.mean(np.square(flat)))),
        "p95": float(np.quantile(flat, 0.95)),
        "maximum": float(np.max(flat)),
    }


def _vector_error(
    reference: np.ndarray, actual: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    values = np.linalg.norm(
        np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64), axis=-1
    )
    return values, _error_stats(values)


def _contact_scores(
    reference: np.ndarray, predicted: np.ndarray, channel_names: tuple[str, ...]
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=bool)
    pred = np.asarray(predicted, dtype=bool)
    if ref.shape != pred.shape or ref.ndim != 2 or ref.shape[1] != len(channel_names):
        raise ValidationError(
            "contact-score tensors and channel names have incompatible shapes: "
            f"reference={ref.shape}, predicted={pred.shape}, names={channel_names}"
        )

    def score(y: np.ndarray, z: np.ndarray) -> dict[str, Any]:
        tp = int(np.count_nonzero(y & z))
        fp = int(np.count_nonzero(~y & z))
        fn = int(np.count_nonzero(y & ~z))
        tn = int(np.count_nonzero(~y & ~z))
        precision = tp / (tp + fp) if tp + fp else (1.0 if not np.any(y) else 0.0)
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    result = {
        name: score(ref[:, index], pred[:, index])
        for index, name in enumerate(channel_names)
    }
    result["aggregate"] = score(ref.reshape(-1), pred.reshape(-1))
    return result


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _active_contact_penetration(
    support_clearance: np.ndarray,
    contact_labels: np.ndarray,
    contact_layout: str,
    contact_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Measure support-sphere penetration selected only by contact labels."""

    clearance = np.asarray(support_clearance, dtype=np.float64)
    contacts = np.asarray(contact_labels, dtype=bool)
    if clearance.ndim != 3 or clearance.shape[1:] != (2, 4):
        raise ValidationError(
            f"support clearance must have shape [T,2,4], got {clearance.shape}"
        )
    if len(clearance) != len(contacts):
        raise ValidationError("support clearance and contact labels differ in length")
    active = np.zeros_like(clearance, dtype=bool)
    if contacts.shape == (len(clearance), 4):
        active[:, 0, :2] = contacts[:, 0, None]
        active[:, 0, 2:] = contacts[:, 1, None]
        active[:, 1, :2] = contacts[:, 2, None]
        active[:, 1, 2:] = contacts[:, 3, None]
    elif contacts.shape == (len(clearance), 2):
        active[:, 0, :] = contacts[:, 0, None]
        active[:, 1, :] = contacts[:, 1, None]
    else:
        raise ValidationError(
            f"contact labels must have shape [T,4] or [T,2], got {contacts.shape}"
        )
    selected = np.maximum(-clearance, 0.0)[active]
    return {
        "samples": int(selected.size),
        "mean_m": float(np.mean(selected)) if selected.size else 0.0,
        "p95_m": float(np.quantile(selected, 0.95)) if selected.size else 0.0,
        "maximum_m": float(np.max(selected)) if selected.size else 0.0,
        "label_source": contact_layout,
        "label_keys": list(contact_keys),
        "label_semantics": "physical contact; stance labels are intentionally not used",
        "selection": "active heel/toe support-sphere pairs from contact labels",
    }


def _stance_slip(
    points: np.ndarray,
    stance_labels: np.ndarray,
    fps: float,
    channel_names: tuple[str, ...],
    label_source: str,
    label_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Measure support locking across consecutive stance frames of each channel.

    Central-difference velocity at contact boundaries mixes flight and stance and
    falsely reports the swing-to-contact transition as stance slip.  A support
    point is therefore selected only when its own stance label is true at both
    endpoints.  Explicit stance labels are support-lock semantics; contact labels
    are used only as a backwards-compatible fallback selected by ``_stance_layout``.
    """

    position = np.asarray(points, dtype=np.float64)
    stance = np.asarray(stance_labels, dtype=bool)
    if (
        position.ndim != 3
        or position.shape[-1] != 3
        or stance.shape != position.shape[:2]
        or position.shape[1] != len(channel_names)
        or len(label_keys) != len(channel_names)
    ):
        raise ValidationError(
            "stance-slip tensors and channel names have incompatible shapes: "
            f"points={position.shape}, stance={stance.shape}, names={channel_names}, "
            f"keys={label_keys}"
        )
    step = np.linalg.norm(position[1:, :, :2] - position[:-1, :, :2], axis=-1)
    consecutive = stance[1:] & stance[:-1]

    def stats(selected_steps: np.ndarray) -> dict[str, Any]:
        selected_speeds = selected_steps * fps
        if selected_steps.size == 0:
            return {
                "samples": 0,
                "mean_step_m": 0.0,
                "p95_step_m": 0.0,
                "maximum_step_m": 0.0,
                "mean_m_s": 0.0,
                "p95_m_s": 0.0,
                "maximum_m_s": 0.0,
            }
        return {
            "samples": int(selected_steps.size),
            "mean_step_m": float(np.mean(selected_steps)),
            "p95_step_m": float(np.quantile(selected_steps, 0.95)),
            "maximum_step_m": float(np.max(selected_steps)),
            "mean_m_s": float(np.mean(selected_speeds)),
            "p95_m_s": float(np.quantile(selected_speeds, 0.95)),
            "maximum_m_s": float(np.max(selected_speeds)),
        }

    channel_metrics: dict[str, Any] = {}
    maximum_run_drift = 0.0
    total_runs = 0
    for index, name in enumerate(channel_names):
        selected_steps = step[:, index][consecutive[:, index]]
        channel = stats(selected_steps)
        runs: list[dict[str, Any]] = []
        channel_maximum_drift = 0.0
        for start, stop in _true_runs(stance[:, index]):
            xy = position[start:stop, index, :2]
            anchor_drift = np.linalg.norm(xy - xy[0], axis=-1)
            run_drift = float(np.max(anchor_drift))
            path_length = float(
                np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=-1))
            )
            channel_maximum_drift = max(channel_maximum_drift, run_drift)
            runs.append(
                {
                    "start_frame": start,
                    "stop_frame_exclusive": stop,
                    "frames": stop - start,
                    "consecutive_pairs": max(0, stop - start - 1),
                    "maximum_anchor_drift_xy_m": run_drift,
                    "path_length_xy_m": path_length,
                }
            )
        channel.update(
            {
                "stance_frames": int(np.count_nonzero(stance[:, index])),
                "consecutive_pair_count": int(
                    np.count_nonzero(consecutive[:, index])
                ),
                "stance_run_count": len(runs),
                "maximum_stance_run_drift_xy_m": channel_maximum_drift,
                "stance_runs": runs,
            }
        )
        channel_metrics[name] = channel
        maximum_run_drift = max(maximum_run_drift, channel_maximum_drift)
        total_runs += len(runs)

    aggregate = stats(step[consecutive])
    aggregate.update(
        {
            "channels": channel_metrics,
            "stance_run_count": total_runs,
            "maximum_stance_run_drift_xy_m": maximum_run_drift,
            "selection": "consecutive_same_channel_stance",
            "label_source": label_source,
            "label_keys": list(label_keys),
            "label_semantics": (
                "support-lock stance labels; used for slip, consecutive step, "
                "and run drift only"
            ),
            "uses_contact_fallback": label_source.startswith("contact_fallback"),
        }
    )
    return aggregate


def _reference_checks(
    arrays: dict[str, np.ndarray], contract: Any, frame_count: int
) -> dict[str, Any]:
    q = np.asarray(arrays["joint_pos"][:frame_count], dtype=np.float64)
    qd = np.asarray(arrays["joint_vel"][:frame_count], dtype=np.float64)
    lower = np.asarray([joint.lower for joint in contract.joints], dtype=np.float64)
    upper = np.asarray([joint.upper for joint in contract.joints], dtype=np.float64)
    velocity = np.asarray([joint.velocity for joint in contract.joints], dtype=np.float64)
    hard_mask = (q < lower) | (q > upper)
    velocity_mask = np.abs(qd) > velocity
    return {
        "finite": bool(np.isfinite(q).all() and np.isfinite(qd).all()),
        "hard_joint_limit_violations": int(np.count_nonzero(hard_mask)),
        "velocity_limit_violations": int(np.count_nonzero(velocity_mask)),
        "minimum_hard_limit_margin_rad": float(np.min(np.minimum(q - lower, upper - q))),
        "maximum_velocity_ratio": float(np.max(np.abs(qd) / velocity)),
    }


def _runtime() -> dict[str, Any]:
    sim_version_file = PROJECT_ROOT.parent / "_isaac_sim" / "VERSION"
    return {
        "isaac_sim": (
            sim_version_file.read_text(encoding="utf-8").strip()
            if sim_version_file.is_file()
            else "unknown"
        ),
        "isaaclab_python_package": importlib.metadata.version("isaaclab"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(ARGS.device),
        "physics_dt_s": 0.005,
        "physics_frequency_hz": 200.0,
        "reference_frequency_hz": 50.0,
    }


def _make_env(duration_s: float) -> Any:
    cfg = load_cfg_from_registry(TASK_NAME, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    # Force the exact read-only parent asset rather than accepting a registry override.
    cfg.scene.robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.seed = int(ARGS.seed)
    cfg.sim.device = str(ARGS.device)
    cfg.sim.dt = 0.005
    cfg.sim.render_interval = 4
    cfg.decimation = 4
    cfg.episode_length_s = max(float(duration_s) + 1.0, 2.0)
    cfg.observations.policy.enable_corruption = False
    # No randomization, reset perturbation, commands, or pushes are allowed in a
    # deterministic reference validator.
    for name in (
        "physics_material",
        "add_base_mass",
        "base_com",
        "actuator_gains",
        "joint_friction",
        "base_external_force_torque",
        "reset_base",
        "reset_robot_joints",
        "push_robot",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    if cfg.curriculum is not None:
        for name in ("terrain_levels", "lin_vel_cmd_levels"):
            if hasattr(cfg.curriculum, name):
                setattr(cfg.curriculum, name, None)
    command = cfg.commands.base_velocity
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    return gym.make(
        TASK_NAME,
        cfg=cfg,
        render_mode="rgb_array" if ARGS.video else None,
    )


def _resolve_runtime(env: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    sensor = unwrapped.scene["contact_forces"]
    stored_joints = [str(name) for name in arrays["joint_names"].tolist()]
    stored_bodies = [str(name) for name in arrays["body_names"].tolist()]
    joint_ids, joint_names = robot.find_joints(
        _exact_patterns(stored_joints), preserve_order=True
    )
    body_ids, body_names = robot.find_bodies(
        _exact_patterns(stored_bodies), preserve_order=True
    )
    sensor_foot_ids, sensor_foot_names = sensor.find_bodies(
        _exact_patterns(FOOT_BODY_NAMES), preserve_order=True
    )
    sensor_pelvis_ids, sensor_pelvis_names = sensor.find_bodies(
        _exact_patterns(("pelvis",)), preserve_order=True
    )
    if joint_names != stored_joints or len(joint_ids) != 29:
        raise ValidationError(
            f"Isaac joint name resolution mismatch: expected={stored_joints}, got={joint_names}"
        )
    if body_names != stored_bodies:
        raise ValidationError(
            f"Isaac body name resolution mismatch: expected={stored_bodies}, got={body_names}"
        )
    if sensor_foot_names != list(FOOT_BODY_NAMES) or sensor_pelvis_names != ["pelvis"]:
        raise ValidationError("contact sensor could not resolve pelvis and both feet by name")
    return {
        "robot": robot,
        "sensor": sensor,
        "joint_ids": joint_ids,
        "body_ids": body_ids,
        "sensor_foot_ids": sensor_foot_ids,
        "sensor_pelvis_id": sensor_pelvis_ids[0],
        "stored_joint_names": stored_joints,
        "stored_body_names": stored_bodies,
        "runtime_joint_names": list(robot.joint_names),
        "runtime_body_names": list(robot.body_names),
        "sensor_body_names": list(sensor.body_names),
    }


def _state_tensors(
    arrays: dict[str, np.ndarray], frame: int, device: str
) -> dict[str, torch.Tensor]:
    tensor = lambda key: torch.as_tensor(  # noqa: E731
        arrays[key][frame], dtype=torch.float32, device=device
    ).unsqueeze(0)
    return {
        "joint_pos": tensor("joint_pos"),
        "joint_vel": tensor("joint_vel"),
        "root_pose": torch.cat((tensor("root_pos_w"), tensor("root_quat_wxyz")), dim=-1),
        "root_link_velocity": torch.cat(
            (tensor("root_lin_vel_w"), tensor("root_ang_vel_w")), dim=-1
        ),
    }


def _write_kinematic_state(
    env: Any, resolved: dict[str, Any], arrays: dict[str, np.ndarray], frame: int
) -> None:
    unwrapped = env.unwrapped
    robot = resolved["robot"]
    state = _state_tensors(arrays, frame, unwrapped.device)
    robot.write_root_pose_to_sim(state["root_pose"])
    # The archive stores pelvis/root-link velocity.  The legacy
    # write_root_velocity_to_sim alias expects COM velocity and must not be used.
    robot.write_root_link_velocity_to_sim(state["root_link_velocity"])
    robot.write_joint_state_to_sim(
        state["joint_pos"],
        state["joint_vel"],
        joint_ids=resolved["joint_ids"],
    )
    robot.set_joint_position_target(state["joint_pos"], joint_ids=resolved["joint_ids"])
    robot.set_joint_velocity_target(
        torch.zeros_like(state["joint_vel"]), joint_ids=resolved["joint_ids"]
    )
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()


def _snapshot(resolved: dict[str, Any]) -> dict[str, np.ndarray]:
    robot = resolved["robot"]
    data = robot.data
    joint_ids = resolved["joint_ids"]
    body_ids = resolved["body_ids"]
    return {
        "joint_pos": _cpu(data.joint_pos[0, joint_ids]),
        "joint_vel": _cpu(data.joint_vel[0, joint_ids]),
        "root_pos": _cpu(data.root_link_pos_w[0]),
        "root_quat": _cpu(data.root_link_quat_w[0]),
        "root_lin_vel": _cpu(data.root_link_lin_vel_w[0]),
        "root_ang_vel": _cpu(data.root_link_ang_vel_w[0]),
        "body_pos": _cpu(data.body_link_pos_w[0, body_ids]),
        "body_quat": _cpu(data.body_link_quat_w[0, body_ids]),
        "body_lin_vel": _cpu(data.body_link_lin_vel_w[0, body_ids]),
        "body_ang_vel": _cpu(data.body_link_ang_vel_w[0, body_ids]),
    }


class _Video:
    def __init__(self, env: Any, path: Path | None):
        self.env = env
        self.path = path
        self.writer = None
        self.frames = 0
        self.black_frames = 0
        self._warmed = False
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(
                str(path), fps=50, codec="libx264", quality=8, macro_block_size=None
            )

    def append(self, root_position: np.ndarray) -> None:
        if self.writer is None:
            return
        root = np.asarray(root_position, dtype=np.float64)
        self.env.unwrapped.sim.set_camera_view(
            eye=[float(root[0] - 3.5), float(root[1] - 3.0), 2.2],
            target=[float(root[0]), float(root[1]), 0.9],
        )
        if not self._warmed:
            for _ in range(5):
                self.env.render()
            self._warmed = True
        frame = np.asarray(self.env.render(), dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValidationError(f"renderer returned invalid RGB frame shape {frame.shape}")
        self.black_frames += int(np.max(frame) == 0)
        self.writer.append_data(frame)
        self.frames += 1

    def close(self) -> dict[str, Any]:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        return {
            "enabled": self.path is not None,
            "path": str(self.path) if self.path is not None else None,
            "frames": self.frames,
            "all_black_frames": self.black_frames,
        }


def _stack(snapshots: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.stack([sample[key] for sample in snapshots], axis=0)


def _common_metrics(
    arrays: dict[str, np.ndarray],
    snapshots: list[dict[str, np.ndarray]],
    body_names: list[str],
    frame_count: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    actual_joint = _stack(snapshots, "joint_pos")
    actual_joint_vel = _stack(snapshots, "joint_vel")
    actual_root_pos = _stack(snapshots, "root_pos")
    actual_root_quat = _stack(snapshots, "root_quat")
    actual_root_lin = _stack(snapshots, "root_lin_vel")
    actual_root_ang = _stack(snapshots, "root_ang_vel")
    actual_body_pos = _stack(snapshots, "body_pos")
    actual_body_quat = _stack(snapshots, "body_quat")
    actual_body_lin = _stack(snapshots, "body_lin_vel")
    actual_body_ang = _stack(snapshots, "body_ang_vel")
    actual_sole = _sole_positions(actual_body_pos, actual_body_quat, body_names)
    actual_support = _support_positions(actual_body_pos, actual_body_quat, body_names)
    actual_contact_points = _representative_contact_points(actual_support)

    _, joint_stats = _vector_error(
        arrays["joint_pos"][:frame_count, :, None], actual_joint[:, :, None]
    )
    _, joint_velocity_stats = _vector_error(
        arrays["joint_vel"][:frame_count, :, None], actual_joint_vel[:, :, None]
    )
    _, root_pos_stats = _vector_error(arrays["root_pos_w"][:frame_count], actual_root_pos)
    root_angle = _quat_angle_error(arrays["root_quat_wxyz"][:frame_count], actual_root_quat)
    _, root_lin_stats = _vector_error(
        arrays["root_lin_vel_w"][:frame_count], actual_root_lin
    )
    _, root_ang_stats = _vector_error(
        arrays["root_ang_vel_w"][:frame_count], actual_root_ang
    )
    _, body_pos_stats = _vector_error(arrays["body_pos_w"][:frame_count], actual_body_pos)
    body_angle = _quat_angle_error(
        arrays["body_quat_wxyz"][:frame_count], actual_body_quat
    )
    _, body_lin_stats = _vector_error(
        arrays["body_lin_vel_w"][:frame_count], actual_body_lin
    )
    _, body_ang_stats = _vector_error(
        arrays["body_ang_vel_w"][:frame_count], actual_body_ang
    )
    _, sole_stats = _vector_error(
        arrays["foot_sole_position_w"][:frame_count], actual_sole
    )
    geometry_metrics: dict[str, Any] = {}
    if "foot_support_position_w" in arrays:
        _, support_stats = _vector_error(
            arrays["foot_support_position_w"][:frame_count].reshape(
                frame_count, 2, 4, 3
            ),
            actual_support,
        )
        _, contact_point_stats = _vector_error(
            arrays["contact_point_position_w"][:frame_count],
            actual_contact_points,
        )
        geometry_metrics.update(
            {
                "support_sphere_center_position_error_m": support_stats,
                "heel_toe_representative_position_error_m": contact_point_stats,
            }
        )
    return (
        {
            "joint_position_error_rad": joint_stats,
            "joint_velocity_error_rad_s": joint_velocity_stats,
            "root_link_position_error_m": root_pos_stats,
            "root_link_orientation_error_rad": _error_stats(root_angle),
            "root_link_linear_velocity_error_m_s": root_lin_stats,
            "root_link_angular_velocity_error_rad_s": root_ang_stats,
            "body_link_position_error_m": body_pos_stats,
            "body_link_orientation_error_rad": _error_stats(body_angle),
            "body_link_linear_velocity_error_m_s": body_lin_stats,
            "body_link_angular_velocity_error_rad_s": body_ang_stats,
            "sole_center_position_error_m": sole_stats,
            **geometry_metrics,
        },
        {
            "sole": actual_sole,
            "support": actual_support,
            "contact_points": actual_contact_points,
        },
    )


def _soft_limit_metrics(
    resolved: dict[str, Any], actual_joint: np.ndarray, reference_joint: np.ndarray
) -> dict[str, Any]:
    limits = _cpu(
        resolved["robot"].data.soft_joint_pos_limits[0, resolved["joint_ids"]]
    ).astype(np.float64)
    reference_mask = (reference_joint < limits[:, 0]) | (reference_joint > limits[:, 1])
    actual_mask = (actual_joint < limits[:, 0]) | (actual_joint > limits[:, 1])
    return {
        "limits_rad": limits,
        "reference_violation_samples": int(np.count_nonzero(reference_mask)),
        "actual_violation_samples": int(np.count_nonzero(actual_mask)),
    }


def _validate_kinematic(
    env: Any,
    resolved: dict[str, Any],
    arrays: dict[str, np.ndarray],
    frame_count: int,
    video: _Video,
) -> tuple[dict[str, Any], list[str], list[str]]:
    snapshots: list[dict[str, np.ndarray]] = []
    for frame in range(frame_count):
        if not SIMULATION_APP.is_running():
            raise ValidationError(f"Isaac app stopped at kinematic frame {frame}")
        _write_kinematic_state(env, resolved, arrays, frame)
        sample = _snapshot(resolved)
        snapshots.append(sample)
        video.append(sample["root_pos"])

    body_names = resolved["stored_body_names"]
    metrics, geometry = _common_metrics(
        arrays, snapshots, body_names, frame_count
    )
    contacts, contact_names, contact_layout, contact_keys = _contact_layout(
        arrays, frame_count
    )
    stance, stance_source, stance_keys = _stance_layout(
        arrays, frame_count, contacts, contact_keys
    )
    contact_points = (
        geometry["contact_points"]
        if contact_layout == "heel_toe_v3"
        else geometry["sole"]
    )
    contact_velocity = _linear_velocity(contact_points, 50.0)
    radius = float(_scalar(arrays, "foot_collision_radius_m"))
    support_clearance = geometry["support"][..., 2] - radius
    representative_clearance = contact_points[..., 2] - radius
    geometric_contact = (
        representative_clearance <= float(ARGS.geometric_contact_height)
    ) & (
        np.linalg.norm(contact_velocity, axis=-1)
        <= float(ARGS.geometric_contact_speed)
    )
    contact = _contact_scores(contacts, geometric_contact, contact_names)
    contact.update(
        {
            "label_source": contact_layout,
            "label_keys": list(contact_keys),
            "label_semantics": "physical/geometric contact, not support-lock stance",
        }
    )
    slip = _stance_slip(
        contact_points,
        stance,
        50.0,
        contact_names,
        stance_source,
        stance_keys,
    )
    active_penetration = _active_contact_penetration(
        support_clearance, contacts, contact_layout, contact_keys
    )
    actual_joint = _stack(snapshots, "joint_pos")
    soft_limits = _soft_limit_metrics(
        resolved, actual_joint, arrays["joint_pos"][:frame_count]
    )
    metrics.update(
        {
            "contact_layout": contact_layout,
            "label_semantics": _label_semantics(
                contact_layout, contact_keys, stance_source, stance_keys
            ),
            "support_sphere_clearance_m": {
                "minimum": float(np.min(support_clearance)),
                "p05": float(np.quantile(support_clearance, 0.05)),
                "maximum": float(np.max(support_clearance)),
                "maximum_penetration": float(
                    max(0.0, -np.min(support_clearance))
                ),
                "penetration_semantics": "all support spheres; report-only",
                "sphere_count_per_frame": 8,
            },
            "contact_active_support_sphere_penetration_m": active_penetration,
            "contact_representative_clearance_m": {
                "minimum": float(np.min(representative_clearance)),
                "p05": float(np.quantile(representative_clearance, 0.05)),
                "maximum": float(np.max(representative_clearance)),
            },
            "geometric_contact": contact,
            "reference_stance_support_point_slip": slip,
            "isaac_soft_joint_limits": soft_limits,
            "all_state_values_finite": bool(
                all(np.isfinite(_stack(snapshots, key)).all() for key in snapshots[0])
            ),
        }
    )

    failures: list[str] = []
    warnings: list[str] = []
    if stance_source.startswith("contact_fallback"):
        warnings.append(
            "explicit heel/toe stance channels are absent; stance slip/step/run-drift "
            "use heel/toe contact labels as a backwards-compatible fallback"
        )
    checks = (
        (
            metrics["joint_position_error_rad"]["maximum"],
            ARGS.kinematic_joint_tol,
            "joint position FK/write error",
        ),
        (
            metrics["root_link_position_error_m"]["maximum"],
            ARGS.kinematic_root_pos_tol,
            "root-link position error",
        ),
        (
            metrics["root_link_orientation_error_rad"]["maximum"],
            ARGS.kinematic_root_angle_tol,
            "root-link orientation error",
        ),
        (
            metrics["body_link_position_error_m"]["maximum"],
            ARGS.kinematic_body_pos_tol,
            "body-link position FK error",
        ),
        (
            metrics["body_link_orientation_error_rad"]["maximum"],
            ARGS.kinematic_body_angle_tol,
            "body-link orientation FK error",
        ),
        (
            metrics["sole_center_position_error_m"]["maximum"],
            ARGS.kinematic_sole_pos_tol,
            "sole-center FK error",
        ),
        (
            metrics["contact_active_support_sphere_penetration_m"]["maximum_m"],
            ARGS.max_sole_penetration,
            "contact-active support-sphere penetration",
        ),
        (
            metrics["reference_stance_support_point_slip"]["p95_m_s"],
            ARGS.kinematic_max_stance_slip_p95,
            "reference stance slip p95",
        ),
    )
    if "support_sphere_center_position_error_m" in metrics:
        checks += (
            (
                metrics["support_sphere_center_position_error_m"]["maximum"],
                ARGS.kinematic_sole_pos_tol,
                "support-sphere FK error",
            ),
            (
                metrics["heel_toe_representative_position_error_m"]["maximum"],
                ARGS.kinematic_sole_pos_tol,
                "heel/toe representative FK error",
            ),
        )
    for actual, limit, label in checks:
        if actual > limit:
            failures.append(f"{label} {actual:.6g} exceeds {limit:.6g}")
    f1 = metrics["geometric_contact"]["aggregate"]["f1"]
    if f1 < ARGS.kinematic_min_contact_f1:
        failures.append(
            f"geometric contact F1 {f1:.6g} is below {ARGS.kinematic_min_contact_f1:.6g}"
        )
    if not metrics["all_state_values_finite"]:
        failures.append("Isaac produced NaN/Inf during kinematic replay")
    if soft_limits["reference_violation_samples"]:
        warnings.append(
            "reference remains inside URDF hard limits but crosses the locomotion "
            f"task's 0.9 soft limits at {soft_limits['reference_violation_samples']} samples"
        )
    return metrics, failures, warnings


def _initialize_pd(
    env: Any, resolved: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    unwrapped = env.unwrapped
    robot = resolved["robot"]
    unwrapped.scene.reset()
    state = _state_tensors(arrays, 0, unwrapped.device)
    robot.write_root_pose_to_sim(state["root_pose"])
    robot.write_root_link_velocity_to_sim(state["root_link_velocity"])
    robot.write_joint_state_to_sim(
        state["joint_pos"], state["joint_vel"], joint_ids=resolved["joint_ids"]
    )
    robot.set_joint_position_target(state["joint_pos"], joint_ids=resolved["joint_ids"])
    robot.set_joint_velocity_target(
        torch.zeros_like(state["joint_vel"]), joint_ids=resolved["joint_ids"]
    )
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()


def _pd_force_state(resolved: dict[str, Any]) -> dict[str, Any]:
    sensor = resolved["sensor"]
    robot = resolved["robot"]
    forces = _cpu(sensor.data.net_forces_w[0])
    force_norm = np.linalg.norm(forces, axis=-1)
    foot_force = force_norm[resolved["sensor_foot_ids"]]
    pelvis_force = float(force_norm[resolved["sensor_pelvis_id"]])
    excluded = set(resolved["sensor_foot_ids"])
    undesired = np.asarray(
        [value for index, value in enumerate(force_norm) if index not in excluded],
        dtype=np.float64,
    )
    projected = _cpu(robot.data.projected_gravity_b[0])
    tilt = float(math.acos(float(np.clip(-projected[2], -1.0, 1.0))))
    computed = _cpu(robot.data.computed_torque[0, resolved["joint_ids"]])
    applied = _cpu(robot.data.applied_torque[0, resolved["joint_ids"]])
    return {
        "foot_force": foot_force,
        "pelvis_force": pelvis_force,
        "max_undesired_force": float(np.max(undesired)) if undesired.size else 0.0,
        "root_height": float(robot.data.root_link_pos_w[0, 2]),
        "tilt": tilt,
        "computed_torque": computed,
        "applied_torque": applied,
    }


def _validate_pd(
    env: Any,
    resolved: dict[str, Any],
    arrays: dict[str, np.ndarray],
    frame_count: int,
    contract: Any,
    video: _Video,
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    _initialize_pd(env, resolved, arrays)
    unwrapped = env.unwrapped
    robot = resolved["robot"]
    snapshots = [_snapshot(resolved)]
    initial_force = _pd_force_state(resolved)
    foot_forces = [initial_force["foot_force"]]
    min_height = initial_force["root_height"]
    max_tilt = initial_force["tilt"]
    max_pelvis_force = initial_force["pelvis_force"]
    max_undesired_force = initial_force["max_undesired_force"]
    max_computed_torque = np.abs(initial_force["computed_torque"])
    max_applied_torque = np.abs(initial_force["applied_torque"])
    saturated_samples = 0
    torque_samples = 0
    first_fall_time: float | None = None
    stopped_non_finite = False
    video.append(snapshots[0]["root_pos"])

    effort_limits = np.asarray([joint.effort for joint in contract.joints])
    for frame in range(frame_count - 1):
        q0 = np.asarray(arrays["joint_pos"][frame], dtype=np.float64)
        q1 = np.asarray(arrays["joint_pos"][frame + 1], dtype=np.float64)
        for substep in range(1, 5):
            if not SIMULATION_APP.is_running():
                raise ValidationError(
                    f"Isaac app stopped at PD frame {frame}, substep {substep}"
                )
            alpha = substep / 4.0
            target_np = (1.0 - alpha) * q0 + alpha * q1
            target = torch.as_tensor(
                target_np, dtype=torch.float32, device=unwrapped.device
            ).unsqueeze(0)
            robot.set_joint_position_target(target, joint_ids=resolved["joint_ids"])
            # Real deployment and the existing action term use qd_des=0.
            robot.set_joint_velocity_target(
                torch.zeros_like(target), joint_ids=resolved["joint_ids"]
            )
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.step(render=False)
            unwrapped.scene.update(0.005)
            force = _pd_force_state(resolved)
            min_height = min(min_height, force["root_height"])
            max_tilt = max(max_tilt, force["tilt"])
            max_pelvis_force = max(max_pelvis_force, force["pelvis_force"])
            max_undesired_force = max(
                max_undesired_force, force["max_undesired_force"]
            )
            max_computed_torque = np.maximum(
                max_computed_torque, np.abs(force["computed_torque"])
            )
            max_applied_torque = np.maximum(
                max_applied_torque, np.abs(force["applied_torque"])
            )
            saturated_samples += int(
                np.count_nonzero(np.abs(force["computed_torque"]) > effort_limits + 1.0e-5)
            )
            torque_samples += len(effort_limits)
            fell = (
                force["root_height"] < float(ARGS.fall_height)
                or force["tilt"] > float(ARGS.fall_tilt)
                or force["pelvis_force"] > float(ARGS.contact_force_threshold)
            )
            if fell and first_fall_time is None:
                first_fall_time = (frame * 4 + substep) * 0.005
            state_finite = bool(
                torch.isfinite(robot.data.joint_pos).all()
                and torch.isfinite(robot.data.joint_vel).all()
                and torch.isfinite(robot.data.root_link_pose_w).all()
            )
            if not state_finite:
                stopped_non_finite = True
                break
        if stopped_non_finite:
            break
        sample = _snapshot(resolved)
        snapshots.append(sample)
        foot_forces.append(force["foot_force"])
        video.append(sample["root_pos"])

    completed = len(snapshots)
    body_names = resolved["stored_body_names"]
    metrics, geometry = _common_metrics(arrays, snapshots, body_names, completed)
    contacts, contact_names, contact_layout, contact_keys = _contact_layout(
        arrays, completed
    )
    stance, stance_source, stance_keys = _stance_layout(
        arrays, completed, contacts, contact_keys
    )
    contact_points = (
        geometry["contact_points"]
        if contact_layout == "heel_toe_v3"
        else geometry["sole"]
    )
    contact_velocity = _linear_velocity(contact_points, 50.0)
    radius = float(_scalar(arrays, "foot_collision_radius_m"))
    representative_clearance = contact_points[..., 2] - radius
    geometric_contact_mask = (
        representative_clearance <= float(ARGS.geometric_contact_height)
    ) & (
        np.linalg.norm(contact_velocity, axis=-1)
        <= float(ARGS.geometric_contact_speed)
    )
    geometric_contact = _contact_scores(
        contacts, geometric_contact_mask, contact_names
    )
    geometric_contact.update(
        {
            "label_source": contact_layout,
            "label_keys": list(contact_keys),
            "label_semantics": "physical/geometric contact, not support-lock stance",
        }
    )
    foot_contacts = np.stack(
        (
            arrays["left_foot_contact"][:completed],
            arrays["right_foot_contact"][:completed],
        ),
        axis=1,
    ).astype(bool)
    force_contact = np.asarray(foot_forces) > float(ARGS.contact_force_threshold)
    force_contact_scores = _contact_scores(
        foot_contacts, force_contact, ("left_foot", "right_foot")
    )
    force_contact_scores.update(
        {
            "label_source": "left/right composite contact",
            "label_keys": ["left_foot_contact", "right_foot_contact"],
            "label_semantics": "physical contact, not support-lock stance",
        }
    )
    slip = _stance_slip(
        contact_points,
        stance,
        50.0,
        contact_names,
        stance_source,
        stance_keys,
    )
    support_clearance = geometry["support"][..., 2] - radius
    active_penetration = _active_contact_penetration(
        support_clearance, contacts, contact_layout, contact_keys
    )
    actual_joint = _stack(snapshots, "joint_pos")
    actual_joint_vel = _stack(snapshots, "joint_vel")
    lower = np.asarray([joint.lower for joint in contract.joints])
    upper = np.asarray([joint.upper for joint in contract.joints])
    velocity = np.asarray([joint.velocity for joint in contract.joints])
    hard_violations = int(
        np.count_nonzero((actual_joint < lower) | (actual_joint > upper))
    )
    velocity_violations = int(np.count_nonzero(np.abs(actual_joint_vel) > velocity))
    soft_limits = _soft_limit_metrics(
        resolved, actual_joint, arrays["joint_pos"][:completed]
    )
    metrics.update(
        {
            "contact_layout": contact_layout,
            "label_semantics": _label_semantics(
                contact_layout, contact_keys, stance_source, stance_keys
            ),
            "completed_frames": completed,
            "requested_frames": frame_count,
            "completed_full_clip": completed == frame_count,
            "stopped_on_non_finite": stopped_non_finite,
            "execution_complete": completed == frame_count and not stopped_non_finite,
            "first_fall_time_s": first_fall_time,
            "minimum_root_link_height_m": float(min_height),
            "maximum_tilt_rad": float(max_tilt),
            "maximum_pelvis_contact_force_n": float(max_pelvis_force),
            "maximum_non_foot_contact_force_n": float(max_undesired_force),
            "force_contact": force_contact_scores,
            "geometric_contact": geometric_contact,
            "support_sphere_clearance_m": {
                "minimum": float(np.min(support_clearance)),
                "p05": float(np.quantile(support_clearance, 0.05)),
                "maximum": float(np.max(support_clearance)),
                "maximum_penetration": float(
                    max(0.0, -np.min(support_clearance))
                ),
                "penetration_semantics": "all support spheres; report-only",
                "sphere_count_per_frame": 8,
            },
            "contact_active_support_sphere_penetration_m": active_penetration,
            "reference_stance_support_point_slip": slip,
            "actual_hard_joint_limit_violations": hard_violations,
            "actual_velocity_limit_violations": velocity_violations,
            "isaac_soft_joint_limits": soft_limits,
            "maximum_computed_torque_nm_by_joint": max_computed_torque,
            "maximum_applied_torque_nm_by_joint": max_applied_torque,
            "torque_saturation_fraction": (
                float(saturated_samples / torque_samples) if torque_samples else 0.0
            ),
        }
    )

    execution_failures: list[str] = []
    dynamic_failures: list[str] = []
    warnings: list[str] = []
    if stance_source.startswith("contact_fallback"):
        warnings.append(
            "explicit heel/toe stance channels are absent; stance slip/step/run-drift "
            "use heel/toe contact labels as a backwards-compatible fallback"
        )
    if completed != frame_count:
        execution_failures.append(f"PD replay completed {completed}/{frame_count} frames")
    if stopped_non_finite:
        execution_failures.append("Isaac produced NaN/Inf during PD replay")
    if first_fall_time is not None:
        dynamic_failures.append(f"fall detected at {first_fall_time:.3f} s")
    if hard_violations:
        dynamic_failures.append(
            f"actual joints crossed URDF hard limits {hard_violations} times"
        )
    if velocity_violations:
        dynamic_failures.append(
            f"actual joints crossed velocity limits {velocity_violations} times"
        )
    joint_rmse = metrics["joint_position_error_rad"]["rmse"]
    joint_max = metrics["joint_position_error_rad"]["maximum"]
    root_rmse = metrics["root_link_position_error_m"]["rmse"]
    f1 = metrics["force_contact"]["aggregate"]["f1"]
    slip_p95 = metrics["reference_stance_support_point_slip"]["p95_m_s"]
    if joint_rmse > ARGS.pd_max_joint_rmse:
        dynamic_failures.append(
            f"PD joint RMSE {joint_rmse:.6g} exceeds {ARGS.pd_max_joint_rmse:.6g} rad"
        )
    if joint_max > ARGS.pd_max_joint_error:
        dynamic_failures.append(
            f"PD max joint error {joint_max:.6g} exceeds {ARGS.pd_max_joint_error:.6g} rad"
        )
    if root_rmse > ARGS.pd_max_root_position_rmse:
        dynamic_failures.append(
            f"PD root position RMSE {root_rmse:.6g} exceeds "
            f"{ARGS.pd_max_root_position_rmse:.6g} m"
        )
    if f1 < ARGS.pd_min_contact_f1:
        dynamic_failures.append(
            f"force-contact F1 {f1:.6g} is below {ARGS.pd_min_contact_f1:.6g}"
        )
    if slip_p95 > ARGS.pd_max_stance_slip_p95:
        dynamic_failures.append(
            f"PD stance slip p95 {slip_p95:.6g} exceeds "
            f"{ARGS.pd_max_stance_slip_p95:.6g} m/s"
        )
    if soft_limits["reference_violation_samples"]:
        warnings.append(
            "reference crosses the locomotion task's 0.9 soft limits at "
            f"{soft_limits['reference_violation_samples']} samples"
        )
    if metrics["torque_saturation_fraction"] > 0.05:
        warnings.append(
            f"PD torque saturation fraction is {metrics['torque_saturation_fraction']:.3%}"
        )
    return metrics, execution_failures, dynamic_failures, warnings


def _thresholds() -> dict[str, Any]:
    return {
        "contact_force_n": ARGS.contact_force_threshold,
        "geometric_contact_height_m": ARGS.geometric_contact_height,
        "geometric_contact_speed_m_s": ARGS.geometric_contact_speed,
        "fall_height_m": ARGS.fall_height,
        "fall_tilt_rad": ARGS.fall_tilt,
        "kinematic": {
            "joint_error_rad": ARGS.kinematic_joint_tol,
            "root_position_error_m": ARGS.kinematic_root_pos_tol,
            "root_orientation_error_rad": ARGS.kinematic_root_angle_tol,
            "body_position_error_m": ARGS.kinematic_body_pos_tol,
            "body_orientation_error_rad": ARGS.kinematic_body_angle_tol,
            "sole_position_error_m": ARGS.kinematic_sole_pos_tol,
            "contact_active_support_sphere_penetration_m": ARGS.max_sole_penetration,
            "contact_f1_minimum": ARGS.kinematic_min_contact_f1,
            "stance_slip_p95_m_s": ARGS.kinematic_max_stance_slip_p95,
        },
        "pd_replay": {
            "joint_rmse_rad": ARGS.pd_max_joint_rmse,
            "joint_max_error_rad": ARGS.pd_max_joint_error,
            "root_position_rmse_m": ARGS.pd_max_root_position_rmse,
            "contact_f1_minimum": ARGS.pd_min_contact_f1,
            "stance_slip_p95_m_s": ARGS.pd_max_stance_slip_p95,
        },
    }


def run() -> dict[str, Any]:
    arrays = _load_archive(INPUT_PATH)
    total_frames = int(_scalar(arrays, "frame_count"))
    if ARGS.max_frames < 0:
        raise ValidationError("--max-frames cannot be negative")
    frame_count = (
        min(total_frames, int(ARGS.max_frames)) if ARGS.max_frames else total_frames
    )
    if frame_count < 2:
        raise ValidationError("validation requires at least two frames")
    contract = load_joint_contract()
    archive_joint_names = [str(name) for name in arrays["joint_names"].tolist()]
    if archive_joint_names != contract.policy_names:
        raise ValidationError("archive joint_names differ from the parent G1 policy contract")
    if str(_scalar(arrays, "g1_contract_checksum_sha256")) != _sha256(CONTRACT_PATH):
        raise ValidationError("archive was generated from a different G1 joint contract")
    if str(_scalar(arrays, "g1_urdf_checksum_sha256")) != _sha256(URDF_PATH):
        raise ValidationError("archive was generated from a different G1 URDF")
    reference = _reference_checks(arrays, contract, frame_count)

    env = None
    video = None
    try:
        env = _make_env((frame_count - 1) / 50.0)
        env.reset(seed=int(ARGS.seed))
        resolved = _resolve_runtime(env, arrays)
        video = _Video(env, VIDEO_PATH)
        if ARGS.mode == "kinematic":
            metrics, failures, warnings = _validate_kinematic(
                env, resolved, arrays, frame_count, video
            )
            dynamic_failures: list[str] = []
        else:
            metrics, failures, dynamic_failures, warnings = _validate_pd(
                env, resolved, arrays, frame_count, contract, video
            )
        if reference["hard_joint_limit_violations"]:
            failures.append(
                "reference contains "
                f"{reference['hard_joint_limit_violations']} URDF hard-limit violations"
            )
        if reference["velocity_limit_violations"]:
            failures.append(
                "reference contains "
                f"{reference['velocity_limit_violations']} joint velocity-limit violations"
            )
        video_report = video.close()
        video = None
        if video_report["enabled"] and video_report["all_black_frames"]:
            warnings.append(
                f"renderer produced {video_report['all_black_frames']} all-black frames"
            )
        execution_complete = not failures and (
            ARGS.mode == "kinematic" or bool(metrics["execution_complete"])
        )
        dynamics_passed = (
            not dynamic_failures if ARGS.mode == "pd-replay" else None
        )
        passed = execution_complete and not dynamic_failures
        # A fully completed, finite PD diagnostic is a successful invocation
        # even when it proves that open-loop dynamics are inadequate.  The JSON
        # keeps that scientific result explicit via passed/dynamics_passed.
        exit_code = (
            0
            if (ARGS.mode == "pd-replay" and execution_complete)
            else (0 if passed else 2)
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": ARGS.mode,
            "passed": passed,
            "execution_complete": execution_complete,
            "dynamics_passed": dynamics_passed,
            "exit_code": exit_code,
            "failure_reasons": failures,
            "dynamic_failure_reasons": dynamic_failures,
            "warnings": warnings,
            "validator": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "input": {
                "path": str(INPUT_PATH),
                "sha256": _sha256(INPUT_PATH),
                "archive_schema": _scalar(arrays, "schema_version"),
                "total_frames": total_frames,
                "validated_frames": frame_count,
                "full_clip": frame_count == total_frames,
                "duration_s": float((frame_count - 1) / 50.0),
            },
            "output": str(OUTPUT_PATH),
            "runtime": _runtime(),
            "asset": {
                "urdf": str(URDF_PATH),
                "urdf_sha256": _sha256(URDF_PATH),
                "joint_contract": str(CONTRACT_PATH),
                "joint_contract_sha256": _sha256(CONTRACT_PATH),
                "spawn_asset_path": str(G1_29DOF_CFG.spawn.asset_path),
            },
            "name_resolution": {
                "archive_joint_names": resolved["stored_joint_names"],
                "archive_body_names": resolved["stored_body_names"],
                "runtime_joint_order": resolved["runtime_joint_names"],
                "runtime_body_order": resolved["runtime_body_names"],
                "contact_sensor_body_order": resolved["sensor_body_names"],
                "resolved_by_name": True,
            },
            "reference_checks": reference,
            "thresholds": _thresholds(),
            "metrics": metrics,
            "video": video_report,
            "interpretation": (
                "Independent state-write/FK agreement; this is not a dynamics claim."
                if ARGS.mode == "kinematic"
                else (
                    "Open-loop joint-position PD with a free floating root; no policy "
                    "or root teleportation. A completed diagnostic exits normally even "
                    "when passed/dynamics_passed are false."
                )
            ),
        }
    finally:
        if video is not None:
            video.close()
        if env is not None:
            env.close()


def _error_report(exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": ARGS.mode,
        "passed": False,
        "execution_complete": False,
        "dynamics_passed": None,
        "exit_code": 1,
        "failure_reasons": [f"{type(exc).__name__}: {exc}"],
        "dynamic_failure_reasons": [],
        "warnings": [],
        "validator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "input": {"path": str(INPUT_PATH)},
        "output": str(OUTPUT_PATH),
        "runtime": _runtime(),
        "traceback": traceback.format_exc(),
    }


EXIT_CODE = 1
try:
    REPORT = run()
    _write_json_atomic(OUTPUT_PATH, REPORT)
    EXIT_CODE = int(REPORT["exit_code"])
    print(
        json.dumps(_json_value(REPORT), ensure_ascii=False, indent=2, sort_keys=True),
        flush=True,
    )
    print(f"Report: {OUTPUT_PATH}", flush=True)
except BaseException as error:
    REPORT = _error_report(error)
    try:
        _write_json_atomic(OUTPUT_PATH, REPORT)
        print(f"Report: {OUTPUT_PATH}", file=sys.stderr, flush=True)
    finally:
        traceback.print_exc()
        EXIT_CODE = 1
finally:
    # Kit's binary launcher can replace Python's final exit status during
    # shutdown.  The Gym environment/SimulationContext was already closed by
    # run(); on an error, preserve its non-zero status before Kit can mask it.
    sys.stdout.flush()
    sys.stderr.flush()
    if EXIT_CODE:
        os._exit(EXIT_CODE)
    SIMULATION_APP.close()
    os._exit(0)
