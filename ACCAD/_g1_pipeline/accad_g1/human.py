"""Strict ACCAD stage-II loading and licensed SMPL-X reconstruction.

This module intentionally keeps two boundaries explicit:

* AMASS/ACCAD ``*_stageii.npz`` files are motion parameters, not joint
  positions.  :func:`load_stageii_numeric` validates only the numeric fields
  needed to reconstruct those parameters and never enables pickle loading.
* Human reconstruction needs both the exact ``smplx`` Python package version
  and the separately licensed SMPL-X neutral model.  The preflight report is
  designed to fail before a long batch job starts when either is absent.

The generated archive is a *human reconstruction basis*.  It contains named
joint positions, local/global rotations, velocities, and contact labels built
only after semantic heel/toe names are verified.  Visual Z-up/ground/contact
validation is still required before it is treated as a completed Gate-2
artefact.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


REQUIRED_SMPLX_VERSION = "0.1.28"
MODEL_FILENAME = "SMPLX_NEUTRAL.npz"
MODEL_SUBDIRECTORY = "smplx"
NUM_BETAS = 16
BODY_POSE_SIZE = 21 * 3
HAND_POSE_SIZE = 15 * 3
FULL_POSE_SIZE = 165
KINEMATIC_JOINT_COUNT = 55

_FRAME_FIELDS = {
    "trans": 3,
    "root_orient": 3,
    "pose_body": BODY_POSE_SIZE,
    "pose_hand": 2 * HAND_POSE_SIZE,
    "pose_jaw": 3,
    "pose_eye": 6,
}


class StageIIFormatError(ValueError):
    """Raised when an input is not a strict numeric AMASS stage-II motion."""


class HumanPrerequisiteError(RuntimeError):
    """Raised when the pinned package or licensed body model is unavailable."""


class HumanReconstructionError(RuntimeError):
    """Raised when SMPL-X output cannot be mapped without guessing."""


def _as_path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def _read_scalar(array: np.ndarray, name: str) -> Any:
    if array.shape != ():
        raise StageIIFormatError(f"{name} must be a scalar; got shape {array.shape}")
    if array.dtype.hasobject:
        raise StageIIFormatError(f"{name} must not use object dtype")
    return array.item()


def _read_text_scalar(array: np.ndarray, name: str) -> str:
    value = _read_scalar(array, name)
    if array.dtype.kind not in {"U", "S"}:
        raise StageIIFormatError(
            f"{name} must be a fixed-width string, not dtype {array.dtype}"
        )
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _read_float_array(
    archive: np.lib.npyio.NpzFile,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    try:
        value = archive[name]
    except ValueError as exc:
        raise StageIIFormatError(
            f"{name} could not be read with allow_pickle=False: {exc}"
        ) from exc
    if value.dtype.kind != "f" or value.dtype.hasobject:
        raise StageIIFormatError(f"{name} must be floating point; got {value.dtype}")
    if value.shape != expected_shape:
        raise StageIIFormatError(
            f"{name} must have shape {expected_shape}; got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise StageIIFormatError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(value, dtype=np.float32)


def load_stageii_numeric(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the SMPL-X fields of an AMASS stage-II archive without pickle.

    Unknown fields are never indexed.  This is important because AMASS files
    can contain object-valued marker metadata such as ``markers_latent_vids``;
    those fields are neither required nor trusted by this pipeline.

    The returned ``left_hand_pose`` and ``right_hand_pose`` are the exact first
    and last 45 columns of ``pose_hand`` respectively.
    """

    input_path = _as_path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"stage-II motion not found: {input_path}")

    required = {
        *_FRAME_FIELDS,
        "poses",
        "betas",
        "num_betas",
        "mocap_frame_rate",
        "mocap_time_length",
        "gender",
        "surface_model_type",
    }
    try:
        archive_context = np.load(input_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise StageIIFormatError(f"cannot open {input_path}: {exc}") from exc

    with archive_context as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise StageIIFormatError(f"missing required keys: {', '.join(missing)}")

        # Read only a safe numeric array first to establish the frame count.
        try:
            trans_raw = archive["trans"]
        except ValueError as exc:
            raise StageIIFormatError(
                f"trans could not be read with allow_pickle=False: {exc}"
            ) from exc
        if trans_raw.dtype.kind != "f" or trans_raw.dtype.hasobject:
            raise StageIIFormatError(
                f"trans must be floating point; got {trans_raw.dtype}"
            )
        if trans_raw.ndim != 2 or trans_raw.shape[1] != 3:
            raise StageIIFormatError(f"trans must have shape (T, 3); got {trans_raw.shape}")
        frame_count = int(trans_raw.shape[0])
        if frame_count < 2:
            raise StageIIFormatError("a motion must contain at least two frames")
        if not np.isfinite(trans_raw).all():
            raise StageIIFormatError("trans contains NaN or infinity")

        fields: dict[str, np.ndarray] = {
            "trans": np.ascontiguousarray(trans_raw, dtype=np.float32)
        }
        for name, width in _FRAME_FIELDS.items():
            if name == "trans":
                continue
            fields[name] = _read_float_array(
                archive, name, (frame_count, width)
            )

        poses = _read_float_array(
            archive, "poses", (frame_count, FULL_POSE_SIZE)
        )
        betas = _read_float_array(archive, "betas", (NUM_BETAS,))

        num_betas_raw = archive["num_betas"]
        num_betas = _read_scalar(num_betas_raw, "num_betas")
        if num_betas_raw.dtype.kind not in {"i", "u"} or int(num_betas) != NUM_BETAS:
            raise StageIIFormatError(
                f"num_betas must be integer {NUM_BETAS}; got {num_betas!r}"
            )

        fps_raw = archive["mocap_frame_rate"]
        fps = _read_scalar(fps_raw, "mocap_frame_rate")
        if fps_raw.dtype.kind not in {"f", "i", "u"}:
            raise StageIIFormatError(
                f"mocap_frame_rate must be numeric; got {fps_raw.dtype}"
            )
        fps = float(fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise StageIIFormatError(f"invalid mocap_frame_rate: {fps!r}")

        duration_raw = archive["mocap_time_length"]
        duration = _read_scalar(duration_raw, "mocap_time_length")
        if duration_raw.dtype.kind not in {"f", "i", "u"}:
            raise StageIIFormatError(
                f"mocap_time_length must be numeric; got {duration_raw.dtype}"
            )
        duration = float(duration)
        expected_duration = frame_count / fps
        if not np.isfinite(duration) or not np.isclose(
            duration, expected_duration, rtol=1e-7, atol=1e-9
        ):
            raise StageIIFormatError(
                "mocap_time_length must equal frame_count / mocap_frame_rate; "
                f"got {duration}, expected {expected_duration}"
            )

        gender = _read_text_scalar(archive["gender"], "gender").strip().lower()
        surface_model_type = _read_text_scalar(
            archive["surface_model_type"], "surface_model_type"
        ).strip().lower()
        if gender not in {"neutral", "male", "female"}:
            raise StageIIFormatError(f"unsupported gender metadata: {gender!r}")
        if surface_model_type != "smplx":
            raise StageIIFormatError(
                f"surface_model_type must be 'smplx'; got {surface_model_type!r}"
            )

        full_pose = np.concatenate(
            (
                fields["root_orient"],
                fields["pose_body"],
                fields["pose_jaw"],
                fields["pose_eye"],
                fields["pose_hand"],
            ),
            axis=1,
        )
        if not np.allclose(poses, full_pose, rtol=1e-6, atol=1e-7):
            raise StageIIFormatError(
                "poses is inconsistent with root/body/jaw/eye/hand components"
            )

    left_hand = np.ascontiguousarray(fields["pose_hand"][:, :HAND_POSE_SIZE])
    right_hand = np.ascontiguousarray(fields["pose_hand"][:, HAND_POSE_SIZE:])
    if left_hand.shape != (frame_count, HAND_POSE_SIZE) or right_hand.shape != (
        frame_count,
        HAND_POSE_SIZE,
    ):
        # The fixed pose_hand validation above should make this unreachable; it
        # remains an explicit invariant because a reversed/uneven hand split is
        # especially damaging and difficult to diagnose downstream.
        raise StageIIFormatError("pose_hand did not split into exact 45/45 columns")

    return {
        "path": input_path,
        "frame_count": frame_count,
        "mocap_frame_rate": fps,
        "mocap_time_length": duration,
        "gender": gender,
        "surface_model_type": surface_model_type,
        "num_betas": NUM_BETAS,
        "betas": betas,
        "trans": fields["trans"],
        "root_orient": fields["root_orient"],
        "body_pose": fields["pose_body"],
        "jaw_pose": fields["pose_jaw"],
        "left_eye_pose": np.ascontiguousarray(fields["pose_eye"][:, :3]),
        "right_eye_pose": np.ascontiguousarray(fields["pose_eye"][:, 3:]),
        "left_hand_pose": left_hand,
        "right_hand_pose": right_hand,
        "full_pose": np.ascontiguousarray(full_pose),
    }


def _inspect_model_file(path: Path) -> dict[str, Any]:
    """Inspect enough of a model archive to reject a wrong/corrupt file."""

    report: dict[str, Any] = {
        "path": str(path),
        "available": path.is_file(),
        "valid": False,
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "keys_count": None,
        "shapes": {},
        "dtypes": {},
        "error": None,
    }
    if not path.is_file():
        report["error"] = f"licensed model file is missing: {path}"
        return report

    required_keys = {
        "v_template",
        "shapedirs",
        "posedirs",
        "J_regressor",
        "weights",
        "kintree_table",
        "f",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            report["keys_count"] = len(archive.files)
            missing = sorted(required_keys.difference(archive.files))
            if missing:
                raise ValueError(f"missing SMPL-X model keys: {', '.join(missing)}")

            # shapedirs and posedirs can be very large.  Their presence is
            # checked above; only the comparatively small structural arrays are
            # materialized during this preflight.
            inspected = {
                key: archive[key]
                for key in ("v_template", "J_regressor", "weights", "kintree_table", "f")
            }
            for key, value in inspected.items():
                if value.dtype.hasobject or value.dtype.kind not in {"f", "i", "u"}:
                    raise ValueError(f"model key {key} has unsafe dtype {value.dtype}")
                if not np.isfinite(value).all():
                    raise ValueError(f"model key {key} contains NaN or infinity")
                report["shapes"][key] = list(value.shape)
                report["dtypes"][key] = str(value.dtype)

            vertices = inspected["v_template"]
            regressor = inspected["J_regressor"]
            weights = inspected["weights"]
            tree = inspected["kintree_table"]
            faces = inspected["f"]
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                raise ValueError(f"v_template has invalid shape {vertices.shape}")
            vertex_count = vertices.shape[0]
            if regressor.ndim != 2 or regressor.shape[1] != vertex_count:
                raise ValueError(f"J_regressor has invalid shape {regressor.shape}")
            if weights.ndim != 2 or weights.shape[0] != vertex_count:
                raise ValueError(f"weights has invalid shape {weights.shape}")
            if tree.ndim != 2 or tree.shape[0] != 2:
                raise ValueError(f"kintree_table has invalid shape {tree.shape}")
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError(f"f has invalid shape {faces.shape}")
            if regressor.shape[0] < KINEMATIC_JOINT_COUNT:
                raise ValueError(
                    "J_regressor has fewer than 55 SMPL-X kinematic joints"
                )
            if weights.shape[1] < KINEMATIC_JOINT_COUNT:
                raise ValueError("weights has fewer than 55 joint columns")
            if tree.shape[1] < KINEMATIC_JOINT_COUNT:
                raise ValueError("kintree_table has fewer than 55 joints")
    except (OSError, ValueError, KeyError) as exc:
        report["error"] = f"invalid SMPL-X neutral model: {exc}"
        return report

    report["valid"] = True
    return report


def check_human_prerequisites(
    pipeline_root: str | os.PathLike[str],
    model_root: str | os.PathLike[str],
    vendor_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return a JSON-serializable SMPL-X package/model preflight report.

    ``model_root`` is the parent directory expected by ``smplx.create``.  The
    licensed file must therefore be located at
    ``model_root/smplx/SMPLX_NEUTRAL.npz``.
    """

    pipeline_path = _as_path(pipeline_root)
    model_root_path = _as_path(model_root)
    vendor_path = _as_path(vendor_root)
    vendor_added = False
    if vendor_path.is_dir():
        vendor_text = str(vendor_path)
        while vendor_text in sys.path:
            sys.path.remove(vendor_text)
        sys.path.insert(0, vendor_text)
        importlib.invalidate_caches()
        vendor_added = True

    package_report: dict[str, Any] = {
        "available": False,
        "required_version": REQUIRED_SMPLX_VERSION,
        "version": None,
        "version_ok": False,
        "module_path": None,
        "error": None,
    }
    errors: list[str] = []
    try:
        module = importlib.import_module("smplx")
        package_report["available"] = True
        module_path = getattr(module, "__file__", None)
        package_report["module_path"] = str(module_path) if module_path else None
        try:
            version = importlib.metadata.version("smplx")
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", None)
        package_report["version"] = str(version) if version is not None else None
        package_report["version_ok"] = version == REQUIRED_SMPLX_VERSION
        if version is None:
            package_report["error"] = (
                "smplx is importable but its installed distribution version "
                "cannot be verified"
            )
        elif version != REQUIRED_SMPLX_VERSION:
            package_report["error"] = (
                f"smplx version {version} is installed; exact version "
                f"{REQUIRED_SMPLX_VERSION} is required"
            )
    except Exception as exc:  # Import can fail because of a missing transitive dependency.
        package_report["error"] = f"cannot import smplx: {type(exc).__name__}: {exc}"

    if package_report["error"]:
        errors.append(str(package_report["error"]))

    model_path = model_root_path / MODEL_SUBDIRECTORY / MODEL_FILENAME
    model_report = _inspect_model_file(model_path)
    if model_report["error"]:
        errors.append(str(model_report["error"]))

    next_actions: list[str] = []
    if not package_report["version_ok"]:
        next_actions.append(
            "Install the pinned package only inside the pipeline vendor folder: "
            f"python -m pip install --no-deps --target {vendor_path} "
            f"smplx=={REQUIRED_SMPLX_VERSION}"
        )
    if not model_report["valid"]:
        next_actions.append(
            "After accepting the SMPL-X license, place the neutral NPZ at: "
            f"{model_path}"
        )

    ready = bool(package_report["version_ok"] and model_report["valid"])
    return {
        # ``ok`` is the direct API result; ``ready`` is retained as an explicit
        # gate-state spelling for the pipeline orchestrator and status report.
        "ok": ready,
        "ready": ready,
        "pipeline_root": str(pipeline_path),
        "model_root": str(model_root_path),
        "vendor_root": str(vendor_path),
        "vendor_exists": vendor_path.is_dir(),
        "vendor_added_to_sys_path": vendor_added,
        "smplx": package_report,
        "model": model_report,
        "errors": errors,
        "next_actions": next_actions,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_sample_times(
    frame_count: int, source_fps: float, target_fps: float
) -> np.ndarray:
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"target_fps must be positive; got {target_fps!r}")
    # A frame represents t=i/fps.  Keep a uniform target grid and never create
    # an irregular final interval just to hit the source endpoint.
    endpoint = (frame_count - 1) / source_fps
    target_count = int(np.floor(endpoint * target_fps + 1e-9)) + 1
    return np.arange(max(target_count, 1), dtype=np.float64) / target_fps


def _resample_translation(
    values: np.ndarray,
    source_fps: float,
    target_times: np.ndarray,
) -> np.ndarray:
    from .math3d import resample_translation_antialiased, source_time_grid

    source_times = source_time_grid(len(values), source_fps)
    return np.asarray(
        resample_translation_antialiased(values, source_times, target_times),
        dtype=np.float32,
    )


def _resample_axis_angle(
    values: np.ndarray,
    source_fps: float,
    target_times: np.ndarray,
) -> np.ndarray:
    from .math3d import (
        quaternion_wxyz_to_rotvec,
        resample_quaternion_slerp,
        rotvec_to_quaternion_wxyz,
        source_time_grid,
    )

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"axis-angle input must have shape (T, J, 3); got {values.shape}")
    source_times = source_time_grid(values.shape[0], source_fps)
    source_quaternions = rotvec_to_quaternion_wxyz(values)
    target_quaternions = resample_quaternion_slerp(
        source_quaternions, source_times, target_times
    )
    return np.asarray(
        quaternion_wxyz_to_rotvec(target_quaternions), dtype=np.float32
    )


def _rotation_matrices(axis_angle: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    shape = axis_angle.shape
    return Rotation.from_rotvec(axis_angle.reshape(-1, 3)).as_matrix().reshape(
        *shape[:-1], 3, 3
    )


def _quaternion_wxyz(matrices: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    shape = matrices.shape
    xyzw = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(
        *shape[:-2], 4
    )
    return np.ascontiguousarray(xyzw[..., [3, 0, 1, 2]], dtype=np.float32)


def _continuous_quaternion_wxyz(matrices: np.ndarray) -> np.ndarray:
    """Convert matrices and enforce temporal sign continuity on axis zero."""

    from .math3d import quaternion_sign_continuity

    return np.asarray(
        quaternion_sign_continuity(_quaternion_wxyz(matrices), axis=0),
        dtype=np.float32,
    )


def _global_rotations(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    joint_count = local.shape[1]
    if parents.shape != (joint_count,):
        raise HumanReconstructionError(
            f"parents must have shape ({joint_count},); got {parents.shape}"
        )
    if np.any(parents < -1) or np.any(parents >= joint_count):
        raise HumanReconstructionError("SMPL-X parent table contains invalid indices")

    result = np.empty_like(local)
    state = np.zeros(joint_count, dtype=np.int8)

    def visit(joint: int) -> None:
        if state[joint] == 2:
            return
        if state[joint] == 1:
            raise HumanReconstructionError("SMPL-X parent table contains a cycle")
        state[joint] = 1
        parent = int(parents[joint])
        if parent < 0:
            result[:, joint] = local[:, joint]
        else:
            visit(parent)
            result[:, joint] = result[:, parent] @ local[:, joint]
        state[joint] = 2

    for joint_index in range(joint_count):
        visit(joint_index)
    if np.count_nonzero(parents < 0) != 1:
        raise HumanReconstructionError("SMPL-X parent table must have exactly one root")
    return result


def _linear_velocity(values: np.ndarray, fps: float) -> np.ndarray:
    """Frame-aligned finite-difference velocity along axis zero."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 2:
        raise ValueError("velocity input must contain at least two frames")
    edge_order = 2 if array.shape[0] >= 3 else 1
    return np.asarray(
        np.gradient(array, 1.0 / float(fps), axis=0, edge_order=edge_order),
        dtype=np.float32,
    )


def _angular_velocity_world(quaternions: np.ndarray, fps: float) -> np.ndarray:
    """Convert interval quaternion rates to a frame-aligned world velocity."""

    from .math3d import quaternion_angular_velocity_world

    interval = quaternion_angular_velocity_world(quaternions, 1.0 / float(fps))
    output = np.empty((quaternions.shape[0],) + interval.shape[1:], dtype=np.float32)
    output[0] = interval[0]
    output[-1] = interval[-1]
    if len(output) > 2:
        output[1:-1] = 0.5 * (interval[:-1] + interval[1:])
    return output


def _remove_short_true_runs(values: np.ndarray, minimum_frames: int) -> np.ndarray:
    result = np.asarray(values, dtype=bool).copy()
    start: int | None = None
    for index in range(len(result) + 1):
        active = index < len(result) and bool(result[index])
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start < minimum_frames:
                result[start:index] = False
            start = None
    return result


def _hysteresis_contact(
    height: np.ndarray,
    speed: np.ndarray,
    *,
    enter_height: float = 0.04,
    exit_height: float = 0.07,
    enter_speed: float = 0.35,
    exit_speed: float = 0.60,
    minimum_frames: int = 3,
) -> np.ndarray:
    contact = np.zeros(len(height), dtype=bool)
    active = False
    for frame in range(len(height)):
        if active:
            active = not (
                height[frame] > exit_height or speed[frame] > exit_speed
            )
        else:
            active = bool(
                height[frame] < enter_height and speed[frame] < enter_speed
            )
        contact[frame] = active
    return _remove_short_true_runs(contact, minimum_frames)


def _adaptive_ground_baseline(
    points: dict[str, np.ndarray],
    speeds: dict[str, np.ndarray],
    fps: float,
    *,
    slow_speed: float = 0.35,
    slow_vertical_speed: float = 0.15,
    window_seconds: float = 1.0,
    percentile: float = 0.20,
    max_drift_speed: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate slow calibration drift without treating a jump as the floor.

    ACCAD/SMPL-X fits can contain a slow common-mode vertical drift even on a
    flat floor.  A single scalar floor then drops valid stance labels near the
    end of a clip.  The estimator only admits low-speed heel/toe observations,
    forms a local lower envelope, and applies a two-sided rate limit.  The rate
    limit is important: a slow foot at the apex of a jump must not pull the
    inferred floor upward.

    The returned confidence flag records frames that have a nearby, low-speed
    foot observation.  Missing/flight intervals are bridged by the constrained
    envelope rather than being declared ground observations.
    """

    names = list(points)
    heights = np.stack([np.asarray(points[name])[:, 2] for name in names], axis=1)
    point_speeds = np.stack([np.asarray(speeds[name]) for name in names], axis=1)
    vertical_speeds = np.stack(
        [np.abs(_linear_velocity(points[name], fps)[:, 2]) for name in names],
        axis=1,
    )
    slow = (point_speeds < slow_speed) & (vertical_speeds < slow_vertical_speed)
    slow_minimum = np.full(heights.shape[0], np.nan, dtype=np.float64)
    for frame in range(heights.shape[0]):
        if np.any(slow[frame]):
            slow_minimum[frame] = float(np.min(heights[frame, slow[frame]]))

    finite = np.isfinite(slow_minimum)
    if np.count_nonzero(finite) < 3:
        # This is deliberately low-confidence.  It keeps the reconstruction
        # finite while ensuring a clip with no stance evidence cannot silently
        # claim a well-observed ground trajectory.
        fallback = float(np.quantile(heights, 0.05))
        return np.full(len(heights), fallback, dtype=np.float32), np.zeros(
            len(heights), dtype=bool
        )

    window = max(5, int(round(window_seconds * float(fps))))
    if window % 2 == 0:
        window += 1
    half_window = window // 2
    local_lower = np.full(len(slow_minimum), np.nan, dtype=np.float64)
    for frame in range(len(slow_minimum)):
        start = max(0, frame - half_window)
        stop = min(len(slow_minimum), frame + half_window + 1)
        candidates = slow_minimum[start:stop]
        candidates = candidates[np.isfinite(candidates)]
        if len(candidates) >= 3:
            local_lower[frame] = float(np.quantile(candidates, percentile))

    valid_lower = np.isfinite(local_lower)
    if not np.any(valid_lower):
        local_lower[:] = float(np.quantile(slow_minimum[finite], percentile))
    else:
        indices = np.arange(len(local_lower), dtype=np.float64)
        local_lower = np.interp(
            indices, indices[valid_lower], local_lower[valid_lower]
        )

    # Smooth only the already speed-gated envelope.  Reflect padding avoids an
    # artificial start/end offset.  The following Lipschitz minorant prevents
    # an elevated flight plateau from being promoted to ground.
    from scipy.ndimage import gaussian_filter1d

    sigma = max(1.0, 0.16 * float(fps))
    baseline = gaussian_filter1d(local_lower, sigma=sigma, mode="reflect")
    maximum_step = max_drift_speed / float(fps)
    for frame in range(1, len(baseline)):
        baseline[frame] = min(baseline[frame], baseline[frame - 1] + maximum_step)
    for frame in range(len(baseline) - 2, -1, -1):
        baseline[frame] = min(baseline[frame], baseline[frame + 1] + maximum_step)

    nearby = np.abs(heights - baseline[:, None]) <= 0.08
    confidence = np.any(slow & nearby, axis=1)
    return np.asarray(baseline, dtype=np.float32), confidence


def _infer_ground_contacts(
    joint_positions: np.ndarray,
    joint_names: list[str],
    fps: float,
) -> dict[str, Any]:
    """Infer ground and four contact tracks using verified semantic names."""

    required = (
        "left_heel",
        "left_big_toe",
        "left_small_toe",
        "right_heel",
        "right_big_toe",
        "right_small_toe",
    )
    missing = [name for name in required if name not in joint_names]
    if missing:
        raise HumanReconstructionError(
            "SMPL-X output lacks contact keypoints: " + ", ".join(missing)
        )
    index = {name: joint_names.index(name) for name in required}
    points = {
        "left_heel": joint_positions[:, index["left_heel"]],
        "left_toe": 0.5
        * (
            joint_positions[:, index["left_big_toe"]]
            + joint_positions[:, index["left_small_toe"]]
        ),
        "right_heel": joint_positions[:, index["right_heel"]],
        "right_toe": 0.5
        * (
            joint_positions[:, index["right_big_toe"]]
            + joint_positions[:, index["right_small_toe"]]
        ),
    }
    velocities = {name: _linear_velocity(value, fps) for name, value in points.items()}
    speeds = {name: np.linalg.norm(value, axis=-1) for name, value in velocities.items()}

    baseline, baseline_confidence = _adaptive_ground_baseline(
        points, speeds, float(fps)
    )
    minimum_contact_frames = max(3, int(round(0.06 * float(fps))))

    contacts = {
        f"{name}_contact": _hysteresis_contact(
            point[:, 2] - baseline,
            speeds[name],
            minimum_frames=minimum_contact_frames,
        )
        for name, point in points.items()
    }
    return {
        "ground_height_w": float(np.quantile(baseline, 0.05)),
        "ground_baseline_height_w": baseline,
        "ground_baseline_confidence": baseline_confidence,
        "ground_baseline_drift_m": float(baseline[-1] - baseline[0]),
        "contact_keypoint_position_w": np.stack(list(points.values()), axis=1).astype(
            np.float32
        ),
        "contact_keypoint_names": list(points),
        **contacts,
    }


def _resample_boolean_majority(
    values: np.ndarray,
    source_fps: float,
    target_times: np.ndarray,
    target_fps: float,
) -> np.ndarray:
    """Resample a boolean sequence by centered-bin majority, nearest on ties."""

    source = np.asarray(values)
    if source.ndim != 1 or source.dtype != np.bool_ or len(source) < 2:
        raise ValueError("contact values must be a one-dimensional bool sequence")
    source_times = np.arange(len(source), dtype=np.float64) / float(source_fps)
    result = np.zeros(len(target_times), dtype=bool)
    half_width = 0.5 / float(target_fps)
    for frame, target_time in enumerate(np.asarray(target_times, dtype=np.float64)):
        selected = np.flatnonzero(
            (source_times >= target_time - half_width)
            & (source_times < target_time + half_width)
        )
        nearest = int(
            np.clip(round(target_time * float(source_fps)), 0, len(source) - 1)
        )
        if len(selected) == 0:
            result[frame] = source[nearest]
            continue
        true_count = int(np.count_nonzero(source[selected]))
        false_count = len(selected) - true_count
        result[frame] = source[nearest] if true_count == false_count else true_count > false_count
    return result


def _validated_upstream_joint_names(
    output_joint_count: int,
    kinematic_joint_count: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    try:
        names_module = importlib.import_module("smplx.joint_names")
        upstream = list(getattr(names_module, "JOINT_NAMES"))
    except Exception as exc:
        raise HumanReconstructionError(
            f"cannot read joint names from pinned smplx package: {exc}"
        ) from exc
    if len(upstream) < output_joint_count:
        raise HumanReconstructionError(
            "SMPL-X returned more joints than its upstream JOINT_NAMES table: "
            f"{output_joint_count} > {len(upstream)}"
        )
    joint_names = [str(name) for name in upstream[:output_joint_count]]
    if len(set(joint_names)) != len(joint_names):
        raise HumanReconstructionError("upstream SMPL-X joint names are not unique")
    if kinematic_joint_count > len(joint_names):
        raise HumanReconstructionError(
            "SMPL-X kinematic tree is larger than its named joint output"
        )

    # These are semantic names, never numeric guesses.  Heel/toe contact names
    # are deliberately absent; a later contact stage must discover and verify
    # such names against this exact upstream table.
    required = {
        "pelvis",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_foot",
        "right_foot",
        "left_wrist",
        "right_wrist",
        "head",
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
    }
    missing = sorted(required.difference(joint_names))
    if missing:
        raise HumanReconstructionError(
            "pinned SMPL-X joint-name table lacks required semantic joints: "
            + ", ".join(missing)
        )
    mapping = {name: joint_names.index(name) for name in sorted(required)}
    return joint_names, joint_names[:kinematic_joint_count], mapping


def _save_npz_atomic(output_path: Path, payload: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", dir=output_path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            np.savez_compressed(temporary, **payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output_path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _run_smplx_fk(
    model: Any,
    local_axis_angle: np.ndarray,
    translation: np.ndarray,
    betas: np.ndarray,
    *,
    chunk_size: int,
    model_batch_size: int,
) -> tuple[np.ndarray, list[str], list[str], dict[str, int]]:
    """Run chunked joint-only SMPL-X FK for one pose/translation sequence."""

    import torch

    frame_count = len(local_axis_angle)
    if translation.shape != (frame_count, 3):
        raise HumanReconstructionError(
            "translation and local pose frame counts do not match"
        )
    expression_count = int(getattr(model, "num_expression_coeffs", 10))
    joint_chunks: list[np.ndarray] = []
    joint_names: list[str] | None = None
    rotation_joint_names: list[str] | None = None
    semantic_mapping: dict[str, int] | None = None

    with torch.inference_mode():
        for start in range(0, frame_count, chunk_size):
            stop = min(start + chunk_size, frame_count)
            batch = stop - start
            pose = local_axis_angle[start:stop]
            chunk_translation = translation[start:stop]
            # smplx 0.1.28 expands landmark buffers using the fixed model batch
            # size.  Pad only the final short chunk and trim its result.
            if batch < model_batch_size:
                padding = model_batch_size - batch
                pose = np.concatenate(
                    (pose, np.repeat(pose[-1:], padding, axis=0)), axis=0
                )
                chunk_translation = np.concatenate(
                    (
                        chunk_translation,
                        np.repeat(chunk_translation[-1:], padding, axis=0),
                    ),
                    axis=0,
                )
            model_batch = len(pose)

            def tensor(array: np.ndarray) -> Any:
                return torch.as_tensor(array, dtype=torch.float32)

            result = model(
                global_orient=tensor(pose[:, 0, :]),
                body_pose=tensor(
                    pose[:, 1:22, :].reshape(model_batch, BODY_POSE_SIZE)
                ),
                jaw_pose=tensor(pose[:, 22, :]),
                leye_pose=tensor(pose[:, 23, :]),
                reye_pose=tensor(pose[:, 24, :]),
                left_hand_pose=tensor(
                    pose[:, 25:40, :].reshape(model_batch, HAND_POSE_SIZE)
                ),
                right_hand_pose=tensor(
                    pose[:, 40:55, :].reshape(model_batch, HAND_POSE_SIZE)
                ),
                transl=tensor(chunk_translation),
                betas=tensor(
                    np.broadcast_to(betas, (model_batch, NUM_BETAS)).copy()
                ),
                expression=torch.zeros(
                    (model_batch, expression_count), dtype=torch.float32
                ),
                return_verts=False,
            )
            joints_tensor = getattr(result, "joints", None)
            if joints_tensor is None:
                raise HumanReconstructionError("SMPL-X did not return joint positions")
            joints = joints_tensor.detach().cpu().numpy()[:batch]
            if joints.ndim != 3 or joints.shape[0] != batch or joints.shape[2] != 3:
                raise HumanReconstructionError(
                    f"SMPL-X returned unexpected joint shape {joints.shape}"
                )
            if not np.isfinite(joints).all():
                raise HumanReconstructionError("SMPL-X returned NaN or infinite joints")
            if joint_names is None:
                joint_names, rotation_joint_names, semantic_mapping = (
                    _validated_upstream_joint_names(
                        joints.shape[1], KINEMATIC_JOINT_COUNT
                    )
                )
            elif joints.shape[1] != len(joint_names):
                raise HumanReconstructionError(
                    "SMPL-X joint count changed between reconstruction chunks"
                )
            joint_chunks.append(np.asarray(joints, dtype=np.float32))

    assert joint_names is not None
    assert rotation_joint_names is not None
    assert semantic_mapping is not None
    return (
        np.concatenate(joint_chunks, axis=0),
        joint_names,
        rotation_joint_names,
        semantic_mapping,
    )


def reconstruct_human_motion(
    input_path: str | os.PathLike[str],
    model_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    target_fps: float = 50,
    chunk_size: int = 128,
) -> dict[str, Any]:
    """Reconstruct and resample one stage-II motion with pinned SMPL-X.

    The returned dictionary is a small execution summary; the numeric result is
    written atomically to ``output_path``.  Contact labels are inferred at the
    source rate with verified upstream names and then resampled.  The serialized
    ``gate2_complete`` remains false because generation and independent Gate-2
    validation are deliberately separate operations.
    """

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer; got {chunk_size!r}")
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"target_fps must be positive; got {target_fps!r}")

    pipeline_root = Path(__file__).resolve().parents[1]
    vendor_root = pipeline_root / "vendor"
    output_file = _as_path(output_path)
    if output_file.suffix.lower() != ".npz":
        raise ValueError(f"output_path must end in .npz; got {output_file}")
    try:
        output_file.relative_to(pipeline_root)
    except ValueError as exc:
        raise ValueError(
            f"output_path must stay inside {pipeline_root}; got {output_file}"
        ) from exc

    preflight = check_human_prerequisites(
        pipeline_root=pipeline_root,
        model_root=model_root,
        vendor_root=vendor_root,
    )
    if not preflight["ok"]:
        details = "\n- ".join(preflight["errors"] + preflight["next_actions"])
        raise HumanPrerequisiteError(
            "SMPL-X reconstruction prerequisites are not satisfied:\n- " + details
        )

    source = load_stageii_numeric(input_path)
    input_file = Path(source["path"])

    source_fps = float(source["mocap_frame_rate"])
    target_times = _target_sample_times(
        int(source["frame_count"]), source_fps, float(target_fps)
    )
    local_source = np.asarray(source["full_pose"], dtype=np.float32).reshape(
        int(source["frame_count"]), KINEMATIC_JOINT_COUNT, 3
    )
    local_axis_angle = _resample_axis_angle(local_source, source_fps, target_times)

    # Import only after the exact-version preflight has passed and any local
    # vendor path has been placed first on sys.path.
    import smplx

    target_count = len(target_times)
    model_batch_size = min(
        chunk_size, max(int(source["frame_count"]), target_count)
    )
    model = smplx.create(
        model_path=str(_as_path(model_root)),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        num_betas=NUM_BETAS,
        flat_hand_mean=True,
        batch_size=model_batch_size,
    )
    model.eval()

    parents_tensor = getattr(model, "parents", None)
    if parents_tensor is None:
        raise HumanReconstructionError("SMPL-X model does not expose its parent table")
    if hasattr(parents_tensor, "detach"):
        parents = parents_tensor.detach().cpu().numpy().astype(np.int64)
    else:
        parents = np.asarray(parents_tensor, dtype=np.int64)
    parents = parents.reshape(-1)
    if len(parents) != KINEMATIC_JOINT_COUNT:
        raise HumanReconstructionError(
            f"expected 55 SMPL-X kinematic joints; model exposes {len(parents)}"
        )

    betas = np.asarray(source["betas"], dtype=np.float32)

    # The Gate-2 contract deliberately performs FK/contact at the original
    # 120 Hz before any resampling.  This preserves short stance transitions.
    (
        source_joint_positions,
        source_joint_names,
        source_rotation_joint_names,
        source_semantic_mapping,
    ) = _run_smplx_fk(
        model,
        local_source,
        np.asarray(source["trans"], dtype=np.float32),
        betas,
        chunk_size=chunk_size,
        model_batch_size=model_batch_size,
    )
    source_contacts = _infer_ground_contacts(
        source_joint_positions, source_joint_names, source_fps
    )

    # Verify the semantic mapping without assuming that the performer is
    # upright.  ACCAD includes lying, rolling, crawling, and acrobatic clips;
    # head-above-pelvis is therefore a motion-class property, not a coordinate
    # system test.  The source collection contract is Z-up and the identity
    # basis is recorded explicitly below.
    pelvis_source = source_joint_positions[
        :, source_semantic_mapping["pelvis"]
    ]
    head_source = source_joint_positions[:, source_semantic_mapping["head"]]
    head_offset = head_source - pelvis_source
    head_above_fraction = float(np.mean(head_offset[:, 2] > 0.0))
    median_head_offset = np.median(head_offset, axis=0)
    head_distance = np.linalg.norm(head_offset, axis=-1)
    median_head_distance = float(np.median(head_distance))
    if not np.isfinite(head_distance).all() or not 0.35 <= median_head_distance <= 1.0:
        raise HumanReconstructionError(
            "semantic pelvis/head distance is incompatible with the pinned SMPL-X model"
        )

    # Remove only the inferred common-mode floor drift from source translation.
    # During flight the rate-limited, confidence-aware baseline is bridged, so
    # the motion's actual vertical displacement is retained.
    source_ground_baseline = np.asarray(
        source_contacts["ground_baseline_height_w"], dtype=np.float32
    )
    normalized_source_translation = np.asarray(
        source["trans"], dtype=np.float32
    ).copy()
    normalized_source_translation[:, 2] -= source_ground_baseline
    translation = _resample_translation(
        normalized_source_translation, source_fps, target_times
    )
    original_translation_target = _resample_translation(
        source["trans"], source_fps, target_times
    )
    ground_correction_target = np.asarray(
        original_translation_target[:, 2] - translation[:, 2], dtype=np.float32
    )

    (
        joint_positions,
        joint_names,
        rotation_joint_names,
        semantic_mapping,
    ) = _run_smplx_fk(
        model,
        local_axis_angle,
        translation,
        betas,
        chunk_size=chunk_size,
        model_batch_size=model_batch_size,
    )
    if (
        joint_names != source_joint_names
        or rotation_joint_names != source_rotation_joint_names
        or semantic_mapping != source_semantic_mapping
    ):
        raise HumanReconstructionError(
            "SMPL-X semantic joint contract changed between 120 Hz and 50 Hz FK"
        )

    local_matrices = _rotation_matrices(local_axis_angle)
    global_matrices = _global_rotations(local_matrices, parents)
    local_quaternions = _continuous_quaternion_wxyz(local_matrices)
    global_quaternions = _continuous_quaternion_wxyz(global_matrices)

    kinematic_positions = np.asarray(
        joint_positions[:, :KINEMATIC_JOINT_COUNT], dtype=np.float32
    )
    joint_linear_velocity = _linear_velocity(kinematic_positions, target_fps)
    joint_angular_velocity = _angular_velocity_world(
        global_quaternions, target_fps
    )
    pelvis_index = semantic_mapping["pelvis"]
    root_position = np.asarray(joint_positions[:, pelvis_index], dtype=np.float32)
    root_quaternion = global_quaternions[:, 0]
    target_contact_geometry = _infer_ground_contacts(
        joint_positions, joint_names, float(target_fps)
    )
    contact_keys = (
        "left_heel_contact",
        "left_toe_contact",
        "right_heel_contact",
        "right_toe_contact",
    )
    minimum_target_contact_frames = max(3, int(round(0.06 * float(target_fps))))
    contacts = {
        key: _remove_short_true_runs(
            _resample_boolean_majority(
                source_contacts[key], source_fps, target_times, float(target_fps)
            ),
            minimum_target_contact_frames,
        )
        for key in contact_keys
    }
    target_ground_confidence = _resample_boolean_majority(
        source_contacts["ground_baseline_confidence"],
        source_fps,
        target_times,
        float(target_fps),
    )

    mapping_names = list(semantic_mapping)
    mapping_indices = [semantic_mapping[name] for name in mapping_names]
    source_checksum = _sha256(input_file)
    model_file = _as_path(model_root) / MODEL_SUBDIRECTORY / MODEL_FILENAME
    model_checksum = _sha256(model_file)
    try:
        source_file = input_file.relative_to(pipeline_root.parent).as_posix()
    except ValueError:
        source_file = str(input_file)
    payload = {
        "schema_version": np.asarray("accad_human_reconstruction_v1"),
        "processing_version": np.asarray("0.3.0"),
        "coordinate_system": np.asarray("right-handed_z-up"),
        "basis_transform_source_to_world": np.eye(3, dtype=np.float32),
        "ground_normal_w": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "quaternion_order": np.asarray("wxyz"),
        "position_unit": np.asarray("meter"),
        "angle_unit": np.asarray("radian"),
        "fps": np.asarray(float(target_fps), dtype=np.float64),
        "source_file": np.asarray(source_file),
        "source_path": np.asarray(str(input_file)),
        "source_checksum_sha256": np.asarray(source_checksum),
        # Backward-compatible alias for early local debugging artifacts.
        "source_sha256": np.asarray(source_checksum),
        "source_frame_count": np.asarray(source["frame_count"], dtype=np.int64),
        "source_fps": np.asarray(source_fps, dtype=np.float64),
        "source_mocap_time_length": np.asarray(
            source["mocap_time_length"], dtype=np.float64
        ),
        "target_frame_count": np.asarray(target_count, dtype=np.int64),
        "target_fps": np.asarray(float(target_fps), dtype=np.float64),
        "time": target_times,
        "source_last_frame_time": np.asarray(
            (int(source["frame_count"]) - 1) / source_fps, dtype=np.float64
        ),
        "target_last_frame_time": np.asarray(target_times[-1], dtype=np.float64),
        "unsampled_source_tail_seconds": np.asarray(
            (int(source["frame_count"]) - 1) / source_fps - target_times[-1],
            dtype=np.float64,
        ),
        "gender": np.asarray(source["gender"]),
        "gender_metadata": np.asarray(source["gender"]),
        "surface_model_type": np.asarray(source["surface_model_type"]),
        "model_gender": np.asarray("neutral"),
        "smplx_version": np.asarray(REQUIRED_SMPLX_VERSION),
        "smplx_model_path": np.asarray(str(model_file)),
        "smplx_model_checksum_sha256": np.asarray(model_checksum),
        "smplx_use_pca": np.asarray(False),
        "smplx_num_betas": np.asarray(NUM_BETAS, dtype=np.int64),
        "smplx_flat_hand_mean": np.asarray(True),
        "betas": betas,
        "root_translation": translation,
        "ground_correction_z_w": ground_correction_target,
        "root_position_w": root_position,
        "root_quaternion_wxyz": root_quaternion,
        "root_linear_velocity_w": _linear_velocity(root_position, target_fps),
        "root_angular_velocity_w": _angular_velocity_world(
            root_quaternion, target_fps
        ),
        "joint_names": np.asarray(rotation_joint_names),
        "all_joint_names": np.asarray(joint_names),
        "all_joint_position_w": joint_positions,
        # Backward-compatible debug alias for all SMPL-X output joints.
        "joint_positions_world": joint_positions,
        "rotation_joint_names": np.asarray(rotation_joint_names),
        "joint_position_w": kinematic_positions,
        "joint_quaternion_wxyz": global_quaternions,
        "joint_local_quaternion_wxyz": local_quaternions,
        "joint_linear_velocity_w": joint_linear_velocity,
        "joint_angular_velocity_w": joint_angular_velocity,
        "parents": parents,
        "local_rotation_quat_wxyz": local_quaternions,
        "global_rotation_quat_wxyz": global_quaternions,
        "validated_mapping_names": np.asarray(mapping_names),
        "validated_mapping_indices": np.asarray(mapping_indices, dtype=np.int64),
        "resampling_method": np.asarray(
            "root zero-phase low-pass+PCHIP; joint quaternion SLERP"
        ),
        "coordinate_status": np.asarray(
            "ACCAD stageii Z-up source contract; identity basis; pose-independent "
            "SMPL-X body-length audit; floor drift removed; no G1 alignment"
        ),
        "coordinate_evidence_method": np.asarray(
            "ACCAD_stageii_contract_plus_identity_basis_and_body_length"
        ),
        "semantic_head_above_pelvis_fraction": np.asarray(
            head_above_fraction, dtype=np.float32
        ),
        "semantic_median_head_minus_pelvis_w": np.asarray(
            median_head_offset, dtype=np.float32
        ),
        "semantic_median_head_pelvis_distance_m": np.asarray(
            median_head_distance, dtype=np.float32
        ),
        "ground_height_w": np.asarray(0.0, dtype=np.float32),
        "ground_baseline_height_w": np.zeros(target_count, dtype=np.float32),
        "ground_baseline_confidence": target_ground_confidence,
        "source_fk_performed": np.asarray(True),
        "source_fk_frame_count": np.asarray(
            int(source["frame_count"]), dtype=np.int64
        ),
        "source_ground_baseline_height_w": source_ground_baseline,
        "source_ground_baseline_confidence": source_contacts[
            "ground_baseline_confidence"
        ],
        "source_ground_baseline_drift_m": np.asarray(
            source_contacts["ground_baseline_drift_m"], dtype=np.float32
        ),
        "source_left_heel_contact": source_contacts["left_heel_contact"],
        "source_left_toe_contact": source_contacts["left_toe_contact"],
        "source_right_heel_contact": source_contacts["right_heel_contact"],
        "source_right_toe_contact": source_contacts["right_toe_contact"],
        "contact_source_fps": np.asarray(source_fps, dtype=np.float64),
        "contact_keypoint_names": np.asarray(
            target_contact_geometry["contact_keypoint_names"]
        ),
        "contact_keypoint_position_w": target_contact_geometry[
            "contact_keypoint_position_w"
        ],
        "left_heel_contact": contacts["left_heel_contact"],
        "left_toe_contact": contacts["left_toe_contact"],
        "right_heel_contact": contacts["right_heel_contact"],
        "right_toe_contact": contacts["right_toe_contact"],
        "contact_detection": np.asarray(
            "120Hz name-verified heel/toe; speed-gated adaptive lower envelope; "
            "2cm/s two-sided drift limit; height/speed hysteresis"
        ),
        "contact_resampling": np.asarray(
            "120Hz centered 50Hz-bin majority; nearest source sample on tie"
        ),
        "contact_enter_height_m": np.asarray(0.04, dtype=np.float32),
        "contact_exit_height_m": np.asarray(0.07, dtype=np.float32),
        "contact_enter_speed_mps": np.asarray(0.35, dtype=np.float32),
        "contact_exit_speed_mps": np.asarray(0.60, dtype=np.float32),
        "ground_baseline_max_drift_speed_mps": np.asarray(0.02, dtype=np.float32),
        "contact_labels_present": np.asarray(True),
        "gate2_complete": np.asarray(False),
        "gate2_status": np.asarray(
            "canonical basis generated; automated Gate-2 validation pending"
        ),
    }
    _save_npz_atomic(output_file, payload)

    return {
        "ok": True,
        "output_path": str(output_file),
        "source_frames": int(source["frame_count"]),
        "target_frames": target_count,
        "source_fps": source_fps,
        "target_fps": float(target_fps),
        "joint_count": len(joint_names),
        "rotation_joint_count": len(rotation_joint_names),
        "contact_labels_present": True,
        "source_fk_performed": True,
        "source_ground_baseline_drift_m": float(
            source_contacts["ground_baseline_drift_m"]
        ),
        "source_ground_confidence_fraction": float(
            np.mean(source_contacts["ground_baseline_confidence"])
        ),
        "gate2_complete": False,
        "gate2_status": (
            "canonical basis generated; automated Gate-2 validation pending"
        ),
    }
