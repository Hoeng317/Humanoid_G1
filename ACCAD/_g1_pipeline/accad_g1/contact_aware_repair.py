"""Contact-aware downstream repair for immutable ACCAD G1 v28 archives.

This stage changes only root translation and a named lower-body balance set.
Root orientation, all non-balance joints, frame count, time, contact labels, and
the v28 source-coordinate map remain immutable.  Candidate trajectories are
published only after the exact v29 centroidal-ZMP/capture audit, the v28
dynamic prefilter, and independent Gate-3 validation all pass.

The optimization is deliberately fail-closed.  Exact v29 signals generate a
frozen centroidal seed; differentiable whole-sequence contact IK realizes that
seed; then the complete nonlinear v29 audit is recomputed from serialized FK
state.  A failed cycle is evidence for quarantine, never a partially repaired
reference.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
import torch

from .contact_feasibility import (
    CONTACT_GATE_VERSION,
    DEFAULT_DYNAMIC_POLICY_PATH,
    DEFAULT_V27_MANIFESTS,
    DEFAULT_V28_MANIFESTS,
    FOOT_SUPPORT_NAMES,
    POLICY_ALGORITHM as V29_ALGORITHM,
    ClipSignals,
    ResolvedInertialModel,
    _contact_arrays,
    _convex_hull,
    _interpolate_scalar_same_phase,
    _safe_member,
    _safe_npz,
    _scalar,
    audit_contact_pair,
    compute_clip_signals,
    load_urdf_asset,
    resolve_inertial_model,
    signed_distance_to_support,
)
from .dynamic_retime import audit_dynamic_feasibility
from .g1_kinematics import (
    G1_CONTRACT_PATH,
    G1_URDF_PATH,
    TorchG1Kinematics,
    load_g1_contract,
    matrix_to_quaternion_wxyz,
    policy_to_sdk,
)
from .human import _angular_velocity_world, _linear_velocity, _save_npz_atomic
from .paths import PIPELINE_ROOT, PROJECT_ROOT, assert_output_is_local
from .retarget import FOOT_COLLISION_RADIUS_M


REPAIR_SCHEMA_VERSION = "accad_g1_contact_aware_repair_archive_v1"
POLICY_SCHEMA_VERSION = "accad_g1_contact_aware_repair_policy_v1"
PLAN_SCHEMA_VERSION = "accad_g1_contact_aware_repair_plan_v1"
REPORT_SCHEMA_VERSION = "accad_g1_contact_aware_repair_report_v1"
MANIFEST_SCHEMA_VERSION = "accad_g1_contact_aware_repair_manifest_v1"
REPAIR_VERSION = "g1-sequence-retarget-v30-contact-aware-sequence-ik-v1"
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "work" / "contact_v30"
IMPLEMENTATION_PATH = Path(__file__).resolve()
_TEST_AUTHORITY = object()

# Public, precommitted input metadata.  Validation and test are deliberately
# not allowed to select manifests at runtime: doing so would let a caller
# cherry-pick an easier holdout after observing train results.  The v28 test
# entry remains absent until that upstream artifact is published and its
# public metadata is deliberately committed here; no test NPZ is needed to
# establish or verify this registry.
CANONICAL_MANIFEST_REGISTRY: dict[str, dict[str, dict[str, Any] | None]] = {
    "v27": {
        "train": {
            "path": "work/manifests/g1_train.json",
            "checksum_sha256": "a78c26f1d82f1cfe96b38682bd40f217749bdf4cb6fd0c2c34ba3a18ae4255ea",
            "motion_set_checksum_sha256": "a153fa565e2e1641420b6258d9af70abaf8f9018ac33947590b4f8eacfa29270",
            "num_motions": 215,
            "num_frames": 66076,
            "duration_sec": 1317.2199999999993,
        },
        "validation": {
            "path": "work/manifests/g1_validation.json",
            "checksum_sha256": "c4d43072b3a4c82f76b17a627e8a156365bebcf446d368ef6b50b63362a6f8b2",
            "motion_set_checksum_sha256": "a153fa565e2e1641420b6258d9af70abaf8f9018ac33947590b4f8eacfa29270",
            "num_motions": 5,
            "num_frames": 1952,
            "duration_sec": 38.94,
        },
        "test": {
            "path": "work/manifests/g1_test.json",
            "checksum_sha256": "786daea165d826c36fbdad0c8132d0d04bfd92fdcedfd89124110f769aaa7627",
            "motion_set_checksum_sha256": "a153fa565e2e1641420b6258d9af70abaf8f9018ac33947590b4f8eacfa29270",
            "num_motions": 29,
            "num_frames": 11490,
            "duration_sec": 229.22,
        },
    },
    "v28": {
        "train": {
            "path": "work/dynamic/g1_train.json",
            "checksum_sha256": "ba8c59b6edb49c82dfc39c7242857b24fe6260b2c171582ff6ae706f95d3a3de",
            "motion_set_checksum_sha256": "5f929faae8f87373cd17afb32a40c87d54dce2efbadce8f191a1890d46452c36",
            "num_motions": 103,
            "num_source_motions": 215,
            "num_frames": 66258,
            "duration_sec": 1323.1000000000004,
        },
        "validation": {
            "path": "work/dynamic/g1_validation.json",
            "checksum_sha256": "0ecc4ae4d456613c4b6355c36d22285272ee5d98d6b75791eb8f1bbaf04137a2",
            "motion_set_checksum_sha256": "1ecba1501045209f540c2ddf14c762ec96fb94cec8fa617282c3fe08f9f9c875",
            "num_motions": 2,
            "num_source_motions": 5,
            "num_frames": 2136,
            "duration_sec": 42.68,
        },
        "test": None,
    },
}
DEPENDENCY_PATHS = tuple(
    PIPELINE_ROOT / "accad_g1" / name
    for name in (
        "contact_feasibility.py",
        "dynamic_retime.py",
        "g1_kinematics.py",
        "g1_validation.py",
        "human.py",
        "retarget.py",
    )
)

LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)
# Waist roll/pitch are balance coordinates, not a relaxation of the 12-leg
# contract.  They are named explicitly and yaw remains frozen.
WAIST_BALANCE_JOINT_NAMES = ("waist_roll_joint", "waist_pitch_joint")

REPAIR_ALGORITHM: dict[str, Any] = {
    "schema_version": "accad_g1_contact_aware_repair_formula_v1",
    "fps": 50.0,
    "dtype": "float64",
    "device": "cpu",
    "deterministic": True,
    "variables": {
        "root_translation": ["x", "y", "z"],
        "leg_joint_names": list(LEG_JOINT_NAMES),
        "waist_balance_joint_names": list(WAIST_BALANCE_JOINT_NAMES),
        "root_quaternion": "immutable_v28",
        "all_other_joints": "immutable_v28",
    },
    "seed": {
        "source": "exact_v29_centroidal_zmp_and_preview_capture",
        "target_interior_margin_m": 0.010,
        "comparison_buffer_m": 0.005,
        # The validation Qk clip requires a >0.17 m seed and some terminal
        # corridor points exceed 0.5 m.  This bound is therefore an explicit
        # escalation domain, not a hidden clamp to a walking-only heuristic.
        "maximum_trust_region_m": 0.65,
        "smoothing_first_difference": 8.0,
        "smoothing_second_difference": 24.0,
        "out_of_domain_action": "quarantine_without_clipping",
    },
    "ik": {
        "outer_cycles": 4,
        "iterations_per_cycle": 240,
        "learning_rate": 0.012,
        "soft_joint_limit_factor": 0.9,
        "maximum_joint_delta_rad": 0.70,
        "maximum_root_xy_delta_m": 0.35,
        "maximum_root_z_delta_m": 0.12,
        "weights": {
            "centroidal_seed": 1800.0,
            "zmp_seed": 700.0,
            "capture_seed": 700.0,
            "contact_lock": 5000.0,
            "ground": 5000.0,
            "joint_style": 1.0,
            "root_style_xy": 2.0,
            "root_style_z": 20.0,
            "correction_velocity": 12.0,
            "correction_acceleration": 3.0,
            "joint_velocity_headroom": 10.0,
            "positive_com_height": 5000.0,
            "positive_normal_force": 5000.0,
        },
        "joint_velocity_limit_fraction": 0.8,
        "minimum_com_height_m": 0.10,
        "minimum_normal_force_weight_fraction": 0.10,
        "gradient_clip_norm": 1000.0,
        "convergence_delta": 1.0e-10,
        "convergence_patience": 30,
    },
    "acceptance": {
        "exact_contact_gate": CONTACT_GATE_VERSION,
        "dynamic_prefilter_scale_pre": 1.0,
        "gate3_kinematic_complete": True,
        "terminal_frames_may_be_cropped": False,
        "active_contact_surface_height_maximum_m": 0.060,
        "active_contact_support_displacement_maximum_m": 0.012,
        "active_stance_anchor_displacement_maximum_m": 0.012,
        "support_surface_penetration_minimum_m": -0.002,
        "contact_anchor_threshold_basis": {
            "surface_height": (
                "0.060 m bounds the frozen-train v28 maximum 0.055175 m with "
                "headroom while rejecting uniform airborne contact labels"
            ),
            "anchor_displacement": (
                "one 0.60 m/s semantic-contact step at the 50 Hz control rate"
            ),
            "penetration": "existing independent Gate-3 0.002 m tolerance",
        },
    },
}


class ContactAwareRepairError(RuntimeError):
    """A contract or provenance error that prevents deterministic repair."""


class ContactAwareRepairQuarantine(ContactAwareRepairError):
    """A valid input whose bounded repair failed the frozen acceptance path."""

    def __init__(self, reason: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.evidence = dict(evidence)


@dataclass(frozen=True)
class RepairTargets:
    com_position: np.ndarray
    zmp_position: np.ndarray
    capture_position: np.ndarray
    zmp_mask: np.ndarray
    capture_mask: np.ndarray
    support_plane_z: np.ndarray
    active_support_mask: np.ndarray
    maximum_raw_correction_m: float
    maximum_smoothed_com_seed_correction_m: float
    capture_preview_missing_frames: tuple[int, ...]
    capture_preview_missing_terminal_frames: tuple[int, ...]


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


def _exclusive_json(path: Path, document: Mapping[str, Any]) -> None:
    """Create a durable JSON receipt exactly once using filesystem O_EXCL."""

    output = assert_output_is_local(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ContactAwareRepairError(
            "sealed test split has already been consumed once"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(
                document,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            output.unlink()
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContactAwareRepairError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ContactAwareRepairError(f"{description} must be a JSON object")
    return document


def _pipeline_path(path: str | Path, description: str) -> Path:
    """Resolve a path while rejecting escapes from this self-contained pipeline."""

    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(PIPELINE_ROOT)
    except ValueError as exc:
        raise ContactAwareRepairError(
            f"{description} escapes the ACCAD G1 pipeline: {resolved}"
        ) from exc
    return resolved


def _canonical_commitment(stage: str, split: str) -> dict[str, Any]:
    try:
        commitment = CANONICAL_MANIFEST_REGISTRY[stage][split]
    except KeyError as exc:
        raise ContactAwareRepairError(
            f"no canonical {stage} {split} manifest registry entry exists"
        ) from exc
    if commitment is None:
        raise ContactAwareRepairError(
            f"{split} remains sealed: canonical {stage} public metadata commitment "
            "has not been published"
        )
    return commitment


def _verify_canonical_manifest_commitment(
    *,
    stage: str,
    split: str,
    path: Path,
    document: Mapping[str, Any],
) -> None:
    """Bind a holdout manifest to its source-code-committed public metadata."""

    commitment = _canonical_commitment(stage, split)
    default_paths = DEFAULT_V27_MANIFESTS if stage == "v27" else DEFAULT_V28_MANIFESTS
    expected_default = Path(default_paths[split]).resolve()
    expected_registry_path = _pipeline_path(
        PIPELINE_ROOT / str(commitment["path"]),
        f"canonical {stage} {split} manifest",
    )
    if path != expected_default or expected_registry_path != expected_default:
        raise ContactAwareRepairError(
            f"{stage} {split} must use the precommitted canonical default manifest"
        )
    if not path.is_file() or _sha256(path) != commitment["checksum_sha256"]:
        raise ContactAwareRepairError(
            f"canonical {stage} {split} manifest is missing or its checksum changed"
        )
    exact_fields = (
        "motion_set_checksum_sha256",
        "num_motions",
        "num_frames",
    )
    for name in exact_fields:
        if document.get(name) != commitment[name]:
            raise ContactAwareRepairError(
                f"canonical {stage} {split} manifest {name} commitment mismatch"
            )
    if "num_source_motions" in commitment and document.get(
        "num_source_motions"
    ) != commitment["num_source_motions"]:
        raise ContactAwareRepairError(
            f"canonical {stage} {split} manifest num_source_motions commitment mismatch"
        )
    try:
        duration = float(document.get("duration_sec"))
    except (TypeError, ValueError) as exc:
        raise ContactAwareRepairError(
            f"canonical {stage} {split} manifest duration_sec is invalid"
        ) from exc
    if not math.isclose(
        duration, float(commitment["duration_sec"]), rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ContactAwareRepairError(
            f"canonical {stage} {split} manifest duration_sec commitment mismatch"
        )


def _archive_scalar(arrays: Mapping[str, np.ndarray], name: str) -> Any:
    try:
        value = np.asarray(arrays[name])
    except KeyError as exc:
        raise ContactAwareRepairError(f"archive is missing scalar {name!r}") from exc
    if value.shape != () or value.dtype.hasobject:
        raise ContactAwareRepairError(
            f"archive scalar {name!r} is unsafe or non-scalar: "
            f"shape={value.shape}, dtype={value.dtype}"
        )
    return value.item()


def _serialized_archive_metadata(
    arrays: Mapping[str, np.ndarray], *, split_field: str
) -> dict[str, Any]:
    """Recompute record metadata solely from safe serialized archive fields."""

    raw_frame_count = _archive_scalar(arrays, "frame_count")
    raw_fps = _archive_scalar(arrays, "fps")
    if isinstance(raw_frame_count, (bool, np.bool_)):
        raise ContactAwareRepairError("archive frame_count must be an integer")
    try:
        frame_count = int(raw_frame_count)
        fps = float(raw_fps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContactAwareRepairError("archive frame_count/fps scalar is invalid") from exc
    if frame_count < 1 or frame_count != raw_frame_count:
        raise ContactAwareRepairError("archive frame_count must be a positive integer")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ContactAwareRepairError("archive fps must be finite and positive")
    split = str(_archive_scalar(arrays, split_field))
    try:
        time = np.asarray(arrays["time"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContactAwareRepairError("archive time field is missing or invalid") from exc
    expected_time = np.arange(frame_count, dtype=np.float64) / fps
    if time.shape != (frame_count,) or not np.array_equal(time, expected_time):
        raise ContactAwareRepairError(
            "archive time is not the exact frame_count/fps sampling grid"
        )
    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration_sec": float((frame_count - 1) / fps),
        "split": split,
    }


def _validate_v28_record_metadata(
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    split: str,
) -> dict[str, Any]:
    """Reject a v28 manifest record whose scalars disagree with its archive."""

    actual = _serialized_archive_metadata(arrays, split_field="dynamic_split")
    try:
        recorded_frame_count = int(record["frame_count"])
        recorded_fps = float(record["fps"])
        recorded_duration = float(record["duration_sec"])
        recorded_split = str(record["split"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ContactAwareRepairError("v28 record metadata is missing or invalid") from exc
    mismatches: list[str] = []
    if recorded_frame_count != actual["frame_count"]:
        mismatches.append("frame_count")
    if not math.isclose(recorded_fps, actual["fps"], rel_tol=0.0, abs_tol=1.0e-12):
        mismatches.append("fps")
    if not math.isclose(
        recorded_duration, actual["duration_sec"], rel_tol=0.0, abs_tol=1.0e-12
    ):
        mismatches.append("duration_sec")
    if recorded_split != actual["split"] or actual["split"] != split:
        mismatches.append("split")
    if mismatches:
        raise ContactAwareRepairError(
            "v28 manifest record disagrees with serialized archive metadata: "
            + ", ".join(mismatches)
        )
    return actual


def _report_status(split: str, counts: Mapping[str, Any]) -> str:
    """Separate successful execution from strict holdout acceptance."""

    published = int(counts.get("published", 0))
    quarantined = int(counts.get("quarantined", 0))
    if split in {"validation", "test"} and (published <= 0 or quarantined != 0):
        return "failed_acceptance"
    return "completed"


def _balance_indices() -> tuple[np.ndarray, np.ndarray]:
    contract = load_g1_contract()
    name_to_index = {name: index for index, name in enumerate(contract.policy_names)}
    try:
        leg = np.asarray([name_to_index[name] for name in LEG_JOINT_NAMES], dtype=np.int64)
        waist = np.asarray(
            [name_to_index[name] for name in WAIST_BALANCE_JOINT_NAMES], dtype=np.int64
        )
    except KeyError as exc:
        raise ContactAwareRepairError(f"G1 joint contract is missing {exc.args[0]}") from exc
    if len(set((*leg.tolist(), *waist.tolist()))) != 14:
        raise ContactAwareRepairError("balance joint names do not resolve uniquely")
    return leg, waist


def inertial_com_torch(
    transforms: Mapping[str, torch.Tensor], model: ResolvedInertialModel
) -> torch.Tensor:
    """Return exact URDF mass-weighted COM positions from differentiable FK."""

    missing = [name for name in model.body_names if name not in transforms]
    if missing:
        raise KeyError(f"FK transforms are missing inertial anchors: {missing}")
    body_position = torch.stack(
        [transforms[name][..., :3, 3] for name in model.body_names], dim=-2
    )
    body_rotation = torch.stack(
        [transforms[name][..., :3, :3] for name in model.body_names], dim=-3
    )
    device = body_position.device
    dtype = body_position.dtype
    anchor_index = torch.as_tensor(model.anchor_indices, dtype=torch.long, device=device)
    anchor_position = body_position.index_select(-2, anchor_index)
    anchor_rotation = body_rotation.index_select(-3, anchor_index)
    offsets = torch.as_tensor(
        model.com_offsets_in_anchor, dtype=dtype, device=device
    )
    inertial_position = anchor_position + torch.einsum(
        "...eij,ej->...ei", anchor_rotation, offsets
    )
    masses = torch.as_tensor(model.masses, dtype=dtype, device=device)
    return torch.sum(inertial_position * masses[..., None], dim=-2) / float(
        model.total_mass_kg
    )


def _gradient_torch(values: torch.Tensor, fps: float) -> torch.Tensor:
    """Exact torch equivalent of ``np.gradient(..., edge_order=2)`` on axis 0."""

    if values.ndim < 2 or len(values) < 3:
        raise ValueError("second-order frame gradient requires at least three frames")
    scale = float(fps) * 0.5
    return torch.cat(
        (
            ((-3.0 * values[0] + 4.0 * values[1] - values[2]) * scale).unsqueeze(0),
            (values[2:] - values[:-2]) * scale,
            ((3.0 * values[-1] - 4.0 * values[-2] + values[-3]) * scale).unsqueeze(0),
        ),
        dim=0,
    )


def _angular_velocity_from_rotations_torch(
    rotations: torch.Tensor, fps: float
) -> torch.Tensor:
    """Match the serialized shortest-path world angular-velocity convention."""

    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3) or len(rotations) < 2:
        raise ValueError("body rotations must have shape [T,B,3,3] with T>=2")
    relative = rotations[1:] @ rotations[:-1].transpose(-1, -2)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1.0e-9, 1.0 - 1.0e-9)
    theta = torch.acos(cosine)
    vee = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.sin(theta)
    factor = torch.where(
        theta.abs() > 1.0e-6,
        theta / (2.0 * sine).clamp_min(1.0e-12),
        torch.full_like(theta, 0.5),
    )
    interval = vee * factor[..., None] * float(fps)
    if len(rotations) == 2:
        return torch.stack((interval[0], interval[0]), dim=0)
    return torch.cat(
        (
            interval[:1],
            0.5 * (interval[:-1] + interval[1:]),
            interval[-1:],
        ),
        dim=0,
    )


def differentiable_centroidal_signals(
    transforms: Mapping[str, torch.Tensor],
    model: ResolvedInertialModel,
    *,
    fps: float,
) -> dict[str, torch.Tensor]:
    """Differentiable counterpart of v29's full URDF centroidal state path.

    It includes fixed-child inertials, link rotational inertia, orbital angular
    momentum, serialized body-rate conventions, and the same five-frame rate
    operator used by :func:`compute_clip_signals`.
    """

    body_position = torch.stack(
        [transforms[name][..., :3, 3] for name in model.body_names], dim=-2
    )
    body_rotation = torch.stack(
        [transforms[name][..., :3, :3] for name in model.body_names], dim=-3
    )
    if body_position.ndim != 3:
        raise ValueError("centroidal sequence requires transforms with leading time axis")
    body_linear_velocity = _gradient_torch(body_position, fps)
    body_angular_velocity = _angular_velocity_from_rotations_torch(body_rotation, fps)
    device, dtype = body_position.device, body_position.dtype
    anchor_index = torch.as_tensor(model.anchor_indices, dtype=torch.long, device=device)
    anchor_position = body_position.index_select(1, anchor_index)
    anchor_rotation = body_rotation.index_select(1, anchor_index)
    anchor_linear_velocity = body_linear_velocity.index_select(1, anchor_index)
    angular_velocity = body_angular_velocity.index_select(1, anchor_index)
    offsets = torch.as_tensor(model.com_offsets_in_anchor, dtype=dtype, device=device)
    offset_world = torch.einsum("teij,ej->tei", anchor_rotation, offsets)
    inertial_position = anchor_position + offset_world
    inertial_velocity = anchor_linear_velocity + torch.linalg.cross(
        angular_velocity, offset_world, dim=-1
    )
    masses = torch.as_tensor(model.masses, dtype=dtype, device=device)
    mass = masses[None, :, None]
    total_mass = float(model.total_mass_kg)
    com_position = torch.sum(mass * inertial_position, dim=1) / total_mass
    com_velocity = torch.sum(mass * inertial_velocity, dim=1) / total_mass
    com_acceleration = _five_frame_torch(com_velocity, fps)[0]

    rotation_anchor_inertial = torch.as_tensor(
        model.rotations_anchor_inertial, dtype=dtype, device=device
    )
    rotation_world_inertial = torch.einsum(
        "teij,ejk->teik", anchor_rotation, rotation_anchor_inertial
    )
    inertia_in_inertial = torch.as_tensor(
        model.inertias_in_inertial, dtype=dtype, device=device
    )
    inertia_world = torch.einsum(
        "teik,ekl,tejl->teij",
        rotation_world_inertial,
        inertia_in_inertial,
        rotation_world_inertial,
    )
    spin = torch.einsum("teij,tej->tei", inertia_world, angular_velocity)
    relative_position = inertial_position - com_position[:, None, :]
    relative_velocity = inertial_velocity - com_velocity[:, None, :]
    orbital = torch.linalg.cross(
        relative_position, mass * relative_velocity, dim=-1
    )
    momentum = torch.sum(spin + orbital, dim=1)
    momentum_rate = _five_frame_torch(momentum, fps)[0]
    return {
        "body_position": body_position,
        "body_rotation": body_rotation,
        "body_linear_velocity": body_linear_velocity,
        "body_angular_velocity": body_angular_velocity,
        "com_position": com_position,
        "com_velocity": com_velocity,
        "com_acceleration": com_acceleration,
        "centroidal_momentum": momentum,
        "centroidal_momentum_rate": momentum_rate,
    }


def _five_frame_torch(values: torch.Tensor, fps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or len(values) < 5:
        raise ValueError("five-frame torch derivatives require [T,D] with T>=5")
    dt = 1.0 / float(fps)
    velocity_interior = (
        -2.0 * values[:-4] - values[1:-3] + values[3:-1] + 2.0 * values[4:]
    ) / (10.0 * dt)
    acceleration_interior = (
        2.0 * values[:-4]
        - values[1:-3]
        - 2.0 * values[2:-2]
        - values[3:-1]
        + 2.0 * values[4:]
    ) / (7.0 * dt * dt)
    pad = torch.zeros((2, values.shape[1]), dtype=values.dtype, device=values.device)
    return (
        torch.cat((pad, velocity_interior, pad), dim=0),
        torch.cat((pad, acceleration_interior, pad), dim=0),
    )


def _closest_interior_point(
    point: np.ndarray, support_points: np.ndarray, required_margin: float
) -> np.ndarray:
    """Move a finite 2-D point minimally toward the hull centroid.

    Bisection is deterministic and never claims an unattainable margin: when
    the requested inset exceeds the polygon inradius, the hull centroid is the
    explicit best bounded seed and the exact downstream gate remains decisive.
    """

    query = np.asarray(point, dtype=np.float64)
    hull = _convex_hull(np.asarray(support_points, dtype=np.float64))
    if len(hull) < 3:
        return np.mean(hull, axis=0)
    if signed_distance_to_support(query, hull) >= required_margin:
        return query.copy()
    center = np.mean(hull, axis=0)
    if signed_distance_to_support(center, hull) < required_margin:
        return center
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        candidate = (1.0 - middle) * query + middle * center
        if signed_distance_to_support(candidate, hull) >= required_margin:
            high = middle
        else:
            low = middle
    return (1.0 - high) * query + high * center


def _smooth_seed(raw: np.ndarray, active: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(active, dtype=bool)
    frame_count = len(values)
    if values.shape != (frame_count, 2) or mask.shape != (frame_count,):
        raise ValueError("seed smoothing expects [T,2] values and [T] mask")
    first_weight = float(REPAIR_ALGORITHM["seed"]["smoothing_first_difference"])
    second_weight = float(REPAIR_ALGORITHM["seed"]["smoothing_second_difference"])
    data_weight = np.where(mask, 1.0, 50.0)
    d1 = diags((-np.ones(frame_count - 1), np.ones(frame_count - 1)), (0, 1), shape=(frame_count - 1, frame_count))
    d2 = diags(
        (np.ones(frame_count - 2), -2.0 * np.ones(frame_count - 2), np.ones(frame_count - 2)),
        (0, 1, 2),
        shape=(frame_count - 2, frame_count),
    )
    system = diags(data_weight) + first_weight * (d1.T @ d1) + second_weight * (d2.T @ d2) + 1.0e-9 * eye(frame_count)
    result = np.empty_like(values)
    for axis in range(2):
        result[:, axis] = spsolve(system.tocsc(), data_weight * values[:, axis])
    return result


def _support_plane_and_corridor(
    arrays: Mapping[str, np.ndarray], signals: ClipSignals
) -> tuple[np.ndarray, list[np.ndarray | None], np.ndarray]:
    frame_count = signals.frame_count
    channels, _ = _contact_arrays(arrays, frame_count)
    support = np.asarray(arrays["foot_support_position_w"], dtype=np.float64)
    radii = load_urdf_asset().support_radii.reshape(8)
    plane = np.full(frame_count, np.nan, dtype=np.float64)
    corridor: list[np.ndarray | None] = [None] * frame_count
    active_mask = np.zeros((frame_count, 8), dtype=bool)
    for frame in np.flatnonzero(signals.eligible_support):
        left = bool(np.any(channels[frame, :2]))
        right = bool(np.any(channels[frame, 2:]))
        optimistic: list[int] = []
        strict: list[int] = []
        if left:
            optimistic.extend(range(4))
        if right:
            optimistic.extend(range(4, 8))
        for channel in np.flatnonzero(channels[frame]):
            foot = 0 if channel < 2 else 1
            pair = (0, 1) if channel % 2 == 0 else (2, 3)
            strict.extend(foot * 4 + offset for offset in pair)
        strict_index = np.asarray(sorted(set(strict)), dtype=np.int64)
        optimistic_index = np.asarray(optimistic, dtype=np.int64)
        active_mask[frame, optimistic_index] = True
        plane[frame] = float(
            np.median(support[frame, strict_index, 2] - radii[strict_index])
        )
        if not signals.eligible_single_support[frame]:
            continue
        current_foot = 0 if left else 1
        opposite_foot = 1 - current_foot
        points = support[frame, np.arange(current_foot * 4, current_foot * 4 + 4), :2]
        height = signals.com_height[frame]
        if not math.isfinite(float(height)) or height <= 0.0:
            corridor[frame] = points
            continue
        omega = math.sqrt(float(V29_ALGORITHM["gravity_mps2"]) / float(height))
        preview_frames = int(
            math.ceil(
                float(V29_ALGORITHM["capture_preview"]["horizon_natural_time_constants"])
                / (omega / signals.fps)
            )
        )
        opposite_slice = slice(opposite_foot * 2, opposite_foot * 2 + 2)
        for future in range(frame + 1, min(frame_count, frame + preview_frames + 1)):
            if np.any(channels[future, opposite_slice]):
                opposite = np.arange(opposite_foot * 4, opposite_foot * 4 + 4)
                points = np.concatenate((points, support[future, opposite, :2]), axis=0)
                break
        corridor[frame] = points
    return plane, corridor, active_mask


def plan_repair_targets(
    source_arrays: Mapping[str, np.ndarray],
    candidate_arrays: Mapping[str, np.ndarray],
) -> RepairTargets:
    """Build a deterministic seed from exact v29 signals without optimizing."""

    source = compute_clip_signals(source_arrays)
    candidate = compute_clip_signals(candidate_arrays)
    realized_scale = float(_scalar(candidate_arrays, "dynamic_realized_scale"))
    source_coordinate = np.arange(candidate.frame_count, dtype=np.float64) / realized_scale
    source_coordinate[-1] = source.frame_count - 1
    source_zmp_margin, zmp_compare = _interpolate_scalar_same_phase(
        source.zmp_margin_optimistic,
        source.contact_state,
        candidate.contact_state,
        source_coordinate,
    )
    source_capture_margin, capture_compare = _interpolate_scalar_same_phase(
        source.capture_margin_optimistic,
        source.contact_state,
        candidate.contact_state,
        source_coordinate,
    )
    plane, corridors, active_support = _support_plane_and_corridor(
        candidate_arrays, candidate
    )
    support = np.asarray(candidate_arrays["foot_support_position_w"], dtype=np.float64)
    absolute_target = float(REPAIR_ALGORITHM["seed"]["target_interior_margin_m"])
    comparison_buffer = float(REPAIR_ALGORITHM["seed"]["comparison_buffer_m"])
    degradation_tolerance = float(
        V29_ALGORITHM["thresholds"]["semantic_contact_speed_budget_mps"]
    ) / candidate.fps
    zmp_target = candidate.centroidal_zmp.copy()
    capture_target = candidate.capture_point.copy()
    zmp_mask = np.zeros(candidate.frame_count, dtype=bool)
    capture_mask = np.zeros(candidate.frame_count, dtype=bool)
    corrections = np.zeros((candidate.frame_count, 2), dtype=np.float64)
    correction_counts = np.zeros(candidate.frame_count, dtype=np.float64)
    maximum_individual_correction = 0.0

    for frame in np.flatnonzero(candidate.eligible_support):
        point = candidate.centroidal_zmp[frame]
        if not np.isfinite(point).all():
            continue
        required = absolute_target
        if zmp_compare[frame] and math.isfinite(float(source_zmp_margin[frame])):
            required = max(
                required,
                float(source_zmp_margin[frame]) - degradation_tolerance + comparison_buffer,
            )
        support_points = support[frame, active_support[frame], :2]
        target = _closest_interior_point(point, support_points, required)
        delta = target - point
        maximum_individual_correction = max(
            maximum_individual_correction, float(np.linalg.norm(delta))
        )
        if float(np.linalg.norm(delta)) > 1.0e-10:
            zmp_target[frame] = target
            zmp_mask[frame] = True
            corrections[frame] += delta
            correction_counts[frame] += 1.0

    for frame in np.flatnonzero(candidate.eligible_single_support):
        point = candidate.capture_point[frame]
        points = corridors[frame]
        if points is None or not np.isfinite(point).all():
            continue
        required = absolute_target
        if capture_compare[frame] and math.isfinite(float(source_capture_margin[frame])):
            required = max(
                required,
                float(source_capture_margin[frame])
                - degradation_tolerance
                + comparison_buffer,
            )
        target = _closest_interior_point(point, points, required)
        delta = target - point
        maximum_individual_correction = max(
            maximum_individual_correction, float(np.linalg.norm(delta))
        )
        if float(np.linalg.norm(delta)) > 1.0e-10:
            capture_target[frame] = target
            capture_mask[frame] = True
            corrections[frame] += delta
            correction_counts[frame] += 1.0

    nonzero = correction_counts > 0.0
    corrections[nonzero] /= correction_counts[nonzero, None]
    maximum_combined = float(np.max(np.linalg.norm(corrections, axis=1)))
    maximum_raw = max(maximum_individual_correction, maximum_combined)
    trust = float(REPAIR_ALGORITHM["seed"]["maximum_trust_region_m"])
    if maximum_raw > trust + 1.0e-12:
        raise ContactAwareRepairQuarantine(
            "exact centroidal seed exceeds the frozen repair trust region",
            {
                "maximum_raw_correction_m": maximum_raw,
                "maximum_individual_target_correction_m": maximum_individual_correction,
                "maximum_combined_target_correction_m": maximum_combined,
                "maximum_trust_region_m": trust,
                "action": "quarantine_without_clipping",
            },
        )
    smoothed = _smooth_seed(corrections, candidate.eligible_support)
    maximum_smoothed = float(np.max(np.linalg.norm(smoothed, axis=1)))
    if maximum_smoothed > trust + 1.0e-12:
        raise ContactAwareRepairQuarantine(
            "smoothed COM seed exceeds the frozen repair trust region",
            {
                "maximum_smoothed_com_seed_correction_m": maximum_smoothed,
                "maximum_trust_region_m": trust,
                "action": "quarantine_without_clipping",
            },
        )
    com_target = candidate.com_position.copy()
    com_target[:, :2] += smoothed
    preview_missing = tuple(
        int(frame)
        for frame in np.flatnonzero(
            candidate.eligible_single_support & ~candidate.capture_preview_available
        )
    )
    terminal_missing = tuple(
        int(frame)
        for frame in preview_missing
        if frame >= candidate.frame_count - int(math.ceil(candidate.fps))
    )
    return RepairTargets(
        com_position=com_target,
        zmp_position=zmp_target,
        capture_position=capture_target,
        zmp_mask=zmp_mask,
        capture_mask=capture_mask,
        support_plane_z=plane,
        active_support_mask=active_support,
        maximum_raw_correction_m=maximum_raw,
        maximum_smoothed_com_seed_correction_m=maximum_smoothed,
        capture_preview_missing_frames=preview_missing,
        capture_preview_missing_terminal_frames=terminal_missing,
    )


def _optimize_contact_ik(
    current_arrays: Mapping[str, np.ndarray],
    immutable_v28: Mapping[str, np.ndarray],
    targets: RepairTargets,
    *,
    iterations: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Realize one frozen centroidal seed with contact-locked sequence IK."""

    frame_count = int(_scalar(immutable_v28, "frame_count"))
    fps = float(_scalar(immutable_v28, "fps"))
    if frame_count < 7 or fps != 50.0:
        raise ContactAwareRepairError("contact IK requires a canonical >=7-frame 50 Hz clip")
    q_original_np = np.asarray(immutable_v28["joint_pos"], dtype=np.float64)
    root_original_np = np.asarray(immutable_v28["root_pos_w"], dtype=np.float64)
    q_current_np = np.asarray(current_arrays["joint_pos"], dtype=np.float64)
    root_current_np = np.asarray(current_arrays["root_pos_w"], dtype=np.float64)
    root_quaternion_np = np.asarray(
        immutable_v28["root_quat_wxyz"], dtype=np.float64
    )
    if (
        q_original_np.shape != (frame_count, 29)
        or root_original_np.shape != (frame_count, 3)
        or q_current_np.shape != (frame_count, 29)
        or root_current_np.shape != (frame_count, 3)
        or root_quaternion_np.shape != (frame_count, 4)
    ):
        raise ContactAwareRepairError("repair input tensors violate the G1 frame contract")

    algorithm = REPAIR_ALGORITHM["ik"]
    maximum_iterations = int(
        algorithm["iterations_per_cycle"] if iterations is None else iterations
    )
    if maximum_iterations < 1:
        raise ValueError("contact IK iterations must be positive")
    device = torch.device(str(REPAIR_ALGORITHM["device"]))
    dtype = torch.float64
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    kinematics = TorchG1Kinematics(device=device, dtype=dtype)
    body_names = tuple(str(name) for name in immutable_v28["body_names"].tolist())
    inertial_model = resolve_inertial_model(body_names)
    leg_indices, waist_indices = _balance_indices()
    variable_indices_np = np.concatenate((leg_indices, waist_indices))
    variable_indices = torch.as_tensor(variable_indices_np, dtype=torch.long, device=device)

    q_original = torch.as_tensor(q_original_np, dtype=dtype, device=device)
    root_original = torch.as_tensor(root_original_np, dtype=dtype, device=device)
    root_quaternion = torch.as_tensor(root_quaternion_np, dtype=dtype, device=device)
    variable_q = torch.nn.Parameter(
        torch.as_tensor(q_current_np[:, variable_indices_np], dtype=dtype, device=device).clone()
    )
    variable_root = torch.nn.Parameter(
        torch.as_tensor(root_current_np, dtype=dtype, device=device).clone()
    )
    optimizer = torch.optim.Adam(
        (variable_q, variable_root), lr=float(algorithm["learning_rate"])
    )

    contract = load_g1_contract()
    midpoint = 0.5 * (contract.lower + contract.upper)
    half_range = (
        0.5
        * float(algorithm["soft_joint_limit_factor"])
        * (contract.upper - contract.lower)
    )
    soft_lower_np = midpoint - half_range
    soft_upper_np = midpoint + half_range
    maximum_joint_delta = float(algorithm["maximum_joint_delta_rad"])
    q_lower_np = np.maximum(
        soft_lower_np[variable_indices_np],
        q_original_np[:, variable_indices_np] - maximum_joint_delta,
    )
    q_upper_np = np.minimum(
        soft_upper_np[variable_indices_np],
        q_original_np[:, variable_indices_np] + maximum_joint_delta,
    )
    if np.any(q_lower_np > q_upper_np):
        raise ContactAwareRepairError("source pose lies outside the frozen balance trust region")
    q_lower = torch.as_tensor(q_lower_np, dtype=dtype, device=device)
    q_upper = torch.as_tensor(q_upper_np, dtype=dtype, device=device)
    root_lower_np = root_original_np.copy()
    root_upper_np = root_original_np.copy()
    root_lower_np[:, :2] -= float(algorithm["maximum_root_xy_delta_m"])
    root_upper_np[:, :2] += float(algorithm["maximum_root_xy_delta_m"])
    root_lower_np[:, 2] -= float(algorithm["maximum_root_z_delta_m"])
    root_upper_np[:, 2] += float(algorithm["maximum_root_z_delta_m"])
    root_lower = torch.as_tensor(root_lower_np, dtype=dtype, device=device)
    root_upper = torch.as_tensor(root_upper_np, dtype=dtype, device=device)

    com_target = torch.as_tensor(targets.com_position, dtype=dtype, device=device)
    zmp_target = torch.as_tensor(targets.zmp_position, dtype=dtype, device=device)
    capture_target = torch.as_tensor(
        targets.capture_position, dtype=dtype, device=device
    )
    zmp_mask = torch.as_tensor(targets.zmp_mask, dtype=torch.bool, device=device)
    capture_mask = torch.as_tensor(
        targets.capture_mask, dtype=torch.bool, device=device
    )
    eligible = torch.as_tensor(
        np.isfinite(targets.support_plane_z), dtype=torch.bool, device=device
    )
    support_plane = torch.as_tensor(
        np.nan_to_num(targets.support_plane_z, nan=0.0), dtype=dtype, device=device
    )
    support_target = torch.as_tensor(
        np.asarray(immutable_v28["foot_support_position_w"], dtype=np.float64).reshape(
            frame_count, 2, 4, 3
        ),
        dtype=dtype,
        device=device,
    )
    active_support = torch.as_tensor(
        targets.active_support_mask.reshape(frame_count, 2, 4),
        dtype=torch.bool,
        device=device,
    )
    radii = torch.as_tensor(
        load_urdf_asset().support_radii, dtype=dtype, device=device
    )
    velocity_limits = torch.as_tensor(
        contract.velocity[variable_indices_np], dtype=dtype, device=device
    )
    weights = algorithm["weights"]
    dt = 1.0 / fps
    best_loss = math.inf
    patience = 0
    iterations_completed = 0
    best_components: dict[str, float] = {}
    best_q: torch.Tensor | None = None
    best_root: torch.Tensor | None = None

    for iteration in range(maximum_iterations):
        optimizer.zero_grad(set_to_none=True)
        q = q_original.clone()
        q[:, variable_indices] = variable_q
        transforms = kinematics.forward(q, variable_root, root_quaternion)
        centroidal = differentiable_centroidal_signals(
            transforms, inertial_model, fps=fps
        )
        com = centroidal["com_position"]
        com_velocity = centroidal["com_velocity"]
        com_acceleration = centroidal["com_acceleration"]
        momentum_rate = centroidal["centroidal_momentum_rate"]
        support = kinematics.foot_support_positions(transforms)
        height = com[:, 2] - support_plane
        gravity = float(V29_ALGORITHM["gravity_mps2"])
        normal_force = float(inertial_model.total_mass_kg) * (
            com_acceleration[:, 2] + gravity
        )
        safe_normal_force = normal_force.clamp_min(1.0)
        force_xy = float(inertial_model.total_mass_kg) * com_acceleration[:, :2]
        zmp_surrogate = torch.stack(
            (
                com[:, 0]
                - (height * force_xy[:, 0] + momentum_rate[:, 1])
                / safe_normal_force,
                com[:, 1]
                - (height * force_xy[:, 1] - momentum_rate[:, 0])
                / safe_normal_force,
            ),
            dim=-1,
        )
        omega = torch.sqrt(
            gravity / height.clamp_min(0.1)
        )
        capture_surrogate = com[:, :2] + com_velocity[:, :2] / omega[:, None]

        def masked_mean_square(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            if bool(torch.any(mask)):
                return torch.mean(torch.square(values[mask]))
            return torch.zeros((), dtype=dtype, device=device)

        com_loss = masked_mean_square(com - com_target, eligible)
        zmp_loss = masked_mean_square(zmp_surrogate - zmp_target, zmp_mask)
        capture_loss = masked_mean_square(
            capture_surrogate - capture_target, capture_mask
        )
        contact_loss = masked_mean_square(support - support_target, active_support)
        ground_violation = torch.relu(
            radii[None, :, :] + 0.0005 - support[..., 2]
        )
        ground_loss = torch.mean(torch.square(ground_violation))
        joint_delta = variable_q - q_original[:, variable_indices]
        root_delta = variable_root - root_original
        joint_style = torch.mean(torch.square(joint_delta))
        root_style_xy = torch.mean(torch.square(root_delta[:, :2]))
        root_style_z = torch.mean(torch.square(root_delta[:, 2]))
        correction_velocity = torch.mean(torch.square(joint_delta[1:] - joint_delta[:-1]))
        correction_velocity = correction_velocity + torch.mean(
            torch.square(root_delta[1:] - root_delta[:-1])
        )
        if frame_count >= 3:
            correction_acceleration = torch.mean(
                torch.square(joint_delta[2:] - 2.0 * joint_delta[1:-1] + joint_delta[:-2])
            ) + torch.mean(
                torch.square(root_delta[2:] - 2.0 * root_delta[1:-1] + root_delta[:-2])
            )
        else:
            correction_acceleration = torch.zeros((), dtype=dtype, device=device)
        qdot = (variable_q[1:] - variable_q[:-1]) / dt
        velocity_excess = torch.relu(
            torch.abs(qdot)
            - float(algorithm["joint_velocity_limit_fraction"])
            * velocity_limits[None, :]
        )
        velocity_loss = torch.mean(torch.square(velocity_excess))
        positive_height_loss = masked_mean_square(
            torch.relu(float(algorithm["minimum_com_height_m"]) - height), eligible
        )
        minimum_normal_force = (
            float(algorithm["minimum_normal_force_weight_fraction"])
            * float(inertial_model.total_mass_kg)
            * gravity
        )
        positive_force_loss = masked_mean_square(
            torch.relu(minimum_normal_force - normal_force), eligible
        )

        terms = {
            "centroidal_seed": com_loss,
            "zmp_seed": zmp_loss,
            "capture_seed": capture_loss,
            "contact_lock": contact_loss,
            "ground": ground_loss,
            "joint_style": joint_style,
            "root_style_xy": root_style_xy,
            "root_style_z": root_style_z,
            "correction_velocity": correction_velocity,
            "correction_acceleration": correction_acceleration,
            "joint_velocity_headroom": velocity_loss,
            "positive_com_height": positive_height_loss,
            "positive_normal_force": positive_force_loss,
        }
        loss = sum(float(weights[name]) * term for name, term in terms.items())
        if not bool(torch.isfinite(loss)):
            raise ContactAwareRepairQuarantine(
                "contact IK produced a non-finite objective",
                {"iteration": iteration, "action": "quarantine"},
            )
        value = float(loss.detach().cpu())
        current_components = {
            name: float(term.detach().cpu()) for name, term in terms.items()
        }
        if best_loss - value > float(algorithm["convergence_delta"]):
            best_loss = value
            best_components = current_components
            best_q = variable_q.detach().clone()
            best_root = variable_root.detach().clone()
            patience = 0
        else:
            patience += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (variable_q, variable_root), float(algorithm["gradient_clip_norm"])
        )
        optimizer.step()
        with torch.no_grad():
            variable_q.clamp_(q_lower, q_upper)
            variable_root.clamp_(root_lower, root_upper)
            root_xy_delta = variable_root[:, :2] - root_original[:, :2]
            root_xy_norm = torch.linalg.vector_norm(
                root_xy_delta, dim=-1, keepdim=True
            ).clamp_min(1.0e-12)
            maximum_root_xy = float(algorithm["maximum_root_xy_delta_m"])
            root_xy_scale = torch.clamp(maximum_root_xy / root_xy_norm, max=1.0)
            variable_root[:, :2] = (
                root_original[:, :2] + root_xy_delta * root_xy_scale
            )
            # Endpoint identity prevents implicit clipping/cropping and keeps
            # the v28 source-coordinate boundary exact.
            variable_q[0] = q_original[0, variable_indices]
            variable_q[-1] = q_original[-1, variable_indices]
            variable_root[0] = root_original[0]
            variable_root[-1] = root_original[-1]
        iterations_completed = iteration + 1
        if patience >= int(algorithm["convergence_patience"]):
            break

    if best_q is None or best_root is None:
        raise ContactAwareRepairError("contact IK did not retain a finite best state")
    q_result = q_original_np.copy()
    q_result[:, variable_indices_np] = best_q.cpu().numpy()
    root_result = best_root.cpu().numpy()
    non_variable = np.setdiff1d(np.arange(29, dtype=np.int64), variable_indices_np)
    if not np.array_equal(q_result[:, non_variable], q_original_np[:, non_variable]):
        raise ContactAwareRepairError("contact IK changed a frozen non-balance joint")
    maximum_radial_root_delta = float(
        np.max(np.linalg.norm(root_result[:, :2] - root_original_np[:, :2], axis=1))
    )
    if maximum_radial_root_delta > float(
        algorithm["maximum_root_xy_delta_m"]
    ) + 1.0e-10:
        raise ContactAwareRepairError("root XY radial trust region was not enforced")
    return q_result, root_result, {
        "iterations": iterations_completed,
        "best_weighted_loss": best_loss,
        "best_unweighted_loss_components": best_components,
        "variable_joint_names": [contract.policy_names[index] for index in variable_indices_np],
        "leg_joint_count": len(leg_indices),
        "waist_balance_joint_count": len(waist_indices),
        "maximum_joint_delta_rad": float(
            np.max(np.abs(q_result[:, variable_indices_np] - q_original_np[:, variable_indices_np]))
        ),
        "maximum_root_xy_delta_m": maximum_radial_root_delta,
        "maximum_root_z_delta_m": float(
            np.max(np.abs(root_result[:, 2] - root_original_np[:, 2]))
        ),
    }


def _recompute_payload(
    immutable_v28: Mapping[str, np.ndarray],
    joint_position: np.ndarray,
    root_position: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recompute every FK/rate field while retaining immutable v28 semantics."""

    payload = {name: np.asarray(value).copy() for name, value in immutable_v28.items()}
    frame_count = int(_scalar(immutable_v28, "frame_count"))
    fps = float(_scalar(immutable_v28, "fps"))
    q = np.asarray(joint_position, dtype=np.float64)
    root = np.asarray(root_position, dtype=np.float64)
    root_quaternion = np.asarray(immutable_v28["root_quat_wxyz"], dtype=np.float64)
    body_names = tuple(str(name) for name in immutable_v28["body_names"].tolist())
    if q.shape != (frame_count, 29) or root.shape != (frame_count, 3):
        raise ContactAwareRepairError("recompute received a wrong-shaped trajectory")
    kinematics = TorchG1Kinematics(device="cpu", dtype=torch.float64)

    def solve(root_value: np.ndarray) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        with torch.inference_mode():
            transform_value = kinematics.forward(
                torch.as_tensor(q, dtype=torch.float64),
                torch.as_tensor(root_value, dtype=torch.float64),
                torch.as_tensor(root_quaternion, dtype=torch.float64),
            )
            support_value = kinematics.foot_support_positions(transform_value)
        return transform_value, support_value

    transforms, support = solve(root)
    preliminary_support = support.cpu().numpy()
    required_z_shift = np.maximum(
        FOOT_COLLISION_RADIUS_M
        + 0.0005
        - np.min(preliminary_support[..., 2], axis=(1, 2)),
        0.0,
    )
    projected_root = root.copy()
    projected_root[:, 2] += required_z_shift
    original_root = np.asarray(immutable_v28["root_pos_w"], dtype=np.float64)
    maximum_root_z_delta = float(
        np.max(np.abs(projected_root[:, 2] - original_root[:, 2]))
    )
    if maximum_root_z_delta > float(
        REPAIR_ALGORITHM["ik"]["maximum_root_z_delta_m"]
    ) + 1.0e-12:
        raise ContactAwareRepairQuarantine(
            "ground projection exceeds the root-Z trust region",
            {
                "maximum_root_z_delta_m": maximum_root_z_delta,
                "maximum_allowed_root_z_delta_m": REPAIR_ALGORITHM["ik"][
                    "maximum_root_z_delta_m"
                ],
            },
        )
    transforms, support = solve(projected_root)
    root = projected_root
    with torch.inference_mode():
        _, body_position, body_rotation = kinematics.stacked(transforms, body_names)
        sole = kinematics.sole_positions(transforms)
    support_np64 = support.cpu().numpy()
    minimum_surface = float(
        np.min(support_np64[..., 2] - load_urdf_asset().support_radii[None, :, :])
    )
    if minimum_surface < -0.002:
        raise ContactAwareRepairQuarantine(
            "contact IK penetrates the exact URDF support plane",
            {
                "minimum_support_sphere_surface_z_m": minimum_surface,
                "tolerance_m": -0.002,
            },
        )
    body_position_np = body_position.cpu().numpy().astype(np.float32)
    body_quaternion = matrix_to_quaternion_wxyz(body_rotation.cpu().numpy())
    support_flat = support_np64.reshape(frame_count, 8, 3)
    contact_points = np.stack(
        (
            support_flat[:, :4][:, :2].mean(axis=1),
            support_flat[:, :4][:, 2:].mean(axis=1),
            support_flat[:, 4:][:, :2].mean(axis=1),
            support_flat[:, 4:][:, 2:].mean(axis=1),
        ),
        axis=1,
    )
    contract = load_g1_contract()
    q32 = q.astype(np.float32)
    root32 = root.astype(np.float32)
    payload.update(
        {
            "retarget_version": np.asarray(REPAIR_VERSION),
            "joint_pos": q32,
            "joint_pos_sdk": policy_to_sdk(q32, contract),
            "joint_vel": _linear_velocity(q32, fps),
            "joint_acc": _linear_velocity(_linear_velocity(q32, fps), fps),
            "root_pos_w": root32,
            "root_quat_wxyz": np.asarray(immutable_v28["root_quat_wxyz"]).copy(),
            "root_lin_vel_w": _linear_velocity(root32, fps),
            "root_ang_vel_w": _angular_velocity_world(root_quaternion, fps),
            "body_pos_w": body_position_np,
            "body_quat_wxyz": body_quaternion,
            "body_lin_vel_w": _linear_velocity(body_position_np, fps),
            "body_ang_vel_w": _angular_velocity_world(body_quaternion, fps),
            "foot_sole_position_w": sole.cpu().numpy().astype(np.float32),
            "foot_support_position_w": support_flat.astype(np.float32),
            "contact_point_position_w": contact_points.astype(np.float32),
            "solver_name": np.asarray("v30_exact_v29_seed+torch_contact_sequence_ik"),
            "ground_projection_maximum_z_shift_m": np.asarray(
                float(np.max(required_z_shift)), dtype=np.float32
            ),
            "ground_projection_ground_margin_m": np.asarray(0.0005, dtype=np.float32),
            "gate3_complete": np.asarray(False),
            "gate3_status": np.asarray(
                "v30 contact-aware derivative generated; independent validation pending"
            ),
        }
    )
    return payload


def _assert_immutable_contract(
    source_v27: Mapping[str, np.ndarray],
    immutable_v28: Mapping[str, np.ndarray],
    repaired: Mapping[str, np.ndarray],
) -> np.ndarray:
    frame_count = int(_scalar(immutable_v28, "frame_count"))
    if int(_scalar(repaired, "frame_count")) != frame_count:
        raise ContactAwareRepairError("repair changed frame_count")
    for name in (
        "time",
        "root_quat_wxyz",
        "left_foot_contact",
        "right_foot_contact",
        "left_heel_contact",
        "left_toe_contact",
        "right_heel_contact",
        "right_toe_contact",
    ):
        if not np.array_equal(repaired[name], immutable_v28[name]):
            raise ContactAwareRepairError(f"repair changed immutable field {name}")
    leg, waist = _balance_indices()
    variable = np.concatenate((leg, waist))
    frozen = np.setdiff1d(np.arange(29, dtype=np.int64), variable)
    if not np.array_equal(
        np.asarray(repaired["joint_pos"])[:, frozen],
        np.asarray(immutable_v28["joint_pos"])[:, frozen],
    ):
        raise ContactAwareRepairError("repair changed an immutable non-balance joint")
    root_delta = np.asarray(repaired["root_pos_w"], dtype=np.float64) - np.asarray(
        immutable_v28["root_pos_w"], dtype=np.float64
    )
    maximum_root_xy = float(np.max(np.linalg.norm(root_delta[:, :2], axis=1)))
    maximum_root_z = float(np.max(np.abs(root_delta[:, 2])))
    if maximum_root_xy > float(
        REPAIR_ALGORITHM["ik"]["maximum_root_xy_delta_m"]
    ) + 1.0e-6:
        raise ContactAwareRepairError("serialized root exceeds the radial XY trust region")
    if maximum_root_z > float(
        REPAIR_ALGORITHM["ik"]["maximum_root_z_delta_m"]
    ) + 1.0e-6:
        raise ContactAwareRepairError("serialized root exceeds the Z trust region")
    joint_delta = np.asarray(repaired["joint_pos"], dtype=np.float64)[
        :, variable
    ] - np.asarray(immutable_v28["joint_pos"], dtype=np.float64)[:, variable]
    if float(np.max(np.abs(joint_delta))) > float(
        REPAIR_ALGORITHM["ik"]["maximum_joint_delta_rad"]
    ) + 1.0e-6:
        raise ContactAwareRepairError("serialized joints exceed the repair trust region")
    scale = float(_scalar(immutable_v28, "dynamic_realized_scale"))
    coordinate = np.arange(frame_count, dtype=np.float64) / scale
    coordinate[-1] = int(_scalar(source_v27, "frame_count")) - 1
    if not np.all(np.diff(coordinate) > 0.0):
        raise ContactAwareRepairError("v28 source-frame coordinate is not strictly monotonic")
    return coordinate


def audit_contact_anchor_postconditions(
    immutable_v28: Mapping[str, np.ndarray],
    repaired: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Hard world/contact checks missing from the generic v29/Gate-3 gates."""

    frame_count = int(_scalar(immutable_v28, "frame_count"))
    if int(_scalar(repaired, "frame_count")) != frame_count:
        raise ContactAwareRepairError("contact-anchor audit frame counts differ")
    source_channels, _ = _contact_arrays(immutable_v28, frame_count)
    repaired_channels, _ = _contact_arrays(repaired, frame_count)
    if not np.array_equal(source_channels, repaired_channels):
        raise ContactAwareRepairError("contact-anchor audit requires immutable contacts")
    source_support = np.asarray(
        immutable_v28["foot_support_position_w"], dtype=np.float64
    )
    repaired_support = np.asarray(repaired["foot_support_position_w"], dtype=np.float64)
    if source_support.shape != (frame_count, 8, 3) or repaired_support.shape != (
        frame_count,
        8,
        3,
    ):
        raise ContactAwareRepairError("contact-anchor support tensors have wrong shapes")
    radii = load_urdf_asset().support_radii.reshape(8)
    active_indices: list[tuple[int, int]] = []
    for frame in range(frame_count):
        for channel in np.flatnonzero(source_channels[frame]):
            foot = 0 if channel < 2 else 1
            pair = (0, 1) if channel % 2 == 0 else (2, 3)
            active_indices.extend((frame, foot * 4 + offset) for offset in pair)
    if active_indices:
        frame_index = np.asarray([value[0] for value in active_indices], dtype=np.int64)
        point_index = np.asarray([value[1] for value in active_indices], dtype=np.int64)
        surfaces = (
            repaired_support[frame_index, point_index, 2] - radii[point_index]
        )
        support_displacement = np.linalg.norm(
            repaired_support[frame_index, point_index]
            - source_support[frame_index, point_index],
            axis=1,
        )
        surface_minimum = float(np.min(surfaces))
        surface_maximum = float(np.max(surfaces))
        support_displacement_maximum = float(np.max(support_displacement))
    else:
        surface_minimum = math.nan
        surface_maximum = math.nan
        support_displacement_maximum = 0.0

    stance_fields = (
        "left_heel_stance",
        "left_toe_stance",
        "right_heel_stance",
        "right_toe_stance",
    )
    source_stance_presence = [name in immutable_v28 for name in stance_fields]
    repaired_stance_presence = [name in repaired for name in stance_fields]
    if source_stance_presence != repaired_stance_presence:
        raise ContactAwareRepairError("repair changed stance-field presence")
    if any(source_stance_presence) and not all(source_stance_presence):
        raise ContactAwareRepairError("stance-anchor audit received partial stance fields")
    stance_displacement_maximum = 0.0
    stance_count = 0
    if all(source_stance_presence):
        source_stance = np.stack(
            [np.asarray(immutable_v28[name], dtype=bool) for name in stance_fields], axis=1
        )
        repaired_stance = np.stack(
            [np.asarray(repaired[name], dtype=bool) for name in stance_fields], axis=1
        )
        if not np.array_equal(source_stance, repaired_stance):
            raise ContactAwareRepairError("repair changed immutable stance labels")
        source_contact_point = np.asarray(
            immutable_v28["contact_point_position_w"], dtype=np.float64
        )
        repaired_contact_point = np.asarray(
            repaired["contact_point_position_w"], dtype=np.float64
        )
        stance_count = int(np.count_nonzero(source_stance))
        if stance_count:
            stance_displacement_maximum = float(
                np.max(
                    np.linalg.norm(
                        repaired_contact_point - source_contact_point, axis=-1
                    )[source_stance]
                )
            )

    thresholds = REPAIR_ALGORITHM["acceptance"]
    violations: list[str] = []
    if active_indices:
        if surface_minimum < float(
            thresholds["support_surface_penetration_minimum_m"]
        ):
            violations.append("active_contact_support_penetration")
        if surface_maximum > float(
            thresholds["active_contact_surface_height_maximum_m"]
        ):
            violations.append("active_contact_support_above_ground")
    if support_displacement_maximum > float(
        thresholds["active_contact_support_displacement_maximum_m"]
    ):
        violations.append("active_contact_support_anchor_displacement")
    if stance_displacement_maximum > float(
        thresholds["active_stance_anchor_displacement_maximum_m"]
    ):
        violations.append("active_stance_anchor_displacement")
    return {
        "passed": not violations,
        "violations": sorted(violations),
        "active_support_sample_count": len(active_indices),
        "active_support_surface_minimum_m": (
            None if not math.isfinite(surface_minimum) else surface_minimum
        ),
        "active_support_surface_maximum_m": (
            None if not math.isfinite(surface_maximum) else surface_maximum
        ),
        "active_support_displacement_maximum_m": support_displacement_maximum,
        "active_stance_sample_count": stance_count,
        "active_stance_anchor_displacement_maximum_m": stance_displacement_maximum,
        "thresholds": {
            name: thresholds[name]
            for name in (
                "active_contact_surface_height_maximum_m",
                "active_contact_support_displacement_maximum_m",
                "active_stance_anchor_displacement_maximum_m",
                "support_surface_penetration_minimum_m",
            )
        },
    }


def derive_contact_repaired_payload(
    source_v27: Mapping[str, np.ndarray],
    candidate_v28: Mapping[str, np.ndarray],
    *,
    split: str,
    source_v27_checksum_sha256: str,
    source_v28_checksum_sha256: str,
    source_v27_manifest_checksum_sha256: str,
    source_v28_manifest_checksum_sha256: str,
    policy_checksum_sha256: str,
    _test_authority: object | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Repair one in-memory v28 candidate or quarantine it without writing."""

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be exactly train, validation, or test")
    if split == "test" and _test_authority is not _TEST_AUTHORITY:
        raise ContactAwareRepairError(
            "low-level test repair is sealed; use the witnessed one-shot split builder"
        )
    initial_audit = audit_contact_pair(source_v27, candidate_v28)
    immutable = {name: np.asarray(value).copy() for name, value in candidate_v28.items()}
    current = {name: np.asarray(value).copy() for name, value in candidate_v28.items()}
    cycle_limit = int(REPAIR_ALGORITHM["ik"]["outer_cycles"])
    history: list[dict[str, Any]] = []
    final_audit = initial_audit
    total_iterations = 0
    if not initial_audit["passed"]:
        for cycle in range(1, cycle_limit + 1):
            try:
                targets = plan_repair_targets(source_v27, current)
            except ContactAwareRepairQuarantine as exc:
                raise ContactAwareRepairQuarantine(
                    exc.reason,
                    {
                        **exc.evidence,
                        "failed_outer_cycle": cycle,
                        "optimization_history": history,
                        "terminal_frames_cropped": False,
                    },
                ) from exc
            q, root, optimization = _optimize_contact_ik(
                current,
                immutable,
                targets,
            )
            total_iterations += int(optimization["iterations"])
            current = _recompute_payload(immutable, q, root)
            final_audit = audit_contact_pair(source_v27, current)
            history.append(
                {
                    "cycle": cycle,
                    "maximum_raw_seed_correction_m": targets.maximum_raw_correction_m,
                    "maximum_smoothed_com_seed_correction_m": (
                        targets.maximum_smoothed_com_seed_correction_m
                    ),
                    "capture_preview_missing_frames": list(
                        targets.capture_preview_missing_frames
                    ),
                    "capture_preview_missing_terminal_frames": list(
                        targets.capture_preview_missing_terminal_frames
                    ),
                    "optimization": optimization,
                    "exact_v29_passed": bool(final_audit["passed"]),
                    "exact_v29_quarantine_reasons": final_audit["quarantine_reasons"],
                    "exact_v29_audit_checksum_sha256": _canonical_checksum(final_audit),
                }
            )
            if final_audit["passed"]:
                break
    if not final_audit["passed"]:
        raise ContactAwareRepairQuarantine(
            "bounded contact-aware sequence repair did not pass the exact v29 gate",
            {
                "initial_audit": initial_audit,
                "final_audit": final_audit,
                "optimization_history": history,
                "terminal_frames_cropped": False,
            },
        )
    # Even an already feasible v28 clip is published as a byte-preserving
    # numerical identity derivative in the v30 namespace.
    current["retarget_version"] = np.asarray(REPAIR_VERSION)
    current["solver_name"] = np.asarray(
        "v30_identity_exact_v29_pass"
        if not history
        else "v30_exact_v29_seed+torch_contact_sequence_ik"
    )
    source_coordinate = _assert_immutable_contract(source_v27, immutable, current)
    contact_anchor_audit = audit_contact_anchor_postconditions(immutable, current)
    if not contact_anchor_audit["passed"]:
        raise ContactAwareRepairQuarantine(
            "repaired trajectory violates the hard world/contact anchor contract",
            {
                "contact_anchor_audit": contact_anchor_audit,
                "exact_v29_audit": final_audit,
                "optimization_history": history,
            },
        )
    dynamic_audit = audit_dynamic_feasibility(current)
    if float(dynamic_audit["scale_pre"]) != 1.0:
        raise ContactAwareRepairQuarantine(
            "contact repair violates the frozen v28 dynamic prefilter",
            {
                "dynamic_audit": dynamic_audit,
                "exact_v29_audit": final_audit,
            },
        )
    current.update(
        {
            "contact_repair_schema_version": np.asarray(REPAIR_SCHEMA_VERSION),
            "contact_repair_split": np.asarray(split),
            "contact_repair_source_v27_checksum_sha256": np.asarray(
                source_v27_checksum_sha256
            ),
            "contact_repair_source_v28_checksum_sha256": np.asarray(
                source_v28_checksum_sha256
            ),
            "contact_repair_source_v27_manifest_checksum_sha256": np.asarray(
                source_v27_manifest_checksum_sha256
            ),
            "contact_repair_source_v28_manifest_checksum_sha256": np.asarray(
                source_v28_manifest_checksum_sha256
            ),
            "contact_repair_policy_checksum_sha256": np.asarray(
                policy_checksum_sha256
            ),
            "contact_repair_algorithm_checksum_sha256": np.asarray(
                _canonical_checksum(REPAIR_ALGORITHM)
            ),
            "contact_repair_source_frame_coordinate": source_coordinate,
            "contact_repair_exact_v29_gate_version": np.asarray(CONTACT_GATE_VERSION),
            "contact_repair_exact_v29_audit_checksum_sha256": np.asarray(
                _canonical_checksum(final_audit)
            ),
            "contact_repair_outer_cycles": np.asarray(len(history), dtype=np.int64),
            "contact_repair_total_iterations": np.asarray(total_iterations, dtype=np.int64),
            "contact_repair_terminal_frames_cropped": np.asarray(False),
            "solver_iterations": np.asarray(total_iterations, dtype=np.int64),
        }
    )
    return current, {
        "status": "repaired" if history else "identity_exact_v29_pass",
        "initial_exact_v29_audit": initial_audit,
        "final_exact_v29_audit": final_audit,
        "contact_anchor_audit": contact_anchor_audit,
        "dynamic_audit": dynamic_audit,
        "optimization_history": history,
        "preservation": {
            "frame_count": True,
            "time": True,
            "source_frame_coordinate": True,
            "root_quaternion": True,
            "non_balance_joints": True,
            "contact_labels": True,
            "terminal_frames_cropped": False,
        },
    }


def derive_contact_repaired_archive(
    source_v27_path: str | Path,
    candidate_v28_path: str | Path,
    output_path: str | Path,
    *,
    split: str,
    source_v27_manifest_checksum_sha256: str,
    source_v28_manifest_checksum_sha256: str,
    policy_checksum_sha256: str,
    expected_human_path: str | Path | None = None,
    expected_candidate_record: Mapping[str, Any] | None = None,
    _test_authority: object | None = None,
) -> dict[str, Any]:
    """Derive and independently validate one archive before atomic publication."""

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be exactly train, validation, or test")
    if split == "test" and _test_authority is not _TEST_AUTHORITY:
        raise ContactAwareRepairError(
            "low-level test archive access is sealed; use the witnessed one-shot builder"
        )
    source_path = Path(source_v27_path).expanduser().resolve()
    candidate_path = Path(candidate_v28_path).expanduser().resolve()
    output = assert_output_is_local(output_path)
    if output in {source_path, candidate_path}:
        raise ContactAwareRepairError("v30 output must not overwrite a v27/v28 archive")
    source_checksum = _sha256(source_path)
    candidate_checksum = _sha256(candidate_path)
    source = _safe_npz(source_path)
    candidate = _safe_npz(candidate_path)
    if _sha256(source_path) != source_checksum or _sha256(candidate_path) != candidate_checksum:
        raise ContactAwareRepairError("v27/v28 archive changed while it was being loaded")
    if expected_candidate_record is not None:
        if expected_candidate_record.get("g1_checksum_sha256") != candidate_checksum:
            raise ContactAwareRepairError(
                "v28 manifest record checksum does not bind the selected archive"
            )
        try:
            recorded_size = int(expected_candidate_record["g1_size_bytes"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ContactAwareRepairError(
                "v28 manifest record archive size is missing or invalid"
            ) from exc
        if recorded_size != candidate_path.stat().st_size:
            raise ContactAwareRepairError(
                "v28 manifest record size does not bind the selected archive"
            )
        _validate_v28_record_metadata(
            expected_candidate_record,
            candidate,
            split=split,
        )
    if str(_scalar(candidate, "dynamic_split")) != split:
        raise ContactAwareRepairError("v28 archive split provenance does not match invocation")
    if str(_scalar(candidate, "dynamic_source_g1_checksum_sha256")) != source_checksum:
        raise ContactAwareRepairError("v28 archive does not bind the selected v27 source")
    if str(
        _scalar(candidate, "dynamic_source_manifest_checksum_sha256")
    ) != source_v27_manifest_checksum_sha256:
        raise ContactAwareRepairError(
            "v28 archive does not bind the selected v27 manifest checksum"
        )
    payload, derivation = derive_contact_repaired_payload(
        source,
        candidate,
        split=split,
        source_v27_checksum_sha256=source_checksum,
        source_v28_checksum_sha256=candidate_checksum,
        source_v27_manifest_checksum_sha256=source_v27_manifest_checksum_sha256,
        source_v28_manifest_checksum_sha256=source_v28_manifest_checksum_sha256,
        policy_checksum_sha256=policy_checksum_sha256,
        _test_authority=_test_authority,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", dir=output.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
        temporary_path = Path(temporary_name)
        _save_npz_atomic(temporary_path, payload)
        from .g1_validation import validate_g1_archive

        validation = validate_g1_archive(
            temporary_path,
            expected_source_path=(
                None
                if expected_human_path is None
                else Path(expected_human_path).expanduser().resolve()
            ),
        )
        if validation.get("gate3_kinematic_complete") is not True:
            raise ContactAwareRepairQuarantine(
                "repaired archive failed independent Gate-3 validation",
                {
                    "required_failures": validation.get("required_failures"),
                    "errors": validation.get("errors"),
                    "exact_v29_audit": derivation["final_exact_v29_audit"],
                },
            )
        os.replace(temporary_path, output)
        temporary_name = None
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    serialized_checksum = _sha256(output)
    serialized = _safe_npz(output)
    if _sha256(output) != serialized_checksum:
        raise ContactAwareRepairError("v30 archive changed while metadata was recomputed")
    serialized_metadata = _serialized_archive_metadata(
        serialized, split_field="contact_repair_split"
    )
    if serialized_metadata["split"] != split:
        raise ContactAwareRepairError("serialized v30 split provenance mismatch")
    return {
        **derivation,
        "gate3_kinematic_complete": True,
        "path": str(output),
        "bytes": output.stat().st_size,
        "checksum_sha256": serialized_checksum,
        "serialized_metadata": serialized_metadata,
        "retarget_version": REPAIR_VERSION,
    }


def _load_split_manifests(
    split: str, v27_path: Path, v28_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    v27 = _load_json(v27_path, "v27 manifest")
    v28 = _load_json(v28_path, "v28 manifest")
    if (
        v27.get("schema_version") != "accad_g1_gate3_manifest_v1"
        or v27.get("split") != split
        or v27.get("kind") != f"split:{split}"
    ):
        raise ContactAwareRepairError("v27 manifest schema/split contract is invalid")
    if (
        v28.get("schema_version") != "accad_g1_dynamic_manifest_v1"
        or v28.get("split") != split
        or v28.get("kind") != f"split:{split}"
    ):
        raise ContactAwareRepairError("v28 manifest schema/split contract is invalid")
    for name, document in (("v27", v27), ("v28", v28)):
        motions = document.get("motions")
        if not isinstance(motions, list) or int(document.get("num_motions", -1)) != len(
            motions
        ):
            raise ContactAwareRepairError(f"{name} manifest motion count is invalid")
    if v28.get("source_manifest_checksum_sha256") != _sha256(v27_path):
        raise ContactAwareRepairError("v28 manifest does not bind the selected v27 manifest")
    return v27, v28


def _resolved_asset_document() -> dict[str, Any]:
    asset = load_urdf_asset()
    return {
        "g1_contract_path": G1_CONTRACT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "g1_contract_checksum_sha256": _sha256(G1_CONTRACT_PATH),
        "g1_urdf_path": G1_URDF_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "urdf_checksum_sha256": asset.checksum_sha256,
        "total_mass_kg": asset.total_mass_kg,
        "support_centers_m": asset.support_centers.tolist(),
        "support_radii_m": asset.support_radii.tolist(),
    }


def _dependency_document() -> dict[str, Any]:
    missing = [path for path in DEPENDENCY_PATHS if not path.is_file()]
    if missing:
        raise ContactAwareRepairError(f"repair dependency files are missing: {missing}")
    return {
        "files": {
            path.relative_to(PIPELINE_ROOT).as_posix(): _sha256(path)
            for path in DEPENDENCY_PATHS
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "torch")
        },
    }


def _policy_document(
    train_v27_path: Path,
    train_v28_path: Path,
    train_v27: Mapping[str, Any],
    train_v28: Mapping[str, Any],
) -> dict[str, Any]:
    dynamic_policy_checksum = str(train_v28.get("dynamic_policy_checksum_sha256", ""))
    if (
        not DEFAULT_DYNAMIC_POLICY_PATH.is_file()
        or _sha256(DEFAULT_DYNAMIC_POLICY_PATH) != dynamic_policy_checksum
    ):
        raise ContactAwareRepairError(
            "v28 frozen dynamic policy file/checksum is unavailable"
        )
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "status": "frozen_from_train",
        "created_at_utc": _utc_now(),
        "repair_version": REPAIR_VERSION,
        "implementation": {
            "path": IMPLEMENTATION_PATH.relative_to(PIPELINE_ROOT).as_posix(),
            "checksum_sha256": _sha256(IMPLEMENTATION_PATH),
        },
        "algorithm": REPAIR_ALGORITHM,
        "algorithm_checksum_sha256": _canonical_checksum(REPAIR_ALGORITHM),
        "resolved_asset": _resolved_asset_document(),
        "dependencies": _dependency_document(),
        "train_inputs": {
            "v27_manifest_path": train_v27_path.relative_to(PIPELINE_ROOT).as_posix(),
            "v27_manifest_checksum_sha256": _sha256(train_v27_path),
            "v27_motion_set_checksum_sha256": train_v27.get(
                "motion_set_checksum_sha256"
            ),
            "v28_manifest_path": train_v28_path.relative_to(PIPELINE_ROOT).as_posix(),
            "v28_manifest_checksum_sha256": _sha256(train_v28_path),
            "v28_motion_set_checksum_sha256": train_v28.get(
                "motion_set_checksum_sha256"
            ),
            "v28_dynamic_policy_checksum_sha256": train_v28.get(
                "dynamic_policy_checksum_sha256"
            ),
            "v28_dynamic_policy_path": DEFAULT_DYNAMIC_POLICY_PATH.relative_to(
                PIPELINE_ROOT
            ).as_posix(),
        },
        "split_policy": {
            "train": "allowed_and_freezes_policy",
            "validation": "requires_unchanged_train_policy",
            "test": "requires_explicit_validation_pass_witness_and_is_consumed_once",
        },
    }


def _load_or_freeze_policy(
    *,
    split: str,
    v27_path: Path,
    v28_path: Path,
    v27: Mapping[str, Any],
    v28: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    policy_path = output_root / "policy.json"
    if split == "train":
        expected = _policy_document(v27_path, v28_path, v27, v28)
        if policy_path.exists():
            existing = _load_json(policy_path, "v30 frozen policy")
            left, right = dict(existing), dict(expected)
            left.pop("created_at_utc", None)
            right.pop("created_at_utc", None)
            if left != right:
                raise ContactAwareRepairError(
                    "existing v30 policy differs; use a new output root"
                )
            return existing, policy_path
        _atomic_json(policy_path, expected)
        return expected, policy_path
    if not policy_path.is_file():
        raise ContactAwareRepairError(f"{split} repair requires a train-frozen policy")
    policy = _load_json(policy_path, "v30 frozen policy")
    expected_implementation = {
        "path": IMPLEMENTATION_PATH.relative_to(PIPELINE_ROOT).as_posix(),
        "checksum_sha256": _sha256(IMPLEMENTATION_PATH),
    }
    if (
        policy.get("schema_version") != POLICY_SCHEMA_VERSION
        or policy.get("status") != "frozen_from_train"
        or policy.get("repair_version") != REPAIR_VERSION
        or policy.get("implementation") != expected_implementation
        or policy.get("algorithm") != REPAIR_ALGORITHM
        or policy.get("algorithm_checksum_sha256")
        != _canonical_checksum(REPAIR_ALGORITHM)
        or policy.get("resolved_asset") != _resolved_asset_document()
        or policy.get("dependencies") != _dependency_document()
    ):
        raise ContactAwareRepairError("v30 train policy is missing, altered, or unsupported")
    if v28.get("dynamic_policy_checksum_sha256") != policy["train_inputs"].get(
        "v28_dynamic_policy_checksum_sha256"
    ):
        raise ContactAwareRepairError("v28 split uses a different frozen dynamic policy")
    expected_dynamic_policy = str(
        policy["train_inputs"].get("v28_dynamic_policy_checksum_sha256", "")
    )
    if (
        not DEFAULT_DYNAMIC_POLICY_PATH.is_file()
        or _sha256(DEFAULT_DYNAMIC_POLICY_PATH) != expected_dynamic_policy
    ):
        raise ContactAwareRepairError(
            "v28 frozen dynamic policy changed after v30 train freeze"
        )
    return policy, policy_path


def _load_bound_json(
    path: Path, expected_checksum_sha256: str, description: str
) -> tuple[dict[str, Any], str]:
    """Load JSON only when its checksum is stable before and after parsing."""

    if not path.is_file():
        raise ContactAwareRepairError(f"{description} is missing: {path}")
    before = _sha256(path)
    if before != expected_checksum_sha256:
        raise ContactAwareRepairError(f"{description} checksum mismatch")
    document = _load_json(path, description)
    after = _sha256(path)
    if after != before:
        raise ContactAwareRepairError(f"{description} changed while it was loaded")
    return document, after


def _artifact_contract(
    report: Mapping[str, Any],
    *,
    name: str,
    expected_path: Path,
) -> str:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ContactAwareRepairError("validation witness artifacts are invalid")
    artifact = artifacts.get(name)
    if not isinstance(artifact, Mapping):
        raise ContactAwareRepairError(f"validation witness lacks {name} artifact")
    expected_relative = expected_path.relative_to(PIPELINE_ROOT).as_posix()
    checksum = artifact.get("checksum_sha256")
    if artifact.get("path") != expected_relative or not isinstance(checksum, str):
        raise ContactAwareRepairError(
            f"validation witness {name} artifact path/checksum is invalid"
        )
    return checksum


def _strict_counts(document: Mapping[str, Any], description: str) -> dict[str, int]:
    counts = document.get("counts")
    if not isinstance(counts, Mapping):
        raise ContactAwareRepairError(f"{description} counts are invalid")
    try:
        normalized = {
            "source": int(counts["source"]),
            "published": int(counts["published"]),
            "quarantined": int(counts["quarantined"]),
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ContactAwareRepairError(f"{description} counts are invalid") from exc
    if (
        min(normalized.values()) < 0
        or normalized["source"]
        != normalized["published"] + normalized["quarantined"]
    ):
        raise ContactAwareRepairError(f"{description} counts are inconsistent")
    return normalized


def _validate_policy_for_test_unlock(
    policy_path: Path, policy_checksum: str
) -> dict[str, Any]:
    policy, _ = _load_bound_json(
        policy_path, policy_checksum, "train-frozen v30 policy"
    )
    expected_implementation = {
        "path": IMPLEMENTATION_PATH.relative_to(PIPELINE_ROOT).as_posix(),
        "checksum_sha256": _sha256(IMPLEMENTATION_PATH),
    }
    if (
        policy.get("schema_version") != POLICY_SCHEMA_VERSION
        or policy.get("status") != "frozen_from_train"
        or policy.get("repair_version") != REPAIR_VERSION
        or policy.get("implementation") != expected_implementation
        or policy.get("algorithm") != REPAIR_ALGORITHM
        or policy.get("algorithm_checksum_sha256")
        != _canonical_checksum(REPAIR_ALGORITHM)
        or policy.get("resolved_asset") != _resolved_asset_document()
        or policy.get("dependencies") != _dependency_document()
    ):
        raise ContactAwareRepairError(
            "train-frozen v30 policy is altered or unsupported"
        )
    train_inputs = policy.get("train_inputs")
    if not isinstance(train_inputs, Mapping):
        raise ContactAwareRepairError("train-frozen v30 policy inputs are invalid")
    expected_dynamic_checksum = str(
        train_inputs.get("v28_dynamic_policy_checksum_sha256", "")
    )
    if (
        not DEFAULT_DYNAMIC_POLICY_PATH.is_file()
        or _sha256(DEFAULT_DYNAMIC_POLICY_PATH) != expected_dynamic_checksum
    ):
        raise ContactAwareRepairError(
            "upstream v28 frozen dynamic policy changed after v30 train freeze"
        )
    return policy


def _verify_validation_unlock_chain(
    *,
    output_root: Path,
    policy_checksum: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate every current artifact behind a validation-pass witness."""

    policy_path = (output_root / "policy.json").resolve()
    plan_path = (output_root / "g1_validation_plan.json").resolve()
    manifest_path = (output_root / "g1_validation.json").resolve()
    reference_root = (output_root / "g1").resolve()
    policy = _validate_policy_for_test_unlock(policy_path, policy_checksum)

    policy_artifact_checksum = _artifact_contract(
        report, name="policy", expected_path=policy_path
    )
    plan_checksum = _artifact_contract(report, name="plan", expected_path=plan_path)
    manifest_checksum = _artifact_contract(
        report, name="manifest", expected_path=manifest_path
    )
    if policy_artifact_checksum != policy_checksum:
        raise ContactAwareRepairError(
            "validation witness does not bind the current frozen policy"
        )
    artifacts = report["artifacts"]
    if artifacts.get("reference_root") != reference_root.relative_to(
        PIPELINE_ROOT
    ).as_posix():
        raise ContactAwareRepairError("validation witness reference root is invalid")

    plan, plan_checksum = _load_bound_json(
        plan_path, plan_checksum, "validation repair plan"
    )
    manifest, manifest_checksum = _load_bound_json(
        manifest_path, manifest_checksum, "validation repair manifest"
    )
    report_counts = _strict_counts(report, "validation witness")
    plan_counts = _strict_counts(plan, "validation repair plan")
    if report_counts != plan_counts:
        raise ContactAwareRepairError("validation witness/plan counts disagree")
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != "completed"
        or plan.get("split") != "validation"
    ):
        raise ContactAwareRepairError("validation repair plan contract is invalid")
    frozen_policy = plan.get("frozen_policy")
    if not isinstance(frozen_policy, Mapping) or frozen_policy.get(
        "path"
    ) != policy_path.relative_to(PIPELINE_ROOT).as_posix() or frozen_policy.get(
        "checksum_sha256"
    ) != policy_checksum or frozen_policy.get(
        "algorithm_checksum_sha256"
    ) != policy.get(
        "algorithm_checksum_sha256"
    ):
        raise ContactAwareRepairError("validation plan policy chain is invalid")

    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ContactAwareRepairError("validation plan inputs are invalid")
    for stage in ("v27", "v28"):
        commitment = _canonical_commitment(stage, "validation")
        canonical_path = _pipeline_path(
            PIPELINE_ROOT / str(commitment["path"]),
            f"canonical {stage} validation manifest",
        )
        if (
            inputs.get(f"{stage}_manifest_path") != commitment["path"]
            or inputs.get(f"{stage}_manifest_checksum_sha256")
            != commitment["checksum_sha256"]
            or not canonical_path.is_file()
            or _sha256(canonical_path) != commitment["checksum_sha256"]
        ):
            raise ContactAwareRepairError(
                f"validation plan no longer binds canonical {stage} input"
            )

    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("split") != "validation"
        or manifest.get("kind") != "split:validation"
        or manifest.get("reference_root")
        != reference_root.relative_to(PIPELINE_ROOT).as_posix()
        or manifest.get("contact_repair_policy_checksum_sha256") != policy_checksum
        or manifest.get("contact_repair_plan_checksum_sha256") != plan_checksum
    ):
        raise ContactAwareRepairError("validation repair manifest chain is invalid")
    v28_commitment = _canonical_commitment("v28", "validation")
    if (
        manifest.get("source_v28_manifest_checksum_sha256")
        != v28_commitment["checksum_sha256"]
        or manifest.get("source_v28_motion_set_checksum_sha256")
        != v28_commitment["motion_set_checksum_sha256"]
    ):
        raise ContactAwareRepairError(
            "validation manifest does not bind the canonical v28 input"
        )
    try:
        manifest_counts = {
            "source": int(manifest["num_source_motions"]),
            "published": int(manifest["num_motions"]),
            "quarantined": int(manifest["num_quarantined"]),
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ContactAwareRepairError("validation manifest counts are invalid") from exc
    if manifest_counts != report_counts:
        raise ContactAwareRepairError("validation manifest/report counts disagree")

    records = plan.get("records")
    motions = manifest.get("motions")
    if not isinstance(records, list) or len(records) != report_counts["source"]:
        raise ContactAwareRepairError("validation plan records are invalid")
    if not isinstance(motions, list) or len(motions) != report_counts["published"]:
        raise ContactAwareRepairError("validation manifest motions are invalid")
    published_records = [record for record in records if record.get("status") == "published"]
    quarantined_records = [
        record for record in records if record.get("status") == "quarantined"
    ]
    if (
        len(published_records) != report_counts["published"]
        or len(quarantined_records) != report_counts["quarantined"]
        or len(published_records) + len(quarantined_records) != len(records)
    ):
        raise ContactAwareRepairError("validation plan record statuses are inconsistent")
    motion_by_file: dict[str, Mapping[str, Any]] = {}
    for motion in motions:
        if not isinstance(motion, Mapping):
            raise ContactAwareRepairError("validation manifest motion is invalid")
        relative = str(motion.get("g1_file", ""))
        if not relative or relative in motion_by_file:
            raise ContactAwareRepairError(
                "validation manifest contains empty or duplicate archive paths"
            )
        motion_by_file[relative] = motion
    if set(motion_by_file) != {
        str(record.get("g1_file", "")) for record in published_records
    }:
        raise ContactAwareRepairError("validation plan/manifest archive sets disagree")

    total_frames = 0
    total_duration = 0.0
    motion_set_rows: list[list[str]] = []
    for record in published_records:
        relative = str(record["g1_file"])
        motion = motion_by_file[relative]
        derivation = record.get("derivation")
        metadata = derivation.get("serialized_metadata") if isinstance(
            derivation, Mapping
        ) else None
        if not isinstance(derivation, Mapping) or not isinstance(metadata, Mapping):
            raise ContactAwareRepairError(
                "validation plan lacks serialized archive derivation metadata"
            )
        try:
            archive_checksum = str(motion["g1_checksum_sha256"])
            archive_size = int(motion["g1_size_bytes"])
            frame_count = int(motion["frame_count"])
            fps = float(motion["fps"])
            duration = float(motion["duration_sec"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ContactAwareRepairError(
                "validation manifest archive metadata is invalid"
            ) from exc
        if (
            derivation.get("checksum_sha256") != archive_checksum
            or int(derivation.get("bytes", -1)) != archive_size
            or metadata.get("frame_count") != frame_count
            or metadata.get("fps") != fps
            or metadata.get("duration_sec") != duration
            or metadata.get("split") != "validation"
            or motion.get("split") != "validation"
            or motion.get("contact_repair_policy_checksum_sha256")
            != policy_checksum
            or motion.get("contact_repair_exact_v29_gate_complete") is not True
            or motion.get("gate3_kinematic_complete") is not True
        ):
            raise ContactAwareRepairError(
                "validation plan/manifest serialized metadata chain is invalid"
            )
        archive_path = _safe_member(reference_root, relative, "v30 validation archive")
        if (
            not archive_path.is_file()
            or archive_path.stat().st_size != archive_size
            or _sha256(archive_path) != archive_checksum
        ):
            raise ContactAwareRepairError(
                f"validation archive is missing or stale: {relative}"
            )
        archive_arrays = _safe_npz(archive_path)
        if _sha256(archive_path) != archive_checksum:
            raise ContactAwareRepairError(
                f"validation archive changed while it was revalidated: {relative}"
            )
        actual_metadata = _serialized_archive_metadata(
            archive_arrays, split_field="contact_repair_split"
        )
        if (
            actual_metadata != dict(metadata)
            or str(_archive_scalar(archive_arrays, "retarget_version"))
            != REPAIR_VERSION
            or str(
                _archive_scalar(
                    archive_arrays, "contact_repair_policy_checksum_sha256"
                )
            )
            != policy_checksum
            or str(
                _archive_scalar(
                    archive_arrays,
                    "contact_repair_source_v27_manifest_checksum_sha256",
                )
            )
            != _canonical_commitment("v27", "validation")["checksum_sha256"]
            or str(
                _archive_scalar(
                    archive_arrays,
                    "contact_repair_source_v28_manifest_checksum_sha256",
                )
            )
            != v28_commitment["checksum_sha256"]
        ):
            raise ContactAwareRepairError(
                f"validation archive serialized provenance is invalid: {relative}"
            )
        human_path = _pipeline_path(
            PIPELINE_ROOT / str(motion.get("human_file", "")),
            "validation human source archive",
        )
        if (
            not human_path.is_file()
            or _sha256(human_path) != motion.get("human_checksum_sha256")
        ):
            raise ContactAwareRepairError(
                f"validation human source is missing or stale: {relative}"
            )
        from .g1_validation import validate_g1_archive

        gate3 = validate_g1_archive(
            archive_path,
            expected_source_path=human_path,
        )
        if gate3.get("gate3_kinematic_complete") is not True:
            raise ContactAwareRepairError(
                f"validation archive no longer passes independent Gate-3: {relative}"
            )
        total_frames += frame_count
        total_duration += duration
        motion_set_rows.append(["validation", relative, archive_checksum])
    expected_motion_set = hashlib.sha256(
        json.dumps(motion_set_rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        int(manifest.get("num_frames", -1)) != total_frames
        or not math.isclose(
            float(manifest.get("duration_sec", math.nan)),
            total_duration,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or manifest.get("motion_set_checksum_sha256") != expected_motion_set
    ):
        raise ContactAwareRepairError(
            "validation manifest aggregate metadata is invalid"
        )
    return {
        "policy_checksum_sha256": policy_checksum,
        "plan_checksum_sha256": plan_checksum,
        "manifest_checksum_sha256": manifest_checksum,
        "motion_set_checksum_sha256": expected_motion_set,
        "num_archives": len(motions),
    }


def _authorize_test_once(
    *,
    output_root: Path,
    policy_checksum: str,
    witness_path: str | Path | None,
    witness_checksum_sha256: str | None,
) -> dict[str, Any]:
    if witness_path is None or witness_checksum_sha256 is None:
        raise ContactAwareRepairError(
            "test remains sealed: supply an explicit validation-pass witness path and checksum"
        )
    witness = Path(witness_path).expanduser().resolve()
    expected_path = (output_root / "g1_validation_report.json").resolve()
    if witness != expected_path or not witness.is_file():
        raise ContactAwareRepairError("test witness must be this output root's validation report")
    actual_checksum = _sha256(witness)
    if actual_checksum != witness_checksum_sha256:
        raise ContactAwareRepairError("validation-pass witness checksum mismatch")
    document, actual_checksum = _load_bound_json(
        witness, witness_checksum_sha256, "validation-pass witness"
    )
    counts = document.get("counts", {})
    acceptance = document.get("acceptance", {})
    if (
        document.get("schema_version") != REPORT_SCHEMA_VERSION
        or document.get("split") != "validation"
        or document.get("status") != "completed"
        or int(counts.get("quarantined", -1)) != 0
        or int(counts.get("published", -1)) <= 0
        or acceptance.get("ready_for_test") is not True
        or acceptance.get("policy_checksum_sha256") != policy_checksum
    ):
        raise ContactAwareRepairError("validation witness does not authorize test access")
    chain = _verify_validation_unlock_chain(
        output_root=output_root,
        policy_checksum=policy_checksum,
        report=document,
    )
    receipt_path = output_root / "test_consumption_receipt.json"
    receipt = {
        "schema_version": "accad_g1_contact_aware_test_consumption_v1",
        "status": "test_access_committed_before_manifest_open",
        "created_at_utc": _utc_now(),
        "validation_witness_path": witness.relative_to(PIPELINE_ROOT).as_posix(),
        "validation_witness_checksum_sha256": actual_checksum,
        "policy_checksum_sha256": policy_checksum,
        "validation_chain": chain,
    }
    _exclusive_json(receipt_path, receipt)
    return receipt


def build_contact_aware_repair_split(
    *,
    split: str,
    v27_manifest_path: str | Path | None = None,
    v28_manifest_path: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
    validation_pass_witness_path: str | Path | None = None,
    validation_pass_witness_checksum_sha256: str | None = None,
) -> dict[str, Any]:
    """Build exactly one explicit split without any implicit cross-split scan."""

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be exactly train, validation, or test")
    output = assert_output_is_local(output_root)
    plan_path = output / f"g1_{split}_plan.json"
    report_path = output / f"g1_{split}_report.json"
    manifest_path = output / f"g1_{split}.json"
    existing = [path for path in (plan_path, report_path, manifest_path) if path.exists()]
    if existing and (not force or split == "test"):
        raise ContactAwareRepairError(
            "v30 outputs already exist; test is never force-reopened: "
            + ", ".join(str(path) for path in existing)
        )

    if split != "train" and (
        v27_manifest_path is not None or v28_manifest_path is not None
    ):
        raise ContactAwareRepairError(
            f"{split} manifest overrides are forbidden; use the precommitted "
            "canonical v27/v28 defaults"
        )
    if split == "validation":
        # Fail before opening either manifest when the source-code commitment
        # is incomplete.
        _canonical_commitment("v27", split)
        _canonical_commitment("v28", split)

    # For train/validation this selection performs no operation involving the
    # test defaults.  For test, authorization is checked before either test
    # manifest path is resolved or opened.
    test_authority: object | None = None
    if split == "test":
        policy_path = output / "policy.json"
        if not policy_path.is_file():
            raise ContactAwareRepairError("test requires a train-frozen v30 policy")
        policy_checksum_before_open = _sha256(policy_path)
        # A missing public v28 test commitment must not consume the one-shot
        # receipt and must not open a test manifest or archive.
        _canonical_commitment("v27", split)
        _canonical_commitment("v28", split)
        _authorize_test_once(
            output_root=output,
            policy_checksum=policy_checksum_before_open,
            witness_path=validation_pass_witness_path,
            witness_checksum_sha256=validation_pass_witness_checksum_sha256,
        )
        test_authority = _TEST_AUTHORITY
    v27_path = _pipeline_path(
        DEFAULT_V27_MANIFESTS[split]
        if v27_manifest_path is None
        else v27_manifest_path,
        f"v27 {split} manifest",
    )
    v28_path = _pipeline_path(
        DEFAULT_V28_MANIFESTS[split]
        if v28_manifest_path is None
        else v28_manifest_path,
        f"v28 {split} manifest",
    )
    v27, v28 = _load_split_manifests(split, v27_path, v28_path)
    if split != "train":
        _verify_canonical_manifest_commitment(
            stage="v27", split=split, path=v27_path, document=v27
        )
        _verify_canonical_manifest_commitment(
            stage="v28", split=split, path=v28_path, document=v28
        )
    policy, policy_path = _load_or_freeze_policy(
        split=split,
        v27_path=v27_path,
        v28_path=v28_path,
        v27=v27,
        v28=v28,
        output_root=output,
    )
    policy_checksum = _sha256(policy_path)
    if split == "test" and policy_checksum != policy_checksum_before_open:
        raise ContactAwareRepairError("frozen policy changed during test authorization")
    v27_root = (PIPELINE_ROOT / str(v27["reference_root"])).resolve()
    v28_root = (PIPELINE_ROOT / str(v28["reference_root"])).resolve()
    for root, label in ((v27_root, "v27"), (v28_root, "v28")):
        try:
            root.relative_to(PIPELINE_ROOT)
        except ValueError as exc:
            raise ContactAwareRepairError(f"{label} reference root escapes pipeline") from exc
        if not root.is_dir():
            raise ContactAwareRepairError(f"{label} reference root is missing: {root}")
    derived_root = output / "g1"
    if derived_root.resolve() in {v27_root, v28_root}:
        raise ContactAwareRepairError("v30 reference root aliases an immutable input root")
    v27_by_file = {str(record["g1_file"]): record for record in v27["motions"]}
    if len(v27_by_file) != len(v27["motions"]):
        raise ContactAwareRepairError("v27 manifest contains duplicate g1_file entries")
    v28_files = [str(record["g1_file"]) for record in v28["motions"]]
    if len(v28_files) != len(set(v28_files)):
        raise ContactAwareRepairError("v28 manifest contains duplicate g1_file entries")
    v28_source_files = [
        str(record.get("source_v27_g1_file", record["g1_file"]))
        for record in v28["motions"]
    ]
    if len(v28_source_files) != len(set(v28_source_files)):
        raise ContactAwareRepairError(
            "v28 manifest contains duplicate source_v27_g1_file entries"
        )

    records: list[dict[str, Any]] = []
    published: list[dict[str, Any]] = []
    started = _utc_now()
    for candidate_record in v28["motions"]:
        relative_text = str(candidate_record["g1_file"])
        if relative_text not in v27_by_file:
            raise ContactAwareRepairError(f"v28 motion has no v27 source: {relative_text}")
        if str(candidate_record.get("source_v27_g1_file", relative_text)) != relative_text:
            raise ContactAwareRepairError(
                f"v28 source_v27_g1_file differs from g1_file: {relative_text}"
            )
        source_record = v27_by_file[relative_text]
        source_path = _safe_member(v27_root, relative_text, "v27 archive")
        candidate_path = _safe_member(v28_root, relative_text, "v28 archive")
        for path, record, label in (
            (source_path, source_record, "v27"),
            (candidate_path, candidate_record, "v28"),
        ):
            if not path.is_file():
                raise ContactAwareRepairError(f"{label} archive is missing: {path}")
            if _sha256(path) != record.get("g1_checksum_sha256"):
                raise ContactAwareRepairError(f"{label} checksum mismatch: {relative_text}")
            if path.stat().st_size != int(record.get("g1_size_bytes", -1)):
                raise ContactAwareRepairError(f"{label} size mismatch: {relative_text}")
        if candidate_record.get("source_v27_g1_checksum_sha256") != source_record.get(
            "g1_checksum_sha256"
        ):
            raise ContactAwareRepairError(f"v28 record does not bind v27: {relative_text}")
        output_path = assert_output_is_local(derived_root / relative_text)
        if output_path.exists() and not force:
            raise ContactAwareRepairError(f"v30 archive already exists: {output_path}")
        human_path = _pipeline_path(
            PIPELINE_ROOT / str(source_record["human_file"]),
            "v27 human source archive",
        )
        if (
            not human_path.is_file()
            or _sha256(human_path) != source_record.get("human_checksum_sha256")
        ):
            raise ContactAwareRepairError(
                f"v27 human source is missing or stale: {relative_text}"
            )
        item: dict[str, Any] = {
            "g1_file": relative_text,
            "source_v27_checksum_sha256": source_record["g1_checksum_sha256"],
            "source_v28_checksum_sha256": candidate_record["g1_checksum_sha256"],
        }
        try:
            derivation = derive_contact_repaired_archive(
                source_path,
                candidate_path,
                output_path,
                split=split,
                source_v27_manifest_checksum_sha256=_sha256(v27_path),
                source_v28_manifest_checksum_sha256=_sha256(v28_path),
                policy_checksum_sha256=policy_checksum,
                expected_human_path=human_path,
                expected_candidate_record=candidate_record,
                _test_authority=test_authority,
            )
        except ContactAwareRepairQuarantine as exc:
            item.update(
                {
                    "status": "quarantined",
                    "final_action": "quarantine_v30_contact_aware_repair",
                    "quarantine_reason": exc.reason,
                    "quarantine_evidence": exc.evidence,
                }
            )
            records.append(item)
            continue
        item.update(
            {
                "status": "published",
                "final_action": "publish_v30_contact_aware_reference",
                "derivation": derivation,
            }
        )
        records.append(item)
        serialized_metadata = derivation["serialized_metadata"]
        published.append(
            {
                **candidate_record,
                "source_v28_g1_file": relative_text,
                "source_v28_g1_checksum_sha256": candidate_record[
                    "g1_checksum_sha256"
                ],
                "g1_file": relative_text,
                "g1_checksum_sha256": derivation["checksum_sha256"],
                "g1_size_bytes": derivation["bytes"],
                "frame_count": serialized_metadata["frame_count"],
                "fps": serialized_metadata["fps"],
                "duration_sec": serialized_metadata["duration_sec"],
                "split": serialized_metadata["split"],
                "retarget_version": REPAIR_VERSION,
                "contact_repair_policy_checksum_sha256": policy_checksum,
                "contact_repair_exact_v29_gate_complete": True,
                "gate3_kinematic_complete": True,
            }
        )

    records.sort(key=lambda record: record["g1_file"])
    published.sort(key=lambda record: record["g1_file"])
    reason_counts: dict[str, int] = {}
    for record in records:
        if record["status"] == "quarantined":
            reason = str(record["quarantine_reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    counts = {
        "source": len(records),
        "published": len(published),
        "quarantined": len(records) - len(published),
        "quarantine_reason_counts": dict(sorted(reason_counts.items())),
    }
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "completed",
        "split": split,
        "created_at_utc": _utc_now(),
        "inputs": {
            "v27_manifest_path": v27_path.relative_to(PIPELINE_ROOT).as_posix(),
            "v27_manifest_checksum_sha256": _sha256(v27_path),
            "v28_manifest_path": v28_path.relative_to(PIPELINE_ROOT).as_posix(),
            "v28_manifest_checksum_sha256": _sha256(v28_path),
        },
        "frozen_policy": {
            "path": policy_path.relative_to(PIPELINE_ROOT).as_posix(),
            "checksum_sha256": policy_checksum,
            "algorithm_checksum_sha256": policy["algorithm_checksum_sha256"],
        },
        "counts": counts,
        "records": records,
    }
    _atomic_json(plan_path, plan)
    motion_set_checksum = hashlib.sha256(
        json.dumps(
            [[split, record["g1_file"], record["g1_checksum_sha256"]] for record in published],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "ACCAD",
        "kind": f"split:{split}",
        "split": split,
        "gate": "gate3_kinematic+dynamic_v28+exact_contact_v29+repair_v30",
        "reference_root": derived_root.relative_to(PIPELINE_ROOT).as_posix(),
        "retarget_version": REPAIR_VERSION,
        "motion_set_checksum_sha256": motion_set_checksum,
        "source_v28_motion_set_checksum_sha256": v28.get("motion_set_checksum_sha256"),
        "source_v28_manifest_checksum_sha256": _sha256(v28_path),
        "contact_repair_policy_checksum_sha256": policy_checksum,
        "contact_repair_plan_checksum_sha256": _sha256(plan_path),
        "num_source_motions": len(records),
        "num_motions": len(published),
        "num_quarantined": len(records) - len(published),
        "num_frames": sum(int(record["frame_count"]) for record in published),
        "duration_sec": sum(float(record["duration_sec"]) for record in published),
        "motions": published,
    }
    _atomic_json(manifest_path, manifest)
    ready_for_test = split == "validation" and counts["published"] > 0 and counts[
        "quarantined"
    ] == 0
    report_status = _report_status(split, counts)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": report_status,
        "split": split,
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "counts": counts,
        "acceptance": {
            "ready_for_test": ready_for_test,
            "accepted": report_status == "completed",
            "policy_checksum_sha256": policy_checksum,
            "requires_all_validation_candidates": True,
        },
        "artifacts": {
            "policy": {
                "path": policy_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": policy_checksum,
            },
            "plan": {
                "path": plan_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": _sha256(plan_path),
            },
            "manifest": {
                "path": manifest_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": _sha256(manifest_path),
            },
            "reference_root": derived_root.relative_to(PIPELINE_ROOT).as_posix(),
        },
        "sealed_test_contract": (
            "test access requires an exact successful validation report checksum and "
            "is committed before opening the test manifest"
        ),
        "test_opened_by_this_invocation": split == "test",
    }
    _atomic_json(report_path, report)
    return report


__all__ = [
    "CANONICAL_MANIFEST_REGISTRY",
    "ContactAwareRepairError",
    "ContactAwareRepairQuarantine",
    "DEFAULT_OUTPUT_ROOT",
    "LEG_JOINT_NAMES",
    "MANIFEST_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "POLICY_SCHEMA_VERSION",
    "REPAIR_ALGORITHM",
    "REPAIR_SCHEMA_VERSION",
    "REPAIR_VERSION",
    "REPORT_SCHEMA_VERSION",
    "WAIST_BALANCE_JOINT_NAMES",
    "audit_contact_anchor_postconditions",
    "build_contact_aware_repair_split",
    "differentiable_centroidal_signals",
    "derive_contact_repaired_archive",
    "derive_contact_repaired_payload",
    "inertial_com_torch",
    "plan_repair_targets",
]
