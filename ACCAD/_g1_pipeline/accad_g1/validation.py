"""Numeric Gate-2 validation for canonical ACCAD human motion archives."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .human import (
    KINEMATIC_JOINT_COUNT,
    NUM_BETAS,
    _angular_velocity_world,
    _linear_velocity,
    _remove_short_true_runs,
    _resample_boolean_majority,
    load_stageii_numeric,
)
from .math3d import quaternion_wxyz_to_matrix


CONTACT_KEYS = (
    "left_heel_contact",
    "left_toe_contact",
    "right_heel_contact",
    "right_toe_contact",
)
SOURCE_CONTACT_KEYS = tuple(f"source_{name}" for name in CONTACT_KEYS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _maximum_true_run(values: np.ndarray) -> int:
    padded = np.concatenate(([False], np.asarray(values, dtype=bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return int(np.max(stops - starts)) if len(starts) else 0


def _minimum_true_run(values: np.ndarray) -> int:
    padded = np.concatenate(([False], np.asarray(values, dtype=bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return int(np.min(stops - starts)) if len(starts) else 0


def _rotation_error_rad(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(expected, -1, -2) @ actual
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def validate_human_archive(
    archive_path: str | Path,
    accad_root: str | Path,
    *,
    profile: str = "generic",
) -> dict[str, Any]:
    """Validate one generated human archive without trusting serialized flags.

    ``profile='b3-walk'`` adds walking-only requirements.  Generic ACCAD clips
    may legitimately jump, roll, or lie on the floor, so contact coverage and
    pelvis height are not generic hard gates.
    """

    if profile not in {"generic", "b3-walk"}:
        raise ValueError(f"unsupported validation profile: {profile!r}")
    path = Path(archive_path).expanduser().resolve()
    raw_root = Path(accad_root).expanduser().resolve()
    checks: dict[str, list[dict[str, Any]]] = {
        "schema": [],
        "numeric": [],
        "contact_consistency": [],
        "semantic_coordinate": [],
        "provenance_120hz": [],
        "profile": [],
    }
    warnings: list[str] = []

    def check(group: str, name: str, passed: bool, **details: Any) -> None:
        checks[group].append({"name": name, "passed": bool(passed), **details})

    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays: dict[str, np.ndarray] = {}
            unreadable: dict[str, str] = {}
            for name in archive.files:
                try:
                    arrays[name] = archive[name]
                except ValueError as exc:
                    unreadable[name] = str(exc)
    except (OSError, ValueError) as exc:
        return {
            "ok": False,
            "gate2_complete": False,
            "archive_path": str(path),
            "profile": profile,
            "errors": [f"cannot open archive safely: {exc}"],
            "warnings": warnings,
            "checks": checks,
        }

    required = {
        "schema_version",
        "processing_version",
        "coordinate_system",
        "basis_transform_source_to_world",
        "ground_normal_w",
        "quaternion_order",
        "position_unit",
        "angle_unit",
        "fps",
        "source_file",
        "source_checksum_sha256",
        "source_frame_count",
        "source_fps",
        "source_mocap_time_length",
        "target_frame_count",
        "target_fps",
        "time",
        "gender",
        "surface_model_type",
        "smplx_version",
        "smplx_model_path",
        "smplx_model_checksum_sha256",
        "smplx_use_pca",
        "smplx_num_betas",
        "smplx_flat_hand_mean",
        "betas",
        "root_translation",
        "root_position_w",
        "root_quaternion_wxyz",
        "root_linear_velocity_w",
        "root_angular_velocity_w",
        "joint_names",
        "all_joint_names",
        "all_joint_position_w",
        "rotation_joint_names",
        "joint_position_w",
        "joint_quaternion_wxyz",
        "joint_local_quaternion_wxyz",
        "joint_linear_velocity_w",
        "joint_angular_velocity_w",
        "parents",
        "validated_mapping_names",
        "validated_mapping_indices",
        "ground_height_w",
        "ground_baseline_height_w",
        "ground_baseline_confidence",
        "ground_correction_z_w",
        "source_fk_performed",
        "source_fk_frame_count",
        "source_ground_baseline_height_w",
        "source_ground_baseline_confidence",
        "source_ground_baseline_drift_m",
        "contact_source_fps",
        "contact_keypoint_names",
        "contact_keypoint_position_w",
        "contact_resampling",
        *CONTACT_KEYS,
        *SOURCE_CONTACT_KEYS,
    }
    missing = sorted(required.difference(arrays))
    check("schema", "required_keys", not missing, missing=missing)
    check("schema", "safe_no_pickle_load", not unreadable, unreadable=unreadable)
    object_fields = sorted(name for name, value in arrays.items() if value.dtype.hasobject)
    check("schema", "no_object_dtype", not object_fields, fields=object_fields)
    finite_failures = sorted(
        name
        for name, value in arrays.items()
        if value.dtype.kind in {"f", "i", "u"} and not np.isfinite(value).all()
    )
    check("schema", "all_numeric_finite", not finite_failures, fields=finite_failures)
    if missing or unreadable or object_fields:
        statuses = {name: all(item["passed"] for item in items) for name, items in checks.items()}
        return {
            "ok": False,
            "gate2_complete": False,
            "archive_path": str(path),
            "profile": profile,
            "statuses": statuses,
            "errors": ["archive schema is incomplete or unsafe"],
            "warnings": warnings,
            "checks": checks,
        }

    def scalar(name: str) -> Any:
        value = arrays[name]
        if value.shape != ():
            raise ValueError(f"{name} must be scalar; got {value.shape}")
        return value.item()

    scalar_names = (
        "schema_version",
        "processing_version",
        "coordinate_system",
        "quaternion_order",
        "position_unit",
        "angle_unit",
        "fps",
        "source_file",
        "source_checksum_sha256",
        "source_frame_count",
        "source_fps",
        "source_mocap_time_length",
        "target_frame_count",
        "target_fps",
        "gender",
        "surface_model_type",
        "smplx_version",
        "smplx_model_path",
        "smplx_model_checksum_sha256",
        "smplx_use_pca",
        "smplx_num_betas",
        "smplx_flat_hand_mean",
        "ground_height_w",
        "source_fk_performed",
        "source_fk_frame_count",
        "source_ground_baseline_drift_m",
        "contact_source_fps",
        "contact_resampling",
    )
    non_scalars = {name: list(arrays[name].shape) for name in scalar_names if arrays[name].shape != ()}
    check("schema", "metadata_scalars", not non_scalars, fields=non_scalars)
    if non_scalars:
        statuses = {name: all(item["passed"] for item in items) for name, items in checks.items()}
        return {
            "ok": False,
            "gate2_complete": False,
            "archive_path": str(path),
            "profile": profile,
            "statuses": statuses,
            "errors": ["metadata scalar contract failed"],
            "warnings": warnings,
            "checks": checks,
        }

    frame_count = int(scalar("target_frame_count"))
    source_frame_count = int(scalar("source_frame_count"))
    fps = float(scalar("fps"))
    source_fps = float(scalar("source_fps"))
    joint_count = len(arrays["joint_names"])
    all_joint_count = len(arrays["all_joint_names"])

    fixed_metadata = {
        "schema_version": "accad_human_reconstruction_v1",
        "processing_version": "0.3.0",
        "coordinate_system": "right-handed_z-up",
        "quaternion_order": "wxyz",
        "position_unit": "meter",
        "angle_unit": "radian",
        "gender": "neutral",
        "surface_model_type": "smplx",
        "smplx_version": "0.1.28",
    }
    metadata_mismatch = {
        name: {"actual": scalar(name), "expected": expected}
        for name, expected in fixed_metadata.items()
        if scalar(name) != expected
    }
    check("schema", "fixed_conventions", not metadata_mismatch, mismatch=metadata_mismatch)
    check(
        "schema",
        "model_configuration",
        scalar("smplx_use_pca") is False
        and int(scalar("smplx_num_betas")) == NUM_BETAS
        and scalar("smplx_flat_hand_mean") is True,
    )

    expected_shapes = {
        "time": (frame_count,),
        "betas": (NUM_BETAS,),
        "basis_transform_source_to_world": (3, 3),
        "ground_normal_w": (3,),
        "root_translation": (frame_count, 3),
        "ground_correction_z_w": (frame_count,),
        "root_position_w": (frame_count, 3),
        "root_quaternion_wxyz": (frame_count, 4),
        "root_linear_velocity_w": (frame_count, 3),
        "root_angular_velocity_w": (frame_count, 3),
        "joint_names": (KINEMATIC_JOINT_COUNT,),
        "rotation_joint_names": (KINEMATIC_JOINT_COUNT,),
        "all_joint_names": (127,),
        "all_joint_position_w": (frame_count, 127, 3),
        "joint_position_w": (frame_count, KINEMATIC_JOINT_COUNT, 3),
        "joint_quaternion_wxyz": (frame_count, KINEMATIC_JOINT_COUNT, 4),
        "joint_local_quaternion_wxyz": (frame_count, KINEMATIC_JOINT_COUNT, 4),
        "joint_linear_velocity_w": (frame_count, KINEMATIC_JOINT_COUNT, 3),
        "joint_angular_velocity_w": (frame_count, KINEMATIC_JOINT_COUNT, 3),
        "parents": (KINEMATIC_JOINT_COUNT,),
        "ground_baseline_height_w": (frame_count,),
        "ground_baseline_confidence": (frame_count,),
        "contact_keypoint_names": (4,),
        "contact_keypoint_position_w": (frame_count, 4, 3),
        "source_ground_baseline_height_w": (source_frame_count,),
        "source_ground_baseline_confidence": (source_frame_count,),
    }
    expected_shapes.update({name: (frame_count,) for name in CONTACT_KEYS})
    expected_shapes.update({name: (source_frame_count,) for name in SOURCE_CONTACT_KEYS})
    shape_failures = {
        name: {"actual": list(arrays[name].shape), "expected": list(shape)}
        for name, shape in expected_shapes.items()
        if arrays[name].shape != shape
    }
    check("schema", "canonical_shapes", not shape_failures, failures=shape_failures)
    bool_fields = (*CONTACT_KEYS, *SOURCE_CONTACT_KEYS, "ground_baseline_confidence", "source_ground_baseline_confidence")
    wrong_bool = sorted(name for name in bool_fields if arrays[name].dtype != np.dtype(bool))
    check("schema", "contact_and_confidence_bool_dtype", not wrong_bool, fields=wrong_bool)
    check(
        "schema",
        "canonical_counts",
        frame_count >= 2
        and source_frame_count >= 2
        and joint_count == KINEMATIC_JOINT_COUNT
        and all_joint_count == 127,
        target_frames=frame_count,
        source_frames=source_frame_count,
        joint_count=joint_count,
        all_joint_count=all_joint_count,
    )
    if shape_failures:
        statuses = {name: all(item["passed"] for item in items) for name, items in checks.items()}
        return {
            "ok": False,
            "gate2_complete": False,
            "archive_path": str(path),
            "profile": profile,
            "statuses": statuses,
            "errors": ["canonical shape contract failed"],
            "warnings": warnings,
            "checks": checks,
        }

    time = np.asarray(arrays["time"], dtype=np.float64)
    expected_time = np.arange(frame_count, dtype=np.float64) / fps
    time_error = float(np.max(np.abs(time - expected_time)))
    expected_target_count = int(
        np.floor(((source_frame_count - 1) / source_fps) * fps + 1.0e-9)
    ) + 1
    check(
        "schema",
        "time_grid_and_count_policy",
        fps == 50.0
        and float(scalar("target_fps")) == fps
        and source_fps == 120.0
        and float(scalar("contact_source_fps")) == source_fps
        and frame_count == expected_target_count
        and time_error <= 1.0e-12,
        maximum_time_error_seconds=time_error,
        expected_target_frames=expected_target_count,
    )

    joint_names = [str(value) for value in arrays["joint_names"].tolist()]
    rotation_names = [str(value) for value in arrays["rotation_joint_names"].tolist()]
    all_names = [str(value) for value in arrays["all_joint_names"].tolist()]
    names_ok = (
        len(set(all_names)) == len(all_names)
        and joint_names == rotation_names
        and joint_names == all_names[:KINEMATIC_JOINT_COUNT]
        and joint_names[0] == "pelvis"
    )
    check("schema", "joint_name_contract", names_ok)
    mapping_names = [str(value) for value in arrays["validated_mapping_names"].tolist()]
    mapping_indices = np.asarray(arrays["validated_mapping_indices"], dtype=np.int64)
    mapping_ok = (
        len(mapping_names) == len(mapping_indices)
        and len(set(mapping_names)) == len(mapping_names)
        and all(
            0 <= int(index) < len(all_names) and all_names[int(index)] == name
            for name, index in zip(mapping_names, mapping_indices, strict=True)
        )
    )
    check("schema", "validated_semantic_mapping", mapping_ok)

    parents = np.asarray(arrays["parents"], dtype=np.int64)
    tree_ok = np.count_nonzero(parents < 0) == 1 and int(parents[0]) < 0
    for child in range(len(parents)):
        seen: set[int] = set()
        node = child
        while node >= 0 and tree_ok:
            if node >= len(parents) or node in seen:
                tree_ok = False
                break
            seen.add(node)
            node = int(parents[node])
    check("schema", "parent_tree", tree_ok)

    quaternions = {
        "root": np.asarray(arrays["root_quaternion_wxyz"], dtype=np.float64),
        "global": np.asarray(arrays["joint_quaternion_wxyz"], dtype=np.float64),
        "local": np.asarray(arrays["joint_local_quaternion_wxyz"], dtype=np.float64),
    }
    norm_errors = {
        name: float(np.max(np.abs(np.linalg.norm(value, axis=-1) - 1.0)))
        for name, value in quaternions.items()
    }
    minimum_adjacent_dot = {
        name: float(np.min(np.sum(value[1:] * value[:-1], axis=-1)))
        for name, value in quaternions.items()
    }
    check(
        "numeric",
        "quaternion_norm",
        max(norm_errors.values()) <= 1.0e-5,
        maximum_errors=norm_errors,
        tolerance=1.0e-5,
    )
    check(
        "numeric",
        "quaternion_sign_continuity",
        min(minimum_adjacent_dot.values()) >= -1.0e-6,
        minimum_adjacent_dot=minimum_adjacent_dot,
        tolerance=-1.0e-6,
    )

    local_matrices = quaternion_wxyz_to_matrix(quaternions["local"])
    global_matrices = quaternion_wxyz_to_matrix(quaternions["global"])
    maximum_fk_rotation_error = 0.0
    if tree_ok:
        for child, parent in enumerate(parents):
            expected = (
                local_matrices[:, child]
                if parent < 0
                else global_matrices[:, parent] @ local_matrices[:, child]
            )
            maximum_fk_rotation_error = max(
                maximum_fk_rotation_error,
                float(np.max(_rotation_error_rad(expected, global_matrices[:, child]))),
            )
    check(
        "numeric",
        "global_local_rotation_fk",
        tree_ok and maximum_fk_rotation_error <= 1.0e-5,
        maximum_error_rad=maximum_fk_rotation_error,
        tolerance_rad=1.0e-5,
    )

    positions = np.asarray(arrays["joint_position_w"], dtype=np.float64)
    maximum_bone_range = 0.0
    maximum_local_offset_range = 0.0
    if tree_ok:
        for child, parent in enumerate(parents):
            if parent < 0:
                continue
            offset_world = positions[:, child] - positions[:, parent]
            bone_length = np.linalg.norm(offset_world, axis=-1)
            maximum_bone_range = max(maximum_bone_range, float(np.ptp(bone_length)))
            offset_local = np.einsum(
                "tij,tj->ti",
                np.swapaxes(global_matrices[:, parent], -1, -2),
                offset_world,
            )
            maximum_local_offset_range = max(
                maximum_local_offset_range,
                float(np.max(np.ptp(offset_local, axis=0))),
            )
    check(
        "numeric",
        "rigid_bone_geometry",
        tree_ok
        and maximum_bone_range <= 1.0e-4
        and maximum_local_offset_range <= 1.0e-4,
        maximum_bone_length_range_m=maximum_bone_range,
        maximum_parent_local_offset_range_m=maximum_local_offset_range,
        tolerance_m=1.0e-4,
    )

    velocity_errors = {
        "root_linear": float(
            np.max(
                np.abs(
                    arrays["root_linear_velocity_w"]
                    - _linear_velocity(arrays["root_position_w"], fps)
                )
            )
        ),
        "root_angular": float(
            np.max(
                np.abs(
                    arrays["root_angular_velocity_w"]
                    - _angular_velocity_world(arrays["root_quaternion_wxyz"], fps)
                )
            )
        ),
        "joint_linear": float(
            np.max(
                np.abs(
                    arrays["joint_linear_velocity_w"]
                    - _linear_velocity(arrays["joint_position_w"], fps)
                )
            )
        ),
        "joint_angular": float(
            np.max(
                np.abs(
                    arrays["joint_angular_velocity_w"]
                    - _angular_velocity_world(arrays["joint_quaternion_wxyz"], fps)
                )
            )
        ),
    }
    velocity_ok = (
        max(velocity_errors["root_linear"], velocity_errors["joint_linear"])
        <= 1.0e-5
        and max(velocity_errors["root_angular"], velocity_errors["joint_angular"])
        <= 1.0e-4
    )
    check(
        "numeric",
        "stored_velocities",
        velocity_ok,
        maximum_absolute_errors=velocity_errors,
    )

    alias_ok = (
        np.array_equal(arrays["root_position_w"], arrays["joint_position_w"][:, 0])
        and np.array_equal(arrays["root_quaternion_wxyz"], arrays["joint_quaternion_wxyz"][:, 0])
        and np.array_equal(arrays["root_linear_velocity_w"], arrays["joint_linear_velocity_w"][:, 0])
        and np.array_equal(arrays["root_angular_velocity_w"], arrays["joint_angular_velocity_w"][:, 0])
        and np.array_equal(arrays["joint_position_w"], arrays["all_joint_position_w"][:, :KINEMATIC_JOINT_COUNT])
    )
    check("numeric", "root_and_joint_aliases", alias_ok)

    keypoint_names = [str(value) for value in arrays["contact_keypoint_names"].tolist()]
    expected_keypoint_names = ["left_heel", "left_toe", "right_heel", "right_toe"]
    name_index = {name: all_names.index(name) for name in all_names}
    expected_keypoints = np.stack(
        (
            arrays["all_joint_position_w"][:, name_index["left_heel"]],
            0.5
            * (
                arrays["all_joint_position_w"][:, name_index["left_big_toe"]]
                + arrays["all_joint_position_w"][:, name_index["left_small_toe"]]
            ),
            arrays["all_joint_position_w"][:, name_index["right_heel"]],
            0.5
            * (
                arrays["all_joint_position_w"][:, name_index["right_big_toe"]]
                + arrays["all_joint_position_w"][:, name_index["right_small_toe"]]
            ),
        ),
        axis=1,
    )
    keypoint_error = float(
        np.max(np.abs(expected_keypoints - arrays["contact_keypoint_position_w"]))
    )
    check(
        "contact_consistency",
        "semantic_contact_keypoints",
        keypoint_names == expected_keypoint_names and keypoint_error <= 1.0e-6,
        maximum_position_error_m=keypoint_error,
    )

    replay_failures: dict[str, int] = {}
    minimum_contact_frames = max(3, int(round(0.06 * fps)))
    for target_name, source_name in zip(CONTACT_KEYS, SOURCE_CONTACT_KEYS, strict=True):
        replay = _remove_short_true_runs(
            _resample_boolean_majority(
                arrays[source_name], source_fps, time, fps
            ),
            minimum_contact_frames,
        )
        mismatch = int(np.count_nonzero(replay != arrays[target_name]))
        if mismatch:
            replay_failures[target_name] = mismatch
    check(
        "contact_consistency",
        "source_contact_majority_resampling",
        not replay_failures,
        mismatched_frames=replay_failures,
    )

    keypoint_velocity = _linear_velocity(arrays["contact_keypoint_position_w"], fps)
    keypoint_speed = np.linalg.norm(keypoint_velocity, axis=-1)
    clearance = (
        arrays["contact_keypoint_position_w"][:, :, 2]
        - arrays["ground_baseline_height_w"][:, None]
    )
    contact_metrics: dict[str, Any] = {}
    contact_quality_ok = True
    absent_contact_channels: list[str] = []
    for index, name in enumerate(CONTACT_KEYS):
        active = arrays[name]
        if np.any(active):
            height_p95 = float(np.quantile(clearance[active, index], 0.95))
            speed_p95 = float(np.quantile(keypoint_speed[active, index], 0.95))
        else:
            height_p95 = float("inf")
            speed_p95 = float("inf")
            absent_contact_channels.append(name)
        minimum_run = _minimum_true_run(active)
        contact_metrics[name] = {
            "frames": int(np.count_nonzero(active)),
            "minimum_true_run_frames": minimum_run,
            "active_height_p95_m": height_p95,
            "active_speed_p95_mps": speed_p95,
        }
        # A generic ACCAD clip is not necessarily an upright heel-to-toe
        # locomotion clip.  Crawling, toe pivots and airborne motions may have
        # no valid heel contact at all.  An active channel still has to satisfy
        # the same temporal and geometric quality checks; only the requirement
        # that *every* channel be present belongs to the walking profile.
        if np.any(active):
            contact_quality_ok &= (
                minimum_run >= minimum_contact_frames
                and height_p95 <= 0.07
                and speed_p95 <= 0.60
            )
        elif profile == "b3-walk":
            contact_quality_ok = False
    minimum_clearance = float(np.min(clearance))
    active_mask = np.stack(
        [np.asarray(arrays[name], dtype=bool) for name in CONTACT_KEYS], axis=1
    )
    if np.any(active_mask):
        active_clearance = clearance[active_mask]
        minimum_active_clearance = float(np.min(active_clearance))
        active_clearance_q01 = float(np.quantile(active_clearance, 0.01))
    else:
        minimum_active_clearance = None
        active_clearance_q01 = None
    if profile == "b3-walk":
        # The dedicated walking smoke test is deliberately strict.
        contact_quality_ok &= minimum_clearance >= -0.02
    elif active_clearance_q01 is not None:
        # SMPL-X toe/heel markers and a fitted ground baseline are estimates,
        # not collision geometry.  Reject sustained penetration while allowing
        # a few noisy boundary frames; exact G1 collision spheres are checked
        # independently and strictly at Gate 3.
        # SMPL-X heel/toe keypoints and the fitted ground baseline are not
        # collision shapes.  ACCAD's running, crouching and lying clips can
        # carry a few centimetres of fitting bias even when their contact runs
        # are otherwise temporally and geometrically sound.  Keep this generic
        # reconstruction gate separate from both the strict B3 walking smoke
        # profile above (-0.02 m) and Gate-3's exact G1 collision spheres.
        contact_quality_ok &= active_clearance_q01 >= -0.05
    check(
        "contact_consistency",
        "contact_geometry_and_runs",
        contact_quality_ok,
        minimum_clearance_m=minimum_clearance,
        minimum_active_clearance_m=minimum_active_clearance,
        active_clearance_q01_m=active_clearance_q01,
        absent_contact_channels=absent_contact_channels,
        all_channels_required=profile == "b3-walk",
        per_keypoint=contact_metrics,
    )

    basis = np.asarray(arrays["basis_transform_source_to_world"], dtype=np.float64)
    ground_normal = np.asarray(arrays["ground_normal_w"], dtype=np.float64)
    pelvis_index = all_names.index("pelvis")
    head_index = all_names.index("head")
    head_offset = arrays["all_joint_position_w"][:, head_index] - arrays["all_joint_position_w"][:, pelvis_index]
    head_above_fraction = float(np.mean(head_offset[:, 2] > 0.0))
    median_head_offset = np.median(head_offset, axis=0)
    head_distance = np.linalg.norm(head_offset, axis=-1)
    median_head_distance = float(np.median(head_distance))
    pose_independent_semantics_ok = bool(
        np.isfinite(head_distance).all() and 0.35 <= median_head_distance <= 1.0
    )
    upright_profile_ok = bool(
        head_above_fraction >= 0.99 and median_head_offset[2] >= 0.4
    )
    coordinate_ok = (
        np.allclose(basis, np.eye(3), rtol=0.0, atol=1.0e-7)
        and np.allclose(ground_normal, [0.0, 0.0, 1.0], rtol=0.0, atol=1.0e-7)
        and pose_independent_semantics_ok
        and (profile != "b3-walk" or upright_profile_ok)
        and abs(float(scalar("ground_height_w"))) <= 1.0e-6
        and float(np.max(np.abs(arrays["ground_baseline_height_w"]))) <= 1.0e-6
    )
    check(
        "semantic_coordinate",
        "right_handed_z_up_identity_basis_and_fixed_ground",
        coordinate_ok,
        profile=profile,
        pose_independent_semantics_ok=pose_independent_semantics_ok,
        upright_profile_required=profile == "b3-walk",
        upright_profile_ok=upright_profile_ok,
        head_above_pelvis_fraction=head_above_fraction,
        median_head_minus_pelvis_w=median_head_offset.tolist(),
        median_head_pelvis_distance_m=median_head_distance,
        ground_height_w=float(scalar("ground_height_w")),
    )

    source_relative = Path(str(scalar("source_file")))
    source_path = (raw_root / source_relative).resolve()
    source_confined = not source_relative.is_absolute()
    try:
        source_path.relative_to(raw_root)
    except ValueError:
        source_confined = False
    source_exists = source_confined and source_path.is_file()
    checksum_ok = source_exists and _sha256(source_path) == str(scalar("source_checksum_sha256"))
    source_metadata_ok = False
    if checksum_ok:
        source_motion = load_stageii_numeric(source_path)
        source_metadata_ok = (
            int(source_motion["frame_count"]) == source_frame_count
            and float(source_motion["mocap_frame_rate"]) == source_fps
            and np.isclose(
                float(source_motion["mocap_time_length"]),
                float(scalar("source_mocap_time_length")),
            )
            and source_motion["gender"] == scalar("gender")
            and np.array_equal(source_motion["betas"], arrays["betas"])
        )
    check(
        "provenance_120hz",
        "source_file_checksum_and_metadata",
        source_confined and checksum_ok and source_metadata_ok,
        resolved_source_path=str(source_path),
        confined_to_accad=source_confined,
        checksum_matches=checksum_ok,
    )

    model_path = Path(str(scalar("smplx_model_path"))).expanduser().resolve()
    model_confined = True
    try:
        model_path.relative_to(raw_root)
    except ValueError:
        model_confined = False
    model_checksum_ok = (
        model_confined
        and model_path.is_file()
        and _sha256(model_path) == str(scalar("smplx_model_checksum_sha256"))
    )
    check(
        "provenance_120hz",
        "licensed_model_checksum",
        model_checksum_ok,
        resolved_model_path=str(model_path),
        confined_to_accad=model_confined,
    )
    source_baseline = np.asarray(arrays["source_ground_baseline_height_w"], dtype=np.float64)
    maximum_baseline_speed = float(np.max(np.abs(np.diff(source_baseline))) * source_fps)
    confidence_fraction = float(np.mean(arrays["source_ground_baseline_confidence"]))
    stored_drift = float(scalar("source_ground_baseline_drift_m"))
    baseline_ok = (
        scalar("source_fk_performed") is True
        and int(scalar("source_fk_frame_count")) == source_frame_count
        and np.isclose(stored_drift, source_baseline[-1] - source_baseline[0], atol=1.0e-7)
        and maximum_baseline_speed <= 0.0201
    )
    check(
        "provenance_120hz",
        "source_fk_contact_and_ground_baseline",
        baseline_ok,
        baseline_drift_m=stored_drift,
        maximum_baseline_speed_mps=maximum_baseline_speed,
        confidence_fraction=confidence_fraction,
    )

    left_contact = arrays["left_heel_contact"] | arrays["left_toe_contact"]
    right_contact = arrays["right_heel_contact"] | arrays["right_toe_contact"]
    airborne = ~(left_contact | right_contact)
    horizontal_speed = np.linalg.norm(arrays["root_linear_velocity_w"][:, :2], axis=-1)
    moving_airborne = airborne & (horizontal_speed > 0.8)
    profile_metrics = {
        "left_contact_fraction": float(np.mean(left_contact)),
        "right_contact_fraction": float(np.mean(right_contact)),
        "double_support_fraction": float(np.mean(left_contact & right_contact)),
        "airborne_fraction": float(np.mean(airborne)),
        "maximum_airborne_run_frames": _maximum_true_run(airborne),
        "maximum_fast_moving_airborne_run_frames": _maximum_true_run(moving_airborne),
    }
    if profile == "b3-walk":
        profile_ok = (
            _maximum_true_run(airborne) <= int(round(0.2 * fps))
            and _maximum_true_run(moving_airborne) <= 3
            and np.any(left_contact)
            and np.any(right_contact)
            and 0.04 <= stored_drift <= 0.10
            and confidence_fraction >= 0.90
        )
        check("profile", "b3_grounded_walk", profile_ok, **profile_metrics)
    else:
        check("profile", "generic_no_motion_class_assumption", True, **profile_metrics)

    statuses = {
        name: bool(items) and all(item["passed"] for item in items)
        for name, items in checks.items()
    }
    gate2_complete = all(statuses.values())
    if bool(arrays.get("gate2_complete", np.asarray(False)).item()) != gate2_complete:
        warnings.append(
            "The archive's serialized gate2_complete flag is informational; "
            "this independent report is authoritative."
        )
    return {
        "ok": gate2_complete,
        "gate2_complete": gate2_complete,
        "archive_path": str(path),
        "archive_checksum_sha256": _sha256(path),
        "profile": profile,
        "statuses": statuses,
        "errors": [] if gate2_complete else ["one or more Gate-2 checks failed"],
        "warnings": warnings,
        "checks": checks,
    }
