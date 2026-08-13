"""Morphology- and contact-aware Human→Unitree-G1 sequence retargeting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import yaml

from .g1_kinematics import (
    FOOT_SOLE_OFFSET,
    G1_CONTRACT_PATH,
    G1_URDF_PATH,
    TorchG1Kinematics,
    load_g1_contract,
    matrix_to_quaternion_wxyz,
    policy_to_sdk,
)
from .human import _angular_velocity_world, _linear_velocity, _save_npz_atomic
from .math3d import quaternion_wxyz_to_matrix
from .paths import PIPELINE_ROOT, assert_output_is_local


DEFAULT_CORRESPONDENCE = PIPELINE_ROOT / "correspondence.yaml"
FOOT_COLLISION_RADIUS_M = 0.005
MAXIMUM_GROUND_PROJECTION_LIMIT_EXCESS_M = 0.05
RETARGET_VERSION = "g1-sequence-retarget-v27-ground-feasible-output-stance-float64-fk"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_scalar(arrays: dict[str, np.ndarray], name: str) -> Any:
    value = arrays[name]
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{name} must be a safe scalar; got {value.shape}/{value.dtype}")
    return value.item()


def _safe_load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    objects = [name for name, value in arrays.items() if value.dtype.hasobject]
    if objects:
        raise ValueError(f"object arrays are forbidden: {objects}")
    return arrays


def load_correspondence(path: str | Path = DEFAULT_CORRESPONDENCE) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "accad-g1-correspondence-v1":
        raise ValueError(f"unsupported correspondence schema: {config_path}")
    payload["_path"] = str(config_path)
    return payload


def _continuous_quaternion(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    for frame in range(1, len(result)):
        if float(np.dot(result[frame - 1], result[frame])) < 0.0:
            result[frame] *= -1.0
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return np.asarray(result, dtype=np.float32)


def _contact_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive contiguous runs from a one-dimensional contact mask."""

    contact = np.asarray(mask, dtype=bool)
    if contact.ndim != 1:
        raise ValueError(f"contact mask must be one-dimensional; got {contact.shape}")
    padded = np.pad(contact.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _split_long_true_runs(mask: np.ndarray, maximum_frames: int) -> np.ndarray:
    """Bound rigid-anchor duration with one non-stance re-anchor frame.

    The geometric contact label is not changed.  Only the stricter stance mask
    receives a one-frame boundary, preventing multi-second quasi-static clips
    from accumulating unbounded numerical foot drift around one world anchor.
    """

    if maximum_frames < 2:
        raise ValueError("maximum stance anchor duration must be at least two frames")
    result = np.asarray(mask, dtype=bool).copy()
    for start, end in _contact_runs(result):
        boundary = start + maximum_frames
        while boundary <= end:
            result[boundary] = False
            boundary += maximum_frames + 1
    return result


def _lock_contact_point_targets(
    point_targets: np.ndarray,
    contacts: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Lock each active G1 heel/toe target to a robust world-space anchor.

    Heel and toe channels remain independent.  This preserves rigid contact
    while still allowing a physically meaningful heel-strike or toe-off roll.
    """

    locked_position = np.asarray(point_targets, dtype=np.float64).copy()
    contact = np.asarray(contacts, dtype=bool)
    if locked_position.ndim != 3 or locked_position.shape[1:] != (4, 3):
        raise ValueError(
            f"contact targets must have shape (frames, 4, 3); got {locked_position.shape}"
        )
    if contact.shape != locked_position.shape[:2]:
        raise ValueError(
            f"contact mask must have shape {locked_position.shape[:2]}; got {contact.shape}"
        )
    metadata: list[dict[str, Any]] = []
    point_names = ("left_heel", "left_toe", "right_heel", "right_toe")
    for point_index, point_name in enumerate(point_names):
        for start, end in _contact_runs(contact[:, point_index]):
            selection = slice(start, end + 1)
            original = locked_position[selection, point_index].copy()
            anchor_xy = np.median(original[:, :2], axis=0)
            locked_position[selection, point_index, :2] = anchor_xy
            locked_position[selection, point_index, 2] = FOOT_COLLISION_RADIUS_M
            drift = np.linalg.norm(original[:, :2] - anchor_xy, axis=-1)
            metadata.append(
                {
                    "point": point_name,
                    "start_frame": start,
                    "end_frame": end,
                    "frames": end - start + 1,
                    "anchor_xy_m": anchor_xy.tolist(),
                    "source_target_max_drift_m": float(np.max(drift)),
                }
            )
    return locked_position, metadata


def _derive_stance_channels(
    point_targets: np.ndarray,
    contacts: np.ndarray,
    *,
    fps: float,
    speed_threshold_mps: float,
    target_drift_limit_m: float,
    minimum_frames: int,
    eligible_frames: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Separate rigid stance from geometric ground contact.

    AMASS contact labels only mean that a heel or toe is close to the inferred
    ground.  They do not imply a zero-velocity constraint: crawling and toe
    drags are important counterexamples.  This function retains the original
    contact mask for ground/penetration objectives and creates a stricter mask
    for world-space locking.  Long slowly sliding intervals are split with a
    one-frame stance boundary so no rigid anchor spans more than the configured
    source-target drift.
    """

    points = np.asarray(point_targets, dtype=np.float64)
    contact = np.asarray(contacts, dtype=bool)
    if points.ndim != 3 or points.shape[1:] != (4, 3):
        raise ValueError(f"point_targets must have shape (frames, 4, 3); got {points.shape}")
    if contact.shape != points.shape[:2]:
        raise ValueError(f"contacts must have shape {points.shape[:2]}; got {contact.shape}")
    if fps <= 0.0 or speed_threshold_mps <= 0.0 or target_drift_limit_m <= 0.0:
        raise ValueError("stance fps, speed threshold and drift limit must be positive")
    if minimum_frames < 2:
        raise ValueError("minimum stance run must contain at least two frames")

    xy_step = np.zeros(contact.shape, dtype=np.float64)
    if len(points) > 1:
        interval_speed = np.linalg.norm(np.diff(points[..., :2], axis=0), axis=-1) * fps
        # A frame is stationary only when both adjacent intervals are slow.
        xy_step[0] = interval_speed[0]
        xy_step[-1] = interval_speed[-1]
        if len(points) > 2:
            xy_step[1:-1] = np.maximum(interval_speed[:-1], interval_speed[1:])
    if eligible_frames is None:
        eligible = np.ones(len(points), dtype=bool)
    else:
        eligible = np.asarray(eligible_frames, dtype=bool)
        if eligible.shape != (len(points),):
            raise ValueError(
                f"eligible_frames must have shape {(len(points),)}; got {eligible.shape}"
            )
    candidate = contact & eligible[:, None] & (xy_step <= speed_threshold_mps)
    stance = np.zeros_like(candidate)
    metadata: list[dict[str, Any]] = []
    point_names = ("left_heel", "left_toe", "right_heel", "right_toe")

    for point_index, point_name in enumerate(point_names):
        for candidate_start, candidate_end in _contact_runs(candidate[:, point_index]):
            segment_start = candidate_start
            frame = candidate_start
            while frame <= candidate_end:
                drift = float(
                    np.linalg.norm(
                        points[frame, point_index, :2]
                        - points[segment_start, point_index, :2]
                    )
                )
                if drift <= target_drift_limit_m:
                    frame += 1
                    continue

                segment_end = frame - 1
                if segment_end - segment_start + 1 >= minimum_frames:
                    stance[segment_start : segment_end + 1, point_index] = True
                    metadata.append(
                        {
                            "point": point_name,
                            "start_frame": segment_start,
                            "end_frame": segment_end,
                            "frames": segment_end - segment_start + 1,
                            "source": "stationary_contact",
                        }
                    )
                # Keep the threshold-crossing frame false.  It is still a
                # geometric contact, but explicitly separates rigid anchors.
                segment_start = frame + 1
                frame = segment_start

            segment_end = candidate_end
            if segment_end - segment_start + 1 >= minimum_frames:
                stance[segment_start : segment_end + 1, point_index] = True
                metadata.append(
                    {
                        "point": point_name,
                        "start_frame": segment_start,
                        "end_frame": segment_end,
                        "frames": segment_end - segment_start + 1,
                        "source": "stationary_contact",
                    }
                )

    return stance, metadata


def _refine_output_stance_channels(
    point_positions: np.ndarray,
    preliminary_stance: np.ndarray,
    *,
    fps: float,
    solver: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Keep stance only where the *retargeted G1* support point is stationary.

    Human contact is useful while optimizing, but morphology and finite IK
    accuracy can leave a G1 heel/toe moving even when its source marker looked
    stationary.  Gate-3 evaluates the final G1 geometry, so the serialized
    stance contract must be derived from that same geometry.  Starting from
    ``preliminary_stance`` makes this a refinement only: it can never promote a
    source drag, crawl or non-load-bearing contact to rigid stance.
    """

    preliminary = np.asarray(preliminary_stance, dtype=bool)
    stance, _ = _derive_stance_channels(
        point_positions,
        preliminary,
        fps=fps,
        speed_threshold_mps=float(solver["stance_speed_threshold_mps"]),
        target_drift_limit_m=float(solver["stance_target_drift_limit_m"]),
        minimum_frames=int(solver["stance_minimum_frames"]),
    )
    minimum_frames = int(solver["stance_minimum_frames"])
    maximum_frames = int(solver["stance_maximum_anchor_frames"])
    for channel in range(stance.shape[1]):
        stance[:, channel] = _split_long_true_runs(
            stance[:, channel], maximum_frames
        )
        for start, end in _contact_runs(stance[:, channel]):
            if end - start + 1 < minimum_frames:
                stance[start : end + 1, channel] = False

    metadata = [
        {
            "point": point_name,
            "start_frame": start,
            "end_frame": end,
            "frames": end - start + 1,
            "source": "stationary_final_g1_support_subset_of_preliminary_stance",
        }
        for channel, point_name in enumerate(
            ("left_heel", "left_toe", "right_heel", "right_toe")
        )
        for start, end in _contact_runs(stance[:, channel])
    ]
    return stance, metadata


def _root_and_targets(
    human: dict[str, np.ndarray],
    config: dict[str, Any],
    kinematics: TorchG1Kinematics,
) -> dict[str, Any]:
    all_names = [str(value) for value in human["all_joint_names"].tolist()]
    rotation_names = [str(value) for value in human["joint_names"].tolist()]
    all_index = {name: index for index, name in enumerate(all_names)}
    rotation_index = {name: index for index, name in enumerate(rotation_names)}
    required_positions = {
        "pelvis",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    }
    required_rotations = {"spine3"}
    if not required_positions.issubset(all_index):
        raise ValueError(f"missing human position targets: {sorted(required_positions - all_index.keys())}")
    if not required_rotations.issubset(rotation_index):
        raise ValueError(f"missing human rotation targets: {sorted(required_rotations - rotation_index.keys())}")

    human_root_position = np.asarray(human["root_position_w"], dtype=np.float64)
    all_positions = np.asarray(human["all_joint_position_w"], dtype=np.float64)
    human_root_rotation = quaternion_wxyz_to_matrix(human["root_quaternion_wxyz"])
    human_local_from_g1 = np.asarray(config["human_local_from_g1_local"], dtype=np.float64)
    if not np.allclose(human_local_from_g1.T @ human_local_from_g1, np.eye(3), atol=1.0e-8):
        raise ValueError("human/G1 local-axis mapping must be orthonormal")
    if np.linalg.det(human_local_from_g1) < 0.999999:
        raise ValueError("human/G1 local-axis mapping must preserve handedness")
    g1_root_rotation = human_root_rotation @ human_local_from_g1
    g1_root_quaternion = _continuous_quaternion(
        matrix_to_quaternion_wxyz(g1_root_rotation)
    )

    contact_points = np.asarray(human["contact_keypoint_position_w"], dtype=np.float64)
    human_sole = np.stack(
        (0.5 * (contact_points[:, 0] + contact_points[:, 1]),
         0.5 * (contact_points[:, 2] + contact_points[:, 3])),
        axis=1,
    )
    contact_channels = np.stack(
        (
            np.asarray(human["left_heel_contact"], dtype=bool),
            np.asarray(human["left_toe_contact"], dtype=bool),
            np.asarray(human["right_heel_contact"], dtype=bool),
            np.asarray(human["right_toe_contact"], dtype=bool),
        ),
        axis=1,
    )
    left_contact = contact_channels[:, 0] | contact_channels[:, 1]
    right_contact = contact_channels[:, 2] | contact_channels[:, 3]
    contacts = np.stack((left_contact, right_contact), axis=1)
    full_contacts = np.stack(
        (
            contact_channels[:, 0] & contact_channels[:, 1],
            contact_channels[:, 2] & contact_channels[:, 3],
        ),
        axis=1,
    )
    # Estimate embodiment scale from rigid bone lengths, never from the
    # pelvis' vertical height in the current pose.  The latter collapses for
    # lying/crawling sequences and previously caused the 1.10 scale clamp.
    leg_chain_lengths: list[np.ndarray] = []
    for side in ("left", "right"):
        pelvis_to_hip = np.linalg.norm(
            all_positions[:, all_index["pelvis"]]
            - all_positions[:, all_index[f"{side}_hip"]],
            axis=-1,
        )
        hip_to_knee = np.linalg.norm(
            all_positions[:, all_index[f"{side}_hip"]]
            - all_positions[:, all_index[f"{side}_knee"]],
            axis=-1,
        )
        knee_to_ankle = np.linalg.norm(
            all_positions[:, all_index[f"{side}_knee"]]
            - all_positions[:, all_index[f"{side}_ankle"]],
            axis=-1,
        )
        leg_chain_lengths.append(pelvis_to_hip + hip_to_knee + knee_to_ankle)
    pose_independent_leg_height = float(np.median(np.concatenate(leg_chain_lengths)))
    # When the clip actually contains a near-straight upright support pose,
    # its pelvis-to-sole height is the most direct scale calibration (and
    # preserves the validated B3 reference).  Non-upright clips fall back to
    # rigid bone lengths, so crawling/lying can never collapse the estimate.
    ground_baseline = np.asarray(human["ground_baseline_height_w"], dtype=np.float64)
    upright_for_scale = (
        human_root_position[:, 2] - ground_baseline
    ) >= 0.90 * pose_independent_leg_height
    upright_support_heights: list[np.ndarray] = []
    for side in range(2):
        selection = upright_for_scale & contacts[:, side]
        if np.any(selection):
            upright_support_heights.append(
                human_root_position[selection, 2]
                - human_sole[selection, side, 2]
            )
    upright_samples = (
        np.concatenate(upright_support_heights)
        if upright_support_heights
        else np.asarray([], dtype=np.float64)
    )
    human_leg_height = float(
        np.median(upright_samples)
        if len(upright_samples) >= 10
        else pose_independent_leg_height
    )

    contract = kinematics.contract
    default_q = torch.as_tensor(contract.default[None], dtype=kinematics.dtype, device=kinematics.device)
    zero_root = torch.zeros((1, 3), dtype=kinematics.dtype, device=kinematics.device)
    identity_root = torch.as_tensor(
        [[1.0, 0.0, 0.0, 0.0]], dtype=kinematics.dtype, device=kinematics.device
    )
    with torch.inference_mode():
        nominal_transforms = kinematics.forward(default_q, zero_root, identity_root)
        nominal_sole = kinematics.sole_positions(nominal_transforms).cpu().numpy()[0]
    g1_leg_height = float(-np.mean(nominal_sole[:, 2]) + FOOT_COLLISION_RADIUS_M)
    morphology_scale = float(np.clip(g1_leg_height / human_leg_height, 0.70, 1.10))

    g1_root_position = np.asarray(human_root_position * morphology_scale, dtype=np.float64)
    g1_root_position[:, :2] -= g1_root_position[:1, :2]

    def scaled_position(name: str) -> np.ndarray:
        relative = all_positions[:, all_index[name]] - human_root_position
        return g1_root_position + morphology_scale * relative

    position_targets: dict[str, np.ndarray] = {}
    position_weights: dict[str, float] = {}
    for link_name, spec in config["position_targets"].items():
        human_name = str(spec["human"])
        if human_name not in all_index:
            raise ValueError(f"correspondence references unknown human joint {human_name}")
        position_targets[str(link_name)] = scaled_position(human_name)
        position_weights[str(link_name)] = float(spec["weight"])

    foot_targets = g1_root_position[:, None, :] + morphology_scale * (
        human_sole - human_root_position[:, None, :]
    )
    # Targets are collision-sphere centers, not the geometric floor surface.
    foot_targets[:, :, 2] += FOOT_COLLISION_RADIUS_M
    foot_targets[:, :, 2] = np.where(
        full_contacts, FOOT_COLLISION_RADIUS_M, foot_targets[:, :, 2]
    )

    joint_rotations = quaternion_wxyz_to_matrix(human["joint_quaternion_wxyz"])
    orientation_targets: dict[str, np.ndarray] = {}
    orientation_weights: dict[str, float] = {}
    for link_name, spec in config["orientation_targets"].items():
        human_name = str(spec["human"])
        orientation_targets[str(link_name)] = (
            joint_rotations[:, rotation_index[human_name]] @ human_local_from_g1
        )
        orientation_weights[str(link_name)] = float(spec["weight"])

    # Stance soles should be level and point along the projected human heading.
    forward = g1_root_rotation[:, :, 0].copy()
    forward[:, 2] = 0.0
    forward /= np.linalg.norm(forward, axis=-1, keepdims=True).clip(min=1.0e-8)
    up = np.broadcast_to(np.asarray([0.0, 0.0, 1.0]), forward.shape)
    left = np.cross(up, forward)
    foot_orientation = np.stack((forward, left, up), axis=-1)
    # Rear/front collision-center representatives are 0.085 m behind/ahead
    # of the mean sole point for the G1's 0.17 m support geometry.
    contact_targets = foot_targets[:, [0, 0, 1, 1]].copy()
    contact_targets += forward[:, None, :] * np.asarray(
        [-0.085, 0.085, -0.085, 0.085], dtype=np.float64
    )[None, :, None]
    contact_targets[..., 2] = np.where(
        contact_channels, FOOT_COLLISION_RADIUS_M, contact_targets[..., 2]
    )
    solver = config["solver"]
    fps = float(_load_scalar(human, "fps"))
    pelvis_clearance = human_root_position[:, 2] - ground_baseline
    stance_eligible = pelvis_clearance >= (
        float(solver["stance_minimum_pelvis_height_ratio"]) * human_leg_height
    )
    # Contact means near-ground; stance additionally requires a load-bearing
    # posture and locally stationary source target.  This preserves planted
    # walking/gesture feet while keeping pivots, toe drags and shuffles movable.
    stance_channels, _ = _derive_stance_channels(
        contact_targets,
        contact_channels,
        fps=fps,
        speed_threshold_mps=float(solver["stance_speed_threshold_mps"]),
        target_drift_limit_m=float(solver["stance_target_drift_limit_m"]),
        minimum_frames=int(solver["stance_minimum_frames"]),
        eligible_frames=stance_eligible,
    )
    minimum_stance_frames = int(solver["stance_minimum_frames"])
    for channel in range(stance_channels.shape[1]):
        for start, end in _contact_runs(stance_channels[:, channel]):
            if end - start + 1 < minimum_stance_frames:
                stance_channels[start : end + 1, channel] = False
        stance_channels[:, channel] = _split_long_true_runs(
            stance_channels[:, channel],
            int(solver["stance_maximum_anchor_frames"]),
        )
        for start, end in _contact_runs(stance_channels[:, channel]):
            if end - start + 1 < minimum_stance_frames:
                stance_channels[start : end + 1, channel] = False
    stance_detection = [
        {
            "point": point_name,
            "start_frame": start,
            "end_frame": end,
            "frames": end - start + 1,
            "source": "stationary_contact_and_load_bearing_posture",
        }
        for channel, point_name in enumerate(
            ("left_heel", "left_toe", "right_heel", "right_toe")
        )
        for start, end in _contact_runs(stance_channels[:, channel])
    ]
    contact_targets, contact_runs = _lock_contact_point_targets(
        contact_targets, stance_channels
    )

    return {
        "root_position": np.asarray(g1_root_position, dtype=np.float32),
        "root_quaternion": g1_root_quaternion,
        "root_rotation": np.asarray(g1_root_rotation, dtype=np.float32),
        "position_targets": position_targets,
        "position_weights": position_weights,
        "foot_targets": np.asarray(foot_targets, dtype=np.float32),
        "foot_orientation": np.asarray(foot_orientation, dtype=np.float32),
        "contacts": contacts,
        "contact_channels": contact_channels,
        "stance_channels": stance_channels,
        "full_contacts": full_contacts,
        "full_stance": np.stack(
            (
                stance_channels[:, 0] & stance_channels[:, 1],
                stance_channels[:, 2] & stance_channels[:, 3],
            ),
            axis=1,
        ),
        "contact_targets": np.asarray(contact_targets, dtype=np.float32),
        "contact_runs": contact_runs,
        "stance_detection": stance_detection,
        "stance_eligible": stance_eligible,
        "orientation_targets": orientation_targets,
        "orientation_weights": orientation_weights,
        "morphology_scale": morphology_scale,
        "human_leg_height_m": human_leg_height,
        "g1_leg_height_m": g1_leg_height,
    }


def _optimize_sequence(
    targets: dict[str, Any],
    config: dict[str, Any],
    kinematics: TorchG1Kinematics,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    solver = config["solver"]
    device = kinematics.device
    dtype = kinematics.dtype
    torch.manual_seed(int(solver.get("seed", 42)))
    contract = kinematics.contract
    frames = len(targets["root_position"])

    margin = float(solver["limit_margin_rad"])
    soft_factor = float(solver["soft_joint_limit_factor"])
    if not 0.0 < soft_factor <= 1.0:
        raise ValueError("soft_joint_limit_factor must be in (0, 1]")
    hard_center = 0.5 * (contract.lower + contract.upper)
    hard_half_range = 0.5 * (contract.upper - contract.lower)
    lower_np = hard_center - soft_factor * hard_half_range + margin
    upper_np = hard_center + soft_factor * hard_half_range - margin
    if np.any(lower_np >= upper_np):
        raise ValueError("soft joint limits and configured margin leave an empty range")
    lower = torch.as_tensor(lower_np, dtype=dtype, device=device)
    upper = torch.as_tensor(upper_np, dtype=dtype, device=device)
    center = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    default = torch.as_tensor(contract.default, dtype=dtype, device=device)
    normalized_default = ((default - center) / half_range).clamp(-0.999, 0.999)
    raw_q = torch.nn.Parameter(
        torch.atanh(normalized_default).expand(frames, -1).clone()
    )
    raw_root = torch.nn.Parameter(torch.zeros((frames, 3), dtype=dtype, device=device))
    stance_eligible_fraction = float(np.mean(targets["stance_eligible"]))
    root_z_limit = float(solver["root_z_correction_limit_m"])
    if stance_eligible_fraction < 0.95:
        root_z_limit = float(solver["nonupright_root_z_correction_limit_m"])
    root_limits = torch.as_tensor(
        [
            float(solver["root_xy_correction_limit_m"]),
            float(solver["root_xy_correction_limit_m"]),
            root_z_limit,
        ],
        dtype=dtype,
        device=device,
    )
    base_root = torch.as_tensor(targets["root_position"], dtype=dtype, device=device)
    root_quaternion = torch.as_tensor(targets["root_quaternion"], dtype=dtype, device=device)
    position_targets = {
        name: torch.as_tensor(value, dtype=dtype, device=device)
        for name, value in targets["position_targets"].items()
    }
    foot_target = torch.as_tensor(targets["foot_targets"], dtype=dtype, device=device)
    foot_orientation = torch.as_tensor(targets["foot_orientation"], dtype=dtype, device=device)
    contacts = torch.as_tensor(targets["contacts"], dtype=torch.bool, device=device)
    contact_channels = torch.as_tensor(
        targets["contact_channels"], dtype=torch.bool, device=device
    )
    stance_channels = torch.as_tensor(
        targets["stance_channels"], dtype=torch.bool, device=device
    )
    full_contacts = torch.as_tensor(
        targets["full_contacts"], dtype=torch.bool, device=device
    )
    full_stance = torch.as_tensor(
        targets["full_stance"], dtype=torch.bool, device=device
    )
    contact_target = torch.as_tensor(
        targets["contact_targets"], dtype=dtype, device=device
    )
    orientation_targets = {
        name: torch.as_tensor(value, dtype=dtype, device=device)
        for name, value in targets["orientation_targets"].items()
    }
    optimizer = torch.optim.Adam(
        (raw_q, raw_root), lr=float(solver["learning_rate"])
    )
    iterations = int(solver["iterations"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(iterations, 1), eta_min=0.002
    )
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    def decoded() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = center + half_range * torch.tanh(raw_q)
        root_correction = root_limits * torch.tanh(raw_root)
        return q, base_root + root_correction, root_correction

    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        q, root_position, root_correction = decoded()
        transforms = kinematics.forward(q, root_position, root_quaternion)
        terms: dict[str, torch.Tensor] = {}
        position_loss = torch.zeros((), dtype=dtype, device=device)
        for link_name, target in position_targets.items():
            error = transforms[link_name][..., :3, 3] - target
            position_loss = position_loss + float(
                targets["position_weights"][link_name]
            ) * error.square().sum(dim=-1).mean()
        terms["position"] = position_loss

        sole = kinematics.sole_positions(transforms)
        support = kinematics.foot_support_positions(transforms)
        support_representatives = torch.stack(
            (
                support[:, 0, :2].mean(dim=1),
                support[:, 0, 2:].mean(dim=1),
                support[:, 1, :2].mean(dim=1),
                support[:, 1, 2:].mean(dim=1),
            ),
            dim=1,
        )
        terms["foot_position"] = float(solver["foot_position_weight"]) * (
            sole - foot_target
        ).square().sum(dim=-1).mean()
        contact_float = contacts.to(dtype=dtype)
        channel_float = contact_channels.to(dtype=dtype)
        terms["contact_position"] = float(solver["contact_position_weight"]) * (
            (
                (support_representatives - contact_target).square().sum(dim=-1)
                * channel_float
            ).sum()
            / channel_float.sum().clamp_min(1.0)
        )
        support_contact = torch.stack(
            (
                contact_channels[:, 0],
                contact_channels[:, 0],
                contact_channels[:, 1],
                contact_channels[:, 1],
                contact_channels[:, 2],
                contact_channels[:, 2],
                contact_channels[:, 3],
                contact_channels[:, 3],
            ),
            dim=1,
        ).reshape(frames, 2, 4).to(dtype=dtype)
        terms["stance_ground"] = float(solver["stance_ground_weight"]) * (
            (
                (support[..., 2] - FOOT_COLLISION_RADIUS_M).square()
                * support_contact
            ).sum()
            / support_contact.sum().clamp_min(1.0)
        )
        if frames > 1:
            consecutive_stance = stance_channels[1:] & stance_channels[:-1]
            consecutive_float = consecutive_stance.to(dtype=dtype)
            stance_delta = support_representatives[1:] - support_representatives[:-1]
            terms["stance_lock"] = float(solver["stance_lock_weight"]) * (
                (stance_delta[..., :2].square().sum(dim=-1) * consecutive_float).sum()
                / consecutive_float.sum().clamp_min(1.0)
            )
        else:
            terms["stance_lock"] = torch.zeros((), dtype=dtype, device=device)
        terms["penetration"] = float(solver["penetration_weight"]) * torch.relu(
            FOOT_COLLISION_RADIUS_M - support[..., 2]
        ).square().mean()

        orientation_loss = torch.zeros((), dtype=dtype, device=device)
        for link_name, target in orientation_targets.items():
            actual = transforms[link_name][..., :3, :3]
            orientation_loss = orientation_loss + float(
                targets["orientation_weights"][link_name]
            ) * (actual - target).square().mean()
        terms["orientation"] = orientation_loss
        ankle_rotation = torch.stack(
            (
                transforms["left_ankle_roll_link"][..., :3, :3],
                transforms["right_ankle_roll_link"][..., :3, :3],
            ),
            dim=1,
        )
        foot_rotation_error = (ankle_rotation - foot_orientation[:, None]).square().mean(
            dim=(-2, -1)
        )
        full_contact_float = full_stance.to(dtype=dtype)
        terms["stance_orientation"] = float(
            solver["stance_orientation_weight"]
        ) * (foot_rotation_error * full_contact_float).sum() / full_contact_float.sum().clamp_min(1.0)

        terms["nominal"] = float(solver["nominal_pose_weight"]) * (
            q - default
        ).square().mean()
        terms["root_correction"] = float(solver["root_correction_weight"]) * root_correction.square().mean()
        if frames > 1:
            joint_velocity_physical = (q[1:] - q[:-1]) / contract.control_dt
            velocity_limit = torch.as_tensor(
                contract.velocity * float(solver["velocity_limit_factor"]),
                dtype=dtype,
                device=device,
            )
            terms["joint_velocity_limit"] = float(
                solver["joint_velocity_limit_weight"]
            ) * torch.relu(
                joint_velocity_physical.abs() - velocity_limit
            ).square().mean()
            terms["joint_velocity"] = float(solver["joint_velocity_weight"]) * (
                q[1:] - q[:-1]
            ).square().mean()
            terms["root_velocity"] = float(solver["root_velocity_weight"]) * (
                root_correction[1:] - root_correction[:-1]
            ).square().mean()
        else:
            terms["joint_velocity_limit"] = torch.zeros((), dtype=dtype, device=device)
            terms["joint_velocity"] = torch.zeros((), dtype=dtype, device=device)
            terms["root_velocity"] = torch.zeros((), dtype=dtype, device=device)
        if frames > 2:
            terms["joint_acceleration"] = float(
                solver["joint_acceleration_weight"]
            ) * (q[2:] - 2.0 * q[1:-1] + q[:-2]).square().mean()
            terms["root_acceleration"] = float(
                solver["root_acceleration_weight"]
            ) * (
                root_correction[2:]
                - 2.0 * root_correction[1:-1]
                + root_correction[:-2]
            ).square().mean()
        else:
            terms["joint_acceleration"] = torch.zeros((), dtype=dtype, device=device)
            terms["root_acceleration"] = torch.zeros((), dtype=dtype, device=device)
        total = sum(terms.values())
        if not torch.isfinite(total):
            raise RuntimeError(f"retarget loss became non-finite at iteration {iteration}")
        total.backward()
        torch.nn.utils.clip_grad_norm_((raw_q, raw_root), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        if iteration == 0 or (iteration + 1) % 50 == 0 or iteration + 1 == iterations:
            record = {name: float(value.detach().cpu()) for name, value in terms.items()}
            record["total"] = float(total.detach().cpu())
            record["iteration"] = iteration + 1
            history.append(record)

    with torch.inference_mode():
        q_final, root_final, correction_final = decoded()
    optimization = {
        "iterations": iterations,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "dtype": str(dtype),
        "loss_history": history,
        "initial_total_loss": history[0]["total"],
        "final_total_loss": history[-1]["total"],
        "maximum_root_correction_m": float(
            torch.linalg.vector_norm(correction_final, dim=-1).max().cpu()
        ),
        "root_z_correction_limit_m": root_z_limit,
        "stance_eligible_fraction": stance_eligible_fraction,
    }
    return (
        q_final.detach().cpu().numpy().astype(np.float32),
        root_final.detach().cpu().numpy().astype(np.float32),
        optimization,
    )


def _project_root_to_stance_anchors(
    joint_position: np.ndarray,
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    targets: dict[str, Any],
    config: dict[str, Any],
    kinematics: TorchG1Kinematics,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the least-squares XY translation implied by rigid stance anchors.

    Root translation affects every support point linearly and identically.  A
    closed-form projection therefore removes the small residual left by Adam
    without changing joint angles or fabricating foot articulation.  With two
    simultaneous feet this is the least-squares translation across all active
    heel/toe anchors.
    """

    with torch.inference_mode():
        transforms = kinematics.forward(
            torch.as_tensor(
                joint_position, dtype=kinematics.dtype, device=kinematics.device
            ),
            torch.as_tensor(
                root_position, dtype=kinematics.dtype, device=kinematics.device
            ),
            torch.as_tensor(
                root_quaternion, dtype=kinematics.dtype, device=kinematics.device
            ),
        )
        support = kinematics.foot_support_positions(transforms)
        representatives = torch.stack(
            (
                support[:, 0, :2].mean(dim=1),
                support[:, 0, 2:].mean(dim=1),
                support[:, 1, :2].mean(dim=1),
                support[:, 1, 2:].mean(dim=1),
            ),
            dim=1,
        ).cpu().numpy()
    stance = np.asarray(targets["stance_channels"], dtype=bool)
    anchor = np.asarray(targets["contact_targets"], dtype=np.float64)
    error_xy = anchor[..., :2] - representatives[..., :2]
    counts = stance.sum(axis=1)
    shift_xy = np.zeros((len(root_position), 2), dtype=np.float64)
    active = counts > 0
    if np.any(active):
        shift_xy[active] = (
            error_xy[active] * stance[active, :, None]
        ).sum(axis=1) / counts[active, None]
    maximum_shift = float(config["solver"]["post_projection_xy_limit_m"])
    norms = np.linalg.norm(shift_xy, axis=-1)
    scale = np.minimum(1.0, maximum_shift / np.maximum(norms, 1.0e-12))
    shift_xy *= scale[:, None]
    projected = np.asarray(root_position, dtype=np.float64).copy()
    projected[:, :2] += shift_xy
    return np.asarray(projected, dtype=np.float32), {
        "active_frame_fraction": float(np.mean(active)),
        "maximum_xy_shift_m": float(np.max(np.linalg.norm(shift_xy, axis=-1))),
        "mean_active_xy_shift_m": float(
            np.mean(np.linalg.norm(shift_xy[active], axis=-1)) if np.any(active) else 0.0
        ),
        "configured_xy_limit_m": maximum_shift,
    }


def _project_root_above_ground(
    joint_position: np.ndarray,
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    targets: dict[str, Any],
    config: dict[str, Any],
    kinematics: TorchG1Kinematics,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Enforce the exact URDF support-sphere ground half-space.

    The embodiment gap can saturate the soft contact loss in any pose.  A
    vertical root translation is the closed-form feasible projection and
    preserves every joint angle and body-relative pose.
    """

    eligible_fraction = float(np.mean(targets["stance_eligible"]))
    projected = np.asarray(root_position, dtype=np.float64).copy()
    with torch.inference_mode():
        transforms = kinematics.forward(
            torch.as_tensor(
                joint_position, dtype=kinematics.dtype, device=kinematics.device
            ),
            torch.as_tensor(
                root_position, dtype=kinematics.dtype, device=kinematics.device
            ),
            torch.as_tensor(
                root_quaternion, dtype=kinematics.dtype, device=kinematics.device
            ),
        )
        support_z = (
            kinematics.foot_support_positions(transforms)[..., 2].cpu().numpy()
        )
    margin = float(config["solver"]["post_projection_ground_margin_m"])
    required_shift = np.maximum(
        FOOT_COLLISION_RADIUS_M + margin - np.min(support_z, axis=(1, 2)),
        0.0,
    )
    base_root = np.asarray(targets["root_position"], dtype=np.float64)
    z_limit_key = (
        "root_z_correction_limit_m"
        if eligible_fraction >= 0.95
        else "nonupright_root_z_correction_limit_m"
    )
    z_limit = float(config["solver"][z_limit_key])
    # The configured Z limit is an optimization trust region, not permission
    # to leave collision geometry below the floor.  Always apply the exact
    # half-space projection.  A separate small excess guard catches truly
    # implausible retargets while allowing the few-millimetre embodiment-gap
    # correction observed in crouching/crawling ACCAD clips.
    applied_shift = required_shift
    projected[:, 2] += applied_shift
    final_correction = projected[:, 2] - base_root[:, 2]
    limit_excess = np.maximum(final_correction - z_limit, 0.0)
    maximum_limit_excess = float(np.max(limit_excess))
    if maximum_limit_excess > MAXIMUM_GROUND_PROJECTION_LIMIT_EXCESS_M:
        raise RuntimeError(
            "ground feasibility requires an implausible root-Z correction "
            f"beyond the optimization limit: {maximum_limit_excess:.6f} m > "
            f"{MAXIMUM_GROUND_PROJECTION_LIMIT_EXCESS_M:.6f} m"
        )
    return np.asarray(projected, dtype=np.float32), {
        "applied": bool(np.any(applied_shift > 0.0)),
        "maximum_z_shift_m": float(np.max(applied_shift)),
        "mean_z_shift_m": float(np.mean(applied_shift)),
        "maximum_unresolved_shift_m": 0.0,
        "limit_exceeded": bool(np.any(limit_excess > 0.0)),
        "maximum_limit_excess_m": maximum_limit_excess,
        "maximum_allowed_limit_excess_m": MAXIMUM_GROUND_PROJECTION_LIMIT_EXCESS_M,
        "eligible_fraction": eligible_fraction,
        "ground_margin_m": margin,
        "root_z_limit_m": z_limit,
    }


def retarget_human_to_g1(
    human_path: str | Path,
    output_path: str | Path,
    *,
    correspondence_path: str | Path = DEFAULT_CORRESPONDENCE,
) -> dict[str, Any]:
    human_file = Path(human_path).expanduser().resolve()
    output_file = assert_output_is_local(output_path)
    if output_file.suffix.lower() != ".npz":
        raise ValueError("G1 retarget output must end in .npz")
    human = _safe_load_npz(human_file)
    if _load_scalar(human, "schema_version") != "accad_human_reconstruction_v1":
        raise ValueError("retargeter requires a canonical Gate-2 human archive")
    fps = float(_load_scalar(human, "fps"))
    if fps != 50.0:
        raise ValueError(f"G1 contract requires 50 Hz human reference; got {fps}")
    config = load_correspondence(correspondence_path)
    solver_config = config["solver"]
    dtype = torch.float32 if str(solver_config["dtype"]) == "float32" else torch.float64
    kinematics = TorchG1Kinematics(
        device=str(solver_config["device"]), dtype=dtype
    )
    targets = _root_and_targets(human, config, kinematics)
    joint_position, root_position, optimization = _optimize_sequence(
        targets, config, kinematics
    )
    root_quaternion = np.asarray(targets["root_quaternion"], dtype=np.float32)
    # Optimization stays on the configured accelerator.  The final physical
    # projection and every serialized FK quantity are recomputed with one
    # independent CPU/float64 kinematics instance.  This removes float32
    # endpoint-derivative noise while keeping the optimized joint trajectory
    # unchanged.
    final_kinematics = TorchG1Kinematics(device="cpu", dtype=torch.float64)
    root_position, ground_projection = _project_root_above_ground(
        joint_position,
        root_position,
        root_quaternion,
        targets,
        config,
        final_kinematics,
    )
    optimization["ground_projection"] = ground_projection

    with torch.inference_mode():
        transforms = final_kinematics.forward(
            torch.as_tensor(
                joint_position,
                dtype=final_kinematics.dtype,
                device=final_kinematics.device,
            ),
            torch.as_tensor(
                root_position,
                dtype=final_kinematics.dtype,
                device=final_kinematics.device,
            ),
            torch.as_tensor(
                root_quaternion,
                dtype=final_kinematics.dtype,
                device=final_kinematics.device,
            ),
        )
        body_names = tuple(str(name) for name in config["isaac_body_names"])
        _, body_position, body_rotation = final_kinematics.stacked(
            transforms, body_names
        )
        sole_position = final_kinematics.sole_positions(transforms)
        support_position = final_kinematics.foot_support_positions(transforms)
    body_position_np = body_position.cpu().numpy().astype(np.float32)
    body_rotation_np = body_rotation.cpu().numpy()
    body_quaternion = matrix_to_quaternion_wxyz(body_rotation_np)
    sole_position_np = sole_position.cpu().numpy().astype(np.float32)
    support_position_np64 = support_position.cpu().numpy()
    support_position_np = support_position_np64.astype(np.float32)
    contact_point_position_np64 = np.stack(
        (
            support_position_np64[:, 0, :2].mean(axis=1),
            support_position_np64[:, 0, 2:].mean(axis=1),
            support_position_np64[:, 1, :2].mean(axis=1),
            support_position_np64[:, 1, 2:].mean(axis=1),
        ),
        axis=1,
    )
    contact_point_position_np = contact_point_position_np64.astype(np.float32)
    preliminary_stance = np.asarray(targets["stance_channels"], dtype=bool)
    output_stance, output_stance_runs = _refine_output_stance_channels(
        contact_point_position_np64,
        preliminary_stance,
        fps=fps,
        solver=solver_config,
    )
    preliminary_stance_frames = int(np.count_nonzero(preliminary_stance))
    output_stance_frames = int(np.count_nonzero(output_stance))
    stance_retention = (
        float(output_stance_frames / preliminary_stance_frames)
        if preliminary_stance_frames
        else 1.0
    )
    optimization["output_stance_refinement"] = {
        "preliminary_stance_frames": preliminary_stance_frames,
        "output_stance_frames": output_stance_frames,
        "retention_fraction": stance_retention,
        "run_count": len(output_stance_runs),
    }

    joint_velocity = _linear_velocity(joint_position, fps)
    joint_acceleration = _linear_velocity(joint_velocity, fps)
    root_linear_velocity = _linear_velocity(root_position, fps)
    root_angular_velocity = _angular_velocity_world(root_quaternion, fps)
    body_linear_velocity = _linear_velocity(body_position_np, fps)
    body_angular_velocity = _angular_velocity_world(body_quaternion, fps)
    contacts = np.asarray(targets["contacts"], dtype=bool)
    contract = load_g1_contract()
    correspondence_file = Path(config["_path"])
    source_checksum = _sha256(human_file)
    try:
        source_file = human_file.relative_to(PIPELINE_ROOT).as_posix()
    except ValueError:
        source_file = str(human_file)

    payload = {
        "schema_version": np.asarray("accad_g1_reference_v1"),
        "retarget_version": np.asarray(RETARGET_VERSION),
        "fps": np.asarray(fps, dtype=np.float64),
        "time": np.asarray(human["time"], dtype=np.float64),
        "frame_count": np.asarray(len(joint_position), dtype=np.int64),
        "coordinate_system": np.asarray("right-handed_x-forward_y-left_z-up"),
        "quaternion_order": np.asarray("wxyz"),
        "position_unit": np.asarray("meter"),
        "angle_unit": np.asarray("radian"),
        "joint_order": np.asarray("policy_index"),
        "joint_names": np.asarray(contract.policy_names),
        "sdk_joint_names": np.asarray(contract.sdk_names),
        "policy_to_sdk": np.asarray(contract.policy_to_sdk, dtype=np.int64),
        "joint_pos": joint_position,
        "joint_pos_sdk": policy_to_sdk(joint_position, contract),
        "joint_vel": joint_velocity,
        "joint_acc": joint_acceleration,
        "root_pos_w": root_position,
        "root_quat_wxyz": root_quaternion,
        "root_lin_vel_w": root_linear_velocity,
        "root_ang_vel_w": root_angular_velocity,
        "body_names": np.asarray(body_names),
        "body_pos_w": body_position_np,
        "body_quat_wxyz": body_quaternion,
        "body_lin_vel_w": body_linear_velocity,
        "body_ang_vel_w": body_angular_velocity,
        "tracking_body_names": np.asarray(config["tracking_body_names"]),
        "foot_sole_names": np.asarray(["left_sole", "right_sole"]),
        "foot_sole_position_w": sole_position_np,
        "foot_support_names": np.asarray(
            [
                "left_rear_outer",
                "left_rear_inner",
                "left_front_outer",
                "left_front_inner",
                "right_rear_inner",
                "right_rear_outer",
                "right_front_inner",
                "right_front_outer",
            ]
        ),
        "foot_support_position_w": support_position_np.reshape(
            len(support_position_np), 8, 3
        ),
        "foot_collision_radius_m": np.asarray(
            FOOT_COLLISION_RADIUS_M, dtype=np.float32
        ),
        "left_foot_contact": contacts[:, 0],
        "right_foot_contact": contacts[:, 1],
        "contact_point_names": np.asarray(
            ["left_heel", "left_toe", "right_heel", "right_toe"]
        ),
        "contact_point_position_w": contact_point_position_np,
        "left_heel_contact": np.asarray(targets["contact_channels"][:, 0]),
        "left_toe_contact": np.asarray(targets["contact_channels"][:, 1]),
        "right_heel_contact": np.asarray(targets["contact_channels"][:, 2]),
        "right_toe_contact": np.asarray(targets["contact_channels"][:, 3]),
        "left_heel_stance": output_stance[:, 0],
        "left_toe_stance": output_stance[:, 1],
        "right_heel_stance": output_stance[:, 2],
        "right_toe_stance": output_stance[:, 3],
        "morphology_scale": np.asarray(
            targets["morphology_scale"], dtype=np.float32
        ),
        "human_leg_height_m": np.asarray(
            targets["human_leg_height_m"], dtype=np.float32
        ),
        "g1_leg_height_m": np.asarray(
            targets["g1_leg_height_m"], dtype=np.float32
        ),
        "source_file": np.asarray(source_file),
        "source_checksum_sha256": np.asarray(source_checksum),
        "source_human_schema": np.asarray(_load_scalar(human, "schema_version")),
        "source_stageii_file": np.asarray(_load_scalar(human, "source_file")),
        "source_stageii_checksum_sha256": np.asarray(
            _load_scalar(human, "source_checksum_sha256")
        ),
        "g1_contract_path": np.asarray(str(G1_CONTRACT_PATH.resolve())),
        "g1_contract_checksum_sha256": np.asarray(_sha256(G1_CONTRACT_PATH)),
        "g1_urdf_path": np.asarray(str(G1_URDF_PATH.resolve())),
        "g1_urdf_checksum_sha256": np.asarray(_sha256(G1_URDF_PATH)),
        "correspondence_path": np.asarray(str(correspondence_file)),
        "correspondence_checksum_sha256": np.asarray(
            _sha256(correspondence_file)
        ),
        "solver_name": np.asarray("torch_global_sequence_adam"),
        "solver_iterations": np.asarray(
            optimization["iterations"], dtype=np.int64
        ),
        "tracking_action_mode": np.asarray(config["tracking_action"]["mode"]),
        "tracking_residual_scale": np.asarray(
            config["tracking_action"]["residual_scale"], dtype=np.float32
        ),
        "tracking_action_clip": np.asarray(
            config["tracking_action"]["clip"], dtype=np.float32
        ),
        "tracking_normalized_delta_limit": np.asarray(
            config["tracking_action"]["normalized_delta_limit"], dtype=np.float32
        ),
        "contact_lock_run_count": np.asarray(
            len(targets["contact_runs"]), dtype=np.int64
        ),
        "contact_semantics": np.asarray(
            "contact=human_near_ground; stance=stationary_final_g1_support_subset_of_load_bearing_source_stance"
        ),
        "preliminary_stance_frame_count": np.asarray(
            preliminary_stance_frames, dtype=np.int64
        ),
        "output_stance_frame_count": np.asarray(
            output_stance_frames, dtype=np.int64
        ),
        "output_stance_retention_fraction": np.asarray(
            stance_retention, dtype=np.float32
        ),
        "output_stance_run_count": np.asarray(
            len(output_stance_runs), dtype=np.int64
        ),
        "ground_projection_maximum_z_shift_m": np.asarray(
            ground_projection["maximum_z_shift_m"], dtype=np.float32
        ),
        "ground_projection_maximum_limit_excess_m": np.asarray(
            ground_projection["maximum_limit_excess_m"], dtype=np.float32
        ),
        "ground_projection_ground_margin_m": np.asarray(
            ground_projection["ground_margin_m"], dtype=np.float32
        ),
        "stance_speed_threshold_mps": np.asarray(
            solver_config["stance_speed_threshold_mps"], dtype=np.float32
        ),
        "stance_target_drift_limit_m": np.asarray(
            solver_config["stance_target_drift_limit_m"], dtype=np.float32
        ),
        "stance_minimum_frames": np.asarray(
            solver_config["stance_minimum_frames"], dtype=np.int64
        ),
        "stance_maximum_anchor_frames": np.asarray(
            solver_config["stance_maximum_anchor_frames"], dtype=np.int64
        ),
        "stance_minimum_pelvis_height_ratio": np.asarray(
            solver_config["stance_minimum_pelvis_height_ratio"], dtype=np.float32
        ),
        "stance_eligible_fraction": np.asarray(
            np.mean(targets["stance_eligible"]), dtype=np.float32
        ),
        "gate3_complete": np.asarray(False),
        "gate3_status": np.asarray(
            "retarget basis generated; independent Pinocchio and Isaac validation pending"
        ),
    }
    _save_npz_atomic(output_file, payload)
    return {
        "ok": True,
        "gate3_complete": False,
        "output_path": str(output_file),
        "frames": len(joint_position),
        "fps": fps,
        "morphology_scale": targets["morphology_scale"],
        "optimization": optimization,
        "archive_checksum_sha256": _sha256(output_file),
        "source_human_path": str(human_file),
        "source_human_checksum_sha256": source_checksum,
        "g1_contract_checksum_sha256": _sha256(G1_CONTRACT_PATH),
        "g1_urdf_checksum_sha256": _sha256(G1_URDF_PATH),
        "correspondence_checksum_sha256": _sha256(correspondence_file),
        "configuration": json.loads(json.dumps(config, default=str)),
    }
