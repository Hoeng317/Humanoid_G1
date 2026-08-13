"""Isaac Lab multi-motion tracking task for ACCAD-retargeted Unitree G1 clips.

This module is intentionally self-contained inside ``ACCAD/_g1_pipeline``.  It
reads the parent project's :data:`G1_29DOF_CFG`, but never writes outside the
ACCAD tree.  The public action and joint-state axes are *always* the 29-entry
``policy_index`` order from ``configs/robot/g1_29dof.yaml``; articulation and
contact-sensor indices are resolved by exact names at Isaac runtime.

The small quaternion/alignment/action helpers at the beginning of the file have
no Isaac Sim dependency and are covered by ordinary CPU tests.  Isaac classes
are defined when the module is imported after :class:`isaaclab.app.AppLauncher`
has started.  This lets preprocessing CI test the numerical contract without
booting Kit while preserving a normal ManagerBasedRLEnv configuration for
training and evaluation.

Default tensor contracts
------------------------

* raw policy action: ``[N, 29]`` normalized residual in ``[-1, 1]``;
* position target: ``q_ref + 0.25 * rate_limit(clip(action), 0.15 / control step)``;
* velocity target: zero (the empirically selected position-PD contract);
* actor observation: ``[N, 293]`` for horizons ``(0, .04, .08, .16)``;
* critic observation: ``[N, 505]`` for 14 tracking bodies;
* physics/control rates: 200 Hz / 50 Hz.

All dimensions are checked explicitly by :func:`tracking_observation_dims` and
again when the command/action terms bind to a live articulation.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .g1_kinematics import G1_CONTRACT_PATH, G1_URDF_PATH, load_g1_contract
from .motion_library import MotionLibrary
from .paths import ACCAD_ROOT
from .training_io import (
    DEFAULT_TRAINING_MOTION_ASSIGNMENT,
    TRAINING_MOTION_ASSIGNMENT_CHOICES,
    TRAINING_MOTION_ASSIGNMENT_SCHEMA_VERSION,
)


G1_CONTRACT = load_g1_contract()
G1_POLICY_JOINT_NAMES = G1_CONTRACT.policy_names
G1_ACTION_DIM = len(G1_POLICY_JOINT_NAMES)
DEFAULT_FUTURE_HORIZONS_S = (0.0, 0.04, 0.08, 0.16)
DEFAULT_TRACKING_BODY_COUNT = 14
TRACKING_POLICY_CONTRACT_VERSION = (
    "accad_g1_tracking_v7_zero_velocity_target_train_selected"
)
REFERENCE_VELOCITY_FEEDFORWARD_SCALE = 0.0
RECOVERY_PPO_LEARNING_RATE = 1.0e-4
RECOVERY_PPO_SCHEDULE = "fixed"
RECOVERY_PPO_DESIRED_KL = 0.01
DEFAULT_ACTION_NOISE_STD = 0.2
RECOVERY_RESET_STATE_NOISE_SCALE = 0.0
RECOVERY_ALIVE_REWARD_WEIGHT = 0.25
RECOVERY_FAILURE_REWARD_WEIGHT = -10.0
RECOVERY_DIVERGENCE_RISK_REWARD_WEIGHT = -20.0
RECOVERY_ACTION_BOUNDARY_REWARD_WEIGHT = -2.0
RECOVERY_ACTION_LIMITER_GAP_REWARD_WEIGHT = -1.0
DIVERGENCE_RISK_ONSET_FRACTION = 0.60
ACTION_BOUNDARY_ONSET = 0.90
TRACKING_DIVERGENCE_MAX_ROOT_DISTANCE_M = 0.80
TRACKING_DIVERGENCE_MAX_MEAN_JOINT_ERROR_RAD = 1.0
TRACKING_DIVERGENCE_MAX_MEAN_BODY_DISTANCE_M = 0.50
TRAINING_OBJECTIVE_SCHEMA_VERSION = (
    "accad_g1_training_objective_v4_strong_dense_divergence_action_margins"
)
EVALUATION_ISOLATION_SCHEMA_VERSION = (
    "accad_g1_evaluation_isolation_v2_terminal_failure_impulse"
)


def tracking_observation_dims(
    *,
    num_joints: int = G1_ACTION_DIM,
    num_horizons: int = len(DEFAULT_FUTURE_HORIZONS_S),
    num_tracking_bodies: int = DEFAULT_TRACKING_BODY_COUNT,
) -> tuple[int, int]:
    """Return the exact concatenated actor and critic observation dimensions.

    The actor contains future joint errors, positive-horizon root pose previews,
    current joint/root errors, root velocity commands, phase, gravity, joint
    proprioception, future contacts, and the previous normalized residual.  The
    critic contains that complete actor vector plus full tracking-body pose and
    velocity errors and measured foot contacts.
    """

    if num_joints <= 0 or num_horizons <= 0 or num_tracking_bodies <= 0:
        raise ValueError("joint, horizon, and tracking-body counts must be positive")
    actor = (
        num_horizons * num_joints  # reference joint-position errors
        + (num_horizons - 1) * 3  # positive-horizon root-position errors
        + (num_horizons - 1) * 6  # positive-horizon root-orientation errors
        + num_joints  # current reference joint-velocity error
        + 6  # root orientation error, 6-D rotation representation
        + 3  # root position error in the actual-root frame
        + 3  # root linear-velocity error in the actual-root frame
        + 3  # target root linear velocity in the actual-root frame
        + 2  # sin/cos phase
        + 3  # root angular-velocity error in the actual-root frame
        + 3  # target root angular velocity in the actual-root frame
        + 3  # projected gravity
        + num_joints  # measured joint position relative to nominal
        + num_joints  # measured joint velocity
        + 2 * num_horizons  # left/right reference contact schedule
        + num_joints  # previous processed normalized residual
    )
    critic = actor + num_tracking_bodies * (3 + 6 + 3 + 3) + 2
    return actor, critic


DEFAULT_ACTOR_OBS_DIM, DEFAULT_CRITIC_OBS_DIM = tracking_observation_dims()


def tracking_observation_contract() -> dict[str, Any]:
    """Return the JSON-safe canonical v6 observation/action tensor contract.

    Ranges are half-open ``[start, end)`` slices in the concatenated vector.
    Root command and preview quantities use the *measured* G1 pelvis frame at
    the current control step.  In particular, future root observations exclude
    the zero horizon; the current root errors already encode that state.
    """

    positive_horizon_count = len(DEFAULT_FUTURE_HORIZONS_S) - 1
    policy_specs: tuple[tuple[str, int, float, str, str], ...] = (
        (
            "future_joint_position_error",
            len(DEFAULT_FUTURE_HORIZONS_S) * G1_ACTION_DIM,
            1.0,
            "g1_policy_joint_order",
            "q_ref(t+h)-q_actual(t), h in [0,.04,.08,.16]",
        ),
        (
            "future_root_position_error",
            positive_horizon_count * 3,
            1.0,
            "actual_g1_pelvis",
            "R_actual(t)^T[p_ref(t+h)-p_actual(t)], h in [.04,.08,.16]",
        ),
        (
            "future_root_orientation_error",
            positive_horizon_count * 6,
            1.0,
            "actual_g1_pelvis",
            "rot6d(q_actual(t)^* otimes q_ref(t+h)), h in [.04,.08,.16]",
        ),
        (
            "joint_velocity_error",
            G1_ACTION_DIM,
            0.05,
            "g1_policy_joint_order",
            "dq_ref(t)-dq_actual(t)",
        ),
        (
            "root_orientation_error",
            6,
            1.0,
            "actual_g1_pelvis",
            "rot6d(q_actual(t)^* otimes q_ref(t))",
        ),
        (
            "root_position_error",
            3,
            1.0,
            "actual_g1_pelvis",
            "R_actual(t)^T[p_ref(t)-p_actual(t)]",
        ),
        (
            "root_linear_velocity_error",
            3,
            0.2,
            "actual_g1_pelvis",
            "R_actual(t)^T[v_ref(t)-v_actual(t)]",
        ),
        (
            "target_root_linear_velocity",
            3,
            0.2,
            "actual_g1_pelvis",
            "R_actual(t)^T v_ref(t)",
        ),
        ("phase", 2, 1.0, "motion_phase", "[sin(2*pi*phase),cos(2*pi*phase)]"),
        (
            "root_angular_velocity_error",
            3,
            0.2,
            "actual_g1_pelvis",
            "R_actual(t)^T[w_ref(t)-w_actual(t)]",
        ),
        (
            "target_root_angular_velocity",
            3,
            0.2,
            "actual_g1_pelvis",
            "R_actual(t)^T w_ref(t)",
        ),
        (
            "projected_gravity",
            3,
            1.0,
            "actual_g1_pelvis",
            "gravity direction projected into the measured pelvis frame",
        ),
        (
            "joint_position",
            G1_ACTION_DIM,
            1.0,
            "g1_policy_joint_order",
            "q_actual(t)-q_default",
        ),
        (
            "joint_velocity",
            G1_ACTION_DIM,
            0.05,
            "g1_policy_joint_order",
            "dq_actual(t)",
        ),
        (
            "predicted_contacts",
            2 * len(DEFAULT_FUTURE_HORIZONS_S),
            1.0,
            "left_right_feet",
            "reference contacts at h in [0,.04,.08,.16]",
        ),
        (
            "last_action",
            G1_ACTION_DIM,
            1.0,
            "g1_policy_joint_order",
            "last processed normalized joint residual",
        ),
    )
    privileged_specs: tuple[tuple[str, int, float, str, str], ...] = (
        (
            "tracking_body_position_error",
            DEFAULT_TRACKING_BODY_COUNT * 3,
            1.0,
            "actual_g1_pelvis",
            "R_actual_root^T[p_ref_body-p_actual_body]",
        ),
        (
            "tracking_body_orientation_error",
            DEFAULT_TRACKING_BODY_COUNT * 6,
            1.0,
            "actual_tracking_body",
            "rot6d(q_actual_body^* otimes q_ref_body)",
        ),
        (
            "tracking_body_linear_velocity_error",
            DEFAULT_TRACKING_BODY_COUNT * 3,
            0.2,
            "actual_g1_pelvis",
            "R_actual_root^T[v_ref_body-v_actual_body]",
        ),
        (
            "tracking_body_angular_velocity_error",
            DEFAULT_TRACKING_BODY_COUNT * 3,
            0.2,
            "actual_g1_pelvis",
            "R_actual_root^T[w_ref_body-w_actual_body]",
        ),
        (
            "actual_foot_contacts",
            2,
            1.0,
            "left_right_feet",
            "measured binary foot contacts",
        ),
    )

    def group(specs: Sequence[tuple[str, int, float, str, str]]) -> dict[str, Any]:
        offset = 0
        terms: list[dict[str, Any]] = []
        for name, dimension, scale, frame, formula in specs:
            end = offset + dimension
            terms.append(
                {
                    "name": name,
                    "dimension": dimension,
                    "range": [offset, end],
                    "scale": scale,
                    "frame": frame,
                    "formula": formula,
                }
            )
            offset = end
        return {
            "concatenate_terms": True,
            "dimension": offset,
            "terms": terms,
        }

    policy_group = group(policy_specs)
    critic_group = group(policy_specs + privileged_specs)
    if policy_group["dimension"] != DEFAULT_ACTOR_OBS_DIM:
        raise RuntimeError("canonical policy observation terms do not total 293")
    if critic_group["dimension"] != DEFAULT_CRITIC_OBS_DIM:
        raise RuntimeError("canonical critic observation terms do not total 505")
    return {
        "schema_version": "accad_g1_tracking_observation_contract_v6",
        "policy_contract_version": TRACKING_POLICY_CONTRACT_VERSION,
        "range_semantics": "half_open_[start,end)",
        "future_horizons_s": [float(value) for value in DEFAULT_FUTURE_HORIZONS_S],
        "positive_preview_horizons_s": [
            float(value) for value in DEFAULT_FUTURE_HORIZONS_S[1:]
        ],
        "num_policy_joints": G1_ACTION_DIM,
        "num_tracking_bodies": DEFAULT_TRACKING_BODY_COUNT,
        "observation_groups": {
            "policy": policy_group,
            "critic": critic_group,
        },
        "action": {
            "term_names": ["joint_residual"],
            "term_dimensions": [G1_ACTION_DIM],
            "dimension": G1_ACTION_DIM,
            "position_target_formula": (
                "clip_limits(q_ref + 0.25 * rate_limit(clip(action,-1,1), "
                "0.15_per_control_step))"
            ),
            "velocity_target_formula": "zeros_like(reference_joint_velocity)",
            "reference_velocity_feedforward_scale": REFERENCE_VELOCITY_FEEDFORWARD_SCALE,
        },
    }


def resolve_recovery_training_objective(
    reset_state_noise_scale: float,
    alive_reward_weight: float,
    failure_reward_weight: float,
) -> tuple[float, float, float]:
    """Validate and normalize the explicit recovery-training objective."""

    values = (
        (reset_state_noise_scale, "reset_state_noise_scale"),
        (alive_reward_weight, "alive_reward_weight"),
        (failure_reward_weight, "failure_reward_weight"),
    )
    for value, name in values:
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
    scale = float(reset_state_noise_scale)
    alive = float(alive_reward_weight)
    failure = float(failure_reward_weight)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("reset_state_noise_scale must be in [0, 1]")
    if alive <= 0.0:
        raise ValueError("alive_reward_weight must be positive")
    if failure >= 0.0:
        raise ValueError("failure_reward_weight must be negative")
    return scale, alive, failure


def terminal_failure_impulse(
    terminated: torch.Tensor,
    step_dt: float,
) -> torch.Tensor:
    """Return a one-shot non-timeout failure indicator before reward integration.

    Isaac Lab's :class:`RewardManager` multiplies every reward term by the
    environment step duration.  A terminal outcome penalty is an impulse, not
    a per-second rate, so dividing the boolean non-timeout termination signal
    by ``step_dt`` makes the manager-integrated contribution equal the
    configured weight exactly once.  ``TerminationManager.terminated`` excludes
    pure timeouts (including a clean ``motion_end``) but remains true when a
    timeout and a physical failure occur on the same step.
    """

    if terminated.dtype is not torch.bool or terminated.ndim != 1:
        raise ValueError("terminated must be a rank-one boolean tensor")
    if isinstance(step_dt, bool) or not math.isfinite(float(step_dt)) or float(step_dt) <= 0.0:
        raise ValueError("step_dt must be finite and positive")
    return terminated.to(dtype=torch.float32) / float(step_dt)


def resolve_action_noise_std(value: float, *, name: str = "action_noise_std") -> float:
    """Validate a requested independent Gaussian action-noise deviation."""

    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite positive number")
    resolved = float(value)
    if resolved <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return resolved


def resolved_policy_action_noise_contract(policy: Any) -> dict[str, Any]:
    """Return the exact live RSL-RL action-noise parameter layout.

    The ACCAD policy contract supports only action-independent, per-joint
    Gaussian parameters.  Both RSL-RL representations are recognized:
    ``std`` for scalar standard deviation and ``log_std`` for log standard
    deviation.  State-dependent or otherwise ambiguous layouts fail closed.
    """

    if getattr(policy, "state_dependent_std", None) is not False:
        raise RuntimeError(
            "policy action-noise override requires state_dependent_std=False"
        )
    noise_std_type = getattr(policy, "noise_std_type", None)
    if noise_std_type == "scalar":
        parameter_name = "std"
    elif noise_std_type == "log":
        parameter_name = "log_std"
    else:
        raise RuntimeError(
            "policy action-noise layout must use noise_std_type 'scalar' or 'log'"
        )
    parameter = getattr(policy, parameter_name, None)
    if not isinstance(parameter, torch.nn.Parameter):
        raise RuntimeError(
            f"policy action-noise layout is missing Parameter {parameter_name!r}"
        )
    if tuple(parameter.shape) != (G1_ACTION_DIM,):
        raise RuntimeError(
            f"policy {parameter_name} must have shape ({G1_ACTION_DIM},); "
            f"got {tuple(parameter.shape)}"
        )
    detached = parameter.detach()
    if not bool(torch.isfinite(detached).all()):
        raise RuntimeError(f"policy {parameter_name} contains NaN or Inf")
    resolved_std = detached if noise_std_type == "scalar" else torch.exp(detached)
    if not bool(torch.isfinite(resolved_std).all()) or bool(torch.any(resolved_std <= 0.0)):
        raise RuntimeError("resolved policy action standard deviation is not finite and positive")
    parameter_values = [float(value) for value in detached.cpu().tolist()]
    resolved_values = [float(value) for value in resolved_std.cpu().tolist()]
    return {
        "schema_version": "accad_g1_policy_action_noise_contract_v1",
        "noise_std_type": noise_std_type,
        "state_dependent_std": False,
        "parameter_name": parameter_name,
        "parameter_shape": [G1_ACTION_DIM],
        "parameter_values": parameter_values,
        "resolved_std_per_action": resolved_values,
        "resolved_std_min": min(resolved_values),
        "resolved_std_max": max(resolved_values),
    }


def set_policy_action_noise_std(policy: Any, value: float) -> dict[str, Any]:
    """Set every live action-noise axis to ``value`` and return a witness."""

    requested = resolve_action_noise_std(value, name="loaded_action_noise_std")
    before = resolved_policy_action_noise_contract(policy)
    parameter = getattr(policy, before["parameter_name"])
    stored_value = requested if before["noise_std_type"] == "scalar" else math.log(requested)
    with torch.no_grad():
        parameter.fill_(stored_value)
    expected = torch.full_like(parameter.detach(), stored_value)
    if not torch.equal(parameter.detach(), expected):
        raise RuntimeError("failed to set the loaded policy action-noise parameter exactly")
    after = resolved_policy_action_noise_contract(policy)
    return {
        "schema_version": "accad_g1_loaded_action_noise_override_v1",
        "requested_std": requested,
        "stored_parameter_value": float(expected[0].item()),
        "before": before,
        "after": after,
    }


def set_ppo_learning_rate_exact(algorithm: Any, value: float) -> dict[str, Any]:
    """Set the algorithm and every optimizer group LR, with a before/after witness."""

    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError("learning_rate must be a finite positive number")
    requested = float(value)
    optimizer = getattr(algorithm, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, list) or not param_groups:
        raise RuntimeError("PPO optimizer must expose a non-empty param_groups list")

    def snapshot() -> dict[str, Any]:
        algorithm_lr = getattr(algorithm, "learning_rate", None)
        if (
            isinstance(algorithm_lr, bool)
            or not isinstance(algorithm_lr, (int, float))
            or not math.isfinite(float(algorithm_lr))
        ):
            raise RuntimeError("PPO algorithm.learning_rate is not a finite scalar")
        group_lrs: list[float] = []
        for index, group in enumerate(param_groups):
            if not isinstance(group, dict) or "lr" not in group:
                raise RuntimeError(
                    f"PPO optimizer param_group {index} does not expose an lr"
                )
            group_lr = group["lr"]
            if (
                isinstance(group_lr, bool)
                or not isinstance(group_lr, (int, float))
                or not math.isfinite(float(group_lr))
            ):
                raise RuntimeError(
                    f"PPO optimizer param_group {index} lr is not a finite scalar"
                )
            group_lrs.append(float(group_lr))
        return {
            "algorithm_learning_rate": float(algorithm_lr),
            "optimizer_param_group_learning_rates": group_lrs,
        }

    before = snapshot()
    algorithm.learning_rate = requested
    for group in param_groups:
        group["lr"] = requested
    after = snapshot()
    if after["algorithm_learning_rate"] != requested or any(
        group_lr != requested
        for group_lr in after["optimizer_param_group_learning_rates"]
    ):
        raise RuntimeError("failed to set the PPO learning rate exactly")
    return {
        "schema_version": "accad_g1_ppo_learning_rate_override_v1",
        "requested_learning_rate": requested,
        "before": before,
        "after": after,
    }


def _require_last_dim(value: torch.Tensor, size: int, name: str) -> None:
    if value.ndim == 0 or value.shape[-1] != size:
        raise ValueError(f"{name} must end in {size} values; got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf")


def normalize_quat_wxyz(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Normalize a wxyz quaternion tensor and reject degenerate values."""

    _require_last_dim(quaternion, 4, "quaternion")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool(torch.any(norm <= eps)):
        raise ValueError("quaternion norm is zero or numerically degenerate")
    return quaternion / norm


def quat_conjugate_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of a wxyz quaternion."""

    quaternion = normalize_quat_wxyz(quaternion)
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def quat_mul_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product of broadcast-compatible normalized wxyz quaternions."""

    left = normalize_quat_wxyz(left)
    right = normalize_quat_wxyz(right)
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    result = torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )
    return normalize_quat_wxyz(result)


def quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate broadcast-compatible 3-D vectors by normalized wxyz quaternions."""

    quaternion = normalize_quat_wxyz(quaternion)
    _require_last_dim(vector, 3, "vector")
    q_vec = quaternion[..., 1:]
    uv = torch.linalg.cross(q_vec, vector, dim=-1)
    return vector + 2.0 * (
        quaternion[..., :1] * uv + torch.linalg.cross(q_vec, uv, dim=-1)
    )


def yaw_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    """Construct normalized Z-yaw quaternions in wxyz order."""

    if not bool(torch.isfinite(yaw).all()):
        raise ValueError("yaw contains NaN or Inf")
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


def quat_yaw_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Extract intrinsic Z yaw from normalized wxyz quaternions."""

    quaternion = normalize_quat_wxyz(quaternion)
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_to_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized wxyz quaternions to rotation matrices."""

    quaternion = normalize_quat_wxyz(quaternion)
    w, x, y, z = quaternion.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def quat_error_angle_wxyz(reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Return shortest absolute orientation error in radians."""

    error = quat_mul_wxyz(quat_conjugate_wxyz(actual), reference)
    return 2.0 * torch.acos(torch.abs(error[..., 0]).clamp(0.0, 1.0))


def quat_error_6d_wxyz(reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Return relative orientation as the first two matrix columns (six values)."""

    error = quat_mul_wxyz(quat_conjugate_wxyz(actual), reference)
    matrix = quat_to_matrix_wxyz(error)
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,))


def root_frame_vector(
    vector_w: torch.Tensor,
    actual_root_quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Express a world-frame vector in the measured G1 pelvis frame."""

    _require_last_dim(vector_w, 3, "vector_w")
    _require_last_dim(actual_root_quaternion_wxyz, 4, "actual_root_quaternion_wxyz")
    if vector_w.shape[:-1] != actual_root_quaternion_wxyz.shape[:-1]:
        raise ValueError("root quaternion leading dimensions must match vector_w")
    return quat_apply_wxyz(
        quat_conjugate_wxyz(actual_root_quaternion_wxyz),
        vector_w,
    )


def root_frame_vector_error(
    reference_w: torch.Tensor,
    actual_w: torch.Tensor,
    actual_root_quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Return ``reference - actual`` expressed in the actual-root frame.

    This pure helper fixes the sign, link, and coordinate-frame contract shared
    by deployable root position and linear-velocity observations.
    """

    _require_last_dim(reference_w, 3, "reference_w")
    _require_last_dim(actual_w, 3, "actual_w")
    _require_last_dim(actual_root_quaternion_wxyz, 4, "actual_root_quaternion_wxyz")
    if reference_w.shape != actual_w.shape:
        raise ValueError("reference_w and actual_w shapes must match")
    if actual_root_quaternion_wxyz.shape[:-1] != actual_w.shape[:-1]:
        raise ValueError("root quaternion leading dimensions must match the vectors")
    return root_frame_vector(reference_w - actual_w, actual_root_quaternion_wxyz)


def positive_horizon_root_pose_preview(
    future_reference_position_w: torch.Tensor,
    future_reference_quaternion_wxyz: torch.Tensor,
    actual_root_position_w: torch.Tensor,
    actual_root_quaternion_wxyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return h>0 reference root pose relative to the current measured pelvis.

    Inputs use ``[env, horizon, xyz/quat]`` for the future sequence, whose first
    element is h=0, and ``[env, xyz/quat]`` for the current measured G1 pelvis.
    The returned tensors are ``[env, horizon-1, 3]`` position errors and
    ``[env, horizon-1, 6]`` relative-orientation representations.  This helper
    deliberately drops h=0 so the preview cannot duplicate the current root
    pose observation.
    """

    _require_last_dim(future_reference_position_w, 3, "future_reference_position_w")
    _require_last_dim(
        future_reference_quaternion_wxyz,
        4,
        "future_reference_quaternion_wxyz",
    )
    _require_last_dim(actual_root_position_w, 3, "actual_root_position_w")
    _require_last_dim(actual_root_quaternion_wxyz, 4, "actual_root_quaternion_wxyz")
    if future_reference_position_w.ndim != 3:
        raise ValueError("future_reference_position_w must have shape [env, horizon, 3]")
    if future_reference_quaternion_wxyz.ndim != 3:
        raise ValueError(
            "future_reference_quaternion_wxyz must have shape [env, horizon, 4]"
        )
    if actual_root_position_w.ndim != 2 or actual_root_quaternion_wxyz.ndim != 2:
        raise ValueError("actual root position/quaternion must have rank two")
    if future_reference_position_w.shape[:2] != future_reference_quaternion_wxyz.shape[:2]:
        raise ValueError("future root position/quaternion env and horizon dimensions differ")
    env_count, horizon_count = future_reference_position_w.shape[:2]
    if horizon_count < 2:
        raise ValueError("future root preview requires h=0 and at least one positive horizon")
    if actual_root_position_w.shape[0] != env_count or actual_root_quaternion_wxyz.shape[0] != env_count:
        raise ValueError("future and actual root environment counts differ")

    positive_position = future_reference_position_w[:, 1:]
    positive_quaternion = future_reference_quaternion_wxyz[:, 1:]
    root_inverse = quat_conjugate_wxyz(actual_root_quaternion_wxyz).unsqueeze(1)
    root_inverse = root_inverse.expand(-1, horizon_count - 1, -1)
    position_error = quat_apply_wxyz(
        root_inverse,
        positive_position - actual_root_position_w.unsqueeze(1),
    )
    relative_quaternion = quat_mul_wxyz(root_inverse, positive_quaternion)
    orientation_error = quat_to_matrix_wxyz(relative_quaternion)[..., :, :2].reshape(
        env_count, horizon_count - 1, 6
    )
    return position_error, orientation_error


def mix_frame_zero_start_times(
    random_start_times: torch.Tensor,
    frame_zero_probability: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix exact frame-zero starts with valid continuous random start times.

    Returns the selected times and a boolean mask whose true entries are exact
    frame-zero samples.  Edge probabilities do not consume random numbers,
    which keeps seeded sampling reproducible for the two limiting contracts.
    """

    if random_start_times.ndim != 1 or not bool(torch.isfinite(random_start_times).all()):
        raise ValueError("random_start_times must be a finite rank-one tensor")
    if bool(torch.any(random_start_times < 0.0)):
        raise ValueError("random_start_times cannot be negative")
    if isinstance(frame_zero_probability, bool) or not math.isfinite(frame_zero_probability):
        raise ValueError("frame_zero_probability must be a finite number in [0, 1]")
    probability = float(frame_zero_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("frame_zero_probability must be in [0, 1]")
    if probability == 0.0:
        zero_mask = torch.zeros_like(random_start_times, dtype=torch.bool)
    elif probability == 1.0:
        zero_mask = torch.ones_like(random_start_times, dtype=torch.bool)
    else:
        zero_mask = torch.rand(
            random_start_times.shape,
            device=random_start_times.device,
            dtype=random_start_times.dtype,
            generator=generator,
        ) < probability
    return torch.where(zero_mask, torch.zeros_like(random_start_times), random_start_times), zero_mask


def raw_action_excess_l2(raw_action: torch.Tensor, action_clip: float = 1.0) -> torch.Tensor:
    """Mean squared policy output beyond the action term's safe clip envelope."""

    if raw_action.ndim < 1 or not bool(torch.isfinite(raw_action).all()):
        raise ValueError("raw_action must be a finite tensor with an action axis")
    if isinstance(action_clip, bool) or not math.isfinite(action_clip) or action_clip <= 0.0:
        raise ValueError("action_clip must be finite and positive")
    excess = (torch.abs(raw_action) - float(action_clip)).clamp_min(0.0)
    return torch.mean(torch.square(excess), dim=-1)


def action_boundary_margin_l2(
    raw_action: torch.Tensor,
    *,
    onset: float = ACTION_BOUNDARY_ONSET,
    action_clip: float = 1.0,
) -> torch.Tensor:
    """Squared action-boundary barrier that reaches one at the safety clip.

    The old excess-only loss was exactly zero until an action had already left
    the physical envelope.  This bounded margin gives PPO a dense warning from
    ``onset`` to ``action_clip`` without changing or hiding the raw action used
    by deterministic acceptance metrics.
    """

    if raw_action.ndim < 1 or not bool(torch.isfinite(raw_action).all()):
        raise ValueError("raw_action must be a finite tensor with an action axis")
    values = ((onset, "onset"), (action_clip, "action_clip"))
    for value, name in values:
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if onset < 0.0 or action_clip <= onset:
        raise ValueError("action boundary requires 0 <= onset < action_clip")
    normalized = (torch.abs(raw_action) - float(onset)) / float(action_clip - onset)
    return torch.mean(torch.square(normalized.clamp(0.0, 1.0)), dim=-1)


def action_limiter_request_gap_l2(
    raw_action: torch.Tensor,
    normalized_residual: torch.Tensor,
    *,
    action_clip: float = 1.0,
) -> torch.Tensor:
    """Expose policy requests rejected by the normalized residual rate limiter."""

    if raw_action.shape != normalized_residual.shape or raw_action.ndim < 1:
        raise ValueError("raw_action and normalized_residual must share an action shape")
    if not bool(torch.isfinite(raw_action).all()) or not bool(
        torch.isfinite(normalized_residual).all()
    ):
        raise ValueError("action limiter inputs must be finite")
    if isinstance(action_clip, bool) or not math.isfinite(action_clip) or action_clip <= 0.0:
        raise ValueError("action_clip must be finite and positive")
    requested = torch.clamp(raw_action, -float(action_clip), float(action_clip))
    return torch.mean(torch.square(requested - normalized_residual), dim=-1)


def deterministic_suite_actions(
    action_source: str,
    observations: Any,
    action_dim: int,
    *,
    checkpoint_policy: Any | None = None,
) -> torch.Tensor:
    """Resolve deterministic checkpoint or zero-residual suite actions.

    This helper has no Isaac dependency, allowing the diagnostic source
    contract to be tested without starting Kit.  Shape and finiteness checks
    that depend on the evaluator's active first-episode mask remain at the call
    site.
    """

    if action_source not in {"checkpoint", "zero"}:
        raise ValueError("action_source must be 'checkpoint' or 'zero'")
    if isinstance(observations, torch.Tensor):
        if observations.ndim != 2 or observations.shape[0] <= 0:
            raise ValueError("tensor observations must have shape [environment, feature]")
        environment_count = int(observations.shape[0])
        zero_device = observations.device
        zero_dtype = observations.dtype
    else:
        batch_size = getattr(observations, "batch_size", None)
        if batch_size is None or len(batch_size) != 1 or int(batch_size[0]) <= 0:
            raise ValueError(
                "observations must be a tensor or a TensorDict with one environment batch axis"
            )
        environment_count = int(batch_size[0])
        zero_device = getattr(observations, "device", None)
        zero_dtype = torch.float32
        policy_observation = None
        getter = getattr(observations, "get", None)
        if callable(getter):
            policy_observation = getter("policy", None)
        if isinstance(policy_observation, torch.Tensor):
            zero_device = policy_observation.device
            zero_dtype = policy_observation.dtype
    if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
        raise ValueError("action_dim must be a positive integer")
    if action_source == "zero":
        return torch.zeros(
            (environment_count, action_dim),
            device=zero_device,
            dtype=zero_dtype,
        )
    if checkpoint_policy is None or not callable(checkpoint_policy):
        raise ValueError("checkpoint action source requires a callable policy")
    result = checkpoint_policy(observations)
    if not isinstance(result, torch.Tensor):
        raise TypeError("checkpoint policy must return a torch.Tensor")
    return result


def advance_motion_time_after_step(
    motion_time: torch.Tensor,
    reset_mask: torch.Tensor,
    step_dt: float,
) -> torch.Tensor:
    """Advance only episodes that actually executed the just-finished step."""

    if motion_time.ndim != 1 or not bool(torch.isfinite(motion_time).all()):
        raise ValueError("motion_time must be a finite rank-one tensor")
    if reset_mask.shape != motion_time.shape or reset_mask.dtype != torch.bool:
        raise ValueError("reset_mask must be boolean with the same shape as motion_time")
    if isinstance(step_dt, bool) or not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("step_dt must be finite and positive")
    result = motion_time.clone()
    result[~reset_mask] += float(step_dt)
    return result


@dataclass(frozen=True)
class ReferenceAwareFallComponents:
    """Per-environment values and subcauses used by the fall termination."""

    root_height_deficit_m: torch.Tensor
    root_orientation_error_rad: torch.Tensor
    reference_pelvis_height_m: torch.Tensor
    reference_pelvis_tilt_rad: torch.Tensor
    pelvis_contact_force_n: torch.Tensor
    reference_height_eligible: torch.Tensor
    reference_tilt_eligible: torch.Tensor
    pelvis_contact_applicable: torch.Tensor
    pelvis_force_exceeded: torch.Tensor
    height_deficit_exceeded: torch.Tensor
    orientation_error_exceeded: torch.Tensor
    unexpected_pelvis_contact: torch.Tensor
    fall: torch.Tensor


def reference_aware_fall_components(
    actual_root_position_w: torch.Tensor,
    reference_root_position_w: torch.Tensor,
    actual_root_quaternion_wxyz: torch.Tensor,
    reference_root_quaternion_wxyz: torch.Tensor,
    pelvis_contact_force: torch.Tensor,
    *,
    maximum_root_height_deficit: float,
    maximum_orientation_error: float,
    pelvis_contact_threshold: float,
    pelvis_contact_reference_height: float,
    maximum_pelvis_contact_reference_tilt: float,
) -> ReferenceAwareFallComponents:
    """Return the complete reference-aware fall decision and its subcauses.

    An absolute height or world-up test makes valid floor motions (lying,
    crawling, rolls, cartwheels) impossible by definition.  Here a fall means
    losing height or orientation *relative to the retargeted reference*.
    Pelvis collision remains a useful upright safety signal, but is disabled
    when the reference pelvis itself is near the floor.
    """

    for name, value, width in (
        ("actual_root_position_w", actual_root_position_w, 3),
        ("reference_root_position_w", reference_root_position_w, 3),
        ("actual_root_quaternion_wxyz", actual_root_quaternion_wxyz, 4),
        ("reference_root_quaternion_wxyz", reference_root_quaternion_wxyz, 4),
    ):
        _require_last_dim(value, width, name)
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [env, {width}]")
    env_count = actual_root_position_w.shape[0]
    if not (
        reference_root_position_w.shape[0]
        == actual_root_quaternion_wxyz.shape[0]
        == reference_root_quaternion_wxyz.shape[0]
        == env_count
    ):
        raise ValueError("fall-mask inputs have different environment counts")
    if pelvis_contact_force.shape != (env_count,) or not bool(
        torch.isfinite(pelvis_contact_force).all()
    ):
        raise ValueError("pelvis_contact_force must be finite with shape [env]")
    for name, value in (
        ("maximum_root_height_deficit", maximum_root_height_deficit),
        ("maximum_orientation_error", maximum_orientation_error),
        ("pelvis_contact_threshold", pelvis_contact_threshold),
        ("pelvis_contact_reference_height", pelvis_contact_reference_height),
        (
            "maximum_pelvis_contact_reference_tilt",
            maximum_pelvis_contact_reference_tilt,
        ),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    height_deficit = (
        reference_root_position_w[:, 2] - actual_root_position_w[:, 2]
    )
    height_lost = height_deficit > maximum_root_height_deficit
    orientation_error = quat_error_angle_wxyz(
        reference_root_quaternion_wxyz, actual_root_quaternion_wxyz
    )
    orientation_lost = orientation_error > maximum_orientation_error
    reference_up = quat_apply_wxyz(
        reference_root_quaternion_wxyz,
        torch.tensor(
            [0.0, 0.0, 1.0],
            dtype=reference_root_quaternion_wxyz.dtype,
            device=reference_root_quaternion_wxyz.device,
        ).expand(env_count, -1),
    )
    reference_tilt = torch.acos(reference_up[:, 2].clamp(-1.0, 1.0))
    reference_height_eligible = (
        reference_root_position_w[:, 2] >= pelvis_contact_reference_height
    )
    reference_tilt_eligible = reference_tilt <= maximum_pelvis_contact_reference_tilt
    upright_reference = reference_height_eligible & reference_tilt_eligible
    pelvis_force_exceeded = pelvis_contact_force > pelvis_contact_threshold
    unexpected_pelvis_contact = upright_reference & pelvis_force_exceeded
    fall = height_lost | orientation_lost | unexpected_pelvis_contact
    return ReferenceAwareFallComponents(
        root_height_deficit_m=height_deficit,
        root_orientation_error_rad=orientation_error,
        reference_pelvis_height_m=reference_root_position_w[:, 2],
        reference_pelvis_tilt_rad=reference_tilt,
        pelvis_contact_force_n=pelvis_contact_force,
        reference_height_eligible=reference_height_eligible,
        reference_tilt_eligible=reference_tilt_eligible,
        pelvis_contact_applicable=upright_reference,
        pelvis_force_exceeded=pelvis_force_exceeded,
        height_deficit_exceeded=height_lost,
        orientation_error_exceeded=orientation_lost,
        unexpected_pelvis_contact=unexpected_pelvis_contact,
        fall=fall,
    )


def reference_aware_fall_mask(
    actual_root_position_w: torch.Tensor,
    reference_root_position_w: torch.Tensor,
    actual_root_quaternion_wxyz: torch.Tensor,
    reference_root_quaternion_wxyz: torch.Tensor,
    pelvis_contact_force: torch.Tensor,
    *,
    maximum_root_height_deficit: float,
    maximum_orientation_error: float,
    pelvis_contact_threshold: float,
    pelvis_contact_reference_height: float,
    maximum_pelvis_contact_reference_tilt: float,
) -> torch.Tensor:
    """Classify falls relative to the commanded motion embodiment."""

    return reference_aware_fall_components(
        actual_root_position_w,
        reference_root_position_w,
        actual_root_quaternion_wxyz,
        reference_root_quaternion_wxyz,
        pelvis_contact_force,
        maximum_root_height_deficit=maximum_root_height_deficit,
        maximum_orientation_error=maximum_orientation_error,
        pelvis_contact_threshold=pelvis_contact_threshold,
        pelvis_contact_reference_height=pelvis_contact_reference_height,
        maximum_pelvis_contact_reference_tilt=maximum_pelvis_contact_reference_tilt,
    ).fall


@dataclass(frozen=True)
class TrackingDivergenceComponents:
    """Per-environment values and subcauses used by tracking divergence."""

    root_distance_m: torch.Tensor
    mean_absolute_joint_error_rad: torch.Tensor
    mean_tracking_body_distance_m: torch.Tensor
    root_distance_exceeded: torch.Tensor
    mean_joint_error_exceeded: torch.Tensor
    mean_body_distance_exceeded: torch.Tensor
    diverged: torch.Tensor


def tracking_divergence_components(
    actual_root_position_w: torch.Tensor,
    reference_root_position_w: torch.Tensor,
    actual_joint_position: torch.Tensor,
    reference_joint_position: torch.Tensor,
    actual_tracking_body_position_w: torch.Tensor,
    reference_tracking_body_position_w: torch.Tensor,
    *,
    maximum_root_distance: float,
    maximum_mean_joint_error: float,
    maximum_mean_body_distance: float,
) -> TrackingDivergenceComponents:
    """Return divergence magnitudes and the exact strict-threshold decisions."""

    if actual_root_position_w.ndim != 2 or actual_root_position_w.shape[-1] != 3:
        raise ValueError("actual_root_position_w must have shape [env, 3]")
    if reference_root_position_w.shape != actual_root_position_w.shape:
        raise ValueError("reference_root_position_w shape must match the actual root")
    if actual_joint_position.ndim != 2 or actual_joint_position.shape[1] == 0:
        raise ValueError("actual_joint_position must have shape [env, joint]")
    if reference_joint_position.shape != actual_joint_position.shape:
        raise ValueError("reference_joint_position shape must match the actual joints")
    if (
        actual_tracking_body_position_w.ndim != 3
        or actual_tracking_body_position_w.shape[1] == 0
        or actual_tracking_body_position_w.shape[-1] != 3
    ):
        raise ValueError(
            "actual_tracking_body_position_w must have shape [env, body, 3]"
        )
    if reference_tracking_body_position_w.shape != actual_tracking_body_position_w.shape:
        raise ValueError(
            "reference_tracking_body_position_w shape must match the actual bodies"
        )
    env_count = actual_root_position_w.shape[0]
    if (
        actual_joint_position.shape[0] != env_count
        or actual_tracking_body_position_w.shape[0] != env_count
    ):
        raise ValueError("divergence inputs have different environment counts")
    for name, value in (
        ("maximum_root_distance", maximum_root_distance),
        ("maximum_mean_joint_error", maximum_mean_joint_error),
        ("maximum_mean_body_distance", maximum_mean_body_distance),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    root_error = torch.linalg.vector_norm(
        actual_root_position_w - reference_root_position_w, dim=-1
    )
    joint_error = torch.mean(
        torch.abs(actual_joint_position - reference_joint_position), dim=-1
    )
    body_error = torch.mean(
        torch.linalg.vector_norm(
            actual_tracking_body_position_w - reference_tracking_body_position_w,
            dim=-1,
        ),
        dim=-1,
    )
    root_exceeded = root_error > maximum_root_distance
    joint_exceeded = joint_error > maximum_mean_joint_error
    body_exceeded = body_error > maximum_mean_body_distance
    return TrackingDivergenceComponents(
        root_distance_m=root_error,
        mean_absolute_joint_error_rad=joint_error,
        mean_tracking_body_distance_m=body_error,
        root_distance_exceeded=root_exceeded,
        mean_joint_error_exceeded=joint_exceeded,
        mean_body_distance_exceeded=body_exceeded,
        diverged=root_exceeded | joint_exceeded | body_exceeded,
    )


def tracking_divergence_risk_barrier(
    components: TrackingDivergenceComponents,
    *,
    maximum_root_distance: float = TRACKING_DIVERGENCE_MAX_ROOT_DISTANCE_M,
    maximum_mean_joint_error: float = TRACKING_DIVERGENCE_MAX_MEAN_JOINT_ERROR_RAD,
    maximum_mean_body_distance: float = TRACKING_DIVERGENCE_MAX_MEAN_BODY_DISTANCE_M,
    onset_fraction: float = DIVERGENCE_RISK_ONSET_FRACTION,
) -> torch.Tensor:
    """Dense squared risk in ``[0, 1]`` using the exact termination magnitudes.

    Each component is zero through ``onset_fraction`` of its unchanged hard
    termination threshold and rises quadratically to one at that threshold.
    Taking the maximum mirrors the termination's logical OR and prevents three
    modest errors from being summed into a synthetic failure.
    """

    thresholds = (
        (maximum_root_distance, "maximum_root_distance"),
        (maximum_mean_joint_error, "maximum_mean_joint_error"),
        (maximum_mean_body_distance, "maximum_mean_body_distance"),
    )
    for value, name in thresholds:
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        isinstance(onset_fraction, bool)
        or not math.isfinite(onset_fraction)
        or not 0.0 <= onset_fraction < 1.0
    ):
        raise ValueError("onset_fraction must be finite and in [0, 1)")
    values = (
        components.root_distance_m,
        components.mean_absolute_joint_error_rad,
        components.mean_tracking_body_distance_m,
    )
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValueError("divergence component magnitudes must share one shape")
    risks = []
    for value, (threshold, _) in zip(values, thresholds, strict=True):
        normalized = (value / float(threshold) - float(onset_fraction)) / (
            1.0 - float(onset_fraction)
        )
        normalized = torch.nan_to_num(
            normalized,
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        risks.append(torch.square(normalized))
    return torch.amax(torch.stack(risks, dim=-1), dim=-1)


@dataclass(frozen=True)
class TorqueEffortRatioComponents:
    """Joint-wise requested/applied torque ratios against named effort limits."""

    computed_abs_ratio: torch.Tensor
    applied_abs_ratio: torch.Tensor
    computed_limit_exceeded: torch.Tensor
    applied_limit_exceeded: torch.Tensor


def torque_effort_ratio_components(
    computed_torque: torch.Tensor,
    applied_torque: torch.Tensor,
    effort_limits: torch.Tensor,
    *,
    tolerance_nm: float = 1.0e-5,
) -> TorqueEffortRatioComponents:
    """Return joint-wise torque/limit ratios without hiding actuator clipping."""

    if computed_torque.ndim != 2 or computed_torque.shape[1] == 0:
        raise ValueError("computed_torque must have shape [env, joint]")
    if applied_torque.shape != computed_torque.shape:
        raise ValueError("applied_torque shape must match computed_torque")
    if effort_limits.ndim != 1 or effort_limits.shape[0] != computed_torque.shape[1]:
        raise ValueError("effort_limits must have shape [joint]")
    if not bool(torch.isfinite(effort_limits).all()) or bool(torch.any(effort_limits <= 0.0)):
        raise ValueError("effort_limits must be finite and positive")
    if isinstance(tolerance_nm, bool) or not math.isfinite(tolerance_nm) or tolerance_nm < 0.0:
        raise ValueError("tolerance_nm must be finite and non-negative")
    limits = effort_limits.unsqueeze(0)
    computed_abs = torch.abs(computed_torque)
    applied_abs = torch.abs(applied_torque)
    return TorqueEffortRatioComponents(
        computed_abs_ratio=computed_abs / limits,
        applied_abs_ratio=applied_abs / limits,
        computed_limit_exceeded=computed_abs > limits + float(tolerance_nm),
        applied_limit_exceeded=applied_abs > limits + float(tolerance_nm),
    )


def _expand_env_value(value: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """Insert singleton dimensions after an environment axis for broadcasting."""

    if target_ndim < value.ndim:
        raise ValueError("target rank cannot be smaller than source rank")
    return value.reshape(value.shape[:-1] + (1,) * (target_ndim - value.ndim) + value.shape[-1:])


def rotate_by_env_yaw(vector: torch.Tensor, yaw_quaternion: torch.Tensor) -> torch.Tensor:
    """Rotate ``[env, ..., 3]`` values by one yaw quaternion per environment."""

    _require_last_dim(vector, 3, "vector")
    _require_last_dim(yaw_quaternion, 4, "yaw_quaternion")
    if vector.shape[0] != yaw_quaternion.shape[0]:
        raise ValueError("vector and yaw quaternion environment counts differ")
    expanded = _expand_env_value(yaw_quaternion, vector.ndim)
    return quat_apply_wxyz(expanded, vector)


def align_positions_by_env_yaw(
    position: torch.Tensor,
    yaw_quaternion: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Apply a fixed per-environment yaw and translation to world positions."""

    _require_last_dim(translation, 3, "translation")
    if position.shape[0] != translation.shape[0]:
        raise ValueError("position and translation environment counts differ")
    expanded_translation = _expand_env_value(translation, position.ndim)
    return rotate_by_env_yaw(position, yaw_quaternion) + expanded_translation


def align_quaternions_by_env_yaw(
    quaternion: torch.Tensor,
    yaw_quaternion: torch.Tensor,
) -> torch.Tensor:
    """Left-multiply ``[env, ..., 4]`` orientations by per-environment yaw."""

    _require_last_dim(quaternion, 4, "quaternion")
    if quaternion.shape[0] != yaw_quaternion.shape[0]:
        raise ValueError("quaternion and yaw quaternion environment counts differ")
    expanded = _expand_env_value(yaw_quaternion, quaternion.ndim)
    return quat_mul_wxyz(expanded, quaternion)


def alignment_from_reference_anchor(
    reference_position: torch.Tensor,
    reference_quaternion: torch.Tensor,
    target_xy: torch.Tensor,
    target_yaw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a fixed XY/yaw alignment for the current reference root/anchor.

    Z is deliberately not translated: retargeting owns ground clearance.  The
    returned translation maps the rotated reference XY exactly to ``target_xy``.
    """

    _require_last_dim(reference_position, 3, "reference_position")
    _require_last_dim(reference_quaternion, 4, "reference_quaternion")
    _require_last_dim(target_xy, 2, "target_xy")
    if reference_position.ndim != 2 or target_xy.ndim != 2 or target_yaw.ndim != 1:
        raise ValueError("alignment inputs must be [env,3], [env,4], [env,2], and [env]")
    if not (
        reference_position.shape[0]
        == reference_quaternion.shape[0]
        == target_xy.shape[0]
        == target_yaw.shape[0]
    ):
        raise ValueError("alignment input environment counts differ")
    yaw_delta = target_yaw - quat_yaw_wxyz(reference_quaternion)
    yaw_quaternion = yaw_quat_wxyz(yaw_delta)
    rotated = quat_apply_wxyz(yaw_quaternion, reference_position)
    translation = torch.zeros_like(reference_position)
    translation[:, :2] = target_xy - rotated[:, :2]
    return yaw_quaternion, translation


@dataclass(frozen=True)
class ResidualTargetResult:
    """Intermediate and final values of the reference-residual action contract."""

    clipped_action: torch.Tensor
    normalized_residual: torch.Tensor
    unclamped_target: torch.Tensor
    target: torch.Tensor
    rate_limited: torch.Tensor
    target_clamped: torch.Tensor


@dataclass(frozen=True)
class ReferenceVelocityTargetResult:
    """Intermediate and final values of the reference-velocity target contract."""

    unclamped_target: torch.Tensor
    target: torch.Tensor
    target_clamped: torch.Tensor


def reference_velocity_target(
    reference_joint_velocity: torch.Tensor,
    *,
    scale: float = 1.0,
    velocity_limits: torch.Tensor | None = None,
) -> ReferenceVelocityTargetResult:
    """Build the implicit actuator velocity target from the motion reference.

    Isaac Lab initializes articulation velocity targets to zero.  Supplying
    only the moving position reference therefore adds a ``-Kd*dq_actual``
    braking term instead of the intended ``Kd*(dq_ref-dq_actual)`` term.  This
    helper makes ``dq_ref`` explicit and clips it to the live articulation's
    symmetric soft velocity limits.
    """

    if reference_joint_velocity.ndim < 1 or not bool(
        torch.isfinite(reference_joint_velocity).all()
    ):
        raise ValueError(
            "reference_joint_velocity must be a finite tensor with at least one dimension"
        )
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or float(scale) < 0.0:
        raise ValueError("scale must be finite and non-negative")
    if velocity_limits is None:
        raise ValueError("velocity_limits are required")
    if velocity_limits.shape != reference_joint_velocity.shape or not bool(
        torch.isfinite(velocity_limits).all()
    ):
        raise ValueError(
            "velocity_limits must match reference_joint_velocity and be finite"
        )
    if bool(torch.any(velocity_limits <= 0.0)):
        raise ValueError("velocity_limits must be strictly positive")

    unclamped = reference_joint_velocity * float(scale)
    target = torch.maximum(torch.minimum(unclamped, velocity_limits), -velocity_limits)
    return ReferenceVelocityTargetResult(
        unclamped_target=unclamped,
        target=target,
        target_clamped=torch.abs(target - unclamped) > 1.0e-7,
    )


def reference_residual_target(
    raw_action: torch.Tensor,
    reference_joint_pos: torch.Tensor,
    previous_normalized_residual: torch.Tensor,
    *,
    residual_scale: float = 0.25,
    action_clip: float = 1.0,
    normalized_delta_limit: float | None = 0.15,
    hard_limits: torch.Tensor | None = None,
    soft_limits: torch.Tensor | None = None,
    target_limit_mode: str = "soft",
) -> ResidualTargetResult:
    """Compute a safe G1 target from a normalized residual action.

    ``target_limit_mode='soft'`` clamps to the articulation soft interval;
    ``'hard'`` permits the complete URDF interval.  Hard limits are always
    enforced as the final safety envelope.  Rate limiting is expressed in the
    normalized action domain and is therefore exactly 0.15 per 50 Hz step by
    default, independent of the 0.25 rad residual scale.
    """

    for name, value in (
        ("raw_action", raw_action),
        ("reference_joint_pos", reference_joint_pos),
        ("previous_normalized_residual", previous_normalized_residual),
    ):
        if value.ndim < 1 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be a finite tensor with at least one dimension")
    if raw_action.shape != reference_joint_pos.shape or raw_action.shape != previous_normalized_residual.shape:
        raise ValueError("action, reference, and previous residual shapes must match")
    if not math.isfinite(residual_scale) or residual_scale <= 0.0:
        raise ValueError("residual_scale must be finite and positive")
    if not math.isfinite(action_clip) or action_clip <= 0.0:
        raise ValueError("action_clip must be finite and positive")
    if normalized_delta_limit is not None and (
        not math.isfinite(normalized_delta_limit) or normalized_delta_limit <= 0.0
    ):
        raise ValueError("normalized_delta_limit must be positive or None")
    if target_limit_mode not in {"soft", "hard"}:
        raise ValueError("target_limit_mode must be 'soft' or 'hard'")

    def validate_limits(limits: torch.Tensor | None, name: str) -> None:
        if limits is None:
            return
        if limits.shape != raw_action.shape + (2,) or not bool(torch.isfinite(limits).all()):
            raise ValueError(f"{name} must have shape {raw_action.shape + (2,)} and be finite")
        if bool(torch.any(limits[..., 0] > limits[..., 1])):
            raise ValueError(f"{name} has inverted intervals")

    validate_limits(hard_limits, "hard_limits")
    validate_limits(soft_limits, "soft_limits")
    if target_limit_mode == "soft" and soft_limits is None:
        raise ValueError("soft target clamping requires soft_limits")
    if hard_limits is None and target_limit_mode == "hard":
        raise ValueError("hard target clamping requires hard_limits")

    clipped = raw_action.clamp(-action_clip, action_clip)
    if normalized_delta_limit is None:
        residual = clipped
    else:
        delta = (clipped - previous_normalized_residual).clamp(
            -normalized_delta_limit, normalized_delta_limit
        )
        residual = previous_normalized_residual + delta
    rate_limited = torch.abs(residual - clipped) > 1.0e-7
    unclamped = reference_joint_pos + residual_scale * residual

    selected_limits = soft_limits if target_limit_mode == "soft" else hard_limits
    assert selected_limits is not None
    target = torch.maximum(
        torch.minimum(unclamped, selected_limits[..., 1]), selected_limits[..., 0]
    )
    if hard_limits is not None:
        target = torch.maximum(torch.minimum(target, hard_limits[..., 1]), hard_limits[..., 0])
    target_clamped = torch.abs(target - unclamped) > 1.0e-7
    return ResidualTargetResult(
        clipped_action=clipped,
        normalized_residual=residual,
        unclamped_target=unclamped,
        target=target,
        rate_limited=rate_limited,
        target_clamped=target_clamped,
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_path_below_accad(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(ACCAD_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"tracking motion must remain below {ACCAD_ROOT}: {resolved}") from exc
    if not resolved.is_file() or resolved.suffix.lower() != ".npz":
        raise ValueError(f"tracking motion is not an existing NPZ: {resolved}")
    return resolved


def fixed_motion_ids_for_envs(
    *,
    num_envs: int,
    num_motions: int,
    fixed_motion_id: int | None = None,
    fixed_motion_ids: Sequence[int] | None = None,
) -> tuple[int, ...] | None:
    """Resolve scalar or per-environment fixed-motion selection.

    The scalar form is the original command contract and broadcasts one motion
    to every environment.  The sequence form assigns exactly one library index
    to each environment and is intended for deterministic evaluation suites.
    The two forms are deliberately mutually exclusive so a partial reset can
    never silently change which motion an environment owns.

    Returns:
        A ``num_envs``-entry tuple, or ``None`` when motions should be sampled.
    """

    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    if isinstance(num_motions, bool) or not isinstance(num_motions, int) or num_motions <= 0:
        raise ValueError("num_motions must be a positive integer")
    if fixed_motion_id is not None and fixed_motion_ids is not None:
        raise ValueError("fixed_motion_id and fixed_motion_ids are mutually exclusive")

    if fixed_motion_ids is not None:
        values = tuple(fixed_motion_ids)
        if len(values) != num_envs:
            raise ValueError(
                f"fixed_motion_ids must contain one index per environment; "
                f"expected {num_envs}, got {len(values)}"
            )
    elif fixed_motion_id is not None:
        values = (fixed_motion_id,) * num_envs
    else:
        return None

    normalized: list[int] = []
    for env_id, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"fixed motion index for environment {env_id} is not an integer")
        motion_id = int(value)
        if not 0 <= motion_id < num_motions:
            raise ValueError(
                f"fixed motion index {motion_id} for environment {env_id} is outside "
                f"the loaded range [0, {num_motions - 1}]"
            )
        normalized.append(motion_id)
    return tuple(normalized)


def balanced_fixed_motion_ids(*, num_envs: int, num_motions: int) -> tuple[int, ...]:
    """Return the production training assignment ``env_id % num_motions``.

    Production training requires at least one persistent environment per motion.
    The modulo assignment is deterministic and gives every motion either
    ``floor(num_envs / num_motions)`` or ``ceil(...)`` environments, so the
    largest count difference is at most one.
    """

    # Reuse the public fixed-ID validator for strict integer/bool handling.
    fixed_motion_ids_for_envs(num_envs=num_envs, num_motions=num_motions)
    if num_envs < num_motions:
        raise ValueError(
            "balanced-fixed training requires num_envs >= num_motions so every "
            f"motion has a nonzero environment assignment; got {num_envs} "
            f"environments for {num_motions} motions"
        )
    return tuple(env_id % num_motions for env_id in range(num_envs))


def resolve_training_motion_assignment(
    mode: str,
    *,
    num_envs: int,
    num_motions: int,
) -> tuple[int, ...] | None:
    """Resolve the selected training-only motion assignment policy."""

    if not isinstance(mode, str) or mode not in TRAINING_MOTION_ASSIGNMENT_CHOICES:
        raise ValueError(
            "motion assignment must be one of "
            f"{TRAINING_MOTION_ASSIGNMENT_CHOICES}, got {mode!r}"
        )
    # Validate the dimensions for both modes, including Python bool rejection.
    fixed_motion_ids_for_envs(num_envs=num_envs, num_motions=num_motions)
    if mode == "episode-uniform":
        return None
    return balanced_fixed_motion_ids(num_envs=num_envs, num_motions=num_motions)


def training_motion_assignment_contract(
    mode: str,
    *,
    num_envs: int,
    motion_keys: Sequence[str],
    num_steps_per_env: int,
    learning_updates: int,
    fixed_motion_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe, independently reproducible training assignment record.

    For ``balanced-fixed`` the record proves exact finite-run exposure from the
    immutable environment mapping.  ``episode-uniform`` remains available as a
    diagnostic mode, but correctly records that per-motion finite-run exposure
    is stochastic rather than guaranteed.
    """

    keys = tuple(motion_keys)
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("motion_keys must contain non-empty strings")
    if len(set(keys)) != len(keys):
        raise ValueError("motion_keys must be unique and ordered by motion ID")
    for name, value in (
        ("num_steps_per_env", num_steps_per_env),
        ("learning_updates", learning_updates),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    expected = resolve_training_motion_assignment(
        mode,
        num_envs=num_envs,
        num_motions=len(keys),
    )
    if mode == "balanced-fixed":
        supplied = expected if fixed_motion_ids is None else fixed_motion_ids_for_envs(
            num_envs=num_envs,
            num_motions=len(keys),
            fixed_motion_ids=fixed_motion_ids,
        )
        if supplied != expected:
            raise ValueError(
                "balanced-fixed motion IDs must exactly follow "
                "fixed_motion_id[env_id] = env_id % num_motions"
            )
        resolved_ids = expected
    else:
        if fixed_motion_ids is not None:
            raise ValueError("episode-uniform training cannot provide fixed_motion_ids")
        resolved_ids = None

    total_per_rollout = num_envs * num_steps_per_env
    total_training = total_per_rollout * learning_updates
    per_motion: list[dict[str, Any]] = []
    mapping_digest: str | None = None
    maximum_count_difference: int | None = None
    all_nonzero: bool | None = None

    if resolved_ids is not None:
        counts = [0] * len(keys)
        for motion_id in resolved_ids:
            counts[motion_id] += 1
        maximum_count_difference = max(counts) - min(counts)
        all_nonzero = all(count > 0 for count in counts)
        digest_payload = {
            "schema_version": TRAINING_MOTION_ASSIGNMENT_SCHEMA_VERSION,
            "mapping_semantics": "fixed_motion_id[env_id] = env_id % num_motions",
            "motion_keys": list(keys),
            "fixed_motion_ids": list(resolved_ids),
        }
        mapping_digest = hashlib.sha256(
            json.dumps(
                digest_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        for motion_id, (motion_key, environment_count) in enumerate(zip(keys, counts)):
            transitions_per_rollout = environment_count * num_steps_per_env
            per_motion.append(
                {
                    "motion_id": motion_id,
                    "motion_key": motion_key,
                    "environment_count": environment_count,
                    "theoretical_transitions_per_rollout": transitions_per_rollout,
                    "theoretical_training_transitions": (
                        transitions_per_rollout * learning_updates
                    ),
                }
            )
        witness: Mapping[str, Any] = {
            "verified": all_nonzero,
            "covered_motion_count": sum(count > 0 for count in counts),
            "minimum_environment_count": min(counts),
            "missing_motion_ids": [
                motion_id for motion_id, count in enumerate(counts) if count == 0
            ],
        }
    else:
        per_motion = [
            {
                "motion_id": motion_id,
                "motion_key": motion_key,
                "environment_count": None,
                "theoretical_transitions_per_rollout": None,
                "theoretical_training_transitions": None,
            }
            for motion_id, motion_key in enumerate(keys)
        ]
        witness = {
            "verified": None,
            "covered_motion_count": None,
            "minimum_environment_count": None,
            "missing_motion_ids": None,
            "reason": "episode-uniform resampling has no deterministic finite-run witness",
        }

    return {
        "schema_version": TRAINING_MOTION_ASSIGNMENT_SCHEMA_VERSION,
        "mode": mode,
        "production_default": DEFAULT_TRAINING_MOTION_ASSIGNMENT,
        "fixed_environment_assignment": resolved_ids is not None,
        "num_environments": num_envs,
        "num_motions": len(keys),
        "num_steps_per_env": num_steps_per_env,
        "learning_updates": learning_updates,
        "mapping_semantics": (
            "fixed_motion_id[env_id] = env_id % num_motions"
            if resolved_ids is not None
            else "motion_id sampled uniformly on each command resample"
        ),
        "mapping_length": len(resolved_ids) if resolved_ids is not None else None,
        "mapping_digest_sha256": mapping_digest,
        "maximum_environment_count_difference": maximum_count_difference,
        "all_motions_have_nonzero_environments": all_nonzero,
        "all_motions_nonzero_witness": dict(witness),
        "theoretical_total_transitions_per_rollout": total_per_rollout,
        "theoretical_total_training_transitions": total_training,
        "transition_exposure_guarantee": (
            "exact_fixed_environment_assignment"
            if resolved_ids is not None
            else "none_episode_resampled"
        ),
        "per_motion": per_motion,
    }


def clean_motion_end_success(
    termination_flags: Mapping[str, bool],
    *,
    timeout_term_names: Sequence[str],
) -> bool:
    """Return the strict deterministic-suite success decision.

    ``motion_end`` is a timeout-style task completion signal.  Other timeout
    signals may coincide with it, but any simultaneous non-timeout signal is a
    hard failure.  This explicitly prevents ``motion_end + fall`` from being
    counted as success merely because the clip reached its last frame.
    """

    timeout_names = set(timeout_term_names)
    if "motion_end" not in timeout_names:
        raise ValueError("motion_end must be classified as a timeout term")
    return bool(termination_flags.get("motion_end", False)) and not any(
        bool(value) and name not in timeout_names
        for name, value in termination_flags.items()
    )


def resolved_live_tracking_tensor_contract(env: Any) -> dict[str, Any]:
    """Validate and report the exact live v6 observation/action tensor axes.

    The manager metadata is the independent runtime witness: this function
    compares active names, ordering, individual shapes, concatenated shapes,
    concatenation mode, future horizons, and the 29-D ``joint_residual`` action
    against :func:`tracking_observation_contract`.  Any drift fails closed
    before training, evaluation, or export can claim the v6 policy contract.
    """

    expected = tracking_observation_contract()
    observation_manager = getattr(env, "observation_manager", None)
    action_manager = getattr(env, "action_manager", None)
    command_manager = getattr(env, "command_manager", None)
    if observation_manager is None or action_manager is None or command_manager is None:
        raise RuntimeError("live environment is missing observation/action/command managers")

    active_groups = observation_manager.active_terms
    term_shapes_by_group = observation_manager.group_obs_term_dim
    group_shapes = observation_manager.group_obs_dim
    concatenate_by_group = observation_manager.group_obs_concatenate
    expected_groups = expected["observation_groups"]
    if tuple(active_groups) != tuple(expected_groups):
        raise RuntimeError(
            "live observation groups differ from the v6 contract: "
            f"expected {tuple(expected_groups)}, got {tuple(active_groups)}"
        )

    resolved_groups: dict[str, Any] = {}
    for group_name, expected_group in expected_groups.items():
        expected_terms = expected_group["terms"]
        expected_names = tuple(term["name"] for term in expected_terms)
        actual_names = tuple(active_groups[group_name])
        if actual_names != expected_names:
            raise RuntimeError(
                f"live {group_name} observation term order differs from v6: "
                f"expected {expected_names}, got {actual_names}"
            )
        raw_term_shapes = term_shapes_by_group[group_name]
        actual_term_shapes = tuple(
            tuple(int(axis) for axis in shape) for shape in raw_term_shapes
        )
        expected_term_shapes = tuple(
            (int(term["dimension"]),) for term in expected_terms
        )
        if actual_term_shapes != expected_term_shapes:
            raise RuntimeError(
                f"live {group_name} observation term shapes differ from v6: "
                f"expected {expected_term_shapes}, got {actual_term_shapes}"
            )
        actual_group_shape = tuple(int(axis) for axis in group_shapes[group_name])
        expected_group_shape = (int(expected_group["dimension"]),)
        if actual_group_shape != expected_group_shape:
            raise RuntimeError(
                f"live {group_name} concatenated shape differs from v6: "
                f"expected {expected_group_shape}, got {actual_group_shape}"
            )
        if concatenate_by_group[group_name] is not True:
            raise RuntimeError(f"live {group_name} observations must be concatenated")
        resolved_groups[group_name] = {
            "term_names": list(actual_names),
            "term_dimensions": [shape[0] for shape in actual_term_shapes],
            "term_shapes": [list(shape) for shape in actual_term_shapes],
            "group_shape": list(actual_group_shape),
            "dimension": actual_group_shape[0],
            "concatenate_terms": True,
        }

    actual_action_names = tuple(action_manager.active_terms)
    expected_action_names = tuple(expected["action"]["term_names"])
    actual_action_dimensions = tuple(int(value) for value in action_manager.action_term_dim)
    expected_action_dimensions = tuple(int(value) for value in expected["action"]["term_dimensions"])
    if actual_action_names != expected_action_names:
        raise RuntimeError(
            "live action term order differs from v6: "
            f"expected {expected_action_names}, got {actual_action_names}"
        )
    if actual_action_dimensions != expected_action_dimensions:
        raise RuntimeError(
            "live action term dimensions differ from v6: "
            f"expected {expected_action_dimensions}, got {actual_action_dimensions}"
        )
    actual_total_action_dim = int(action_manager.total_action_dim)
    if actual_total_action_dim != G1_ACTION_DIM:
        raise RuntimeError(
            f"live joint_residual action must total {G1_ACTION_DIM}; "
            f"got {actual_total_action_dim}"
        )

    residual_action = action_manager.get_term("joint_residual")
    action_cfg = getattr(residual_action, "cfg", None)
    live_velocity_feedforward_scale = getattr(
        action_cfg, "reference_velocity_feedforward_scale", None
    )
    expected_velocity_feedforward_scale = float(
        expected["action"]["reference_velocity_feedforward_scale"]
    )
    if (
        live_velocity_feedforward_scale is None
        or not math.isfinite(float(live_velocity_feedforward_scale))
        or abs(
            float(live_velocity_feedforward_scale)
            - expected_velocity_feedforward_scale
        )
        > 1.0e-7
    ):
        raise RuntimeError(
            "live reference velocity feedforward scale differs from the v7 contract: "
            f"expected {expected_velocity_feedforward_scale}, "
            f"got {live_velocity_feedforward_scale}"
        )
    velocity_target_buffer_verified = getattr(
        residual_action, "velocity_target_buffer_verified", False
    )
    velocity_target_buffer_max_abs_error = getattr(
        residual_action, "velocity_target_buffer_max_abs_error", None
    )
    if velocity_target_buffer_verified is not True:
        raise RuntimeError(
            "live articulation velocity-target buffer was not verified after physics warm-up"
        )
    if (
        velocity_target_buffer_max_abs_error is None
        or not math.isfinite(float(velocity_target_buffer_max_abs_error))
        or float(velocity_target_buffer_max_abs_error) > 1.0e-7
    ):
        raise RuntimeError(
            "live articulation velocity-target buffer verification error is invalid"
        )

    motion_command = command_manager.get_term("motion")
    live_horizons = getattr(motion_command, "future_horizons_s", None)
    if isinstance(live_horizons, torch.Tensor):
        live_horizon_values = tuple(
            float(value) for value in live_horizons.detach().cpu().tolist()
        )
    elif live_horizons is not None:
        live_horizon_values = tuple(float(value) for value in live_horizons)
    else:
        command_cfg = getattr(motion_command, "cfg", None)
        configured_horizons = getattr(command_cfg, "future_horizons_s", ())
        live_horizon_values = tuple(float(value) for value in configured_horizons)
    if len(live_horizon_values) != len(DEFAULT_FUTURE_HORIZONS_S) or any(
        abs(actual - expected_value) > 1.0e-7
        for actual, expected_value in zip(
            live_horizon_values, DEFAULT_FUTURE_HORIZONS_S, strict=True
        )
    ):
        raise RuntimeError(
            "live future horizons differ from the v6 contract: "
            f"expected {DEFAULT_FUTURE_HORIZONS_S}, got {live_horizon_values}"
        )

    return {
        "schema_version": "accad_g1_live_tracking_tensor_contract_v1",
        "policy_contract_version": TRACKING_POLICY_CONTRACT_VERSION,
        "static_observation_contract_schema_version": expected["schema_version"],
        "future_horizons_s": [float(value) for value in live_horizon_values],
        "observation_groups": resolved_groups,
        "action": {
            "term_names": list(actual_action_names),
            "term_dimensions": list(actual_action_dimensions),
            "dimension": actual_total_action_dim,
            "position_target_formula": expected["action"]["position_target_formula"],
            "velocity_target_formula": expected["action"]["velocity_target_formula"],
            "reference_velocity_feedforward_scale": float(
                live_velocity_feedforward_scale
            ),
            "velocity_target_buffer_verified": True,
            "velocity_target_buffer_max_abs_error": float(
                velocity_target_buffer_max_abs_error
            ),
        },
    }


# Isaac modules are intentionally optional for ordinary CPU helper tests.  A
# real environment must import this file after AppLauncher has initialized Kit.
try:  # pragma: no cover - exercised by Isaac smoke tests, not ordinary pytest
    import isaaclab.sim as sim_utils
    import isaaclab.envs.mdp as isaac_mdp
    from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.managers import ActionTerm, CommandTerm, ManagerTermBase
    from isaaclab.managers import ActionTermCfg, CommandTermCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise

    from humanoid_g1.assets.g1_asset import G1_29DOF_CFG

    _ISAAC_IMPORT_ERROR: BaseException | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - normal CPU path
    _ISAAC_IMPORT_ERROR = exc


def isaac_tracking_available() -> bool:
    """Return whether the live Isaac task classes were importable."""

    return _ISAAC_IMPORT_ERROR is None


def require_isaac_tracking() -> None:
    """Raise a focused error when this module was imported before AppLauncher."""

    if _ISAAC_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Isaac tracking classes are unavailable. Launch through isaaclab.sh and "
            "construct AppLauncher before importing accad_g1.tracking_task."
        ) from _ISAAC_IMPORT_ERROR


if _ISAAC_IMPORT_ERROR is None:  # pragma: no branch

    def _env_ids_tensor(env_ids: Sequence[int] | slice | torch.Tensor, count: int, device: str) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(count, device=device, dtype=torch.long)[env_ids]
        return torch.as_tensor(env_ids, device=device, dtype=torch.long).flatten()


    def _uniform(
        shape: tuple[int, ...],
        bounds: tuple[float, float],
        *,
        device: str,
        generator: torch.Generator,
    ) -> torch.Tensor:
        low, high = (float(bounds[0]), float(bounds[1]))
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"invalid uniform bounds: {bounds}")
        return low + (high - low) * torch.rand(
            shape, device=device, dtype=torch.float32, generator=generator
        )


    @configclass
    class MultiMotionCommandCfg(CommandTermCfg):
        """Configuration for continuous-time, multi-clip G1 reference commands."""

        class_type: type = MISSING
        motion_files: list[str] = MISSING
        asset_name: str = "robot"
        contact_sensor_name: str = "contact_forces"
        root_body_name: str = "pelvis"
        foot_body_names: tuple[str, str] = (
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        )
        future_horizons_s: tuple[float, ...] = DEFAULT_FUTURE_HORIZONS_S
        correspondence_path: str | None = None
        require_gate3_complete: bool = False
        random_start: bool = True
        frame_zero_start_probability: float = 0.50
        minimum_remaining_time_s: float = 1.0
        fixed_motion_id: int | None = None
        fixed_motion_ids: tuple[int, ...] | None = None
        fixed_start_time_s: float = 0.0
        xy_offset_range: tuple[float, float] = (-0.25, 0.25)
        target_yaw_range: tuple[float, float] = (-math.pi, math.pi)
        joint_position_noise: tuple[float, float] = (-0.02, 0.02)
        joint_velocity_noise: tuple[float, float] = (-0.10, 0.10)
        root_linear_velocity_noise: tuple[float, float] = (-0.05, 0.05)
        root_angular_velocity_noise: tuple[float, float] = (-0.05, 0.05)
        contact_force_threshold: float = 20.0
        seed: int = 42
        resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
        debug_vis: bool = False

        def __post_init__(self) -> None:
            # The term is declared below this dataclass so CPU-only imports can
            # skip the complete Isaac branch.  Bind it when an instance exists.
            if self.class_type is MISSING:
                self.class_type = MultiMotionCommand


    class MultiMotionCommand(CommandTerm):
        """Reference command that samples clips and owns reference-state reset.

        The alignment transform is fixed for the episode.  It changes only global
        XY/yaw, never retargeted Z, joint values, or internal body geometry.
        """

        cfg: MultiMotionCommandCfg

        def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRLEnv):
            paths = tuple(_require_path_below_accad(path) for path in cfg.motion_files)
            if not paths:
                raise ValueError("MultiMotionCommand requires at least one motion archive")
            self.library = MotionLibrary(
                paths,
                device=env.device,
                dtype=torch.float32,
                require_gate3_complete=cfg.require_gate3_complete,
            )
            contract = self.library.contract
            if tuple(contract.joint_names) != G1_POLICY_JOINT_NAMES:
                raise RuntimeError("motion joint names do not equal the local G1 policy contract")
            if abs(contract.fps - 1.0 / env.step_dt) > 1.0e-6:
                raise RuntimeError(
                    f"motion/control rate mismatch: motion={contract.fps} Hz, env={1.0 / env.step_dt} Hz"
                )
            if abs(env.physics_dt - G1_CONTRACT.physics_dt) > 1.0e-9 or env.cfg.decimation != G1_CONTRACT.decimation:
                raise RuntimeError("live Isaac timing does not match the G1 200/50 Hz contract")
            if contract.g1_contract_checksum_sha256 != _sha256(G1_CONTRACT_PATH):
                raise RuntimeError("motion archive G1 YAML checksum does not match the local parent contract")
            if contract.g1_urdf_checksum_sha256 != _sha256(G1_URDF_PATH):
                raise RuntimeError("motion archive G1 URDF checksum does not match the local parent asset")
            if cfg.correspondence_path is not None:
                correspondence = Path(cfg.correspondence_path).expanduser().resolve()
                if contract.correspondence_checksum_sha256 != _sha256(correspondence):
                    raise RuntimeError("motion correspondence checksum does not match configured correspondence")
            horizons = tuple(float(value) for value in cfg.future_horizons_s)
            if not horizons or horizons[0] != 0.0 or any(
                not math.isfinite(value) or value < 0.0 for value in horizons
            ) or tuple(sorted(set(horizons))) != horizons:
                raise ValueError("future_horizons_s must be unique, ascending, and start at 0.0")
            if len(contract.tracking_body_names) != DEFAULT_TRACKING_BODY_COUNT:
                raise RuntimeError(
                    f"default observation contract requires 14 tracking bodies; got {len(contract.tracking_body_names)}"
                )
            fixed_motion_ids = fixed_motion_ids_for_envs(
                num_envs=int(env.num_envs),
                num_motions=self.library.num_motions,
                fixed_motion_id=cfg.fixed_motion_id,
                fixed_motion_ids=cfg.fixed_motion_ids,
            )
            if not math.isfinite(cfg.minimum_remaining_time_s) or cfg.minimum_remaining_time_s < 0.0:
                raise ValueError("minimum_remaining_time_s must be finite and non-negative")
            if (
                isinstance(cfg.frame_zero_start_probability, bool)
                or not math.isfinite(cfg.frame_zero_start_probability)
                or not 0.0 <= cfg.frame_zero_start_probability <= 1.0
            ):
                raise ValueError("frame_zero_start_probability must be finite and in [0, 1]")

            self.robot: Articulation = env.scene[cfg.asset_name]
            self.contact_sensor: ContactSensor = env.scene[cfg.contact_sensor_name]
            joint_ids, joint_names = self.robot.find_joints(contract.joint_names, preserve_order=True)
            if tuple(joint_names) != contract.joint_names or len(set(joint_ids)) != G1_ACTION_DIM:
                raise RuntimeError("Isaac articulation cannot resolve the exact 29-joint policy order")
            body_ids, body_names = self.robot.find_bodies(contract.body_names, preserve_order=True)
            if tuple(body_names) != contract.body_names or len(set(body_ids)) != len(contract.body_names):
                raise RuntimeError("Isaac articulation body names do not equal the Gate-3 body contract")
            tracking_body_ids, tracking_names = self.robot.find_bodies(
                contract.tracking_body_names, preserve_order=True
            )
            if tuple(tracking_names) != contract.tracking_body_names:
                raise RuntimeError("Isaac tracking-body name resolution changed order")
            foot_body_ids, foot_names = self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)
            sensor_foot_ids, sensor_foot_names = self.contact_sensor.find_bodies(
                cfg.foot_body_names, preserve_order=True
            )
            sensor_root_ids, sensor_root_names = self.contact_sensor.find_bodies(
                [cfg.root_body_name], preserve_order=True
            )
            if tuple(foot_names) != cfg.foot_body_names or tuple(sensor_foot_names) != cfg.foot_body_names:
                raise RuntimeError("left/right foot body names do not resolve exactly in robot and contact sensor")
            if tuple(sensor_root_names) != (cfg.root_body_name,):
                raise RuntimeError("root body does not resolve exactly in the contact sensor")
            root_ids, root_names = self.robot.find_bodies([cfg.root_body_name], preserve_order=True)
            if tuple(root_names) != (cfg.root_body_name,) or root_ids[0] != body_ids[contract.body_names.index(cfg.root_body_name)]:
                raise RuntimeError("configured root body is inconsistent with archive body mapping")

            self.policy_joint_ids = tuple(int(value) for value in joint_ids)
            self.body_ids = tuple(int(value) for value in body_ids)
            self.tracking_body_ids = tuple(int(value) for value in tracking_body_ids)
            self.tracking_archive_body_ids = tuple(
                contract.body_names.index(name) for name in contract.tracking_body_names
            )
            self.foot_body_ids = tuple(int(value) for value in foot_body_ids)
            self.sensor_foot_body_ids = tuple(int(value) for value in sensor_foot_ids)
            self.sensor_root_body_id = int(sensor_root_ids[0])
            self.root_body_id = int(root_ids[0])
            self._horizons = torch.tensor(horizons, device=env.device, dtype=torch.float32)
            self._generator = torch.Generator(device=env.device)
            self._generator.manual_seed(int(cfg.seed))

            super().__init__(cfg, env)

            self._fixed_motion_ids = (
                None
                if fixed_motion_ids is None
                else torch.tensor(fixed_motion_ids, device=self.device, dtype=torch.long)
            )
            self._motion_sample_counts = torch.zeros(
                self.library.num_motions, device=self.device, dtype=torch.long
            )
            self._frame_zero_start_samples = torch.zeros(
                (), device=self.device, dtype=torch.long
            )
            self._random_start_samples = torch.zeros(
                (), device=self.device, dtype=torch.long
            )
            self._fixed_start_samples = torch.zeros(
                (), device=self.device, dtype=torch.long
            )

            n, j, b, h = self.num_envs, G1_ACTION_DIM, len(contract.body_names), len(horizons)
            self.motion_ids = torch.zeros(n, device=self.device, dtype=torch.long)
            self.motion_time = torch.zeros(n, device=self.device)
            self.alignment_yaw_quat_wxyz = torch.zeros(n, 4, device=self.device)
            self.alignment_yaw_quat_wxyz[:, 0] = 1.0
            self.alignment_translation_w = torch.zeros(n, 3, device=self.device)
            self.phase = torch.zeros(n, device=self.device)
            self.terminal = torch.zeros(n, device=self.device, dtype=torch.bool)
            self.joint_pos = torch.zeros(n, j, device=self.device)
            self.joint_vel = torch.zeros(n, j, device=self.device)
            self.joint_acc = torch.zeros(n, j, device=self.device)
            self.root_pos_w = torch.zeros(n, 3, device=self.device)
            self.root_quat_wxyz = torch.zeros(n, 4, device=self.device)
            self.root_quat_wxyz[:, 0] = 1.0
            self.root_lin_vel_w = torch.zeros(n, 3, device=self.device)
            self.root_ang_vel_w = torch.zeros(n, 3, device=self.device)
            self.body_pos_w = torch.zeros(n, b, 3, device=self.device)
            self.body_quat_wxyz = torch.zeros(n, b, 4, device=self.device)
            self.body_quat_wxyz[..., 0] = 1.0
            self.body_lin_vel_w = torch.zeros(n, b, 3, device=self.device)
            self.body_ang_vel_w = torch.zeros(n, b, 3, device=self.device)
            self.foot_sole_position_w = torch.zeros(n, 2, 3, device=self.device)
            self.foot_contact = torch.zeros(n, 2, device=self.device, dtype=torch.bool)
            self.future_joint_pos = torch.zeros(n, h, j, device=self.device)
            self.future_joint_vel = torch.zeros(n, h, j, device=self.device)
            self.future_root_pos_w = torch.zeros(n, h, 3, device=self.device)
            self.future_root_quat_wxyz = torch.zeros(n, h, 4, device=self.device)
            self.future_root_quat_wxyz[..., 0] = 1.0
            self.future_body_pos_w = torch.zeros(n, h, b, 3, device=self.device)
            self.future_body_quat_wxyz = torch.zeros(n, h, b, 4, device=self.device)
            self.future_body_quat_wxyz[..., 0] = 1.0
            self.future_foot_contact = torch.zeros(n, h, 2, device=self.device, dtype=torch.bool)
            self.metrics["joint_position_error"] = torch.zeros(n, device=self.device)
            self.metrics["root_position_error"] = torch.zeros(n, device=self.device)
            self.metrics["tracking_body_position_error"] = torch.zeros(n, device=self.device)

            # Live URDF and archive limits must describe the same policy axes.
            hard = self.robot.data.joint_pos_limits[0, list(self.policy_joint_ids)]
            local_hard = torch.tensor(
                np.stack((G1_CONTRACT.lower, G1_CONTRACT.upper), axis=-1),
                device=self.device,
                dtype=hard.dtype,
            )
            if not torch.allclose(hard, local_hard, atol=2.0e-4, rtol=0.0):
                raise RuntimeError("Isaac joint limits do not match the local named G1 policy contract")

        @property
        def command(self) -> torch.Tensor:
            return torch.cat(
                (
                    self.joint_pos,
                    self.joint_vel,
                    torch.sin(2.0 * math.pi * self.phase).unsqueeze(-1),
                    torch.cos(2.0 * math.pi * self.phase).unsqueeze(-1),
                ),
                dim=-1,
            )

        @property
        def future_horizons_s(self) -> torch.Tensor:
            return self._horizons

        @property
        def tracking_body_pos_w(self) -> torch.Tensor:
            return self.body_pos_w[:, self.tracking_archive_body_ids]

        @property
        def tracking_body_quat_wxyz(self) -> torch.Tensor:
            return self.body_quat_wxyz[:, self.tracking_archive_body_ids]

        @property
        def tracking_body_lin_vel_w(self) -> torch.Tensor:
            return self.body_lin_vel_w[:, self.tracking_archive_body_ids]

        @property
        def tracking_body_ang_vel_w(self) -> torch.Tensor:
            return self.body_ang_vel_w[:, self.tracking_archive_body_ids]

        @property
        def actual_foot_contact(self) -> torch.Tensor:
            force = self.contact_sensor.data.net_forces_w[:, self.sensor_foot_body_ids]
            return torch.linalg.vector_norm(force, dim=-1) >= self.cfg.contact_force_threshold

        @property
        def motion_sample_counts(self) -> torch.Tensor:
            """Return a defensive copy of lifetime reset samples per library clip."""

            return self._motion_sample_counts.clone()

        @property
        def motion_sample_counts_list(self) -> list[int]:
            """Return lifetime reset samples as a synchronization-safe CPU list."""

            return [int(value) for value in self._motion_sample_counts.detach().cpu().tolist()]

        @property
        def start_time_sample_counts(self) -> dict[str, int]:
            """Return exact lifetime counts for the three start-time modes."""

            return {
                "frame_zero": int(self._frame_zero_start_samples.item()),
                "continuous_random": int(self._random_start_samples.item()),
                "fixed": int(self._fixed_start_samples.item()),
            }

        def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
            ids = (
                torch.arange(self.num_envs, device=self.device, dtype=torch.long)
                if env_ids is None
                else _env_ids_tensor(env_ids, self.num_envs, self.device)
            )
            return super().reset(ids)

        def _sample_motion_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
            count = int(env_ids.numel())
            horizon = max(float(self._horizons[-1]), self.cfg.minimum_remaining_time_s)
            valid = torch.where(self.library.valid_motion_mask(horizon))[0]
            if valid.numel() == 0:
                raise RuntimeError("no loaded motion is long enough for reset/future horizons")
            if self._fixed_motion_ids is not None:
                fixed = self._fixed_motion_ids[env_ids]
                if not bool(self.library.valid_motion_mask(horizon)[fixed].all()):
                    raise RuntimeError(
                        "at least one fixed motion is shorter than the configured remaining horizon"
                    )
                return fixed
            selections = torch.randint(
                valid.numel(), (count,), device=self.device, generator=self._generator
            )
            return valid[selections]

        def _populate_reference(self, ids: torch.Tensor) -> None:
            query = self.library.query(self.motion_ids[ids], self.motion_time[ids])
            future = self.library.query_future(
                self.motion_ids[ids], self.motion_time[ids], self._horizons
            )
            yaw = self.alignment_yaw_quat_wxyz[ids]
            translation = self.alignment_translation_w[ids]
            self.phase[ids] = query.phase
            self.terminal[ids] = query.terminal
            self.joint_pos[ids] = query.joint_pos
            self.joint_vel[ids] = query.joint_vel
            self.joint_acc[ids] = query.joint_acc
            self.root_pos_w[ids] = align_positions_by_env_yaw(query.root_pos_w, yaw, translation)
            self.root_quat_wxyz[ids] = align_quaternions_by_env_yaw(query.root_quat_wxyz, yaw)
            self.root_lin_vel_w[ids] = rotate_by_env_yaw(query.root_lin_vel_w, yaw)
            self.root_ang_vel_w[ids] = rotate_by_env_yaw(query.root_ang_vel_w, yaw)
            self.body_pos_w[ids] = align_positions_by_env_yaw(query.body_pos_w, yaw, translation)
            self.body_quat_wxyz[ids] = align_quaternions_by_env_yaw(query.body_quat_wxyz, yaw)
            self.body_lin_vel_w[ids] = rotate_by_env_yaw(query.body_lin_vel_w, yaw)
            self.body_ang_vel_w[ids] = rotate_by_env_yaw(query.body_ang_vel_w, yaw)
            self.foot_sole_position_w[ids] = align_positions_by_env_yaw(
                query.foot_sole_position_w, yaw, translation
            )
            self.foot_contact[ids] = query.foot_contact
            self.future_joint_pos[ids] = future.joint_pos
            self.future_joint_vel[ids] = future.joint_vel
            self.future_root_pos_w[ids] = align_positions_by_env_yaw(
                future.root_pos_w, yaw, translation
            )
            self.future_root_quat_wxyz[ids] = align_quaternions_by_env_yaw(
                future.root_quat_wxyz, yaw
            )
            self.future_body_pos_w[ids] = align_positions_by_env_yaw(
                future.body_pos_w, yaw, translation
            )
            self.future_body_quat_wxyz[ids] = align_quaternions_by_env_yaw(
                future.body_quat_wxyz, yaw
            )
            self.future_foot_contact[ids] = future.foot_contact

        def _resample_command(self, env_ids: Sequence[int]) -> None:
            ids = _env_ids_tensor(env_ids, self.num_envs, self.device)
            count = ids.numel()
            if count == 0:
                return
            motion_ids = self._sample_motion_ids(ids)
            self._motion_sample_counts += torch.bincount(
                motion_ids, minlength=self.library.num_motions
            )
            if self.cfg.random_start:
                remaining = max(float(self._horizons[-1]), self.cfg.minimum_remaining_time_s)
                random_start_time = self.library.sample_start_times(
                    motion_ids, future_horizon_s=remaining, generator=self._generator
                )
                start_time, frame_zero_mask = mix_frame_zero_start_times(
                    random_start_time,
                    self.cfg.frame_zero_start_probability,
                    generator=self._generator,
                )
                frame_zero_count = torch.count_nonzero(frame_zero_mask)
                self._frame_zero_start_samples += frame_zero_count
                self._random_start_samples += count - frame_zero_count
            else:
                start_time = torch.full(
                    (count,), float(self.cfg.fixed_start_time_s), device=self.device
                )
                if bool(torch.any(start_time < 0.0)) or bool(
                    torch.any(start_time + float(self._horizons[-1]) > self.library.durations[motion_ids])
                ):
                    raise RuntimeError("fixed start time is outside a motion/future interval")
                self._fixed_start_samples += count
            initial = self.library.query(motion_ids, start_time)
            target_xy = self._env.scene.env_origins[ids, :2].clone()
            target_xy += _uniform(
                (count, 2), self.cfg.xy_offset_range, device=self.device, generator=self._generator
            )
            target_yaw = _uniform(
                (count,), self.cfg.target_yaw_range, device=self.device, generator=self._generator
            )
            yaw, translation = alignment_from_reference_anchor(
                initial.root_pos_w, initial.root_quat_wxyz, target_xy, target_yaw
            )
            self.motion_ids[ids] = motion_ids
            self.motion_time[ids] = start_time
            self.alignment_yaw_quat_wxyz[ids] = yaw
            self.alignment_translation_w[ids] = translation
            self._populate_reference(ids)

            root_state = torch.cat(
                (
                    self.root_pos_w[ids],
                    self.root_quat_wxyz[ids],
                    self.root_lin_vel_w[ids]
                    + _uniform(
                        (count, 3),
                        self.cfg.root_linear_velocity_noise,
                        device=self.device,
                        generator=self._generator,
                    ),
                    self.root_ang_vel_w[ids]
                    + _uniform(
                        (count, 3),
                        self.cfg.root_angular_velocity_noise,
                        device=self.device,
                        generator=self._generator,
                    ),
                ),
                dim=-1,
            )
            q = self.joint_pos[ids] + _uniform(
                (count, G1_ACTION_DIM),
                self.cfg.joint_position_noise,
                device=self.device,
                generator=self._generator,
            )
            dq = self.joint_vel[ids] + _uniform(
                (count, G1_ACTION_DIM),
                self.cfg.joint_velocity_noise,
                device=self.device,
                generator=self._generator,
            )
            soft = self.robot.data.soft_joint_pos_limits[ids][:, self.policy_joint_ids]
            q = torch.maximum(torch.minimum(q, soft[..., 1]), soft[..., 0])
            velocity_limit = self.robot.data.soft_joint_vel_limits[ids][:, self.policy_joint_ids]
            dq = torch.maximum(torch.minimum(dq, velocity_limit), -velocity_limit)
            self.robot.write_root_link_state_to_sim(root_state, env_ids=ids)
            self.robot.write_joint_state_to_sim(
                q, dq, joint_ids=self.policy_joint_ids, env_ids=ids
            )

        def _update_command(self) -> None:
            # ManagerBasedRLEnv auto-resets done environments immediately
            # before command_manager.compute().  Their robot/reference state
            # has just been written at the sampled start time and no physics
            # step from the new episode has occurred yet, so advancing those
            # IDs here would return an observation one frame ahead of the
            # robot.  Non-reset environments have completed one control step
            # and advance normally.
            self.motion_time[:] = advance_motion_time_after_step(
                self.motion_time,
                self._env.reset_buf,
                self._env.step_dt,
            )
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self._populate_reference(ids)

        def _update_metrics(self) -> None:
            actual_q = self.robot.data.joint_pos[:, self.policy_joint_ids]
            self.metrics["joint_position_error"] += torch.mean(
                torch.abs(actual_q - self.joint_pos), dim=-1
            )
            self.metrics["root_position_error"] += torch.linalg.vector_norm(
                self.robot.data.root_link_pos_w - self.root_pos_w, dim=-1
            )
            actual_body = self.robot.data.body_pos_w[:, self.tracking_body_ids]
            self.metrics["tracking_body_position_error"] += torch.mean(
                torch.linalg.vector_norm(actual_body - self.tracking_body_pos_w, dim=-1), dim=-1
            )


    MultiMotionCommandCfg.class_type = MultiMotionCommand


    @configclass
    class ReferenceResidualActionCfg(ActionTermCfg):
        """Configuration for 29-D residual position plus reference velocity targets."""

        class_type: type = MISSING
        asset_name: str = "robot"
        command_name: str = "motion"
        joint_names: tuple[str, ...] = G1_POLICY_JOINT_NAMES
        residual_scale: float = 0.25
        reference_velocity_feedforward_scale: float = 1.0
        action_clip: float = 1.0
        normalized_delta_limit: float | None = 0.15
        target_limit_mode: str = "soft"
        preserve_order: bool = True
        debug_vis: bool = False

        def __post_init__(self) -> None:
            if self.class_type is MISSING:
                self.class_type = ReferenceResidualAction


    class ReferenceResidualAction(ActionTerm):
        """Map residuals and the motion reference to named G1 PD targets."""

        cfg: ReferenceResidualActionCfg

        def __init__(self, cfg: ReferenceResidualActionCfg, env: ManagerBasedRLEnv):
            super().__init__(cfg, env)
            self._asset: Articulation
            command = env.command_manager.get_term(cfg.command_name)
            if not isinstance(command, MultiMotionCommand):
                raise TypeError(f"{cfg.command_name!r} is not a MultiMotionCommand")
            self._command = command
            contract = command.library.contract
            if tuple(cfg.joint_names) != contract.joint_names:
                raise RuntimeError("action joint names must exactly equal the archive policy order")
            ids, names = self._asset.find_joints(cfg.joint_names, preserve_order=cfg.preserve_order)
            if tuple(names) != contract.joint_names or tuple(ids) != command.policy_joint_ids:
                raise RuntimeError("action and command articulation mappings disagree")
            action_values = (
                (cfg.residual_scale, contract.tracking_residual_scale, "residual_scale"),
                (cfg.action_clip, contract.tracking_action_clip, "action_clip"),
                (
                    cfg.normalized_delta_limit,
                    contract.tracking_normalized_delta_limit,
                    "normalized_delta_limit",
                ),
            )
            for configured, stored, name in action_values:
                if configured is None or abs(float(configured) - float(stored)) > 1.0e-6:
                    raise RuntimeError(f"configured {name} does not match the Gate-3 archive")
            if contract.tracking_action_mode != "reference_residual":
                raise RuntimeError("archive action mode is not reference_residual")
            if (
                isinstance(cfg.reference_velocity_feedforward_scale, bool)
                or not math.isfinite(float(cfg.reference_velocity_feedforward_scale))
                or float(cfg.reference_velocity_feedforward_scale) < 0.0
            ):
                raise ValueError(
                    "reference_velocity_feedforward_scale must be finite and non-negative"
                )
            self.joint_ids = tuple(int(value) for value in ids)
            self._raw_actions = torch.zeros(self.num_envs, G1_ACTION_DIM, device=self.device)
            self._processed_actions = torch.zeros_like(self._raw_actions)
            self.normalized_residual = torch.zeros_like(self._raw_actions)
            self.previous_normalized_residual = torch.zeros_like(self._raw_actions)
            self.joint_targets = torch.zeros_like(self._raw_actions)
            self.joint_velocity_targets = torch.zeros_like(self._raw_actions)
            self.rate_limited = torch.zeros_like(self._raw_actions, dtype=torch.bool)
            self.target_clamped = torch.zeros_like(self._raw_actions, dtype=torch.bool)
            self.velocity_target_clamped = torch.zeros_like(
                self._raw_actions, dtype=torch.bool
            )
            self.velocity_target_buffer_verified = False
            self.velocity_target_buffer_max_abs_error = math.inf

        @property
        def action_dim(self) -> int:
            return G1_ACTION_DIM

        @property
        def raw_actions(self) -> torch.Tensor:
            return self._raw_actions

        @property
        def processed_actions(self) -> torch.Tensor:
            # Isaac action terms conventionally expose the actuator target here.
            return self._processed_actions

        def process_actions(self, actions: torch.Tensor) -> None:
            if actions.shape != self._raw_actions.shape or not bool(torch.isfinite(actions).all()):
                raise ValueError(
                    f"reference-residual action must be finite {tuple(self._raw_actions.shape)}; "
                    f"got {tuple(actions.shape)}"
                )
            self._raw_actions[:] = actions
            self.previous_normalized_residual[:] = self.normalized_residual
            hard = self._asset.data.joint_pos_limits[:, self.joint_ids]
            soft = self._asset.data.soft_joint_pos_limits[:, self.joint_ids]
            result = reference_residual_target(
                actions,
                self._command.joint_pos,
                self.normalized_residual,
                residual_scale=self.cfg.residual_scale,
                action_clip=self.cfg.action_clip,
                normalized_delta_limit=self.cfg.normalized_delta_limit,
                hard_limits=hard,
                soft_limits=soft,
                target_limit_mode=self.cfg.target_limit_mode,
            )
            self.normalized_residual[:] = result.normalized_residual
            self.joint_targets[:] = result.target
            self._processed_actions[:] = result.target
            self.rate_limited[:] = result.rate_limited
            self.target_clamped[:] = result.target_clamped
            velocity_result = reference_velocity_target(
                self._command.joint_vel,
                scale=self.cfg.reference_velocity_feedforward_scale,
                velocity_limits=self._asset.data.soft_joint_vel_limits[:, self.joint_ids],
            )
            self.joint_velocity_targets[:] = velocity_result.target
            self.velocity_target_clamped[:] = velocity_result.target_clamped

        def apply_actions(self) -> None:
            self._asset.set_joint_position_target(self.joint_targets, joint_ids=self.joint_ids)
            self._asset.set_joint_velocity_target(
                self.joint_velocity_targets, joint_ids=self.joint_ids
            )
            if not self.velocity_target_buffer_verified:
                live_target = self._asset.data.joint_vel_target[:, self.joint_ids]
                if live_target.shape != self.joint_velocity_targets.shape:
                    raise RuntimeError(
                        "live articulation velocity-target buffer shape differs from the action term"
                    )
                error = torch.max(torch.abs(live_target - self.joint_velocity_targets))
                if not bool(torch.isfinite(error)) or float(error.item()) > 1.0e-7:
                    raise RuntimeError(
                        "live articulation velocity-target buffer does not equal clipped dq_ref"
                    )
                self.velocity_target_buffer_max_abs_error = float(error.item())
                self.velocity_target_buffer_verified = True

        def reset(self, env_ids: Sequence[int] | None = None) -> None:
            ids: Sequence[int] | slice = slice(None) if env_ids is None else env_ids
            self._raw_actions[ids] = 0.0
            self._processed_actions[ids] = 0.0
            self.normalized_residual[ids] = 0.0
            self.previous_normalized_residual[ids] = 0.0
            self.joint_targets[ids] = 0.0
            self.joint_velocity_targets[ids] = 0.0
            self.rate_limited[ids] = False
            self.target_clamped[ids] = False
            self.velocity_target_clamped[ids] = False


    ReferenceResidualActionCfg.class_type = ReferenceResidualAction


    def _motion_command(env: ManagerBasedRLEnv, command_name: str = "motion") -> MultiMotionCommand:
        command = env.command_manager.get_term(command_name)
        if not isinstance(command, MultiMotionCommand):
            raise TypeError(f"command {command_name!r} is not MultiMotionCommand")
        return command


    def _residual_action(env: ManagerBasedRLEnv, action_name: str = "joint_residual") -> ReferenceResidualAction:
        action = env.action_manager.get_term(action_name)
        if not isinstance(action, ReferenceResidualAction):
            raise TypeError(f"action {action_name!r} is not ReferenceResidualAction")
        return action


    def _robot(env: ManagerBasedRLEnv, command_name: str = "motion") -> tuple[Articulation, MultiMotionCommand]:
        command = _motion_command(env, command_name)
        return command.robot, command


    def resolved_evaluation_isolation(
        env: ManagerBasedRLEnv,
        command_name: str = "motion",
    ) -> dict[str, Any]:
        """Return JSON-safe live witnesses for deterministic evaluation isolation."""

        command = _motion_command(env, command_name)
        active_events = {
            str(mode): [str(name) for name in names]
            for mode, names in sorted(env.event_manager.active_terms.items())
            if names
        }
        return {
            "schema_version": EVALUATION_ISOLATION_SCHEMA_VERSION,
            "command_random_start": bool(command.cfg.random_start),
            "fixed_start_time_s": float(command.cfg.fixed_start_time_s),
            "xy_offset_range_m": [float(value) for value in command.cfg.xy_offset_range],
            "target_yaw_range_rad": [
                float(value) for value in command.cfg.target_yaw_range
            ],
            "resolved_reset_state_noise": {
                "joint_position_rad": [
                    float(value) for value in command.cfg.joint_position_noise
                ],
                "joint_velocity_radps": [
                    float(value) for value in command.cfg.joint_velocity_noise
                ],
                "root_linear_velocity_mps": [
                    float(value) for value in command.cfg.root_linear_velocity_noise
                ],
                "root_angular_velocity_radps": [
                    float(value) for value in command.cfg.root_angular_velocity_noise
                ],
            },
            "policy_observation_corruption_enabled": bool(
                env.cfg.observations.policy.enable_corruption
            ),
            "active_event_terms": active_events,
            "enhanced_determinism_enabled": bool(
                env.cfg.sim.physx.enable_enhanced_determinism
            ),
            "evaluation_reward_contract": {
                "alive": {
                    "kind": "per_second_rate",
                    "weight": float(env.reward_manager.get_term_cfg("alive").weight),
                },
                "failure": {
                    "kind": "one_shot_non_timeout_impulse",
                    "impulse": float(env.reward_manager.get_term_cfg("failure").weight),
                    "normalization": "divide_by_live_step_dt",
                    "pure_timeouts_excluded": True,
                    "simultaneous_timeout_failure_penalized": True,
                },
            },
            "reward_integration_dt_s": float(env.step_dt),
        }


    def obs_future_joint_position_error(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        """Reference-minus-measured joint positions for all configured horizons."""

        robot, command = _robot(env, command_name)
        actual = robot.data.joint_pos[:, command.policy_joint_ids].unsqueeze(1)
        return (command.future_joint_pos - actual).flatten(1)


    def obs_future_root_position_error(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        """Positive-horizon root positions relative to the measured G1 pelvis."""

        robot, command = _robot(env, command_name)
        position, _ = positive_horizon_root_pose_preview(
            command.future_root_pos_w,
            command.future_root_quat_wxyz,
            robot.data.root_link_pos_w,
            robot.data.root_link_quat_w,
        )
        return position.flatten(1)


    def obs_future_root_orientation_error(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        """Positive-horizon root orientations relative to the measured G1 pelvis."""

        robot, command = _robot(env, command_name)
        _, orientation = positive_horizon_root_pose_preview(
            command.future_root_pos_w,
            command.future_root_quat_wxyz,
            robot.data.root_link_pos_w,
            robot.data.root_link_quat_w,
        )
        return orientation.flatten(1)


    def obs_joint_velocity_error(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return command.joint_vel - robot.data.joint_vel[:, command.policy_joint_ids]


    def obs_root_orientation_error_6d(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return quat_error_6d_wxyz(command.root_quat_wxyz, robot.data.root_link_quat_w)


    def obs_motion_phase(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
        phase = _motion_command(env, command_name).phase
        return torch.stack(
            (torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)), dim=-1
        )


    def obs_root_angular_velocity_error(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return root_frame_vector_error(
            command.root_ang_vel_w,
            robot.data.root_link_ang_vel_w,
            robot.data.root_link_quat_w,
        )


    def obs_target_root_angular_velocity(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        """Commanded angular velocity in the measured G1 pelvis frame."""

        robot, command = _robot(env, command_name)
        return root_frame_vector(
            command.root_ang_vel_w,
            robot.data.root_link_quat_w,
        )


    def obs_projected_gravity(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, _ = _robot(env, command_name)
        return robot.data.projected_gravity_b


    def obs_policy_joint_position_relative(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        ids = command.policy_joint_ids
        return robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]


    def obs_policy_joint_velocity(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return robot.data.joint_vel[:, command.policy_joint_ids]


    def obs_reference_foot_contacts(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        return _motion_command(env, command_name).future_foot_contact.to(torch.float32).flatten(1)


    def obs_last_normalized_residual(
        env: ManagerBasedRLEnv, action_name: str = "joint_residual"
    ) -> torch.Tensor:
        return _residual_action(env, action_name).normalized_residual


    def obs_root_linear_velocity_error(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return root_frame_vector_error(
            command.root_lin_vel_w,
            robot.data.root_link_lin_vel_w,
            robot.data.root_link_quat_w,
        )


    def obs_target_root_linear_velocity(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        """Commanded linear velocity in the measured G1 pelvis frame."""

        robot, command = _robot(env, command_name)
        return root_frame_vector(
            command.root_lin_vel_w,
            robot.data.root_link_quat_w,
        )


    def obs_root_position_error(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return root_frame_vector_error(
            command.root_pos_w,
            robot.data.root_link_pos_w,
            robot.data.root_link_quat_w,
        )


    def obs_tracking_body_position_error(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        actual = robot.data.body_pos_w[:, command.tracking_body_ids]
        error_w = command.tracking_body_pos_w - actual
        root_inverse = quat_conjugate_wxyz(robot.data.root_link_quat_w).unsqueeze(1)
        return quat_apply_wxyz(root_inverse.expand(-1, error_w.shape[1], -1), error_w).flatten(1)


    def obs_tracking_body_orientation_error(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        actual = robot.data.body_quat_w[:, command.tracking_body_ids]
        return quat_error_6d_wxyz(command.tracking_body_quat_wxyz, actual).flatten(1)


    def obs_tracking_body_linear_velocity_error(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        actual = robot.data.body_lin_vel_w[:, command.tracking_body_ids]
        error_w = command.tracking_body_lin_vel_w - actual
        root_inverse = quat_conjugate_wxyz(robot.data.root_link_quat_w).unsqueeze(1)
        return quat_apply_wxyz(root_inverse.expand(-1, error_w.shape[1], -1), error_w).flatten(1)


    def obs_tracking_body_angular_velocity_error(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        actual = robot.data.body_ang_vel_w[:, command.tracking_body_ids]
        error_w = command.tracking_body_ang_vel_w - actual
        root_inverse = quat_conjugate_wxyz(robot.data.root_link_quat_w).unsqueeze(1)
        return quat_apply_wxyz(root_inverse.expand(-1, error_w.shape[1], -1), error_w).flatten(1)


    def obs_actual_foot_contacts(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        return _motion_command(env, command_name).actual_foot_contact.to(torch.float32)


    def _exp_tracking(error: torch.Tensor, std: float) -> torch.Tensor:
        if not math.isfinite(std) or std <= 0.0:
            raise ValueError("tracking reward std must be finite and positive")
        if error.ndim == 1:
            squared = torch.square(error)
        else:
            squared = torch.mean(torch.square(error), dim=tuple(range(1, error.ndim)))
        return torch.exp(-squared / (std * std))


    def reward_joint_position(
        env: ManagerBasedRLEnv, std: float, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return _exp_tracking(
            command.joint_pos - robot.data.joint_pos[:, command.policy_joint_ids], std
        )


    def reward_joint_velocity(
        env: ManagerBasedRLEnv, std: float, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return _exp_tracking(
            command.joint_vel - robot.data.joint_vel[:, command.policy_joint_ids], std
        )


    def reward_root_position(
        env: ManagerBasedRLEnv, std: float, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return _exp_tracking(command.root_pos_w - robot.data.root_link_pos_w, std)


    def reward_root_orientation(
        env: ManagerBasedRLEnv, std: float, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return _exp_tracking(
            quat_error_angle_wxyz(command.root_quat_wxyz, robot.data.root_link_quat_w), std
        )


    def reward_root_linear_velocity(
        env: ManagerBasedRLEnv, std: float, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return _exp_tracking(
            command.root_lin_vel_w - robot.data.root_link_lin_vel_w, std
        )


    def reward_root_angular_velocity(
        env: ManagerBasedRLEnv, std: float, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return _exp_tracking(
            command.root_ang_vel_w - robot.data.root_link_ang_vel_w, std
        )


    def reward_tracking_body_position(
        env: ManagerBasedRLEnv, std: float, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        actual = robot.data.body_pos_w[:, command.tracking_body_ids]
        return _exp_tracking(command.tracking_body_pos_w - actual, std)


    def reward_tracking_body_orientation(
        env: ManagerBasedRLEnv, std: float, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        actual = robot.data.body_quat_w[:, command.tracking_body_ids]
        angle = quat_error_angle_wxyz(command.tracking_body_quat_wxyz, actual)
        return _exp_tracking(angle, std)


    def reward_contact_match(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        command = _motion_command(env, command_name)
        return torch.mean((command.actual_foot_contact == command.foot_contact).to(torch.float32), dim=-1)


    def reward_terminal_failure_impulse(env: ManagerBasedRLEnv) -> torch.Tensor:
        """Penalize each non-timeout failure by the configured one-shot weight."""

        return terminal_failure_impulse(
            env.termination_manager.terminated,
            env.step_dt,
        )


    def penalty_feet_slide(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        tangential_speed_sq = torch.sum(
            torch.square(robot.data.body_lin_vel_w[:, command.foot_body_ids, :2]), dim=-1
        )
        stance = command.actual_foot_contact | command.foot_contact
        return torch.sum(tangential_speed_sq * stance.to(tangential_speed_sq.dtype), dim=-1)


    def penalty_residual_l2(
        env: ManagerBasedRLEnv, action_name: str = "joint_residual"
    ) -> torch.Tensor:
        residual = _residual_action(env, action_name).normalized_residual
        return torch.mean(torch.square(residual), dim=-1)


    def penalty_raw_action_excess(
        env: ManagerBasedRLEnv,
        action_clip: float,
        action_name: str = "joint_residual",
    ) -> torch.Tensor:
        action = _residual_action(env, action_name)
        if abs(float(action_clip) - float(action.cfg.action_clip)) > 1.0e-9:
            raise RuntimeError("raw-action penalty clip differs from the action safety clip")
        return raw_action_excess_l2(action.raw_actions, action_clip=action_clip)


    def penalty_action_boundary_margin(
        env: ManagerBasedRLEnv,
        onset: float,
        action_clip: float,
        action_name: str = "joint_residual",
    ) -> torch.Tensor:
        action = _residual_action(env, action_name)
        if abs(float(action_clip) - float(action.cfg.action_clip)) > 1.0e-9:
            raise RuntimeError("action-boundary penalty clip differs from the action safety clip")
        return action_boundary_margin_l2(
            action.raw_actions,
            onset=onset,
            action_clip=action_clip,
        )


    def penalty_action_limiter_request_gap(
        env: ManagerBasedRLEnv,
        action_clip: float,
        action_name: str = "joint_residual",
    ) -> torch.Tensor:
        action = _residual_action(env, action_name)
        if abs(float(action_clip) - float(action.cfg.action_clip)) > 1.0e-9:
            raise RuntimeError("action-limiter penalty clip differs from the action safety clip")
        return action_limiter_request_gap_l2(
            action.raw_actions,
            action.normalized_residual,
            action_clip=action_clip,
        )


    def penalty_residual_rate_l2(
        env: ManagerBasedRLEnv, action_name: str = "joint_residual"
    ) -> torch.Tensor:
        action = _residual_action(env, action_name)
        return torch.mean(
            torch.square(action.normalized_residual - action.previous_normalized_residual), dim=-1
        )


    def penalty_tracking_divergence_risk(
        env: ManagerBasedRLEnv,
        maximum_root_distance: float,
        maximum_mean_joint_error: float,
        maximum_mean_body_distance: float,
        onset_fraction: float,
        command_name: str = "motion",
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        components = tracking_divergence_components(
            robot.data.root_link_pos_w,
            command.root_pos_w,
            robot.data.joint_pos[:, command.policy_joint_ids],
            command.joint_pos,
            robot.data.body_pos_w[:, command.tracking_body_ids],
            command.tracking_body_pos_w,
            maximum_root_distance=maximum_root_distance,
            maximum_mean_joint_error=maximum_mean_joint_error,
            maximum_mean_body_distance=maximum_mean_body_distance,
        )
        return tracking_divergence_risk_barrier(
            components,
            maximum_root_distance=maximum_root_distance,
            maximum_mean_joint_error=maximum_mean_joint_error,
            maximum_mean_body_distance=maximum_mean_body_distance,
            onset_fraction=onset_fraction,
        )


    def penalty_target_clamp(
        env: ManagerBasedRLEnv, action_name: str = "joint_residual"
    ) -> torch.Tensor:
        return torch.mean(_residual_action(env, action_name).target_clamped.to(torch.float32), dim=-1)


    def penalty_joint_torque(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return torch.sum(torch.square(robot.data.applied_torque[:, command.policy_joint_ids]), dim=-1)


    def penalty_joint_acceleration(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return torch.sum(torch.square(robot.data.joint_acc[:, command.policy_joint_ids]), dim=-1)


    def penalty_joint_position_limits(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        q = robot.data.joint_pos[:, command.policy_joint_ids]
        limits = robot.data.soft_joint_pos_limits[:, command.policy_joint_ids]
        return torch.sum((limits[..., 0] - q).clamp_min(0.0) + (q - limits[..., 1]).clamp_min(0.0), dim=-1)


    def penalty_joint_velocity_limits(
        env: ManagerBasedRLEnv, soft_ratio: float, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        velocity = torch.abs(robot.data.joint_vel[:, command.policy_joint_ids])
        limit = robot.data.soft_joint_vel_limits[:, command.policy_joint_ids] * soft_ratio
        return torch.sum((velocity - limit).clamp(0.0, 1.0), dim=-1)


    def termination_motion_end(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        return _motion_command(env, command_name).terminal


    def termination_fall(
        env: ManagerBasedRLEnv,
        maximum_root_height_deficit: float,
        maximum_orientation_error: float,
        pelvis_contact_threshold: float,
        pelvis_contact_reference_height: float,
        maximum_pelvis_contact_reference_tilt: float,
        command_name: str = "motion",
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        pelvis_force = torch.linalg.vector_norm(
            command.contact_sensor.data.net_forces_w[:, command.sensor_root_body_id], dim=-1
        )
        return reference_aware_fall_components(
            robot.data.root_link_pos_w,
            command.root_pos_w,
            robot.data.root_link_quat_w,
            command.root_quat_wxyz,
            pelvis_force,
            maximum_root_height_deficit=maximum_root_height_deficit,
            maximum_orientation_error=maximum_orientation_error,
            pelvis_contact_threshold=pelvis_contact_threshold,
            pelvis_contact_reference_height=pelvis_contact_reference_height,
            maximum_pelvis_contact_reference_tilt=maximum_pelvis_contact_reference_tilt,
        ).fall


    class ReferenceAwareFallTermination(ManagerTermBase):
        """Fall termination that can retain an exact post-physics witness.

        Isaac computes termination terms before automatically resetting done
        environments.  The detached clones below therefore remain the only
        exact terminal-state witness available after ``env.step()`` returns.
        Witness capture is disabled during training.
        """

        def __init__(self, cfg: DoneTerm, env: ManagerBasedRLEnv):
            super().__init__(cfg, env)
            self.capture_sequence = 0
            self.last_witness: dict[str, Any] | None = None

        def __call__(
            self,
            env: ManagerBasedRLEnv,
            maximum_root_height_deficit: float,
            maximum_orientation_error: float,
            pelvis_contact_threshold: float,
            pelvis_contact_reference_height: float,
            maximum_pelvis_contact_reference_tilt: float,
            capture_witness: bool = False,
            command_name: str = "motion",
        ) -> torch.Tensor:
            robot, command = _robot(env, command_name)
            pelvis_force = torch.linalg.vector_norm(
                command.contact_sensor.data.net_forces_w[:, command.sensor_root_body_id],
                dim=-1,
            )
            components = reference_aware_fall_components(
                robot.data.root_link_pos_w,
                command.root_pos_w,
                robot.data.root_link_quat_w,
                command.root_quat_wxyz,
                pelvis_force,
                maximum_root_height_deficit=maximum_root_height_deficit,
                maximum_orientation_error=maximum_orientation_error,
                pelvis_contact_threshold=pelvis_contact_threshold,
                pelvis_contact_reference_height=pelvis_contact_reference_height,
                maximum_pelvis_contact_reference_tilt=maximum_pelvis_contact_reference_tilt,
            )
            if capture_witness:
                self.capture_sequence += 1
                self.last_witness = {
                    "capture_sequence": self.capture_sequence,
                    "reference_time_s": command.motion_time.detach().clone(),
                    "motion_ids": command.motion_ids.detach().clone(),
                    "root_height_deficit_m": components.root_height_deficit_m.detach().clone(),
                    "root_orientation_error_rad": components.root_orientation_error_rad.detach().clone(),
                    "reference_pelvis_height_m": components.reference_pelvis_height_m.detach().clone(),
                    "reference_pelvis_tilt_rad": components.reference_pelvis_tilt_rad.detach().clone(),
                    "pelvis_contact_force_n": components.pelvis_contact_force_n.detach().clone(),
                    "reference_height_eligible": components.reference_height_eligible.detach().clone(),
                    "reference_tilt_eligible": components.reference_tilt_eligible.detach().clone(),
                    "pelvis_contact_applicable": components.pelvis_contact_applicable.detach().clone(),
                    "pelvis_force_exceeded": components.pelvis_force_exceeded.detach().clone(),
                    "height_deficit_exceeded": components.height_deficit_exceeded.detach().clone(),
                    "orientation_error_exceeded": components.orientation_error_exceeded.detach().clone(),
                    "unexpected_pelvis_contact": components.unexpected_pelvis_contact.detach().clone(),
                    "fall": components.fall.detach().clone(),
                }
            return components.fall


    def termination_tracking_divergence(
        env: ManagerBasedRLEnv,
        maximum_root_distance: float,
        maximum_mean_joint_error: float,
        maximum_mean_body_distance: float,
        command_name: str = "motion",
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        return tracking_divergence_components(
            robot.data.root_link_pos_w,
            command.root_pos_w,
            robot.data.joint_pos[:, command.policy_joint_ids],
            command.joint_pos,
            robot.data.body_pos_w[:, command.tracking_body_ids],
            command.tracking_body_pos_w,
            maximum_root_distance=maximum_root_distance,
            maximum_mean_joint_error=maximum_mean_joint_error,
            maximum_mean_body_distance=maximum_mean_body_distance,
        ).diverged


    class TrackingDivergenceTermination(ManagerTermBase):
        """Divergence termination with optional terminal torque telemetry."""

        def __init__(self, cfg: DoneTerm, env: ManagerBasedRLEnv):
            super().__init__(cfg, env)
            self.capture_sequence = 0
            self.last_witness: dict[str, Any] | None = None

        def __call__(
            self,
            env: ManagerBasedRLEnv,
            maximum_root_distance: float,
            maximum_mean_joint_error: float,
            maximum_mean_body_distance: float,
            capture_witness: bool = False,
            command_name: str = "motion",
        ) -> torch.Tensor:
            robot, command = _robot(env, command_name)
            components = tracking_divergence_components(
                robot.data.root_link_pos_w,
                command.root_pos_w,
                robot.data.joint_pos[:, command.policy_joint_ids],
                command.joint_pos,
                robot.data.body_pos_w[:, command.tracking_body_ids],
                command.tracking_body_pos_w,
                maximum_root_distance=maximum_root_distance,
                maximum_mean_joint_error=maximum_mean_joint_error,
                maximum_mean_body_distance=maximum_mean_body_distance,
            )
            if capture_witness:
                self.capture_sequence += 1
                witness: dict[str, Any] = {
                    "capture_sequence": self.capture_sequence,
                    "reference_time_s": command.motion_time.detach().clone(),
                    "motion_ids": command.motion_ids.detach().clone(),
                    "root_distance_m": components.root_distance_m.detach().clone(),
                    "mean_absolute_joint_error_rad": components.mean_absolute_joint_error_rad.detach().clone(),
                    "mean_tracking_body_distance_m": components.mean_tracking_body_distance_m.detach().clone(),
                    "root_distance_exceeded": components.root_distance_exceeded.detach().clone(),
                    "mean_joint_error_exceeded": components.mean_joint_error_exceeded.detach().clone(),
                    "mean_body_distance_exceeded": components.mean_body_distance_exceeded.detach().clone(),
                    "diverged": components.diverged.detach().clone(),
                    "effort_available": False,
                    "effort_unavailable_reason": None,
                }
                computed = getattr(robot.data, "computed_torque", None)
                applied = getattr(robot.data, "applied_torque", None)
                if isinstance(computed, torch.Tensor) and isinstance(applied, torch.Tensor):
                    computed_policy = computed[:, command.policy_joint_ids]
                    applied_policy = applied[:, command.policy_joint_ids]
                    effort_limits = torch.as_tensor(
                        G1_CONTRACT.effort,
                        device=computed_policy.device,
                        dtype=computed_policy.dtype,
                    )
                    effort = torque_effort_ratio_components(
                        computed_policy,
                        applied_policy,
                        effort_limits,
                    )
                    witness.update(
                        {
                            "effort_available": True,
                            "computed_abs_torque_to_effort_ratio": effort.computed_abs_ratio.detach().clone(),
                            "applied_abs_torque_to_effort_ratio": effort.applied_abs_ratio.detach().clone(),
                            "computed_effort_limit_exceeded": effort.computed_limit_exceeded.detach().clone(),
                            "applied_effort_limit_exceeded": effort.applied_limit_exceeded.detach().clone(),
                        }
                    )
                else:
                    witness["effort_unavailable_reason"] = (
                        "Isaac articulation data exposes neither a compatible computed_torque "
                        "nor applied_torque tensor"
                    )
                self.last_witness = witness
            return components.diverged


    def termination_joint_limits(
        env: ManagerBasedRLEnv,
        position_tolerance: float,
        velocity_ratio: float,
        command_name: str = "motion",
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        q = robot.data.joint_pos[:, command.policy_joint_ids]
        dq = torch.abs(robot.data.joint_vel[:, command.policy_joint_ids])
        hard = robot.data.joint_pos_limits[:, command.policy_joint_ids]
        velocity_limit = robot.data.soft_joint_vel_limits[:, command.policy_joint_ids] * velocity_ratio
        position_bad = torch.any(
            (q < hard[..., 0] - position_tolerance) | (q > hard[..., 1] + position_tolerance), dim=-1
        )
        velocity_bad = torch.any(dq > velocity_limit, dim=-1)
        return position_bad | velocity_bad


    def termination_nonfinite(
        env: ManagerBasedRLEnv, command_name: str = "motion"
    ) -> torch.Tensor:
        robot, command = _robot(env, command_name)
        values = (
            robot.data.root_link_state_w,
            robot.data.joint_pos[:, command.policy_joint_ids],
            robot.data.joint_vel[:, command.policy_joint_ids],
        )
        result = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for value in values:
            result |= ~torch.isfinite(value).flatten(1).all(dim=-1)
        return result


    @configclass
    class G1TrackingSceneCfg(InteractiveSceneCfg):
        """Flat-ground G1 scene with an all-body contact sensor."""

        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True,
            update_period=0.0,
        )
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.9, 0.9, 0.9)),
        )


    @configclass
    class G1TrackingCommandsCfg:
        motion: MultiMotionCommandCfg = MultiMotionCommandCfg(
            class_type=MultiMotionCommand,
            motion_files=[],
        )


    @configclass
    class G1TrackingActionsCfg:
        joint_residual: ReferenceResidualActionCfg = ReferenceResidualActionCfg(
            class_type=ReferenceResidualAction,
            asset_name="robot",
            command_name="motion",
            joint_names=G1_POLICY_JOINT_NAMES,
            residual_scale=0.25,
            reference_velocity_feedforward_scale=REFERENCE_VELOCITY_FEEDFORWARD_SCALE,
            action_clip=1.0,
            normalized_delta_limit=0.15,
            target_limit_mode="soft",
            preserve_order=True,
        )


    @configclass
    class G1TrackingObservationsCfg:
        """293-D deployable actor and 505-D privileged critic observations."""

        @configclass
        class PolicyCfg(ObsGroup):
            future_joint_position_error = ObsTerm(
                func=obs_future_joint_position_error,
                noise=UniformNoise(n_min=-0.01, n_max=0.01),
            )
            future_root_position_error = ObsTerm(
                func=obs_future_root_position_error,
                scale=1.0,
            )
            future_root_orientation_error = ObsTerm(
                func=obs_future_root_orientation_error,
                scale=1.0,
            )
            joint_velocity_error = ObsTerm(
                func=obs_joint_velocity_error,
                scale=0.05,
                noise=UniformNoise(n_min=-0.10, n_max=0.10),
            )
            root_orientation_error = ObsTerm(func=obs_root_orientation_error_6d)
            root_position_error = ObsTerm(func=obs_root_position_error)
            root_linear_velocity_error = ObsTerm(
                func=obs_root_linear_velocity_error,
                scale=0.2,
                noise=UniformNoise(n_min=-0.05, n_max=0.05),
            )
            target_root_linear_velocity = ObsTerm(
                func=obs_target_root_linear_velocity,
                scale=0.2,
            )
            phase = ObsTerm(func=obs_motion_phase)
            root_angular_velocity_error = ObsTerm(
                func=obs_root_angular_velocity_error,
                scale=0.2,
                noise=UniformNoise(n_min=-0.05, n_max=0.05),
            )
            target_root_angular_velocity = ObsTerm(
                func=obs_target_root_angular_velocity,
                scale=0.2,
            )
            projected_gravity = ObsTerm(
                func=obs_projected_gravity,
                noise=UniformNoise(n_min=-0.02, n_max=0.02),
            )
            joint_position = ObsTerm(
                func=obs_policy_joint_position_relative,
                noise=UniformNoise(n_min=-0.01, n_max=0.01),
            )
            joint_velocity = ObsTerm(
                func=obs_policy_joint_velocity,
                scale=0.05,
                noise=UniformNoise(n_min=-0.10, n_max=0.10),
            )
            predicted_contacts = ObsTerm(func=obs_reference_foot_contacts)
            last_action = ObsTerm(func=obs_last_normalized_residual)

            def __post_init__(self) -> None:
                self.enable_corruption = True
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

        @configclass
        class CriticCfg(ObsGroup):
            future_joint_position_error = ObsTerm(func=obs_future_joint_position_error)
            future_root_position_error = ObsTerm(
                func=obs_future_root_position_error,
                scale=1.0,
            )
            future_root_orientation_error = ObsTerm(
                func=obs_future_root_orientation_error,
                scale=1.0,
            )
            joint_velocity_error = ObsTerm(func=obs_joint_velocity_error, scale=0.05)
            root_orientation_error = ObsTerm(func=obs_root_orientation_error_6d)
            root_position_error = ObsTerm(func=obs_root_position_error)
            root_linear_velocity_error = ObsTerm(
                func=obs_root_linear_velocity_error,
                scale=0.2,
            )
            target_root_linear_velocity = ObsTerm(
                func=obs_target_root_linear_velocity,
                scale=0.2,
            )
            phase = ObsTerm(func=obs_motion_phase)
            root_angular_velocity_error = ObsTerm(
                func=obs_root_angular_velocity_error,
                scale=0.2,
            )
            target_root_angular_velocity = ObsTerm(
                func=obs_target_root_angular_velocity,
                scale=0.2,
            )
            projected_gravity = ObsTerm(func=obs_projected_gravity)
            joint_position = ObsTerm(func=obs_policy_joint_position_relative)
            joint_velocity = ObsTerm(func=obs_policy_joint_velocity, scale=0.05)
            predicted_contacts = ObsTerm(func=obs_reference_foot_contacts)
            last_action = ObsTerm(func=obs_last_normalized_residual)
            tracking_body_position_error = ObsTerm(func=obs_tracking_body_position_error)
            tracking_body_orientation_error = ObsTerm(func=obs_tracking_body_orientation_error)
            tracking_body_linear_velocity_error = ObsTerm(
                func=obs_tracking_body_linear_velocity_error,
                scale=0.2,
            )
            tracking_body_angular_velocity_error = ObsTerm(
                func=obs_tracking_body_angular_velocity_error,
                scale=0.2,
            )
            actual_foot_contacts = ObsTerm(func=obs_actual_foot_contacts)

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = True

        critic: CriticCfg = CriticCfg()


    @configclass
    class G1TrackingRewardsCfg:
        """Pose/contact objectives plus conservative dynamics penalties."""

        joint_position = RewTerm(
            func=reward_joint_position, weight=2.0, params={"std": 0.30}
        )
        joint_velocity = RewTerm(
            func=reward_joint_velocity, weight=0.50, params={"std": 2.0}
        )
        root_position = RewTerm(
            func=reward_root_position, weight=1.0, params={"std": 0.25}
        )
        root_orientation = RewTerm(
            func=reward_root_orientation, weight=0.50, params={"std": 0.35}
        )
        root_linear_velocity = RewTerm(
            func=reward_root_linear_velocity, weight=0.75, params={"std": 1.0}
        )
        root_angular_velocity = RewTerm(
            func=reward_root_angular_velocity, weight=0.25, params={"std": 2.0}
        )
        tracking_body_position = RewTerm(
            func=reward_tracking_body_position, weight=2.0, params={"std": 0.20}
        )
        tracking_body_orientation = RewTerm(
            func=reward_tracking_body_orientation, weight=0.75, params={"std": 0.50}
        )
        contact_match = RewTerm(func=reward_contact_match, weight=0.30)
        alive = RewTerm(func=isaac_mdp.is_alive, weight=0.10)
        feet_slide = RewTerm(func=penalty_feet_slide, weight=-0.20)
        raw_action_excess = RewTerm(
            func=penalty_raw_action_excess,
            weight=-0.25,
            params={"action_clip": 1.0},
        )
        action_boundary_margin = RewTerm(
            func=penalty_action_boundary_margin,
            weight=RECOVERY_ACTION_BOUNDARY_REWARD_WEIGHT,
            params={"onset": ACTION_BOUNDARY_ONSET, "action_clip": 1.0},
        )
        action_limiter_request_gap = RewTerm(
            func=penalty_action_limiter_request_gap,
            weight=RECOVERY_ACTION_LIMITER_GAP_REWARD_WEIGHT,
            params={"action_clip": 1.0},
        )
        residual_magnitude = RewTerm(func=penalty_residual_l2, weight=-0.02)
        residual_rate = RewTerm(func=penalty_residual_rate_l2, weight=-0.05)
        divergence_risk = RewTerm(
            func=penalty_tracking_divergence_risk,
            weight=RECOVERY_DIVERGENCE_RISK_REWARD_WEIGHT,
            params={
                "maximum_root_distance": TRACKING_DIVERGENCE_MAX_ROOT_DISTANCE_M,
                "maximum_mean_joint_error": TRACKING_DIVERGENCE_MAX_MEAN_JOINT_ERROR_RAD,
                "maximum_mean_body_distance": TRACKING_DIVERGENCE_MAX_MEAN_BODY_DISTANCE_M,
                "onset_fraction": DIVERGENCE_RISK_ONSET_FRACTION,
            },
        )
        target_clamp = RewTerm(func=penalty_target_clamp, weight=-0.10)
        joint_torque = RewTerm(func=penalty_joint_torque, weight=-2.0e-5)
        joint_acceleration = RewTerm(func=penalty_joint_acceleration, weight=-2.5e-7)
        joint_position_limit = RewTerm(func=penalty_joint_position_limits, weight=-2.0)
        joint_velocity_limit = RewTerm(
            func=penalty_joint_velocity_limits,
            weight=-0.10,
            params={"soft_ratio": 0.90},
        )
        failure = RewTerm(func=reward_terminal_failure_impulse, weight=-5.0)


    @configclass
    class G1TrackingTerminationsCfg:
        time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
        motion_end = DoneTerm(func=termination_motion_end, time_out=True)
        fall = DoneTerm(
            func=ReferenceAwareFallTermination,
            params={
                "maximum_root_height_deficit": 0.25,
                "maximum_orientation_error": 1.0,
                "pelvis_contact_threshold": 50.0,
                "pelvis_contact_reference_height": 0.50,
                "maximum_pelvis_contact_reference_tilt": 0.80,
                "capture_witness": False,
            },
        )
        tracking_divergence = DoneTerm(
            func=TrackingDivergenceTermination,
            params={
                "maximum_root_distance": TRACKING_DIVERGENCE_MAX_ROOT_DISTANCE_M,
                "maximum_mean_joint_error": TRACKING_DIVERGENCE_MAX_MEAN_JOINT_ERROR_RAD,
                "maximum_mean_body_distance": TRACKING_DIVERGENCE_MAX_MEAN_BODY_DISTANCE_M,
                "capture_witness": False,
            },
        )
        joint_limits = DoneTerm(
            func=termination_joint_limits,
            params={"position_tolerance": 0.01, "velocity_ratio": 1.10},
        )
        nonfinite = DoneTerm(func=termination_nonfinite)


    @configclass
    class G1TrackingEventsCfg:
        """Nominal training randomization; no event overwrites reference reset state."""

        physics_material = EventTerm(
            func=isaac_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.7, 1.2),
                "dynamic_friction_range": (0.6, 1.1),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 32,
            },
        )
        torso_mass = EventTerm(
            func=isaac_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "mass_distribution_params": (-0.5, 0.5),
                "operation": "add",
            },
        )
        actuator_gains = EventTerm(
            func=isaac_mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        push = EventTerm(
            func=isaac_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 7.0),
            params={
                "velocity_range": {
                    "x": (-0.40, 0.40),
                    "y": (-0.40, 0.40),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.20, 0.20),
                }
            },
        )


    @configclass
    class G1TrackingEvalEventsCfg:
        """Intentionally empty event set accepted by Isaac's EventManager."""

        pass


    @configclass
    class G1TrackingEnvCfg(ManagerBasedRLEnvCfg):
        """Multi-motion training environment; randomization is opt-in per run."""

        scene: G1TrackingSceneCfg = G1TrackingSceneCfg(num_envs=2048, env_spacing=2.5)
        observations: G1TrackingObservationsCfg = G1TrackingObservationsCfg()
        actions: G1TrackingActionsCfg = G1TrackingActionsCfg()
        commands: G1TrackingCommandsCfg = G1TrackingCommandsCfg()
        rewards: G1TrackingRewardsCfg = G1TrackingRewardsCfg()
        terminations: G1TrackingTerminationsCfg = G1TrackingTerminationsCfg()
        events: G1TrackingEventsCfg | None = G1TrackingEventsCfg()
        curriculum = None

        def __post_init__(self) -> None:
            self.decimation = G1_CONTRACT.decimation
            # Motion-end normally truncates first.  A generous cap preserves
            # unusually long ACCAD clips in deterministic full-clip evaluation.
            self.episode_length_s = 600.0
            self.sim.dt = G1_CONTRACT.physics_dt
            self.sim.render_interval = self.decimation
            self.sim.physics_material = self.scene.terrain.physics_material
            self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
            self.scene.contact_forces.update_period = self.sim.dt


    @configclass
    class G1TrackingEvalEnvCfg(G1TrackingEnvCfg):
        """Deterministic one-environment, zero-randomization evaluation config."""

        events: G1TrackingEvalEventsCfg = G1TrackingEvalEventsCfg()

        def __post_init__(self) -> None:
            super().__post_init__()
            self.scene.num_envs = 1
            self.observations.policy.enable_corruption = False
            self.terminations.fall.params["capture_witness"] = True
            self.terminations.tracking_divergence.params["capture_witness"] = True
            command = self.commands.motion
            command.random_start = False
            command.minimum_remaining_time_s = 0.0
            command.fixed_motion_id = 0
            command.fixed_start_time_s = 0.0
            command.xy_offset_range = (0.0, 0.0)
            command.target_yaw_range = (0.0, 0.0)
            command.joint_position_noise = (0.0, 0.0)
            command.joint_velocity_noise = (0.0, 0.0)
            command.root_linear_velocity_noise = (0.0, 0.0)
            command.root_angular_velocity_noise = (0.0, 0.0)


    def make_tracking_env_cfg(
        motion_files: Sequence[str | Path],
        *,
        num_envs: int | None = None,
        eval_mode: bool = False,
        future_horizons_s: Sequence[float] = DEFAULT_FUTURE_HORIZONS_S,
        correspondence_path: str | Path | None = None,
        fixed_motion_id: int | None = None,
        fixed_motion_ids: Sequence[int] | None = None,
        random_start: bool | None = None,
        frame_zero_start_probability: float = 0.50,
        reset_state_noise_scale: float = RECOVERY_RESET_STATE_NOISE_SCALE,
        alive_reward_weight: float = RECOVERY_ALIVE_REWARD_WEIGHT,
        failure_reward_weight: float = RECOVERY_FAILURE_REWARD_WEIGHT,
        enable_domain_randomization: bool = False,
        require_gate3_complete: bool = False,
        seed: int = 42,
    ) -> G1TrackingEnvCfg:
        """Create a fully bound task config without hardcoded archive paths."""

        paths = [_require_path_below_accad(path) for path in motion_files]
        if not paths:
            raise ValueError("at least one Gate-3 motion archive is required")
        if num_envs is not None and num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if fixed_motion_id is not None and fixed_motion_ids is not None:
            raise ValueError("fixed_motion_id and fixed_motion_ids are mutually exclusive")
        if isinstance(enable_domain_randomization, bool) is False:
            raise ValueError("enable_domain_randomization must be boolean")
        if eval_mode and enable_domain_randomization:
            raise ValueError("evaluation cannot enable domain randomization")
        reset_noise_scale, alive_weight, failure_weight = (
            resolve_recovery_training_objective(
                reset_state_noise_scale,
                alive_reward_weight,
                failure_reward_weight,
            )
        )
        cfg: G1TrackingEnvCfg = G1TrackingEvalEnvCfg() if eval_mode else G1TrackingEnvCfg()
        if not eval_mode and not enable_domain_randomization:
            cfg.events = G1TrackingEvalEventsCfg()
        cfg.scene.num_envs = (1 if eval_mode else 2048) if num_envs is None else int(num_envs)
        command = cfg.commands.motion
        command.motion_files = [str(path) for path in paths]
        command.future_horizons_s = tuple(float(value) for value in future_horizons_s)
        command.correspondence_path = (
            None if correspondence_path is None else str(Path(correspondence_path).expanduser().resolve())
        )
        command.require_gate3_complete = bool(require_gate3_complete)
        command.frame_zero_start_probability = float(frame_zero_start_probability)
        command.seed = int(seed)
        cfg.seed = int(seed)
        if not eval_mode:
            for attribute in (
                "joint_position_noise",
                "joint_velocity_noise",
                "root_linear_velocity_noise",
                "root_angular_velocity_noise",
            ):
                bounds = getattr(command, attribute)
                setattr(
                    command,
                    attribute,
                    tuple(float(value) * reset_noise_scale for value in bounds),
                )
            cfg.rewards.alive.weight = alive_weight
            cfg.rewards.failure.weight = failure_weight
        if fixed_motion_ids is not None:
            # EvalCfg carries the legacy scalar default.  Explicit suite
            # mapping replaces it instead of creating an ambiguous conflict.
            resolved_fixed_motion_ids = fixed_motion_ids_for_envs(
                num_envs=int(cfg.scene.num_envs),
                num_motions=len(paths),
                fixed_motion_ids=fixed_motion_ids,
            )
            assert resolved_fixed_motion_ids is not None
            command.fixed_motion_id = None
            command.fixed_motion_ids = resolved_fixed_motion_ids
        elif fixed_motion_id is not None:
            fixed_motion_ids_for_envs(
                num_envs=int(cfg.scene.num_envs),
                num_motions=len(paths),
                fixed_motion_id=fixed_motion_id,
            )
            command.fixed_motion_id = int(fixed_motion_id)
            command.fixed_motion_ids = None
        if random_start is not None:
            command.random_start = bool(random_start)
        expected_actor, expected_critic = tracking_observation_dims(
            num_horizons=len(command.future_horizons_s)
        )
        # These attributes are report/export metadata. ObservationManager still
        # independently infers and checks live term shapes.
        cfg.actor_observation_dim = expected_actor
        cfg.critic_observation_dim = expected_critic
        cfg.policy_action_dim = G1_ACTION_DIM
        return cfg


    try:
        from isaaclab_rl.rsl_rl import (
            RslRlOnPolicyRunnerCfg,
            RslRlPpoActorCriticCfg,
            RslRlPpoAlgorithmCfg,
        )

        _RSL_IMPORT_ERROR: BaseException | None = None
    except (ImportError, ModuleNotFoundError) as exc:
        _RSL_IMPORT_ERROR = exc
    else:

        @configclass
        class G1TrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
            """RSL-RL PPO baseline for 29-D reference-residual motion tracking."""

            num_steps_per_env = 32
            max_iterations = 5000
            save_interval = 100
            experiment_name = "accad_g1_motion_tracking"
            run_name = ""
            obs_groups = {"policy": ["policy"], "critic": ["critic"]}
            # RSL-RL 3.x uses the explicit actor/critic normalization flags
            # above.  Keeping this legacy switch at ``None`` avoids the
            # deprecated runner-level normalization path.
            empirical_normalization = None
            # The action term owns the physical +/-1 safety envelope.  Leaving
            # the wrapper open lets the reward observe and penalize raw policy
            # excess instead of silently hiding it before the environment.
            clip_actions = None
            policy = RslRlPpoActorCriticCfg(
                init_noise_std=DEFAULT_ACTION_NOISE_STD,
                noise_std_type="log",
                actor_obs_normalization=True,
                critic_obs_normalization=True,
                actor_hidden_dims=[512, 512, 256],
                critic_hidden_dims=[512, 512, 256],
                activation="elu",
            )
            algorithm = RslRlPpoAlgorithmCfg(
                value_loss_coef=1.0,
                use_clipped_value_loss=True,
                clip_param=0.2,
                entropy_coef=0.0,
                num_learning_epochs=5,
                num_mini_batches=4,
                # RSL-RL applies its adaptive update once per minibatch. With
                # 5 epochs x 4 minibatches the previous setting could hit the
                # 1e-5 floor during the first PPO update and remain there for
                # most of training. A fresh fixed-rate run keeps the executed
                # optimizer step equal to the recorded recovery contract.
                learning_rate=RECOVERY_PPO_LEARNING_RATE,
                schedule=RECOVERY_PPO_SCHEDULE,
                gamma=0.99,
                lam=0.95,
                desired_kl=RECOVERY_PPO_DESIRED_KL,
                max_grad_norm=1.0,
            )


    def make_tracking_runner_cfg(
        *, init_noise_std: float = DEFAULT_ACTION_NOISE_STD
    ) -> Any:
        """Return the RSL-RL PPO config, or a focused dependency error."""

        if _RSL_IMPORT_ERROR is not None:
            raise RuntimeError("isaaclab_rl with RSL-RL support is not importable") from _RSL_IMPORT_ERROR
        cfg = G1TrackingPPORunnerCfg()
        cfg.policy.init_noise_std = resolve_action_noise_std(
            init_noise_std, name="init_noise_std"
        )
        return cfg


if _ISAAC_IMPORT_ERROR is not None:

    def make_tracking_env_cfg(*args: Any, **kwargs: Any) -> Any:
        require_isaac_tracking()


    def make_tracking_runner_cfg(
        *, init_noise_std: float = DEFAULT_ACTION_NOISE_STD
    ) -> Any:
        require_isaac_tracking()
