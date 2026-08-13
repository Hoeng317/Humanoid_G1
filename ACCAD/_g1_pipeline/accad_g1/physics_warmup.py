"""Policy-independent Isaac physics/contact cold-start warm-up contract.

The live helper intentionally accepts an already-created, *unwrapped*
``ManagerBasedRLEnv`` without importing Isaac Lab.  This keeps the validation
logic available to ordinary CPU tests while the executable entry points remain
responsible for starting Kit before constructing the environment.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch


PHYSICS_WARMUP_SCHEMA_VERSION = "accad_g1_physics_cold_start_warmup_v1"
DEFAULT_PHYSICS_WARMUP_STEPS = 100


def validate_physics_warmup_steps(value: int) -> int:
    """Validate a requested number of policy-independent control steps."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("physics warm-up steps must be a non-negative integer")
    return value


def observation_finiteness_witness(observations: Any) -> dict[str, Any]:
    """Return a compact JSON-safe witness for a nested tensor observation."""

    leaves: list[tuple[str, torch.Tensor]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, torch.Tensor):
            leaves.append((path, value))
        elif isinstance(value, Mapping):
            for key in sorted(value, key=str):
                visit(value[key], f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        else:
            raise TypeError(
                f"observation leaf {path} must be a torch.Tensor; got {type(value).__name__}"
            )

    visit(observations, "observation")
    if not leaves:
        raise ValueError("observation tree contains no tensors")
    nonfinite_count = sum(
        int(torch.count_nonzero(~torch.isfinite(tensor)).item())
        for _, tensor in leaves
    )
    witness = {
        "tensor_count": len(leaves),
        "element_count": sum(int(tensor.numel()) for _, tensor in leaves),
        "nonfinite_count": nonfinite_count,
        "all_finite": nonfinite_count == 0,
        "tensors": [
            {
                "path": path,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            for path, tensor in leaves
        ],
    }
    if nonfinite_count:
        raise RuntimeError(
            f"physics warm-up observation contains {nonfinite_count} NaN/Inf values"
        )
    return witness


def post_reset_state_witness(
    *,
    observations: Any,
    episode_length_buf: torch.Tensor,
    motion_time: torch.Tensor,
    raw_action: torch.Tensor,
    normalized_residual: torch.Tensor,
    previous_normalized_residual: torch.Tensor,
    num_envs: int,
    action_dim: int,
    reset_buffers: Mapping[str, torch.Tensor] | None = None,
    state_errors: Mapping[str, float] | None = None,
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Validate and report the authoritative state after the final reset."""

    if num_envs <= 0 or action_dim <= 0:
        raise ValueError("num_envs and action_dim must be positive")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    expected_env_shape = (num_envs,)
    expected_action_shape = (num_envs, action_dim)
    tensors = {
        "episode_length_buf": (episode_length_buf, expected_env_shape),
        "motion_time": (motion_time, expected_env_shape),
        "raw_action": (raw_action, expected_action_shape),
        "normalized_residual": (normalized_residual, expected_action_shape),
        "previous_normalized_residual": (
            previous_normalized_residual,
            expected_action_shape,
        ),
    }
    maxima: dict[str, float] = {}
    for name, (tensor, expected_shape) in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
            shape = None if not isinstance(tensor, torch.Tensor) else tuple(tensor.shape)
            raise RuntimeError(f"post-reset {name} shape {shape} != {expected_shape}")
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"post-reset {name} contains NaN or Inf")
        maxima[name] = (
            0.0 if tensor.numel() == 0 else float(torch.abs(tensor).max().item())
        )

    reset_checks: dict[str, dict[str, Any]] = {}
    for name, tensor in (reset_buffers or {}).items():
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_env_shape:
            raise RuntimeError(f"post-reset {name} must have shape {expected_env_shape}")
        nonzero = int(torch.count_nonzero(tensor).item())
        reset_checks[name] = {"nonzero_count": nonzero, "all_clear": nonzero == 0}

    resolved_state_errors: dict[str, float] = {}
    for name, value in (state_errors or {}).items():
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise RuntimeError(f"post-reset state error {name} must be finite and non-negative")
        resolved_state_errors[name] = numeric

    zero_names = tuple(tensors)
    zero_checks = {name: maxima[name] <= tolerance for name in zero_names}
    reset_clear = all(check["all_clear"] for check in reset_checks.values())
    state_close = all(value <= tolerance for value in resolved_state_errors.values())
    passed = all(zero_checks.values()) and reset_clear and state_close
    witness = {
        "authoritative_motion_frame_zero": zero_checks["motion_time"],
        "episode_counters_zero": zero_checks["episode_length_buf"],
        "action_state_zero": all(
            zero_checks[name]
            for name in (
                "raw_action",
                "normalized_residual",
                "previous_normalized_residual",
            )
        ),
        "maximum_absolute_values": maxima,
        "reset_buffers": reset_checks,
        "reference_state_maximum_errors": resolved_state_errors,
        "observations": observation_finiteness_witness(observations),
        "tolerance": float(tolerance),
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"authoritative post-warm-up reset validation failed: {witness}")
    return witness


def _clear_stale_done_buffers(env: Any) -> dict[str, torch.Tensor]:
    buffers: dict[str, torch.Tensor] = {}
    for name in ("reset_buf", "reset_terminated", "reset_time_outs"):
        value = getattr(env, name, None)
        if isinstance(value, torch.Tensor):
            value.zero_()
            buffers[name] = value
    return buffers


def _reference_state_errors(command: Any) -> dict[str, float]:
    robot = command.robot
    actual_joint = robot.data.joint_pos[:, command.policy_joint_ids]
    joint_error = torch.abs(actual_joint - command.joint_pos)
    root_position_error = torch.linalg.vector_norm(
        robot.data.root_link_pos_w - command.root_pos_w, dim=-1
    )
    actual_quaternion = torch.nn.functional.normalize(
        robot.data.root_link_quat_w, dim=-1
    )
    reference_quaternion = torch.nn.functional.normalize(
        command.root_quat_wxyz, dim=-1
    )
    orientation_error = torch.minimum(
        torch.linalg.vector_norm(actual_quaternion - reference_quaternion, dim=-1),
        torch.linalg.vector_norm(actual_quaternion + reference_quaternion, dim=-1),
    )
    return {
        "joint_position_rad": float(joint_error.max().item()),
        "root_position_m": float(root_position_error.max().item()),
        "root_orientation_quaternion_l2": float(orientation_error.max().item()),
    }


def run_physics_cold_start_warmup(
    env: Any,
    steps: int,
    *,
    expected_action_dim: int = 29,
    command_name: str = "motion",
    action_name: str = "joint_residual",
    post_reset_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    """Prime physics/contact caches using zero residuals, then reset cleanly.

    ``steps == 0`` is a true no-op so diagnostic A/B runs retain the historical
    cold-start behavior.  Positive values perform no policy inference and must
    be called before installing video or RSL-RL wrappers.
    """

    steps = validate_physics_warmup_steps(steps)
    if steps == 0:
        return {
            "schema_version": PHYSICS_WARMUP_SCHEMA_VERSION,
            "enabled": False,
            "requested_control_steps": 0,
            "executed_control_steps": 0,
            "policy_inference_used": False,
            "post_reset": None,
            "passed": True,
        }
    if getattr(env, "unwrapped", env) is not env:
        raise ValueError("physics warm-up requires an unwrapped ManagerBasedRLEnv")
    num_envs = int(env.num_envs)
    action_term = env.action_manager.get_term(action_name)
    command = env.command_manager.get_term(command_name)
    live_action_dim = int(action_term.action_dim)
    manager_action_dim = int(sum(env.action_manager.action_term_dim))
    if live_action_dim != expected_action_dim or manager_action_dim != expected_action_dim:
        raise RuntimeError(
            "physics warm-up action dimension mismatch: "
            f"expected={expected_action_dim}, term={live_action_dim}, manager={manager_action_dim}"
        )
    zero_action = torch.zeros(
        (num_envs, expected_action_dim), device=env.device, dtype=torch.float32
    )
    initial_observations, _ = env.reset()
    initial_observation_witness = observation_finiteness_witness(initial_observations)

    termination_names = tuple(env.termination_manager.active_terms)
    termination_counts = {name: 0 for name in termination_names}
    done_signals = 0
    terminated_signals = 0
    timeout_signals = 0
    steps_with_done = 0
    for _ in range(steps):
        observations, _, terminated, time_outs, _ = env.step(zero_action)
        observation_finiteness_witness(observations)
        done = torch.logical_or(terminated, time_outs)
        current_done = int(torch.count_nonzero(done).item())
        done_signals += current_done
        terminated_signals += int(torch.count_nonzero(terminated).item())
        timeout_signals += int(torch.count_nonzero(time_outs).item())
        steps_with_done += int(current_done > 0)
        for name in termination_names:
            values = env.termination_manager.get_term(name)
            termination_counts[name] += int(torch.count_nonzero(values).item())

    # Force only start-time and state-noise fields for the authoritative reset.
    # Spatial alignment and fixed motion assignment remain exactly as configured.
    cfg = command.cfg
    override_names = (
        "random_start",
        "fixed_start_time_s",
        "joint_position_noise",
        "joint_velocity_noise",
        "root_linear_velocity_noise",
        "root_angular_velocity_noise",
    )
    original = {name: getattr(cfg, name) for name in override_names}
    try:
        cfg.random_start = False
        cfg.fixed_start_time_s = 0.0
        cfg.joint_position_noise = (0.0, 0.0)
        cfg.joint_velocity_noise = (0.0, 0.0)
        cfg.root_linear_velocity_noise = (0.0, 0.0)
        cfg.root_angular_velocity_noise = (0.0, 0.0)
        final_observations, _ = env.reset()
    finally:
        for name, value in original.items():
            setattr(cfg, name, value)

    reset_buffers = _clear_stale_done_buffers(env)
    post_reset = post_reset_state_witness(
        observations=final_observations,
        episode_length_buf=env.episode_length_buf,
        motion_time=command.motion_time,
        raw_action=action_term.raw_actions,
        normalized_residual=action_term.normalized_residual,
        previous_normalized_residual=action_term.previous_normalized_residual,
        num_envs=num_envs,
        action_dim=expected_action_dim,
        reset_buffers=reset_buffers,
        state_errors=_reference_state_errors(command),
        tolerance=post_reset_tolerance,
    )
    return {
        "schema_version": PHYSICS_WARMUP_SCHEMA_VERSION,
        "enabled": True,
        "requested_control_steps": steps,
        "executed_control_steps": steps,
        "num_environments": num_envs,
        "action_dimension": expected_action_dim,
        "zero_action_shape": list(zero_action.shape),
        "policy_inference_used": False,
        "purpose": "prime_physics_and_contact_caches_only",
        "initial_reset_observations": initial_observation_witness,
        "warmup_done_signals": done_signals,
        "warmup_terminated_signals": terminated_signals,
        "warmup_timeout_signals": timeout_signals,
        "warmup_steps_with_any_done": steps_with_done,
        "warmup_termination_term_counts": termination_counts,
        "final_reset_forced_motion_time_s": 0.0,
        "final_reset_zeroed_reference_state_noise": True,
        "post_reset": post_reset,
        "passed": bool(post_reset["passed"]),
    }
