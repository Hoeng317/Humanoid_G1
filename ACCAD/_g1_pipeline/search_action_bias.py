#!/usr/bin/env python3
"""Search constant deterministic actor-output biases in parallel Isaac envs.

One selected train/validation motion is assigned to every environment at frame
zero.  Environment ``i`` executes deterministic checkpoint inference plus row
``i`` of a seeded 29-D bias table.  Only each environment's first episode is
scored.  This is a diagnostic optimizer, not a replacement for the untouched
held-out suite evaluator.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys


PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
PROJECT_SOURCE = PROJECT_ROOT / "source"
for path in (PIPELINE_ROOT, PROJECT_SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from accad_g1.training_io import (  # noqa: E402
    ACCAD_ROOT,
    WORK_ROOT,
    add_motion_input_args,
    command_metadata,
    new_artifact_dir,
    nonnegative_int,
    path_below,
    positive_int,
    resolve_checkpoint,
    resolve_motion_inputs,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from accad_g1.physics_warmup import (  # noqa: E402
    DEFAULT_PHYSICS_WARMUP_STEPS,
    run_physics_cold_start_warmup,
)

from isaaclab.app import AppLauncher  # noqa: E402


def _positive_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _bounded_yaw_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not -math.pi <= parsed <= math.pi:
        raise argparse.ArgumentTypeError(
            f"yaw must be finite and within [-pi, pi], got {value!r}"
        )
    return parsed


PARSER = argparse.ArgumentParser(
    description=(
        "Search seeded constant 29-D actor-output biases for one train/validation "
        "motion using one deterministic frame-zero first episode per Isaac environment."
    )
)
add_motion_input_args(PARSER, default_split="train")
PARSER.add_argument("--checkpoint", default=None, help="Default: latest local model_*.pt.")
PARSER.add_argument(
    "--num-candidates",
    type=positive_int,
    default=512,
    help="Parallel candidates/environments; candidate 0 is always zero (default: 512).",
)
PARSER.add_argument(
    "--bias-mode",
    choices=("constant", "temporal_gaussian"),
    default="constant",
    help=(
        "constant patches the winning final actor bias; temporal_gaussian applies a "
        "fixed seeded per-step dither schedule and never creates a checkpoint"
    ),
)
PARSER.add_argument(
    "--bias-std",
    type=_positive_finite_float,
    default=0.005,
    help="Standard deviation of seeded Gaussian output-bias candidates (default: 0.005).",
)
PARSER.add_argument(
    "--antithetic",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Emit +/- Gaussian pairs after candidate 0 (default: enabled).",
)
PARSER.add_argument(
    "--max-steps",
    type=nonnegative_int,
    default=0,
    help="Control-step cap; 0 uses archive frame count plus one.",
)
PARSER.add_argument(
    "--target-yaw-rad",
    type=_bounded_yaw_float,
    default=0.0,
    help=(
        "Fixed absolute initial reference-root yaw shared by every candidate, "
        "within [-pi, pi] (default: 0)."
    ),
)
PARSER.add_argument("--seed", type=nonnegative_int, default=42)
PARSER.add_argument(
    "--physics-warmup-steps",
    type=nonnegative_int,
    default=DEFAULT_PHYSICS_WARMUP_STEPS,
    help=(
        "Policy-independent zero-residual control steps before the authoritative "
        "frame-0 candidate episodes (default: 100; pass 0 only for a cold-start "
        "diagnostic)."
    ),
)
PARSER.add_argument("--search-name", default="deterministic_action_bias_search")
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
if ARGS.split == "test":
    PARSER.error("action-bias search is forbidden on the test split")
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import importlib.metadata  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from accad_g1.action_bias_search import (  # noqa: E402
    bias_for_control_step,
    bias_statistics,
    checkpoint_patch_allowed,
    generate_action_bias_candidates,
    generate_temporal_action_bias_schedule,
    locate_actor_output_bias,
    select_best_candidate,
    temporal_bias_statistics,
    tensor_sha256,
    validate_bias_mode,
    validate_target_yaw,
    write_actor_bias_checkpoint,
)
from accad_g1.motion_library import motion_paths_from_manifest  # noqa: E402
from accad_g1.tracking_task import (  # noqa: E402
    clean_motion_end_success,
    make_tracking_env_cfg,
    make_tracking_runner_cfg,
    quat_error_angle_wxyz,
    quat_yaw_wxyz,
    reference_residual_target,
    resolved_evaluation_isolation,
    resolved_live_tracking_tensor_contract,
    tracking_observation_contract,
)
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _rmse(square_sum: float, count: int) -> float | None:
    return None if count == 0 else math.sqrt(square_sum / count)


def _candidate_records(
    *,
    bias_tensor: torch.Tensor,
    bias_mode: str,
    num_candidates: int,
    target_yaw_rad: float,
    frame_count: int,
    step_dt: float,
    episode_returns: torch.Tensor,
    episode_lengths: torch.Tensor,
    finished: torch.Tensor,
    success: list[bool],
    termination_flags: list[dict[str, bool]],
    finish_steps: list[int | None],
    joint_abs_sum: torch.Tensor,
    joint_sq_sum: torch.Tensor,
    joint_samples: torch.Tensor,
    root_distance_sum: torch.Tensor,
    root_distance_sq_sum: torch.Tensor,
    root_orientation_sum: torch.Tensor,
    root_orientation_sq_sum: torch.Tensor,
    root_samples: torch.Tensor,
    body_distance_sum: torch.Tensor,
    body_distance_sq_sum: torch.Tensor,
    body_samples: torch.Tensor,
    action_abs_sum: torch.Tensor,
    action_abs_max: torch.Tensor,
    action_clip_sum: torch.Tensor,
    action_target_clamp_sum: torch.Tensor,
    action_rate_limit_sum: torch.Tensor,
    action_samples: torch.Tensor,
    timeout_term_names: set[str],
    max_steps: int,
    app_stopped: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    mode = validate_bias_mode(bias_mode)
    for candidate_id in range(num_candidates):
        length = int(episode_lengths[candidate_id].item())
        complete = bool(finished[candidate_id].item())
        flags = termination_flags[candidate_id]
        fired = [name for name, value in flags.items() if value]
        if success[candidate_id]:
            outcome = "motion_end_success"
        elif complete:
            outcome = (
                "non_timeout_termination"
                if any(
                    value and name not in timeout_term_names
                    for name, value in flags.items()
                )
                else "ended_without_clean_motion_end"
            )
        elif app_stopped:
            outcome = "simulation_stopped_before_done"
        else:
            outcome = "max_steps_reached_before_done"

        joint_count = int(joint_samples[candidate_id].item())
        root_count = int(root_samples[candidate_id].item())
        body_count = int(body_samples[candidate_id].item())
        action_count = int(action_samples[candidate_id].item())
        if mode == "constant":
            vector = bias_tensor[candidate_id]
            bias_record = {
                "mode": mode,
                **bias_statistics(vector),
                "values": [float(value) for value in vector.tolist()],
            }
        else:
            trajectory = bias_tensor[:, candidate_id, :]
            # Per-candidate temporal values are intentionally omitted here to
            # keep a 512-candidate report compact.  The complete selected best
            # trajectory is written to its own checksummed JSON artifact.
            bias_record = {
                "mode": mode,
                **temporal_bias_statistics(trajectory),
                "values_recorded_here": False,
            }
        records.append(
            {
                "candidate_id": candidate_id,
                "target_yaw_rad": float(target_yaw_rad),
                "bias": bias_record,
                "first_episode": {
                    "finished": complete,
                    "success": bool(success[candidate_id]),
                    "outcome": outcome,
                    "length_steps": length,
                    "duration_s": length * step_dt,
                    "return": float(episode_returns[candidate_id].item()),
                    "completion_ratio_length_over_archive_frames": float(
                        length / frame_count
                    ),
                    "finish_global_step": finish_steps[candidate_id],
                    "termination_flags": flags,
                    "fired_termination_terms": fired,
                    "max_steps": max_steps,
                },
                "metrics": {
                    "mean_absolute_joint_error_rad": _safe_ratio(
                        float(joint_abs_sum[candidate_id].item()), joint_count
                    ),
                    "joint_position_rmse_rad": _rmse(
                        float(joint_sq_sum[candidate_id].item()), joint_count
                    ),
                    "mean_root_position_error_m": _safe_ratio(
                        float(root_distance_sum[candidate_id].item()), root_count
                    ),
                    "root_position_rmse_m": _rmse(
                        float(root_distance_sq_sum[candidate_id].item()), root_count
                    ),
                    "mean_root_orientation_error_rad": _safe_ratio(
                        float(root_orientation_sum[candidate_id].item()), root_count
                    ),
                    "root_orientation_rmse_rad": _rmse(
                        float(root_orientation_sq_sum[candidate_id].item()), root_count
                    ),
                    "mean_tracking_body_position_error_m": _safe_ratio(
                        float(body_distance_sum[candidate_id].item()), body_count
                    ),
                    "tracking_body_position_rmse_m": _rmse(
                        float(body_distance_sq_sum[candidate_id].item()), body_count
                    ),
                    "mean_absolute_raw_action": _safe_ratio(
                        float(action_abs_sum[candidate_id].item()), action_count
                    ),
                    "maximum_absolute_raw_action": None
                    if action_count == 0
                    else float(action_abs_max[candidate_id].item()),
                    "raw_action_clip_fraction": _safe_ratio(
                        int(action_clip_sum[candidate_id].item()), action_count
                    ),
                    "predicted_target_clamp_fraction": _safe_ratio(
                        int(action_target_clamp_sum[candidate_id].item()), action_count
                    ),
                    "normalized_action_rate_limit_fraction": _safe_ratio(
                        int(action_rate_limit_sum[candidate_id].item()), action_count
                    ),
                },
            }
        )
    return records


def main() -> None:
    if ARGS.split == "test":
        raise ValueError("action-bias search is forbidden on the test split")
    # The default train manifest contains 215 motions.  With no explicit input,
    # choose its deterministic lexicographic first motion; explicit multi-motion
    # selections fail below unless the caller deliberately supplies --max-motions 1.
    resolved_max_motions = ARGS.max_motions
    if not ARGS.motion and not ARGS.manifest and resolved_max_motions is None:
        resolved_max_motions = 1
    selection = resolve_motion_inputs(
        motions=ARGS.motion,
        manifests=ARGS.manifest,
        split=ARGS.split,
        reference_root=ARGS.reference_root,
        allow_missing=ARGS.allow_missing,
        max_motions=resolved_max_motions,
    )
    if selection.split == "test":
        raise ValueError("resolved test-split inputs are forbidden for action-bias search")
    if len(selection.paths) != 1:
        raise ValueError(
            "action-bias search requires exactly one motion; pass one --motion or "
            "use --max-motions 1 with a train/validation manifest"
        )
    # Direct --motion inputs must not offer a back door into the unopened test
    # set.  Prove membership and checksum against the promoted manifest for the
    # requested non-test split without reading g1_test.json.
    official_manifest = WORK_ROOT / "manifests" / f"g1_{ARGS.split}.json"
    official_selection = motion_paths_from_manifest(
        official_manifest,
        selection.reference_root,
        split=ARGS.split,
        require_all=True,
    )
    selected_path = selection.paths[0]
    expected_checksum = official_selection.expected_checksums.get(selected_path)
    if expected_checksum is None:
        raise ValueError(
            f"selected motion is not promoted in the official {ARGS.split} manifest: "
            f"{selected_path}"
        )
    if sha256_file(selected_path) != expected_checksum:
        raise ValueError("selected motion checksum differs from the official split manifest")
    correspondence = None
    if not ARGS.skip_correspondence_check:
        correspondence = path_below(ARGS.correspondence, ACCAD_ROOT, must_exist=True)
        if not correspondence.is_file():
            raise FileNotFoundError(correspondence)
    checkpoint = resolve_checkpoint(ARGS.checkpoint, latest_if_missing=True)
    assert checkpoint is not None

    bias_mode = validate_bias_mode(ARGS.bias_mode)
    target_yaw_rad = validate_target_yaw(ARGS.target_yaw_rad)
    with np.load(selected_path, allow_pickle=False) as archive:
        if "frame_count" not in archive or "joint_pos" not in archive:
            raise ValueError("selected G1 archive is missing frame_count or joint_pos")
        archive_frame_count = int(np.asarray(archive["frame_count"]).item())
        joint_frame_count = int(np.asarray(archive["joint_pos"]).shape[0])
    if archive_frame_count <= 0 or archive_frame_count != joint_frame_count:
        raise ValueError(
            "selected G1 archive has an inconsistent frame count: "
            f"metadata={archive_frame_count}, joint_pos={joint_frame_count}"
        )
    planned_max_steps = (
        int(ARGS.max_steps) if ARGS.max_steps > 0 else archive_frame_count + 1
    )
    if bias_mode == "constant":
        bias_tensor = generate_action_bias_candidates(
            num_candidates=ARGS.num_candidates,
            action_dim=29,
            std=ARGS.bias_std,
            seed=ARGS.seed,
            antithetic=ARGS.antithetic,
        )
    else:
        bias_tensor = generate_temporal_action_bias_schedule(
            num_steps=planned_max_steps,
            num_candidates=ARGS.num_candidates,
            action_dim=29,
            std=ARGS.bias_std,
            seed=ARGS.seed,
            antithetic=ARGS.antithetic,
        )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_payload, dict) or not isinstance(
        checkpoint_payload.get("model_state_dict"), dict
    ):
        raise ValueError("checkpoint must contain a model_state_dict mapping")
    actor_output_spec = locate_actor_output_bias(
        checkpoint_payload["model_state_dict"], action_dim=29
    )

    evaluation_dir = new_artifact_dir("evaluations", ARGS.search_name)
    params_dir = evaluation_dir / "params"
    params_dir.mkdir()
    initial_report = command_metadata(sys.argv)
    initial_report.update(
        {
            "schema_version": "accad_g1_deterministic_action_bias_search_v2",
            "status": "initializing",
            "diagnostic_only": True,
            "evaluation_dir": str(evaluation_dir),
            "search_contract": {
                "one_selected_motion": True,
                "one_candidate_per_environment": True,
                "candidate_zero_is_unmodified_policy": True,
                "deterministic_policy_inference": True,
                "bias_mode": bias_mode,
                "constant_actor_output_bias_per_candidate": bias_mode == "constant",
                "seeded_per_step_gaussian_dither": bias_mode == "temporal_gaussian",
                "temporal_schedule_is_fixed_before_simulation": (
                    bias_mode == "temporal_gaussian"
                ),
                "temporal_checkpoint_patch_forbidden": (
                    bias_mode == "temporal_gaussian"
                ),
                "first_episode_only": True,
                "fixed_motion_frame_zero_start": True,
                "fixed_target_yaw_rad": target_yaw_rad,
                "same_target_yaw_for_every_candidate": True,
                "state_and_observation_noise": False,
                "domain_randomization_events": False,
                "physx_enhanced_determinism": True,
                "physics_warmup_steps_requested": int(ARGS.physics_warmup_steps),
                "test_split_forbidden": True,
                "success_definition": (
                    "motion_end fired and no non-timeout termination term fired on the same step"
                ),
                "ranking_objective_lexicographic": [
                    "clean_motion_end_success (maximize)",
                    "completion_ratio (maximize)",
                    "episode_return (maximize)",
                    "root_position_rmse_m (minimize)",
                    "joint_position_rmse_rad (minimize)",
                    "tracking_body_position_rmse_m (minimize)",
                    "bias_l2_norm (minimize)",
                    "candidate_id (minimize)",
                ],
                "official_gate_validation": False,
            },
            "candidate_generation": {
                "bias_mode": bias_mode,
                "num_candidates": ARGS.num_candidates,
                "action_dim": 29,
                "num_schedule_steps": planned_max_steps
                if bias_mode == "temporal_gaussian"
                else None,
                "gaussian_std": ARGS.bias_std,
                "seed": ARGS.seed,
                "antithetic": ARGS.antithetic,
                "tensor_shape": list(bias_tensor.shape),
                "tensor_dtype": str(bias_tensor.dtype),
                "tensor_checksum_sha256": tensor_sha256(bias_tensor),
                "table_checksum_sha256": tensor_sha256(bias_tensor)
                if bias_mode == "constant"
                else None,
                "schedule_checksum_sha256": tensor_sha256(bias_tensor)
                if bias_mode == "temporal_gaussian"
                else None,
                "candidate_zero_identically_zero": bool(
                    torch.count_nonzero(
                        bias_tensor[:, 0, :]
                        if bias_mode == "temporal_gaussian"
                        else bias_tensor[0]
                    ).item()
                    == 0
                ),
                "full_sequence_antithetic_pairs": (
                    bias_mode == "temporal_gaussian" and ARGS.antithetic
                ),
            },
            "checkpoint": {
                "path": str(checkpoint),
                "checksum_sha256": sha256_file(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "actor_output_layer": actor_output_spec.metadata(),
            },
            "motion_input": selection.metadata(),
            "official_split_membership": {
                "split": ARGS.split,
                "manifest_path": str(official_selection.manifest_path),
                "manifest_checksum_sha256": official_selection.manifest_checksum_sha256,
                "selected_archive_checksum_sha256": expected_checksum,
                "verified": True,
                "test_manifest_read": False,
            },
            "correspondence": None
            if correspondence is None
            else {
                "path": str(correspondence),
                "checksum_sha256": sha256_file(correspondence),
            },
        }
    )
    write_json_atomic(evaluation_dir / "action_bias_search_summary.json", initial_report)

    env_cfg = make_tracking_env_cfg(
        selection.paths,
        num_envs=ARGS.num_candidates,
        eval_mode=True,
        correspondence_path=correspondence,
        fixed_motion_id=0,
        random_start=False,
        require_gate3_complete=False,
        seed=ARGS.seed,
    )
    env_cfg.sim.physx.enable_enhanced_determinism = True
    env_cfg.commands.motion.target_yaw_range = (
        target_yaw_rad,
        target_yaw_rad,
    )
    agent_cfg = make_tracking_runner_cfg()
    agent_cfg.seed = ARGS.seed
    if getattr(ARGS, "device", None) is not None:
        env_cfg.sim.device = ARGS.device
        agent_cfg.device = ARGS.device
    env_cfg.log_dir = str(evaluation_dir)
    dump_yaml(str(params_dir / "env.yaml"), env_cfg)
    dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)

    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ARGS.seed)

    base_env: ManagerBasedRLEnv | None = None
    wrapped: RslRlVecEnvWrapper | None = None
    wall_start = time.perf_counter()
    try:
        base_env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        physics_warmup = run_physics_cold_start_warmup(
            base_env,
            int(ARGS.physics_warmup_steps),
            expected_action_dim=env_cfg.policy_action_dim,
        )
        observation_contract = tracking_observation_contract()
        live_tensor_contract = resolved_live_tracking_tensor_contract(base_env)
        wrapped = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        print(f"[INFO] Loading checkpoint: {checkpoint}")
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=wrapped.device)
        policy_module = runner.alg.policy

        command = base_env.command_manager.get_term("motion")
        action_term = base_env.action_manager.get_term("joint_residual")
        assigned_motion_ids = [int(value) for value in command.motion_ids.detach().cpu().tolist()]
        expected_motion_ids = [0] * ARGS.num_candidates
        if assigned_motion_ids != expected_motion_ids:
            raise RuntimeError(
                "per-environment fixed motion mapping changed at runtime: "
                f"expected all zero, got {assigned_motion_ids}"
            )
        start_times = command.motion_time.detach().cpu()
        if not bool(torch.allclose(start_times, torch.zeros_like(start_times), atol=1.0e-8, rtol=0.0)):
            raise RuntimeError("candidate environments did not initialize at motion frame zero")
        resolved_yaw_range = tuple(float(value) for value in command.cfg.target_yaw_range)
        expected_yaw_range = (
            target_yaw_rad,
            target_yaw_rad,
        )
        if resolved_yaw_range != expected_yaw_range:
            raise RuntimeError(
                "fixed target yaw changed at runtime: "
                f"expected {expected_yaw_range}, got {resolved_yaw_range}"
            )
        commanded_initial_yaws = quat_yaw_wxyz(command.root_quat_wxyz)
        target_yaw_tensor = torch.full_like(
            commanded_initial_yaws, target_yaw_rad
        )
        wrapped_yaw_error = torch.atan2(
            torch.sin(commanded_initial_yaws - target_yaw_tensor),
            torch.cos(commanded_initial_yaws - target_yaw_tensor),
        )
        max_initial_yaw_error = float(wrapped_yaw_error.abs().max().item())
        if max_initial_yaw_error > 1.0e-5:
            raise RuntimeError(
                "live initial commanded root yaw differs from the requested fixed yaw: "
                f"maximum wrapped error={max_initial_yaw_error}"
            )
        if env_cfg.policy_action_dim != bias_tensor.shape[-1]:
            raise RuntimeError(
                f"action dimension mismatch: environment={env_cfg.policy_action_dim}, "
                f"bias_tensor={bias_tensor.shape[-1]}"
            )
        runtime_motion = {
            "motion_id": 0,
            "motion_key": command.library.motion_keys[0],
            "path": str(command.library.paths[0]),
            "checksum_sha256": command.library.checksums_sha256[0],
            "frame_count": int(command.library.lengths[0].item()),
            "duration_s": float(command.library.durations[0].item()),
        }
        if runtime_motion["frame_count"] != archive_frame_count:
            raise RuntimeError(
                "preflight and live motion-library frame counts differ: "
                f"preflight={archive_frame_count}, live={runtime_motion['frame_count']}"
            )
        runtime_contract = {
            "motion": runtime_motion,
            "library_checksum_sha256": command.library.library_checksum_sha256,
            "library_contract_checksum_sha256": command.library.contract.digest(),
            "per_environment_motion_ids_all_zero": True,
            "initial_motion_times_s_min": float(start_times.min().item()),
            "initial_motion_times_s_max": float(start_times.max().item()),
            "requested_fixed_target_yaw_rad": target_yaw_rad,
            "resolved_target_yaw_range_rad": list(resolved_yaw_range),
            "initial_commanded_root_yaw_rad_min": float(
                commanded_initial_yaws.min().item()
            ),
            "initial_commanded_root_yaw_rad_max": float(
                commanded_initial_yaws.max().item()
            ),
            "maximum_initial_commanded_root_yaw_error_rad": max_initial_yaw_error,
            "actor_observation_dim": env_cfg.actor_observation_dim,
            "critic_observation_dim": env_cfg.critic_observation_dim,
            "action_dim": env_cfg.policy_action_dim,
            "bias_mode": bias_mode,
            "bias_tensor_shape": list(bias_tensor.shape),
            "bias_tensor_checksum_sha256": tensor_sha256(bias_tensor),
            "observation_contract": observation_contract,
            "live_tensor_contract": live_tensor_contract,
            "physics_dt_s": env_cfg.sim.dt,
            "control_dt_s": env_cfg.sim.dt * env_cfg.decimation,
            "policy_action_clip": agent_cfg.clip_actions,
            "physical_action_clip": action_term.cfg.action_clip,
            "evaluation_isolation": resolved_evaluation_isolation(base_env),
            "physics_cold_start_warmup": physics_warmup,
            "device": str(wrapped.device),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "isaaclab": _package_version("isaaclab"),
                "isaaclab_rl": _package_version("isaaclab-rl"),
                "rsl_rl": _package_version("rsl-rl"),
            },
        }
        write_json_atomic(evaluation_dir / "runtime_contract.json", runtime_contract)

        num_envs = base_env.num_envs
        device = wrapped.device
        runtime_bias_tensor = bias_tensor.to(device=device)
        observations = wrapped.get_observations()
        active = torch.ones(num_envs, device=device, dtype=torch.bool)
        finished = torch.zeros(num_envs, device=device, dtype=torch.bool)
        episode_returns = torch.zeros(num_envs, device=device, dtype=torch.float64)
        episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
        success = [False] * num_envs
        termination_flags: list[dict[str, bool]] = [dict() for _ in range(num_envs)]
        finish_steps: list[int | None] = [None] * num_envs

        float_accumulators = {
            name: torch.zeros(num_envs, device=device, dtype=torch.float64)
            for name in (
                "joint_abs_sum",
                "joint_sq_sum",
                "root_distance_sum",
                "root_distance_sq_sum",
                "root_orientation_sum",
                "root_orientation_sq_sum",
                "body_distance_sum",
                "body_distance_sq_sum",
                "action_abs_sum",
                "action_abs_max",
            )
        }
        count_accumulators = {
            name: torch.zeros(num_envs, device=device, dtype=torch.long)
            for name in (
                "joint_samples",
                "root_samples",
                "body_samples",
                "action_clip_sum",
                "action_target_clamp_sum",
                "action_rate_limit_sum",
                "action_samples",
            )
        }
        term_names = tuple(base_env.termination_manager.active_terms)
        timeout_term_names = {
            name
            for name in term_names
            if bool(base_env.termination_manager.get_term_cfg(name).time_out)
        }
        if "motion_end" not in timeout_term_names:
            raise RuntimeError("motion_end must be an explicit timeout-style termination term")
        max_steps = planned_max_steps
        raw_clip = float(action_term.cfg.action_clip)
        steps_executed = 0
        print(
            f"[INFO] Bias search: mode={bias_mode}, {num_envs} candidates, "
            f"motion={runtime_motion['motion_key']}, std={ARGS.bias_std}, max_steps={max_steps}"
        )

        while (
            SIMULATION_APP.is_running()
            and steps_executed < max_steps
            and bool(active.any())
        ):
            active_before = active.clone()
            active_float = active_before.to(torch.float64)
            robot = command.robot

            actual_q = robot.data.joint_pos[:, command.policy_joint_ids]
            joint_error = actual_q - command.joint_pos
            float_accumulators["joint_abs_sum"] += (
                joint_error.abs().sum(dim=-1).to(torch.float64) * active_float
            )
            float_accumulators["joint_sq_sum"] += (
                joint_error.square().sum(dim=-1).to(torch.float64) * active_float
            )
            count_accumulators["joint_samples"] += (
                active_before.to(torch.long) * joint_error.shape[-1]
            )

            root_distance = torch.linalg.vector_norm(
                robot.data.root_link_pos_w - command.root_pos_w, dim=-1
            )
            root_orientation = quat_error_angle_wxyz(
                command.root_quat_wxyz, robot.data.root_link_quat_w
            )
            float_accumulators["root_distance_sum"] += root_distance.to(torch.float64) * active_float
            float_accumulators["root_distance_sq_sum"] += root_distance.square().to(
                torch.float64
            ) * active_float
            float_accumulators["root_orientation_sum"] += root_orientation.to(
                torch.float64
            ) * active_float
            float_accumulators["root_orientation_sq_sum"] += root_orientation.square().to(
                torch.float64
            ) * active_float
            count_accumulators["root_samples"] += active_before.to(torch.long)

            actual_body = robot.data.body_pos_w[:, command.tracking_body_ids]
            body_distance = torch.linalg.vector_norm(
                actual_body - command.tracking_body_pos_w, dim=-1
            )
            float_accumulators["body_distance_sum"] += (
                body_distance.sum(dim=-1).to(torch.float64) * active_float
            )
            float_accumulators["body_distance_sq_sum"] += (
                body_distance.square().sum(dim=-1).to(torch.float64) * active_float
            )
            count_accumulators["body_samples"] += (
                active_before.to(torch.long) * body_distance.shape[-1]
            )

            with torch.inference_mode():
                base_actions = policy(observations)
                if base_actions.shape != (num_envs, env_cfg.policy_action_dim):
                    raise RuntimeError(
                        "policy action shape changed: expected "
                        f"{(num_envs, env_cfg.policy_action_dim)}, got {tuple(base_actions.shape)}"
                    )
                step_biases = bias_for_control_step(
                    runtime_bias_tensor,
                    bias_mode=bias_mode,
                    step=steps_executed,
                    num_candidates=num_envs,
                    action_dim=env_cfg.policy_action_dim,
                )
                actions = base_actions + step_biases
                if not bool(torch.isfinite(actions[active_before]).all()):
                    raise RuntimeError("biased policy produced NaN or Inf for an active candidate")
                actions = torch.where(
                    active_before.unsqueeze(-1), actions, torch.zeros_like(actions)
                )
                absolute_actions = actions.abs()
                float_accumulators["action_abs_sum"] += (
                    absolute_actions.sum(dim=-1).to(torch.float64) * active_float
                )
                float_accumulators["action_abs_max"] = torch.where(
                    active_before,
                    torch.maximum(
                        float_accumulators["action_abs_max"],
                        absolute_actions.max(dim=-1).values.to(torch.float64),
                    ),
                    float_accumulators["action_abs_max"],
                )
                active_actions = active_before.unsqueeze(-1).expand_as(actions)
                count_accumulators["action_clip_sum"] += (
                    (absolute_actions > raw_clip) & active_actions
                ).sum(dim=-1).to(torch.long)
                effective_actions = actions.clamp(-raw_clip, raw_clip)
                hard = robot.data.joint_pos_limits[:, action_term.joint_ids]
                soft = robot.data.soft_joint_pos_limits[:, action_term.joint_ids]
                predicted_target = reference_residual_target(
                    effective_actions,
                    command.joint_pos,
                    action_term.normalized_residual,
                    residual_scale=action_term.cfg.residual_scale,
                    action_clip=action_term.cfg.action_clip,
                    normalized_delta_limit=action_term.cfg.normalized_delta_limit,
                    hard_limits=hard,
                    soft_limits=soft,
                    target_limit_mode=action_term.cfg.target_limit_mode,
                )
                count_accumulators["action_target_clamp_sum"] += (
                    predicted_target.target_clamped & active_actions
                ).sum(dim=-1).to(torch.long)
                count_accumulators["action_rate_limit_sum"] += (
                    predicted_target.rate_limited & active_actions
                ).sum(dim=-1).to(torch.long)
                count_accumulators["action_samples"] += (
                    active_before.to(torch.long) * actions.shape[-1]
                )

                observations, rewards, dones, _ = wrapped.step(actions)
                policy_module.reset(dones)

            steps_executed += 1
            episode_returns += torch.where(
                active_before, rewards.to(torch.float64), torch.zeros_like(episode_returns)
            )
            episode_lengths += active_before.to(torch.long)
            done_now = (dones > 0) & active_before
            if bool(done_now.any()):
                term_values = {
                    name: base_env.termination_manager.get_term(name).clone()
                    for name in term_names
                }
                for candidate_id in torch.where(done_now)[0].detach().cpu().tolist():
                    flags = {
                        name: bool(term_values[name][candidate_id].item())
                        for name in term_names
                    }
                    termination_flags[candidate_id] = flags
                    success[candidate_id] = clean_motion_end_success(
                        flags, timeout_term_names=timeout_term_names
                    )
                    finish_steps[candidate_id] = steps_executed
                finished |= done_now
                active &= ~done_now

        app_stopped = not SIMULATION_APP.is_running()
        wall_seconds = time.perf_counter() - wall_start
        records = _candidate_records(
            bias_tensor=bias_tensor,
            bias_mode=bias_mode,
            num_candidates=num_envs,
            target_yaw_rad=target_yaw_rad,
            frame_count=runtime_motion["frame_count"],
            step_dt=base_env.step_dt,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            finished=finished,
            success=success,
            termination_flags=termination_flags,
            finish_steps=finish_steps,
            joint_abs_sum=float_accumulators["joint_abs_sum"],
            joint_sq_sum=float_accumulators["joint_sq_sum"],
            joint_samples=count_accumulators["joint_samples"],
            root_distance_sum=float_accumulators["root_distance_sum"],
            root_distance_sq_sum=float_accumulators["root_distance_sq_sum"],
            root_orientation_sum=float_accumulators["root_orientation_sum"],
            root_orientation_sq_sum=float_accumulators["root_orientation_sq_sum"],
            root_samples=count_accumulators["root_samples"],
            body_distance_sum=float_accumulators["body_distance_sum"],
            body_distance_sq_sum=float_accumulators["body_distance_sq_sum"],
            body_samples=count_accumulators["body_samples"],
            action_abs_sum=float_accumulators["action_abs_sum"],
            action_abs_max=float_accumulators["action_abs_max"],
            action_clip_sum=count_accumulators["action_clip_sum"],
            action_target_clamp_sum=count_accumulators["action_target_clamp_sum"],
            action_rate_limit_sum=count_accumulators["action_rate_limit_sum"],
            action_samples=count_accumulators["action_samples"],
            timeout_term_names=timeout_term_names,
            max_steps=max_steps,
            app_stopped=app_stopped,
        )
        best = select_best_candidate(records)
        successful_ids = [
            record["candidate_id"]
            for record in records
            if record["first_episode"]["success"]
        ]
        checkpoint_artifact: dict[str, Any] | None = None
        best_id = int(best["candidate_id"])
        patch_allowed = checkpoint_patch_allowed(bias_mode)
        if successful_ids and patch_allowed:
            checkpoint_artifact = write_actor_bias_checkpoint(
                source=checkpoint,
                destination=evaluation_dir / "best_biased_model.pt",
                action_bias=bias_tensor[best_id],
                action_dim=env_cfg.policy_action_dim,
                allowed_output_root=evaluation_dir,
            )
            checkpoint_artifact.update(
                {
                    "selected_candidate_id": best_id,
                    "selection_requires_clean_motion_end": True,
                    "official_validation_required_before_promotion": True,
                }
            )
            write_json_atomic(
                evaluation_dir / "best_biased_checkpoint_provenance.json",
                checkpoint_artifact,
            )
        temporal_trajectory_artifact: dict[str, Any] | None = None
        if bias_mode == "temporal_gaussian":
            best_trajectory = bias_tensor[:, best_id, :]
            trajectory_payload = {
                "schema_version": "accad_g1_temporal_action_bias_trajectory_v1",
                "created_at_utc": utc_now(),
                "diagnostic_only": True,
                "checkpoint_patch_allowed": False,
                "checkpoint_patch_reason": (
                    "a time-varying per-step dither is not equivalent to a constant "
                    "final actor-layer bias"
                ),
                "source_checkpoint": initial_report["checkpoint"],
                "motion": runtime_motion,
                "candidate_generation": initial_report["candidate_generation"],
                "selected_candidate_id": best_id,
                "selection_record": best,
                "trajectory": {
                    **temporal_bias_statistics(best_trajectory),
                    "shape": list(best_trajectory.shape),
                    "values": [
                        [float(value) for value in step_values]
                        for step_values in best_trajectory.tolist()
                    ],
                },
            }
            trajectory_path = write_json_atomic(
                evaluation_dir / "best_temporal_bias_trajectory.json",
                trajectory_payload,
            )
            temporal_trajectory_artifact = {
                "path": str(trajectory_path),
                "checksum_sha256": sha256_file(trajectory_path),
                "bytes": trajectory_path.stat().st_size,
                "selected_candidate_id": best_id,
                "trajectory_checksum_sha256": tensor_sha256(best_trajectory),
                "checkpoint_created": False,
            }

        aggregate = {
            "candidate_count": num_envs,
            "fixed_target_yaw_rad": target_yaw_rad,
            "finished_first_episodes": int(finished.sum().item()),
            "successful_motion_end_candidates": len(successful_ids),
            "success_rate": float(len(successful_ids) / num_envs),
            "successful_candidate_ids": successful_ids,
            "best_candidate_id": int(best["candidate_id"]),
            "baseline_candidate_zero_success": bool(records[0]["first_episode"]["success"]),
            "best_is_unmodified_baseline": int(best["candidate_id"]) == 0,
            "total_first_episode_environment_steps": int(episode_lengths.sum().item()),
            "global_control_steps_executed": steps_executed,
            "max_steps": max_steps,
            "wall_time_s": wall_seconds,
            "simulation_app_stopped_early": app_stopped,
        }
        wrapped.close()
        wrapped = None
        base_env = None
        report = dict(initial_report)
        report.update(
            {
                "status": "completed",
                "finished_at_utc": utc_now(),
                "clean_success_found": bool(successful_ids),
                "runtime_contract": runtime_contract,
                "aggregate": aggregate,
                "best_candidate": best,
                "biased_checkpoint": checkpoint_artifact,
                "checkpoint_patch_contract": {
                    "allowed_for_bias_mode": patch_allowed,
                    "attempted": checkpoint_artifact is not None,
                    "checkpoint_created": checkpoint_artifact is not None,
                    "reason": (
                        "constant output bias is exactly representable by the actor final bias"
                        if patch_allowed
                        else "temporal Gaussian dither is not a checkpoint-constant actor bias"
                    ),
                },
                "best_temporal_bias_trajectory": temporal_trajectory_artifact,
                "candidates": records,
            }
        )
        report_path = write_json_atomic(
            evaluation_dir / "action_bias_search_summary.json", report
        )
        print(
            f"[INFO] Bias search complete: {len(successful_ids)}/{num_envs} clean successes; "
            f"best candidate={best['candidate_id']}"
        )
        print(f"[INFO] Search report: {report_path}")
        if checkpoint_artifact is not None:
            print(f"[INFO] Biased checkpoint: {checkpoint_artifact['output']['path']}")
        if temporal_trajectory_artifact is not None:
            print(
                "[INFO] Best temporal trajectory: "
                f"{temporal_trajectory_artifact['path']} (no checkpoint created)"
            )
    except BaseException as exc:
        failed = dict(initial_report)
        failed.update(
            {
                "status": "failed",
                "clean_success_found": False,
                "finished_at_utc": utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json_atomic(evaluation_dir / "action_bias_search_summary.json", failed)
        raise
    finally:
        if wrapped is not None:
            wrapped.close()
        elif base_env is not None:
            base_env.close()


try:
    main()
finally:
    SIMULATION_APP.close()
