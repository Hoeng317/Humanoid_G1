#!/usr/bin/env python3
"""Evaluate, replay, record, and export an ACCAD G1 tracking policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import MethodType


PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
PROJECT_SOURCE = PROJECT_ROOT / "source"
for path in (PIPELINE_ROOT, PROJECT_SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from accad_g1.training_io import (  # noqa: E402
    ACCAD_ROOT,
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
    validate_policy_exports,
    write_json_atomic,
)
from accad_g1.physics_warmup import (  # noqa: E402
    DEFAULT_PHYSICS_WARMUP_STEPS,
    run_physics_cold_start_warmup,
)

from isaaclab.app import AppLauncher  # noqa: E402


PARSER = argparse.ArgumentParser(
    description="Play/evaluate an RSL-RL G1 reference-residual tracking checkpoint."
)
add_motion_input_args(PARSER, default_split="train")
PARSER.add_argument("--checkpoint", default=None, help="Default: latest local model_*.pt.")
PARSER.add_argument("--num-envs", type=positive_int, default=1)
PARSER.add_argument("--motion-id", type=nonnegative_int, default=0)
PARSER.add_argument("--steps", type=nonnegative_int, default=500, help="0 disables the step limit.")
PARSER.add_argument(
    "--episodes", type=nonnegative_int, default=0, help="0 disables the completed-episode limit."
)
PARSER.add_argument("--seed", type=nonnegative_int, default=42)
PARSER.add_argument(
    "--physics-warmup-steps",
    type=nonnegative_int,
    default=DEFAULT_PHYSICS_WARMUP_STEPS,
    help=(
        "Policy-independent zero-residual control steps before a clean frame-0 "
        "reset; warm-up is completed before video capture starts (default: 100; "
        "pass 0 only for a cold-start diagnostic)."
    ),
)
PARSER.add_argument("--real-time", action="store_true")
PARSER.add_argument("--video", action="store_true")
PARSER.add_argument("--video-length", type=positive_int, default=500)
PARSER.add_argument(
    "--export",
    choices=("none", "jit", "onnx", "both"),
    default="both",
    help="Export into this evaluation directory.",
)
PARSER.add_argument("--evaluation-name", default="tracking_eval")
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
if ARGS.video:
    ARGS.enable_cameras = True
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import time  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from accad_g1.tracking_task import (  # noqa: E402
    make_tracking_env_cfg,
    make_tracking_runner_cfg,
    quat_error_angle_wxyz,
    reference_residual_target,
    resolved_evaluation_isolation,
    resolved_live_tracking_tensor_contract,
    tracking_observation_contract,
)
from accad_g1.video_validation import inspect_video  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def _export_policy(runner: OnPolicyRunner, export_dir: Path, mode: str) -> list[Path]:
    if mode == "none":
        return []
    policy = runner.alg.policy
    normalizer = getattr(policy, "actor_obs_normalizer", None)
    export_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if mode in {"jit", "both"}:
        export_policy_as_jit(
            policy,
            normalizer=normalizer,
            path=str(export_dir),
            filename="policy.pt",
        )
        outputs.append(export_dir / "policy.pt")
    if mode in {"onnx", "both"}:
        export_policy_as_onnx(
            policy,
            normalizer=normalizer,
            path=str(export_dir),
            filename="policy.onnx",
        )
        outputs.append(export_dir / "policy.onnx")
    return outputs


class _TerminalAwareRecordVideo(gym.wrappers.RecordVideo):
    """Use a post-physics/pre-reset frame when Isaac auto-resets a done env."""

    def __init__(self, *args, terminal_frames: list[np.ndarray], capture_stats: dict, **kwargs):
        self._terminal_frames = terminal_frames
        self._capture_stats = capture_stats
        super().__init__(*args, **kwargs)

    def _capture_frame(self) -> None:
        if not self._terminal_frames:
            super()._capture_frame()
            return
        frame = self._terminal_frames.pop(0)
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise RuntimeError("pre-reset terminal renderer returned an invalid frame")
        self.recorded_frames.append(frame)
        self._capture_stats["terminal_frames_substituted"] += 1


def _install_terminal_frame_capture(
    env: ManagerBasedRLEnv,
) -> tuple[list[np.ndarray], dict[str, int | bool]]:
    """Capture the terminal renderer output immediately before internal reset."""

    terminal_frames: list[np.ndarray] = []
    stats: dict[str, int | bool] = {
        "enabled": False,
        "terminal_frames_captured": 0,
        "terminal_frames_substituted": 0,
    }
    original_reset_idx = env._reset_idx

    def reset_idx_with_capture(self, env_ids):
        if bool(stats["enabled"]):
            frame = self.render()
            if isinstance(frame, list):
                if not frame:
                    raise RuntimeError("pre-reset renderer returned an empty frame list")
                frame = frame[-1]
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                raise RuntimeError("pre-reset renderer did not return an RGB array")
            # A single-environment representative rollout can have at most one
            # reset per step.  Clearing prevents a stale terminal image from
            # leaking into a later capture if a recorder has already stopped.
            terminal_frames.clear()
            terminal_frames.append(np.array(frame, copy=True))
            stats["terminal_frames_captured"] = int(
                stats["terminal_frames_captured"]
            ) + 1
        return original_reset_idx(env_ids)

    env._reset_idx = MethodType(reset_idx_with_capture, env)
    return terminal_frames, stats


def main() -> None:
    if ARGS.steps == 0 and ARGS.episodes == 0 and ARGS.headless:
        raise ValueError("headless evaluation needs --steps > 0 or --episodes > 0")
    selection = resolve_motion_inputs(
        motions=ARGS.motion,
        manifests=ARGS.manifest,
        split=ARGS.split,
        reference_root=ARGS.reference_root,
        allow_missing=ARGS.allow_missing,
        max_motions=ARGS.max_motions,
    )
    if ARGS.motion_id >= len(selection.paths):
        raise ValueError(
            f"motion-id {ARGS.motion_id} is outside selected range [0, {len(selection.paths) - 1}]"
        )
    correspondence = None
    if not ARGS.skip_correspondence_check:
        correspondence = path_below(ARGS.correspondence, ACCAD_ROOT, must_exist=True)
        if not correspondence.is_file():
            raise FileNotFoundError(correspondence)
    checkpoint = resolve_checkpoint(ARGS.checkpoint, latest_if_missing=True)
    assert checkpoint is not None

    evaluation_dir = new_artifact_dir("evaluations", ARGS.evaluation_name)
    params_dir = evaluation_dir / "params"
    params_dir.mkdir()
    initial_report = command_metadata(sys.argv)
    initial_report.update(
        {
            "schema_version": "accad_g1_tracking_evaluation_v2",
            "status": "initializing",
            "evaluation_dir": str(evaluation_dir),
            "checkpoint": {
                "path": str(checkpoint),
                "checksum_sha256": sha256_file(checkpoint),
                "bytes": checkpoint.stat().st_size,
            },
            "motion_input": selection.metadata(),
            "selected_motion_id": ARGS.motion_id,
            "selected_motion_path": str(selection.paths[ARGS.motion_id]),
            "physics_warmup_steps_requested": int(ARGS.physics_warmup_steps),
            "correspondence": None
            if correspondence is None
            else {
                "path": str(correspondence),
                "checksum_sha256": sha256_file(correspondence),
            },
        }
    )
    write_json_atomic(evaluation_dir / "evaluation_summary.json", initial_report)

    env_cfg = make_tracking_env_cfg(
        selection.paths,
        num_envs=ARGS.num_envs,
        eval_mode=True,
        correspondence_path=correspondence,
        fixed_motion_id=ARGS.motion_id,
        random_start=False,
        seed=ARGS.seed,
    )
    env_cfg.sim.physx.enable_enhanced_determinism = True
    agent_cfg = make_tracking_runner_cfg()
    agent_cfg.seed = ARGS.seed
    if getattr(ARGS, "device", None) is not None:
        env_cfg.sim.device = ARGS.device
        agent_cfg.device = ARGS.device
    env_cfg.log_dir = str(evaluation_dir)
    dump_yaml(str(params_dir / "env.yaml"), env_cfg)
    dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)

    base_env: ManagerBasedRLEnv | None = None
    wrapped: RslRlVecEnvWrapper | None = None
    video_capture_stats: dict[str, int | bool] | None = None
    try:
        base_env = ManagerBasedRLEnv(
            cfg=env_cfg,
            render_mode="rgb_array" if ARGS.video else None,
        )
        physics_warmup = run_physics_cold_start_warmup(
            base_env,
            int(ARGS.physics_warmup_steps),
            expected_action_dim=env_cfg.policy_action_dim,
        )
        observation_contract = tracking_observation_contract()
        live_tensor_contract = resolved_live_tracking_tensor_contract(base_env)
        gym_env: gym.Env = base_env
        if ARGS.video:
            if ARGS.num_envs != 1:
                raise ValueError(
                    "terminal-aware policy video recording requires --num-envs 1"
                )
            video_dir = evaluation_dir / "videos"
            video_dir.mkdir()
            terminal_frames, video_capture_stats = _install_terminal_frame_capture(
                base_env
            )
            gym_env = _TerminalAwareRecordVideo(
                gym_env,
                video_folder=str(video_dir),
                step_trigger=lambda step: step == 0,
                video_length=ARGS.video_length,
                disable_logger=True,
                terminal_frames=terminal_frames,
                capture_stats=video_capture_stats,
            )
        wrapped = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
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
        exports = _export_policy(runner, evaluation_dir / "exported", ARGS.export)
        export_validations = validate_policy_exports(
            exports,
            actor_observation_dim=env_cfg.actor_observation_dim,
            action_dim=env_cfg.policy_action_dim,
        )

        command = base_env.command_manager.get_term("motion")
        action_term = base_env.action_manager.get_term("joint_residual")
        runtime_contract = {
            "library_checksum_sha256": command.library.library_checksum_sha256,
            "motion_count": command.library.num_motions,
            "motion_keys": list(command.library.motion_keys),
            "selected_motion_id": int(command.motion_ids[0].item()),
            "contract_checksum_sha256": command.library.contract.digest(),
            "actor_observation_dim": env_cfg.actor_observation_dim,
            "critic_observation_dim": env_cfg.critic_observation_dim,
            "action_dim": env_cfg.policy_action_dim,
            "observation_contract": observation_contract,
            "live_tensor_contract": live_tensor_contract,
            "physics_dt": env_cfg.sim.dt,
            "control_dt": env_cfg.sim.dt * env_cfg.decimation,
            "policy_action_clip": agent_cfg.clip_actions,
            "physical_action_clip": action_term.cfg.action_clip,
            "evaluation_isolation": resolved_evaluation_isolation(base_env),
            "physics_cold_start_warmup": physics_warmup,
            "video_capture_started_after_physics_warmup": bool(ARGS.video),
        }
        write_json_atomic(evaluation_dir / "runtime_contract.json", runtime_contract)

        observations = wrapped.get_observations()
        if video_capture_stats is not None:
            video_capture_stats["enabled"] = True
        episode_returns = torch.zeros(base_env.num_envs, device=wrapped.device)
        episode_lengths = torch.zeros(
            base_env.num_envs, device=wrapped.device, dtype=torch.long
        )
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        action_abs_sum = 0.0
        action_abs_max = 0.0
        action_samples = 0
        completed_episodes = 0
        failed_episodes = 0
        timeout_episodes = 0
        successful_motion_end_episodes = 0
        incomplete_timeout_episodes = 0
        termination_counts = {
            name: 0 for name in base_env.termination_manager.active_terms
        }
        first_failure_step: int | None = None
        first_motion_end_step: int | None = None
        joint_abs_error_sum = 0.0
        joint_square_error_sum = 0.0
        joint_error_samples = 0
        root_distance_sum = 0.0
        root_distance_samples = 0
        root_orientation_error_sum = 0.0
        root_orientation_error_samples = 0
        body_distance_sum = 0.0
        body_distance_samples = 0
        contact_match_sum = 0.0
        contact_match_samples = 0
        contact_true_positive = torch.zeros(2, dtype=torch.long)
        contact_false_positive = torch.zeros(2, dtype=torch.long)
        contact_false_negative = torch.zeros(2, dtype=torch.long)
        stance_slip_samples_mps: list[float] = []
        raw_action_clip_count = 0
        raw_action_samples = 0
        target_clamp_count = 0
        target_clamp_samples = 0
        residual_rate_limit_count = 0
        residual_rate_limit_samples = 0
        first_episode_finished = torch.zeros(
            base_env.num_envs, device=wrapped.device, dtype=torch.bool
        )
        first_episode_success = torch.zeros_like(first_episode_finished)
        first_episode_lengths = torch.zeros(
            base_env.num_envs, device=wrapped.device, dtype=torch.long
        )
        steps_executed = 0
        step_limit = ARGS.steps if ARGS.steps > 0 else 2**31 - 1

        print(
            f"[INFO] Evaluating motion {ARGS.motion_id}/{len(selection.paths) - 1}: "
            f"{selection.paths[ARGS.motion_id]}"
        )
        while SIMULATION_APP.is_running() and steps_executed < step_limit:
            start = time.perf_counter()
            # Measure the state represented by the current observation before
            # stepping.  ManagerBasedRLEnv resets done environments inside
            # ``step``; pre-step measurement therefore avoids accidentally
            # reporting a freshly reset state as the terminal tracking state.
            robot = command.robot
            actual_q = robot.data.joint_pos[:, command.policy_joint_ids]
            joint_error = actual_q - command.joint_pos
            joint_abs_error_sum += float(torch.abs(joint_error).sum().item())
            joint_square_error_sum += float(torch.square(joint_error).sum().item())
            joint_error_samples += joint_error.numel()
            root_distance = torch.linalg.vector_norm(
                robot.data.root_link_pos_w - command.root_pos_w, dim=-1
            )
            root_distance_sum += float(root_distance.sum().item())
            root_distance_samples += root_distance.numel()
            root_orientation_error = quat_error_angle_wxyz(
                command.root_quat_wxyz, robot.data.root_link_quat_w
            )
            root_orientation_error_sum += float(
                root_orientation_error.sum().item()
            )
            root_orientation_error_samples += root_orientation_error.numel()
            actual_body = robot.data.body_pos_w[:, command.tracking_body_ids]
            body_distance = torch.linalg.vector_norm(
                actual_body - command.tracking_body_pos_w, dim=-1
            )
            body_distance_sum += float(body_distance.sum().item())
            body_distance_samples += body_distance.numel()
            actual_contact = command.actual_foot_contact
            reference_contact = command.foot_contact
            contact_match_sum += float(
                (actual_contact == reference_contact)
                .to(torch.float32)
                .sum()
                .item()
            )
            contact_match_samples += reference_contact.numel()
            contact_true_positive += (
                actual_contact & reference_contact
            ).sum(dim=0).detach().cpu()
            contact_false_positive += (
                actual_contact & ~reference_contact
            ).sum(dim=0).detach().cpu()
            contact_false_negative += (
                ~actual_contact & reference_contact
            ).sum(dim=0).detach().cpu()
            foot_speed = torch.linalg.vector_norm(
                robot.data.body_lin_vel_w[:, command.foot_body_ids, :2], dim=-1
            )
            stance_speed = foot_speed[actual_contact | reference_contact]
            if stance_speed.numel() > 0:
                stance_slip_samples_mps.extend(
                    float(value)
                    for value in stance_speed.detach().cpu().tolist()
                )
            with torch.inference_mode():
                actions = policy(observations)
                physical_action_clip = float(action_term.cfg.action_clip)
                raw_action_clip_count += int(
                    (torch.abs(actions) > physical_action_clip).sum().item()
                )
                raw_action_samples += actions.numel()
                effective_actions = actions.clamp(
                    -physical_action_clip, physical_action_clip
                )
                predicted_target = reference_residual_target(
                    effective_actions,
                    command.joint_pos,
                    action_term.normalized_residual,
                    residual_scale=action_term.cfg.residual_scale,
                    action_clip=action_term.cfg.action_clip,
                    normalized_delta_limit=action_term.cfg.normalized_delta_limit,
                    hard_limits=robot.data.joint_pos_limits[:, action_term.joint_ids],
                    soft_limits=robot.data.soft_joint_pos_limits[:, action_term.joint_ids],
                    target_limit_mode=action_term.cfg.target_limit_mode,
                )
                target_clamp_count += int(predicted_target.target_clamped.sum().item())
                target_clamp_samples += predicted_target.target_clamped.numel()
                residual_rate_limit_count += int(predicted_target.rate_limited.sum().item())
                residual_rate_limit_samples += predicted_target.rate_limited.numel()
                observations, rewards, dones, _ = wrapped.step(actions)
                policy_module.reset(dones)
            steps_executed += 1
            episode_returns += rewards
            episode_lengths += 1
            action_abs_sum += float(torch.abs(actions).sum().item())
            action_abs_max = max(action_abs_max, float(torch.abs(actions).max().item()))
            action_samples += actions.numel()
            done_ids = torch.where(dones > 0)[0]
            if done_ids.numel() > 0:
                term_masks = {
                    term_name: base_env.termination_manager.get_term(term_name)[
                        done_ids
                    ]
                    for term_name in termination_counts
                }
                for term_name, term_mask in term_masks.items():
                    term_count = int(term_mask.sum().item())
                    termination_counts[term_name] += term_count
                hard_failure = torch.zeros(
                    done_ids.numel(), device=done_ids.device, dtype=torch.bool
                )
                for term_name in (
                    "fall",
                    "tracking_divergence",
                    "joint_limits",
                    "nonfinite",
                ):
                    if term_name in term_masks:
                        hard_failure |= term_masks[term_name]
                motion_end = term_masks.get(
                    "motion_end", torch.zeros_like(hard_failure)
                )
                successful_motion_end = motion_end & ~hard_failure
                first_done_mask = ~first_episode_finished[done_ids]
                first_done_ids = done_ids[first_done_mask]
                if first_done_ids.numel() > 0:
                    first_episode_finished[first_done_ids] = True
                    first_episode_lengths[first_done_ids] = episode_lengths[first_done_ids]
                    first_episode_success[first_done_ids] = successful_motion_end[
                        first_done_mask
                    ]
                successful_motion_end_episodes += int(
                    successful_motion_end.sum().item()
                )
                if first_motion_end_step is None and bool(
                    successful_motion_end.any()
                ):
                    first_motion_end_step = steps_executed
                completed_returns.extend(
                    float(value) for value in episode_returns[done_ids].detach().cpu().tolist()
                )
                completed_lengths.extend(
                    int(value) for value in episode_lengths[done_ids].detach().cpu().tolist()
                )
                completed_episodes += int(done_ids.numel())
                timeout_flags = base_env.reset_time_outs[done_ids]
                timeout_episodes += int(timeout_flags.sum().item())
                failed_episodes += int(hard_failure.sum().item())
                incomplete_timeout_episodes += int(
                    (timeout_flags & ~successful_motion_end).sum().item()
                )
                if first_failure_step is None and bool(hard_failure.any()):
                    first_failure_step = steps_executed
                episode_returns[done_ids] = 0.0
                episode_lengths[done_ids] = 0
            if ARGS.episodes > 0 and completed_episodes >= ARGS.episodes:
                break
            if ARGS.video and steps_executed >= ARGS.video_length:
                break
            if ARGS.real_time:
                delay = base_env.step_dt - (time.perf_counter() - start)
                if delay > 0.0:
                    time.sleep(delay)

        # Closing before scanning videos guarantees that RecordVideo flushes MP4.
        wrapped.close()
        wrapped = None
        base_env = None
        videos = sorted((evaluation_dir / "videos").glob("*.mp4")) if ARGS.video else []
        contact_metrics: dict[str, dict[str, float | int | None]] = {}
        for foot_index, foot_name in enumerate(("left", "right")):
            tp = int(contact_true_positive[foot_index].item())
            fp = int(contact_false_positive[foot_index].item())
            fn = int(contact_false_negative[foot_index].item())
            precision = None if tp + fp == 0 else tp / (tp + fp)
            recall = None if tp + fn == 0 else tp / (tp + fn)
            f1 = (
                None
                if precision is None or recall is None or precision + recall == 0.0
                else 2.0 * precision * recall / (precision + recall)
            )
            contact_metrics[foot_name] = {
                "true_positive_frames": tp,
                "false_positive_frames": fp,
                "false_negative_frames": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        expected_frame_count = int(command.library.lengths[ARGS.motion_id].item())
        first_episode_length = (
            int(first_episode_lengths[0].item())
            if bool(first_episode_finished[0].item())
            else steps_executed
        )
        first_episode_success_count = int(first_episode_success.sum().item())
        all_first_episodes_completed = bool(first_episode_finished.all().item())
        all_first_episodes_succeeded = bool(first_episode_success.all().item())
        report = dict(initial_report)
        report.update(
            {
                "status": "completed",
                "finished_at_utc": utc_now(),
                "runtime_contract": runtime_contract,
                "evaluation": {
                    "steps_executed": steps_executed,
                    "completed_episodes": completed_episodes,
                    "timeout_episodes": timeout_episodes,
                    "incomplete_timeout_episodes": incomplete_timeout_episodes,
                    "failed_episodes": failed_episodes,
                    "successful_motion_end_episodes": successful_motion_end_episodes,
                    "termination_counts": termination_counts,
                    "first_failure_step": first_failure_step,
                    "first_motion_end_step": first_motion_end_step,
                    "expected_frame_count": expected_frame_count,
                    "first_episode_length": first_episode_length,
                    "first_episode_lengths_by_environment": [
                        int(value)
                        for value in first_episode_lengths.detach().cpu().tolist()
                    ],
                    "first_episode_success_count": first_episode_success_count,
                    "all_first_episodes_completed": all_first_episodes_completed,
                    "first_episode_completion_ratio": min(
                        first_episode_length / max(expected_frame_count, 1), 1.0
                    ),
                    "uninterrupted_full_clip_success": bool(
                        all_first_episodes_completed and all_first_episodes_succeeded
                    ),
                    "completed_return_mean": None
                    if not completed_returns
                    else sum(completed_returns) / len(completed_returns),
                    "completed_return_min": None
                    if not completed_returns
                    else min(completed_returns),
                    "completed_return_max": None
                    if not completed_returns
                    else max(completed_returns),
                    "completed_length_mean": None
                    if not completed_lengths
                    else sum(completed_lengths) / len(completed_lengths),
                    "mean_absolute_action": action_abs_sum / max(action_samples, 1),
                    "maximum_absolute_action": action_abs_max,
                    "mean_absolute_joint_error_rad": joint_abs_error_sum
                    / max(joint_error_samples, 1),
                    "joint_position_rmse_rad": (
                        joint_square_error_sum / max(joint_error_samples, 1)
                    )
                    ** 0.5,
                    "mean_root_position_error_m": root_distance_sum
                    / max(root_distance_samples, 1),
                    "mean_root_orientation_error_rad": root_orientation_error_sum
                    / max(root_orientation_error_samples, 1),
                    "mean_tracking_body_position_error_m": body_distance_sum
                    / max(body_distance_samples, 1),
                    "foot_contact_match_fraction": contact_match_sum
                    / max(contact_match_samples, 1),
                    "foot_contact": contact_metrics,
                    "stance_slip_p95_mps": None
                    if not stance_slip_samples_mps
                    else float(np.quantile(stance_slip_samples_mps, 0.95)),
                    "raw_action_clip_fraction": raw_action_clip_count
                    / max(raw_action_samples, 1),
                    "target_clamp_fraction": target_clamp_count
                    / max(target_clamp_samples, 1),
                    "residual_rate_limit_fraction": residual_rate_limit_count
                    / max(residual_rate_limit_samples, 1),
                    "tracking_metric_sampling": (
                        "pre_action_states_and_pre_step_action-target prediction; simulator_"
                        "terminal_post_physics_state_is_reset_internally_and_not_sampled"
                    ),
                },
                "exports": [
                    {
                        "path": str(path),
                        "checksum_sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in exports
                ],
                "export_validations": export_validations,
                "videos": [
                    {
                        "path": str(path),
                        "checksum_sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                        "inspection": inspect_video(path),
                    }
                    for path in videos
                ],
                "video_capture_contract": None
                if video_capture_stats is None
                else {
                    "mode": "post_physics_pre_reset_terminal_frame_substitution",
                    "single_environment": True,
                    "full_fixed_step_window": ARGS.episodes == 0
                    and ARGS.steps == ARGS.video_length,
                    **video_capture_stats,
                },
            }
        )
        write_json_atomic(evaluation_dir / "evaluation_summary.json", report)
        print(f"[INFO] Evaluation report: {evaluation_dir / 'evaluation_summary.json'}")
    except BaseException as exc:
        failed = dict(initial_report)
        failed.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json_atomic(evaluation_dir / "evaluation_summary.json", failed)
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
