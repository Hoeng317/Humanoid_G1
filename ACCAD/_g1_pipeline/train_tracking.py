#!/usr/bin/env python3
"""Train the ACCAD-to-G1 reference-residual policy with RSL-RL PPO.

All generated logs, checkpoints, videos, configs, and exports are confined to
``ACCAD/_g1_pipeline/work/training``.  Direct ``--motion`` inputs may be
repeated; one or more ``--manifest`` files may also be merged.  With neither
option, the authoritative ACCAD train manifest is selected.
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
    DEFAULT_TRAINING_MOTION_ASSIGNMENT,
    TRAINING_MOTION_ASSIGNMENT_CHOICES,
    TRAINING_MOTION_ASSIGNMENT_SCHEMA_VERSION,
    add_motion_input_args,
    add_training_checkpoint_args,
    command_metadata,
    initialize_runner_policy_from_source,
    load_policy_initialization_source,
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


def _probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be a finite probability in [0, 1]")
    return parsed


def _positive_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _negative_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed >= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and negative")
    return parsed


def _positive_unit_interval(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be finite and in (0, 1]")
    return parsed


PARSER = argparse.ArgumentParser(
    description="Train a G1 reference-residual motion-tracking policy with RSL-RL PPO."
)
add_motion_input_args(PARSER, default_split="train")
PARSER.add_argument("--num-envs", type=positive_int, default=2048)
PARSER.add_argument(
    "--motion-assignment",
    choices=TRAINING_MOTION_ASSIGNMENT_CHOICES,
    default=DEFAULT_TRAINING_MOTION_ASSIGNMENT,
    help=(
        "Training-only clip assignment. Production defaults to balanced-fixed: "
        "environment i permanently owns motion i modulo motion count. "
        "episode-uniform preserves stochastic per-reset sampling for diagnostics."
    ),
)
PARSER.add_argument("--max-iterations", type=positive_int, default=None)
PARSER.add_argument("--num-steps-per-env", type=positive_int, default=None)
PARSER.add_argument("--save-interval", type=positive_int, default=None)
PARSER.add_argument("--seed", type=nonnegative_int, default=42)
PARSER.add_argument("--run-name", default="tracking")
PARSER.add_argument(
    "--physics-warmup-steps",
    type=nonnegative_int,
    default=DEFAULT_PHYSICS_WARMUP_STEPS,
    help=(
        "Policy-independent zero-residual control steps before a clean frame-0 "
        "reset; completed before the RSL-RL/video wrappers are created (default: "
        "100; pass 0 only for a cold-start diagnostic)."
    ),
)
PARSER.add_argument(
    "--learning-rate",
    type=_positive_finite,
    default=1.0e-4,
    help="PPO learning rate; reapplied after checkpoint optimizer loading.",
)
PARSER.add_argument(
    "--ppo-clip-param",
    type=_positive_unit_interval,
    default=0.2,
    help="PPO clipped-surrogate epsilon in the interval (0, 1].",
)
PARSER.add_argument(
    "--ppo-learning-epochs",
    type=positive_int,
    default=5,
    help="PPO optimization epochs per rollout.",
)
PARSER.add_argument(
    "--action-noise-std",
    type=_positive_finite,
    default=0.2,
    help="Initial independent Gaussian policy action standard deviation.",
)
PARSER.add_argument(
    "--loaded-action-noise-std",
    type=_positive_finite,
    default=None,
    help=(
        "After loading a resumed or policy-initialization checkpoint, overwrite "
        "every scalar/log action-noise parameter with this standard deviation."
    ),
)
PARSER.add_argument(
    "--disable-policy-observation-corruption",
    action="store_true",
    help="Disable policy observation corruption for a bounded diagnostic/fine-tune run.",
)
PARSER.add_argument(
    "--frame-zero-start-probability",
    type=_probability,
    default=0.50,
    help="Fraction of training resets that begin at the exact first motion frame.",
)
PARSER.add_argument(
    "--reset-state-noise-scale",
    type=_probability,
    default=0.0,
    help="Scale the four reference-state reset noise ranges; final recovery uses 0.",
)
PARSER.add_argument(
    "--alive-reward-weight",
    type=_positive_finite,
    default=0.25,
    help="Per-second alive reward weight for the final recovery objective.",
)
PARSER.add_argument(
    "--failure-reward-weight",
    type=_negative_finite,
    default=-10.0,
    help="Terminal failure reward weight for the final recovery objective.",
)
PARSER.add_argument(
    "--enable-domain-randomization",
    action="store_true",
    help="Opt in to friction/mass/gain randomization and interval pushes.",
)
add_training_checkpoint_args(PARSER)
PARSER.add_argument("--video", action="store_true")
PARSER.add_argument("--video-length", type=positive_int, default=200)
PARSER.add_argument("--video-interval", type=positive_int, default=2000)
PARSER.add_argument(
    "--random-episode-length",
    action="store_true",
    help=(
        "Opt in to RSL-RL initial episode-length randomization. The default is off "
        "because the motion command already samples random reference start frames."
    ),
)
PARSER.add_argument(
    "--export-after-training",
    choices=("none", "jit", "onnx", "both"),
    default="none",
)
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
if ARGS.video:
    ARGS.enable_cameras = True
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from accad_g1.tracking_task import (  # noqa: E402
    TRACKING_POLICY_CONTRACT_VERSION,
    TRAINING_OBJECTIVE_SCHEMA_VERSION,
    make_tracking_env_cfg,
    make_tracking_runner_cfg,
    resolve_training_motion_assignment,
    resolved_live_tracking_tensor_contract,
    resolved_policy_action_noise_contract,
    set_policy_action_noise_std,
    set_ppo_learning_rate_exact,
    training_motion_assignment_contract,
    tracking_observation_contract,
)
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


def main() -> None:
    selection = resolve_motion_inputs(
        motions=ARGS.motion,
        manifests=ARGS.manifest,
        split=ARGS.split,
        reference_root=ARGS.reference_root,
        allow_missing=ARGS.allow_missing,
        max_motions=ARGS.max_motions,
    )
    training_fixed_motion_ids = resolve_training_motion_assignment(
        ARGS.motion_assignment,
        num_envs=ARGS.num_envs,
        num_motions=len(selection.paths),
    )
    correspondence = None
    if not ARGS.skip_correspondence_check:
        correspondence = path_below(ARGS.correspondence, ACCAD_ROOT, must_exist=True)
        if not correspondence.is_file():
            raise FileNotFoundError(correspondence)

    resume_path = resolve_checkpoint(
        ARGS.checkpoint,
        latest_if_missing=bool(ARGS.resume),
    )
    policy_initialization_source = load_policy_initialization_source(
        ARGS.initialize_policy_from
    )
    if (
        ARGS.loaded_action_noise_std is not None
        and resume_path is None
        and policy_initialization_source is None
    ):
        raise ValueError(
            "--loaded-action-noise-std requires --checkpoint, --resume, or "
            "--initialize-policy-from"
        )
    run_dir = new_artifact_dir("runs", ARGS.run_name)
    params_dir = run_dir / "params"
    params_dir.mkdir()
    initial_report = command_metadata(sys.argv)
    initial_report.update(
        {
            "schema_version": "accad_g1_tracking_train_v2",
            "status": "initializing",
            "run_dir": str(run_dir),
            "resume_checkpoint": None if resume_path is None else str(resume_path),
            "policy_initialization_checkpoint": None
            if policy_initialization_source is None
            else policy_initialization_source.metadata(),
            "physics_warmup_steps_requested": int(ARGS.physics_warmup_steps),
            "motion_assignment": {
                "schema_version": TRAINING_MOTION_ASSIGNMENT_SCHEMA_VERSION,
                "mode": ARGS.motion_assignment,
                "production_default": DEFAULT_TRAINING_MOTION_ASSIGNMENT,
                "status": "pending_live_command_verification",
            },
            "motion_input": selection.metadata(),
            "correspondence": None
            if correspondence is None
            else {
                "path": str(correspondence),
                "checksum_sha256": sha256_file(correspondence),
            },
        }
    )
    write_json_atomic(run_dir / "run_summary.json", initial_report)

    env_cfg = make_tracking_env_cfg(
        selection.paths,
        num_envs=ARGS.num_envs,
        eval_mode=False,
        correspondence_path=correspondence,
        fixed_motion_ids=training_fixed_motion_ids,
        frame_zero_start_probability=ARGS.frame_zero_start_probability,
        reset_state_noise_scale=ARGS.reset_state_noise_scale,
        alive_reward_weight=ARGS.alive_reward_weight,
        failure_reward_weight=ARGS.failure_reward_weight,
        enable_domain_randomization=ARGS.enable_domain_randomization,
        seed=ARGS.seed,
    )
    if ARGS.disable_policy_observation_corruption:
        env_cfg.observations.policy.enable_corruption = False
    agent_cfg = make_tracking_runner_cfg(init_noise_std=ARGS.action_noise_std)
    agent_cfg.algorithm.learning_rate = float(ARGS.learning_rate)
    agent_cfg.algorithm.clip_param = float(ARGS.ppo_clip_param)
    agent_cfg.algorithm.num_learning_epochs = int(ARGS.ppo_learning_epochs)
    agent_cfg.seed = ARGS.seed
    if getattr(ARGS, "device", None) is not None:
        env_cfg.sim.device = ARGS.device
        agent_cfg.device = ARGS.device
    if ARGS.max_iterations is not None:
        agent_cfg.max_iterations = ARGS.max_iterations
    if ARGS.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = ARGS.num_steps_per_env
    if ARGS.save_interval is not None:
        agent_cfg.save_interval = ARGS.save_interval
    agent_cfg.run_name = ARGS.run_name
    env_cfg.log_dir = str(run_dir)
    dump_yaml(str(params_dir / "env.yaml"), env_cfg)
    dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)

    base_env: ManagerBasedRLEnv | None = None
    wrapped: RslRlVecEnvWrapper | None = None
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
            video_dir = run_dir / "videos" / "train"
            video_dir.mkdir(parents=True)
            gym_env = gym.wrappers.RecordVideo(
                gym_env,
                video_folder=str(video_dir),
                step_trigger=lambda step: step % ARGS.video_interval == 0,
                video_length=ARGS.video_length,
                disable_logger=True,
            )
        wrapped = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped,
            agent_cfg.to_dict(),
            log_dir=str(run_dir),
            device=agent_cfg.device,
        )
        loaded_action_noise_override = None
        policy_initialization_witness = None
        if resume_path is not None:
            print(f"[INFO] Loading checkpoint: {resume_path}")
            runner.load(str(resume_path), load_optimizer=True)
        elif policy_initialization_source is not None:
            print(
                "[INFO] Initializing policy only from checkpoint: "
                f"{policy_initialization_source.path}"
            )
            policy_initialization_witness = initialize_runner_policy_from_source(
                runner, policy_initialization_source
            )
        if ARGS.loaded_action_noise_std is not None:
            loaded_action_noise_override = set_policy_action_noise_std(
                runner.alg.policy, ARGS.loaded_action_noise_std
            )
        learning_rate_override = set_ppo_learning_rate_exact(
            runner.alg, ARGS.learning_rate
        )
        live_policy_action_noise = resolved_policy_action_noise_contract(
            runner.alg.policy
        )

        command = base_env.command_manager.get_term("motion")
        live_motion_ids = tuple(
            int(value) for value in command.motion_ids.detach().cpu().tolist()
        )
        configured_fixed_motion_ids = (
            None
            if command.cfg.fixed_motion_ids is None
            else tuple(int(value) for value in command.cfg.fixed_motion_ids)
        )
        if training_fixed_motion_ids is not None:
            if configured_fixed_motion_ids != training_fixed_motion_ids:
                raise RuntimeError(
                    "live motion command config changed the balanced-fixed assignment"
                )
            if live_motion_ids != training_fixed_motion_ids:
                raise RuntimeError(
                    "live motion command IDs do not match the balanced-fixed assignment"
                )
        elif configured_fixed_motion_ids is not None:
            raise RuntimeError(
                "episode-uniform training unexpectedly resolved fixed motion IDs"
            )
        motion_assignment = training_motion_assignment_contract(
            ARGS.motion_assignment,
            num_envs=base_env.num_envs,
            motion_keys=command.library.motion_keys,
            num_steps_per_env=int(agent_cfg.num_steps_per_env),
            learning_updates=int(agent_cfg.max_iterations),
            fixed_motion_ids=training_fixed_motion_ids,
        )
        motion_assignment["live_mapping_witness"] = {
            "verified": True,
            "command_cfg_fixed_motion_ids_match": (
                configured_fixed_motion_ids == training_fixed_motion_ids
            ),
            "live_motion_ids_match": (
                live_motion_ids == training_fixed_motion_ids
                if training_fixed_motion_ids is not None
                else None
            ),
            "resolved_live_motion_id_count": len(live_motion_ids),
        }
        runtime_contract = {
            "tracking_policy_contract_version": TRACKING_POLICY_CONTRACT_VERSION,
            "resume_checkpoint": None if resume_path is None else str(resume_path),
            "policy_initialization": policy_initialization_witness,
            "library_checksum_sha256": command.library.library_checksum_sha256,
            "motion_count": command.library.num_motions,
            "motion_keys": list(command.library.motion_keys),
            "motion_assignment": motion_assignment,
            "contract_checksum_sha256": command.library.contract.digest(),
            "actor_observation_dim": env_cfg.actor_observation_dim,
            "critic_observation_dim": env_cfg.critic_observation_dim,
            "action_dim": env_cfg.policy_action_dim,
            "observation_contract": observation_contract,
            "live_tensor_contract": live_tensor_contract,
            "physics_dt": env_cfg.sim.dt,
            "control_dt": env_cfg.sim.dt * env_cfg.decimation,
            "physical_action_clip": command.library.contract.tracking_action_clip,
            "policy_wrapper_action_clip": agent_cfg.clip_actions,
            "frame_zero_start_probability": command.cfg.frame_zero_start_probability,
            "domain_randomization_events_enabled": bool(ARGS.enable_domain_randomization),
            "random_episode_length_enabled": bool(ARGS.random_episode_length),
            "physics_cold_start_warmup": physics_warmup,
            "video_capture_started_after_physics_warmup": bool(ARGS.video),
            "training_objective": {
                "schema_version": TRAINING_OBJECTIVE_SCHEMA_VERSION,
                "reset_state_noise_scale": float(ARGS.reset_state_noise_scale),
                "resolved_reset_state_noise": {
                    "joint_position_rad": list(command.cfg.joint_position_noise),
                    "joint_velocity_radps": list(command.cfg.joint_velocity_noise),
                    "root_linear_velocity_mps": list(
                        command.cfg.root_linear_velocity_noise
                    ),
                    "root_angular_velocity_radps": list(
                        command.cfg.root_angular_velocity_noise
                    ),
                },
                "reward_weights": {
                    "alive": float(env_cfg.rewards.alive.weight),
                    "failure": float(env_cfg.rewards.failure.weight),
                    "divergence_risk": float(
                        env_cfg.rewards.divergence_risk.weight
                    ),
                    "action_boundary_margin": float(
                        env_cfg.rewards.action_boundary_margin.weight
                    ),
                    "action_limiter_request_gap": float(
                        env_cfg.rewards.action_limiter_request_gap.weight
                    ),
                },
                "reward_semantics": {
                    "alive": {
                        "kind": "per_second_rate",
                        "weight": float(env_cfg.rewards.alive.weight),
                    },
                    "failure": {
                        "kind": "one_shot_non_timeout_impulse",
                        "impulse": float(env_cfg.rewards.failure.weight),
                        "normalization": "divide_by_live_step_dt",
                        "pure_timeouts_excluded": True,
                        "simultaneous_timeout_failure_penalized": True,
                    },
                    "divergence_risk": {
                        "kind": "per_second_dense_squared_margin_barrier",
                        "aggregation": "maximum_across_exact_termination_components",
                        "onset_fraction": float(
                            env_cfg.rewards.divergence_risk.params[
                                "onset_fraction"
                            ]
                        ),
                        "maximum_root_distance_m": float(
                            env_cfg.rewards.divergence_risk.params[
                                "maximum_root_distance"
                            ]
                        ),
                        "maximum_mean_joint_error_rad": float(
                            env_cfg.rewards.divergence_risk.params[
                                "maximum_mean_joint_error"
                            ]
                        ),
                        "maximum_mean_body_distance_m": float(
                            env_cfg.rewards.divergence_risk.params[
                                "maximum_mean_body_distance"
                            ]
                        ),
                        "weight": float(env_cfg.rewards.divergence_risk.weight),
                        "hard_termination_thresholds_unchanged": True,
                    },
                    "action_boundary_margin": {
                        "kind": "per_second_dense_squared_action_margin",
                        "onset": float(
                            env_cfg.rewards.action_boundary_margin.params["onset"]
                        ),
                        "action_clip": float(
                            env_cfg.rewards.action_boundary_margin.params[
                                "action_clip"
                            ]
                        ),
                        "weight": float(
                            env_cfg.rewards.action_boundary_margin.weight
                        ),
                        "raw_action_preserved_for_acceptance": True,
                    },
                    "action_limiter_request_gap": {
                        "kind": "per_second_clipped_request_minus_rate_limited_residual_l2",
                        "action_clip": float(
                            env_cfg.rewards.action_limiter_request_gap.params[
                                "action_clip"
                            ]
                        ),
                        "weight": float(
                            env_cfg.rewards.action_limiter_request_gap.weight
                        ),
                    },
                },
                "reward_integration_dt_s": float(base_env.step_dt),
                "policy_observation_corruption": {
                    "disable_requested": bool(
                        ARGS.disable_policy_observation_corruption
                    ),
                    "resolved_enabled": bool(
                        env_cfg.observations.policy.enable_corruption
                    ),
                },
                "action_noise": {
                    "initial_std_requested": float(ARGS.action_noise_std),
                    "initial_std_resolved_in_runner_cfg": float(
                        agent_cfg.policy.init_noise_std
                    ),
                    "loaded_override_std_requested": None
                    if ARGS.loaded_action_noise_std is None
                    else float(ARGS.loaded_action_noise_std),
                    "loaded_override": loaded_action_noise_override,
                    "live_policy_after_load_or_initialization": live_policy_action_noise,
                },
            },
            "ppo": {
                "init_noise_std": agent_cfg.policy.init_noise_std,
                "noise_std_type": agent_cfg.policy.noise_std_type,
                "entropy_coef": agent_cfg.algorithm.entropy_coef,
                "learning_rate_requested": float(ARGS.learning_rate),
                "learning_rate": agent_cfg.algorithm.learning_rate,
                "learning_rate_post_load_witness": learning_rate_override,
                "optimizer_param_group_learning_rates_live": learning_rate_override[
                    "after"
                ]["optimizer_param_group_learning_rates"],
                "clip_param_requested": float(ARGS.ppo_clip_param),
                "clip_param": float(agent_cfg.algorithm.clip_param),
                "num_learning_epochs_requested": int(ARGS.ppo_learning_epochs),
                "num_learning_epochs": int(agent_cfg.algorithm.num_learning_epochs),
                "schedule": agent_cfg.algorithm.schedule,
                "desired_kl": agent_cfg.algorithm.desired_kl,
            },
        }
        write_json_atomic(run_dir / "runtime_contract.json", runtime_contract)

        print(f"[INFO] Run directory: {run_dir}")
        print(
            f"[INFO] Motions/envs/iterations: "
            f"{command.library.num_motions}/{base_env.num_envs}/{agent_cfg.max_iterations}"
        )
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=ARGS.random_episode_length,
        )
        motion_sample_counts = command.motion_sample_counts_list
        if len(motion_sample_counts) != command.library.num_motions:
            raise RuntimeError("motion sampling counter does not match the loaded library")
        start_time_counts = command.start_time_sample_counts
        total_episode_samples = sum(motion_sample_counts)
        if sum(start_time_counts.values()) != total_episode_samples:
            raise RuntimeError("start-time sampling counters do not match motion reset samples")
        zero_sampled_motion_keys = [
            key
            for key, count in zip(
                command.library.motion_keys, motion_sample_counts, strict=True
            )
            if count == 0
        ]
        motion_sampling = {
            "sampler": (
                "fixed_environment_motion_then_frame_zero_or_uniform_valid_start_time"
                if training_fixed_motion_ids is not None
                else "uniform_clip_then_frame_zero_or_uniform_valid_start_time"
            ),
            "motion_assignment_mode": ARGS.motion_assignment,
            "total_episode_samples": total_episode_samples,
            "minimum_samples_per_motion": min(motion_sample_counts),
            "maximum_samples_per_motion": max(motion_sample_counts),
            "zero_sampled_motion_count": len(zero_sampled_motion_keys),
            "zero_sampled_motion_keys": zero_sampled_motion_keys,
            "start_time": {
                "frame_zero_probability": command.cfg.frame_zero_start_probability,
                "frame_zero_samples": start_time_counts["frame_zero"],
                "continuous_random_samples": start_time_counts["continuous_random"],
                "fixed_samples": start_time_counts["fixed"],
                "total_samples": sum(start_time_counts.values()),
            },
            "per_motion": [
                {"motion_id": index, "motion_key": key, "samples": count}
                for index, (key, count) in enumerate(
                    zip(
                        command.library.motion_keys,
                        motion_sample_counts,
                        strict=True,
                    )
                )
            ],
        }
        checkpoints = sorted(
            run_dir.glob("model_*.pt"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
        )
        if not checkpoints:
            raise RuntimeError("RSL-RL completed without writing a model_*.pt checkpoint")
        exports = _export_policy(runner, run_dir / "exported", ARGS.export_after_training)
        export_validations = validate_policy_exports(
            exports,
            actor_observation_dim=env_cfg.actor_observation_dim,
            action_dim=env_cfg.policy_action_dim,
        )
        report = dict(initial_report)
        report.update(
            {
                "status": "completed",
                "finished_at_utc": utc_now(),
                "policy_initialization": policy_initialization_witness,
                "motion_assignment": motion_assignment,
                "runtime_contract": runtime_contract,
                "training": {
                    "num_envs": base_env.num_envs,
                    "num_steps_per_env": agent_cfg.num_steps_per_env,
                    "learning_updates_requested": agent_cfg.max_iterations,
                    "learning_updates_completed": agent_cfg.max_iterations,
                    "final_checkpoint_iteration_index": int(
                        checkpoints[-1].stem.rsplit("_", 1)[-1]
                    ),
                    "runner_current_learning_iteration": runner.current_learning_iteration,
                    "total_timesteps": runner.tot_timesteps,
                    "ppo_execution": {
                        "algorithm_learning_rate_final": runner.alg.learning_rate,
                        "optimizer_param_group_learning_rates_final": [
                            group["lr"] for group in runner.alg.optimizer.param_groups
                        ],
                    },
                },
                "motion_sampling": motion_sampling,
                "checkpoints": [
                    {
                        "path": str(path),
                        "checksum_sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in checkpoints
                ],
                "exports": [
                    {
                        "path": str(path),
                        "checksum_sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in exports
                ],
                "export_validations": export_validations,
            }
        )
        write_json_atomic(run_dir / "run_summary.json", report)
        print(f"[INFO] Final checkpoint: {checkpoints[-1]}")
    except BaseException as exc:
        failed = dict(initial_report)
        failed.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json_atomic(run_dir / "run_summary.json", failed)
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
