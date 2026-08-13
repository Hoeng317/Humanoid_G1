#!/usr/bin/env python3
"""G1 physics smoke simulation with deterministic validation modes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Humanoid-G1-29DoF-Velocity-Flat-v0")
parser.add_argument(
    "--mode",
    choices=(
        "spawn-only",
        "hold-pose",
        "sine-joint-test",
        "random-action",
        "free_fall",
        "default_pose",
        "sine",
        "random_action",
    ),
    default="hold-pose",
)
parser.add_argument("--config", type=Path, default=Path("configs/sim/g1_debug.yaml"))
parser.add_argument("--joint", default="left_hip_pitch_joint")
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--report", type=Path, default=None)
parser.add_argument("--capture-observations", type=Path, default=None)
parser.add_argument("--ankle-action-sweep", type=float, nargs="*", default=None, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import yaml

import humanoid_g1.tasks  # noqa: F401, E402
from humanoid_g1.assets.joint_contract import load_joint_contract  # noqa: E402
from humanoid_g1.utils.paths import ARTIFACTS  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def main() -> None:
    mode = {
        "spawn-only": "default_pose",
        "hold-pose": "default_pose",
        "sine-joint-test": "sine",
        "random-action": "random_action",
    }.get(args.mode, args.mode)
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    simulation = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg.sim.dt = float(simulation["physics_dt"])
    cfg.decimation = int(simulation["decimation"])
    cfg.episode_length_s = float(simulation["episode_length_s"])
    cfg.sim.render_interval = int(simulation.get("render_interval", cfg.decimation))
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.observations.policy.enable_corruption = False
    cfg.events.physics_material = None
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.actuator_gains = None
    cfg.events.joint_friction = None
    cfg.events.push_robot = None
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    if mode == "free_fall":
        for actuator in cfg.scene.robot.actuators.values():
            actuator.stiffness = 0.0
            actuator.damping = 0.0
    env = gym.make(args.task, cfg=cfg)
    env.reset()
    device = env.unwrapped.device
    action_dim = env.unwrapped.action_manager.total_action_dim
    joint_contract = load_joint_contract()
    if args.joint not in joint_contract.policy_names:
        raise ValueError(f"unknown policy joint: {args.joint}")
    sine_joint_index = joint_contract.policy_names.index(args.joint)
    min_height = float("inf")
    max_tilt = 0.0
    gravity_xy_min = [float("inf"), float("inf")]
    gravity_xy_max = [float("-inf"), float("-inf")]
    non_finite = False
    fall_terminations = 0
    timeouts = 0
    termination_reasons: dict[str, int] = {}
    terminations_per_env = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
    captured_observations = []
    steps = args.steps if args.steps > 0 else 2**31 - 1
    for step in range(steps):
        if not app.is_running():
            break
        if mode == "random_action":
            actions = torch.empty((args.num_envs, action_dim), device=device).uniform_(-0.2, 0.2)
        elif mode == "sine":
            actions = torch.zeros((args.num_envs, action_dim), device=device)
            actions[:, sine_joint_index] = 0.15 * math.sin(
                step * env.unwrapped.step_dt * 2.0 * math.pi * 0.5
            )
        else:
            actions = torch.zeros((args.num_envs, action_dim), device=device)
        if args.ankle_action_sweep:
            if len(args.ankle_action_sweep) != args.num_envs:
                raise ValueError("--ankle-action-sweep requires exactly --num-envs values")
            ankle = torch.tensor(args.ankle_action_sweep, device=device)
            actions[:, 13] = ankle
            actions[:, 14] = ankle
        observations, _, terminated, truncated, _ = env.step(actions)
        if args.capture_observations is not None:
            captured_observations.append(observations["policy"].detach().cpu())
        fall_terminations += int(terminated.sum())
        terminations_per_env += terminated.to(torch.int64)
        timeouts += int(truncated.sum())
        for name in env.unwrapped.termination_manager.active_terms:
            count = int(env.unwrapped.termination_manager.get_term(name).sum())
            termination_reasons[name] = termination_reasons.get(name, 0) + count
        data = env.unwrapped.scene["robot"].data
        min_height = min(min_height, float(data.root_pos_w[:, 2].min()))
        tilt = torch.linalg.vector_norm(data.projected_gravity_b[:, :2], dim=-1)
        max_tilt = max(max_tilt, float(tilt.max()))
        for axis in range(2):
            gravity_xy_min[axis] = min(gravity_xy_min[axis], float(data.projected_gravity_b[:, axis].min()))
            gravity_xy_max[axis] = max(gravity_xy_max[axis], float(data.projected_gravity_b[:, axis].max()))
        non_finite |= not bool(torch.isfinite(data.joint_pos).all() and torch.isfinite(data.joint_vel).all())
    report = {
        "mode": mode,
        "requested_steps": args.steps,
        "minimum_base_height_m": min_height,
        "maximum_gravity_xy_norm": max_tilt,
        "projected_gravity_xy_min": gravity_xy_min,
        "projected_gravity_xy_max": gravity_xy_max,
        "non_finite_state": non_finite,
        "fall_terminations": fall_terminations,
        "timeout_resets": timeouts,
        "termination_reasons": termination_reasons,
        "ankle_action_sweep": args.ankle_action_sweep,
        "terminations_per_env": terminations_per_env.cpu().tolist(),
        "action_dimension": action_dim,
        "passed": not non_finite and action_dim == 29 and (mode == "free_fall" or fall_terminations == 0),
    }
    destination = args.report or ARTIFACTS / "reports" / f"stability_{mode}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.capture_observations is not None:
        import numpy as np

        args.capture_observations.parent.mkdir(parents=True, exist_ok=True)
        values = torch.cat(captured_observations).numpy()
        np.savez_compressed(args.capture_observations, actor_observation=values)
        print(f"Actor observations: {args.capture_observations} ({values.shape[0]} samples)")
    print(json.dumps(report, indent=2))
    print(f"Report: {destination}")
    env.close()


try:
    main()
finally:
    app.close()
