#!/usr/bin/env python3
"""Launch one environment and verify live Isaac action/observation contracts."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Humanoid-G1-29DoF-Velocity-Flat-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True if args.headless is None else args.headless
app = AppLauncher(args).app

import gymnasium as gym

import humanoid_g1.tasks  # noqa: F401, E402
from humanoid_g1.assets.joint_contract import load_joint_contract  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def main() -> None:
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    env = gym.make(args.task, cfg=cfg)
    env.reset()
    contract = load_joint_contract()
    robot = env.unwrapped.scene["robot"]
    action_dim = env.unwrapped.action_manager.total_action_dim
    obs = env.unwrapped.observation_manager.compute()
    policy_dim = obs["policy"].shape[-1]
    critic_dim = obs["critic"].shape[-1]
    if action_dim != 29 or policy_dim != 480 or critic_dim != 495:
        raise RuntimeError(f"live schema mismatch: actions={action_dim}, actor={policy_dim}, critic={critic_dim}")
    resolved = list(env.unwrapped.action_manager.get_term("joint_position")._joint_names)
    if resolved != contract.policy_names:
        raise RuntimeError(f"Isaac action order differs from contract: {resolved}")
    print(f"Isaac articulation joints: {robot.num_joints}")
    print(f"Action dimension/order    : {action_dim} / contract match")
    print(f"Actor/Critic observations : {policy_dim}/{critic_dim}")
    print("Task live inspection PASSED")
    env.close()


try:
    main()
finally:
    app.close()

