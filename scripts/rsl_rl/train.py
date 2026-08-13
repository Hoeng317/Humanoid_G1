#!/usr/bin/env python3
"""Train the custom G1 task with baseline RSL-RL PPO."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import yaml

from isaaclab.app import AppLauncher

from common import add_experiment_args, latest_checkpoint, new_run_dir, prepare_experiment

parser = argparse.ArgumentParser()
add_experiment_args(parser, checkpoint=True)
parser.add_argument(
    "--resume",
    nargs="?",
    const="latest",
    default=None,
    help="Resume latest checkpoint or the checkpoint path supplied after --resume",
)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video-length", type=int, default=200)
parser.add_argument("--video-interval", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
exp = prepare_experiment(args)
if not args.headless and bool(exp.data["overrides"].get("headless", False)):
    args.headless = True
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import humanoid_g1.tasks  # noqa: F401, E402
from common import load_isaac_configs  # noqa: E402
from humanoid_g1.utils.reproducibility import create_manifest  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402


def main() -> None:
    env_cfg, agent_cfg = load_isaac_configs(exp)
    if args.run_name:
        agent_cfg.run_name = args.run_name
    run_dir = new_run_dir(exp, args.run_name)
    run_dir.mkdir(parents=True)
    (run_dir / "params").mkdir()
    exp.save(run_dir / "params" / "experiment_resolved.yaml")
    (run_dir / "params" / "agent.yaml").write_text(
        yaml.safe_dump(agent_cfg.to_dict(), sort_keys=False), encoding="utf-8"
    )
    create_manifest(run_dir, sys.argv, exp.source)
    env = gym.make(exp.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(run_dir / "videos" / "train"),
            step_trigger=lambda step: step % args.video_interval == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device)
    resume_path = args.checkpoint
    if args.resume is not None and resume_path is None:
        resume_path = latest_checkpoint(exp.name) if args.resume == "latest" else Path(args.resume)
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        print(f"[INFO] Resume checkpoint: {resume_path}")
        runner.load(str(resume_path))
    print(f"[INFO] Task: {exp.task}")
    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Environments/iterations: {env_cfg.scene.num_envs}/{agent_cfg.max_iterations}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


try:
    main()
finally:
    app.close()
