#!/usr/bin/env python3
"""Play a trained G1 policy."""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher
from common import add_experiment_args, latest_checkpoint, prepare_experiment

parser = argparse.ArgumentParser()
add_experiment_args(parser, checkpoint=True)
parser.add_argument("--steps", type=int, default=0, help="0 means run until GUI closes")
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--video", action="store_true")
parser.add_argument("--video-length", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
exp = prepare_experiment(args)
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import humanoid_g1.tasks  # noqa: F401, E402
from common import load_isaac_configs  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402


def main() -> None:
    env_cfg, agent_cfg = load_isaac_configs(exp, play=False)
    if args.num_envs is None:
        env_cfg.scene.num_envs = 32
    checkpoint = args.checkpoint or latest_checkpoint(exp.name)
    env = gym.make(exp.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(checkpoint.parent / "videos" / "play"),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=wrapped.device)
    obs = wrapped.get_observations()
    limit = args.steps if args.steps > 0 else 2**31 - 1
    print(f"[INFO] Playing {checkpoint}")
    for step in range(limit):
        if not app.is_running():
            break
        start = time.perf_counter()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = wrapped.step(actions)
            runner.alg.policy.reset(dones)
        if args.video and step + 1 >= args.video_length:
            break
        delay = wrapped.unwrapped.step_dt - (time.perf_counter() - start)
        if args.real_time and delay > 0:
            time.sleep(delay)
    env.close()


try:
    main()
finally:
    app.close()
