#!/usr/bin/env python3
"""Quantitative policy evaluation with JSON and Markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher
from common import add_experiment_args, latest_checkpoint, prepare_experiment

parser = argparse.ArgumentParser()
add_experiment_args(parser, checkpoint=True)
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--max-steps", type=int, default=100000)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
exp = prepare_experiment(args)
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import humanoid_g1.tasks  # noqa: F401, E402
from common import load_isaac_configs  # noqa: E402
from humanoid_g1.utils.paths import ARTIFACTS  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402


def main() -> None:
    env_cfg, agent_cfg = load_isaac_configs(exp, play=False)
    env_cfg.scene.num_envs = args.num_envs or min(64, args.episodes)
    checkpoint = args.checkpoint or latest_checkpoint(exp.name)
    env = gym.make(exp.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=wrapped.device)
    obs = wrapped.get_observations()
    returns = torch.zeros(wrapped.num_envs, device=wrapped.device)
    completed: list[float] = []
    falls = 0
    saturation_sum = 0.0
    action_count = 0
    for _ in range(args.max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            saturation_sum += float((actions.abs() >= 0.999).float().sum())
            action_count += actions.numel()
            obs, reward, dones, _ = wrapped.step(actions)
            returns += reward
            indices = dones.nonzero(as_tuple=False).flatten()
            for index in indices.tolist():
                completed.append(float(returns[index]))
                returns[index] = 0.0
            falls += int(indices.numel())
            runner.alg.policy.reset(dones)
        if len(completed) >= args.episodes:
            break
    selected = completed[: args.episodes]
    report = {
        "checkpoint": str(checkpoint),
        "task": exp.task,
        "episodes": len(selected),
        "mean_return": sum(selected) / len(selected) if selected else None,
        "minimum_return": min(selected) if selected else None,
        "maximum_return": max(selected) if selected else None,
        "terminations_observed": falls,
        "action_saturation_fraction": saturation_sum / action_count if action_count else None,
        "passed": len(selected) == args.episodes,
    }
    output = args.output or ARTIFACTS / "reports" / f"evaluation_{checkpoint.parent.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(
        "# G1 policy evaluation\n\n" + "\n".join(f"- {key}: {value}" for key, value in report.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    env.close()


try:
    main()
finally:
    app.close()
