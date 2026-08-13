"""Shared RSL-RL CLI helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import re

import yaml

from humanoid_g1.utils.config import ExperimentConfig, load_experiment, validate_config
from humanoid_g1.utils.paths import LOGS, ROOT


def add_experiment_args(parser: Any, checkpoint: bool = False) -> None:
    parser.add_argument("--config", default="configs/experiments/g1_flat_ppo.yaml")
    parser.add_argument("--task", default=None)
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-iterations", "--max_iterations", dest="max_iterations", type=int, default=None)
    parser.add_argument("--terrain", choices=("flat", "rough", "stairs", "mixed_curriculum"), default=None)
    parser.add_argument(
        "--command-range",
        nargs=6,
        type=float,
        metavar=("VX_MIN", "VX_MAX", "VY_MIN", "VY_MAX", "WZ_MIN", "WZ_MAX"),
    )
    parser.add_argument("--randomization", choices=("nominal", "sim2real", "off"), default=None)
    parser.add_argument("--run-name", "--run_name", dest="run_name", default=None)
    if checkpoint:
        parser.add_argument("--checkpoint", type=Path, default=None)


def prepare_experiment(args: Any) -> ExperimentConfig:
    overrides = {
        "num_envs": getattr(args, "num_envs", None),
        "seed": getattr(args, "seed", None),
        "max_iterations": getattr(args, "max_iterations", None),
        "device": getattr(args, "device", None),
        "headless": getattr(args, "headless", None),
    }
    exp = load_experiment(args.config, overrides)
    if getattr(args, "task", None):
        exp.data["task"] = args.task
    if getattr(args, "terrain", None):
        exp.data["terrain"] = yaml.safe_load(
            (ROOT / "configs" / "terrain" / f"{args.terrain}.yaml").read_text(encoding="utf-8")
        )
        exp.data["task"] = (
            "Humanoid-G1-29DoF-Velocity-Flat-v0"
            if args.terrain == "flat"
            else "Humanoid-G1-29DoF-Velocity-Rough-v0"
        )
    if getattr(args, "randomization", None):
        exp.data["randomization"] = yaml.safe_load(
            (ROOT / "configs" / "randomization" / f"{args.randomization}.yaml").read_text(encoding="utf-8")
        )
    values = getattr(args, "command_range", None)
    if values:
        exp.data["commands"] = {
            "lin_vel_x": values[0:2],
            "lin_vel_y": values[2:4],
            "ang_vel_z": values[4:6],
        }
    validate_config(exp.data)
    return exp


def new_run_dir(exp: ExperimentConfig, run_name: str | None = None) -> Path:
    suffix = f"_{re.sub(r'[^A-Za-z0-9_.-]+', '_', run_name)}" if run_name else ""
    return LOGS / "rsl_rl" / exp.name / f"{datetime.now():%Y-%m-%d_%H-%M-%S}{suffix}"


def latest_checkpoint(exp_name: str) -> Path:
    root = LOGS / "rsl_rl" / exp_name
    candidates = sorted(root.glob("*/model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no model_*.pt checkpoint under {root}")
    def iteration(path: Path) -> tuple[str, int]:
        match = re.search(r"model_(\d+)", path.name)
        return (path.parent.name, int(match.group(1)) if match else -1)
    return max(candidates, key=iteration)


def load_isaac_configs(exp: ExperimentConfig, play: bool = False) -> tuple[Any, Any]:
    from humanoid_g1.utils.config import apply_to_isaac_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_key = "play_env_cfg_entry_point" if play else "env_cfg_entry_point"
    env_cfg = load_cfg_from_registry(exp.task, env_key)
    agent_cfg = load_cfg_from_registry(exp.task, "rsl_rl_cfg_entry_point")
    return apply_to_isaac_cfg(exp, env_cfg, agent_cfg)


def policy_module_and_normalizer(runner: Any) -> tuple[Any, Any]:
    policy = runner.alg.policy
    normalizer = getattr(policy, "actor_obs_normalizer", None)
    return policy, normalizer
