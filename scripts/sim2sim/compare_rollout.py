#!/usr/bin/env python3
"""Compare Isaac and MuJoCo rollout recordings with the same scripted command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED = ("command", "base_quaternion_wxyz", "joint_position", "action")


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        missing = [name for name in REQUIRED if name not in archive]
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        return {name: np.asarray(archive[name]) for name in REQUIRED}


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac", type=Path, required=True)
    parser.add_argument("--mujoco", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    isaac = load(args.isaac)
    mujoco = load(args.mujoco)
    count = min(isaac["action"].shape[0], mujoco["action"].shape[0])
    if count < 1:
        raise ValueError("rollouts contain no comparable samples")
    for name in REQUIRED:
        if isaac[name].shape[1:] != mujoco[name].shape[1:]:
            raise ValueError(f"{name} shape mismatch")
    metrics = {
        "samples": count,
        "command_rmse": rmse(isaac["command"][:count], mujoco["command"][:count]),
        "base_quaternion_rmse": rmse(
            isaac["base_quaternion_wxyz"][:count], mujoco["base_quaternion_wxyz"][:count]
        ),
        "joint_position_rmse_rad": rmse(
            isaac["joint_position"][:count], mujoco["joint_position"][:count]
        ),
        "action_rmse": rmse(isaac["action"][:count], mujoco["action"][:count]),
        "isaac_action_saturation_ratio": float(np.mean(np.abs(isaac["action"][:count]) >= 0.999)),
        "mujoco_action_saturation_ratio": float(np.mean(np.abs(mujoco["action"][:count]) >= 0.999)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# Isaac–MuJoCo rollout comparison\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in metrics.items())
        + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
