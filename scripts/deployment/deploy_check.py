#!/usr/bin/env python3
"""Offline deployment pipeline through target generation, with no DDS imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from humanoid_g1.assets.joint_contract import load_joint_contract
from humanoid_g1.deployment.action_adapter import ActionAdapter
from humanoid_g1.deployment.observation_adapter import LowState, ObservationAdapter
from humanoid_g1.deployment.policy_runtime import PolicyRuntime
from humanoid_g1.deployment.safety_state_machine import SafetyGate


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--policy", type=Path, required=True)
    value.add_argument("--input", type=Path, default=None)
    value.add_argument("--command", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    return value


def load_state(path: Path | None) -> LowState:
    contract = load_joint_contract()
    if path is None:
        q = contract.reorder_policy_to_sdk([joint.default for joint in contract.joints])
        return LowState.from_arrays(time.monotonic(), q, [0.0] * 29, [1.0, 0.0, 0.0, 0.0], [0.0] * 3)
    payload = np.load(path)
    return LowState.from_arrays(
        time.monotonic(),
        payload["joint_position"],
        payload["joint_velocity"],
        payload["quaternion_wxyz"],
        payload["angular_velocity"],
    )


def main() -> None:
    args = parser().parse_args()
    state = load_state(args.input)
    observation_adapter = ObservationAdapter()
    action_adapter = ActionAdapter()
    safety = SafetyGate()
    runtime = PolicyRuntime(args.policy)
    safety.validate_state(state, now_s=state.timestamp_s)
    observation = observation_adapter.build(state, args.command, [0.0] * 29)
    action, inference_s = runtime.infer(observation)
    result = action_adapter.apply(action)
    safety.validate_action(result.clipped_policy, result.target_policy, inference_s)
    report = {
        "dds_publisher_created": False,
        "observation_dimension": int(observation.size),
        "action_dimension": int(action.size),
        "inference_ms": inference_s * 1000.0,
        "saturated_actions": result.saturated_count,
        "target_sdk_min": float(result.target_sdk.min()),
        "target_sdk_max": float(result.target_sdk.max()),
        "passed": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

