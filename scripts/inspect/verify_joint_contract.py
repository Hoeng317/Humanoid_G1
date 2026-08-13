#!/usr/bin/env python3
"""Validate and materialize the cross-engine joint contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from humanoid_g1.assets.joint_contract import load_joint_contract
from humanoid_g1.utils.paths import ARTIFACTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="g1_29dof", choices=("g1_29dof", "g1_29dof_rev_1_0"))
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "contracts" / "joint_contract.json")
    args = parser.parse_args()
    contract = load_joint_contract()
    contract.validate_urdf()
    contract.validate_mujoco()
    contract.write_json(args.output)
    canonical = ARTIFACTS / "contracts" / "g1_29dof_joint_contract.json"
    if args.output.resolve() != canonical.resolve():
        contract.write_json(canonical)
    else:
        canonical = args.output
    payload = json.loads(canonical.read_text(encoding="utf-8"))
    markdown = [
        "# G1 29-DoF joint contract",
        "",
        "|policy|Isaac|MuJoCo|SDK2|joint|default|limits|kp/kd|",
        "|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for joint in payload["joints"]:
        markdown.append(
            f"|{joint['policy_index']}|{joint['isaac_index']}|{joint['mujoco_index']}|"
            f"{joint['sdk_motor_index']}|{joint['joint_name']}|{joint['default_position_rad']}|"
            f"[{joint['lower_limit_rad']}, {joint['upper_limit_rad']}]|{joint['kp']}/{joint['kd']}|"
        )
    (ARTIFACTS / "contracts" / "g1_29dof_joint_contract.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(f"G1 variant       : {contract.variant}")
    print(f"Body/action DoF  : {contract.action_dim} (hands excluded)")
    print(f"Physics/policy Hz: {1 / contract.physics_dt:.0f}/{1 / contract.policy_dt:.0f}")
    print(f"Policy -> SDK    : {contract.policy_to_sdk}")
    for joint in contract.joints:
        print(
            f"{joint.policy_index:02d} -> SDK {joint.sdk_index:02d}  {joint.name:<30} "
            f"[{joint.lower: .4f}, {joint.upper: .4f}] default={joint.default: .3f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
