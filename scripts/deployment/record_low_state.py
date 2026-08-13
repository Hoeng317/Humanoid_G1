#!/usr/bin/env python3
"""Create a synthetic safe LowState sample; no network or DDS is accessed."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from humanoid_g1.assets.joint_contract import load_joint_contract
from humanoid_g1.utils.paths import ARTIFACTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "contracts" / "sample_low_state.npz")
    args = parser.parse_args()
    contract = load_joint_contract()
    q = contract.reorder_policy_to_sdk([joint.default for joint in contract.joints])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        joint_position=np.asarray(q, dtype=np.float32),
        joint_velocity=np.zeros(29, dtype=np.float32),
        quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        angular_velocity=np.zeros(3, dtype=np.float32),
    )
    print(args.output)


if __name__ == "__main__":
    main()

