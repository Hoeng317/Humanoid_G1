#!/usr/bin/env python3
"""Guarded entry point for the physical G1 C++ controller."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from humanoid_g1.deployment.real_guard import RealModeDenied, assert_real_mode_allowed
from humanoid_g1.utils.paths import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--acknowledge-hardware-risk", action="store_true")
    parser.add_argument("--interface", "--network", dest="interface", required=True)
    parser.add_argument("--domain-id", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stand", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.domain_id != 0:
        raise PermissionError("physical G1 deployment contract requires DDS domain 0")
    assert_real_mode_allowed(
        real=args.real,
        acknowledge_hardware_risk=args.acknowledge_hardware_risk,
        interface=args.interface,
        policy_path=args.policy.resolve(),
    )
    controller = ROOT / "deploy" / "generated" / "g1_controller"
    if not controller.is_file():
        raise FileNotFoundError("C++ controller is not built; run scripts/setup/build_controller.sh")
    command = [
        str(controller),
        "--policy",
        str(args.policy.resolve()),
        "--real",
        "--acknowledge-hardware-risk",
        "--interface",
        args.interface,
        "--domain-id",
        "0",
        "--hardware-profile",
        str(ROOT / "configs" / "robot" / "g1_variant.yaml"),
    ]
    if args.stand:
        command.append("--stand")
    if args.run:
        command.append("--run")
    # Replaces this process only after all checksum/contract/profile guards pass.
    # With no --stand/--run the C++ process receives LowState read-only and does
    # not construct a LowCmd publisher.
    os.execv(command[0], command)


if __name__ == "__main__":
    try:
        main()
    except RealModeDenied as error:
        print(f"REAL MODE DENIED: {error}", file=sys.stderr)
        raise SystemExit(2) from None
