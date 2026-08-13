"""Hard gate that runs before any DDS publisher can be constructed."""

from __future__ import annotations

from pathlib import Path
import json
import socket

import yaml

from humanoid_g1.assets.joint_contract import load_joint_contract
from humanoid_g1.deployment.policy_runtime import verify_bundle_checksums
from humanoid_g1.utils.paths import CONFIGS


class RealModeDenied(PermissionError):
    """Physical LowCmd transport was rejected before publisher construction."""


def assert_real_mode_allowed(
    *,
    real: bool,
    acknowledge_hardware_risk: bool,
    interface: str | None,
    policy_path: Path,
    hardware_profile_path: Path = CONFIGS / "robot" / "g1_variant.yaml",
) -> None:
    failures = []
    if not real:
        failures.append("--real is missing")
    if not acknowledge_hardware_risk:
        failures.append("--acknowledge-hardware-risk is missing")
    if not interface or interface == "lo":
        failures.append("an explicit non-loopback --interface is required")
    elif not (Path("/sys/class/net") / interface).exists():
        failures.append(f"network interface does not exist: {interface}")
    profile = yaml.safe_load(hardware_profile_path.read_text(encoding="utf-8"))
    hardware = profile.get("hardware", {})
    if not hardware.get("verified") or not hardware.get("motor_index_verified"):
        failures.append("hardware profile and motor index are not verified")
    contract = load_joint_contract()
    try:
        contract.validate_urdf()
        contract.validate_mujoco()
    except Exception as exc:
        failures.append(f"joint contract failed: {exc}")
    try:
        verify_bundle_checksums(policy_path.parent)
        metadata = json.loads((policy_path.parent / "policy_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("output_dimension") != 29:
            failures.append("policy output dimension is not 29")
    except Exception as exc:
        failures.append(f"policy verification failed: {exc}")
    if failures:
        raise RealModeDenied("LowCmd publisher NOT created: " + "; ".join(failures))
