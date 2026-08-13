#!/usr/bin/env python3
"""Offline asset inspection; does not start Isaac Sim."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from humanoid_g1.assets.joint_contract import load_joint_contract
from humanoid_g1.utils.paths import ARTIFACTS, MUJOCO_MODEL_PATH, URDF_PATH


def main() -> None:
    contract = load_joint_contract()
    contract.validate_urdf()
    contract.validate_mujoco()
    root = ET.parse(URDF_PATH).getroot()
    links = root.findall("link")
    collisions = root.findall(".//collision")
    visuals = root.findall(".//visual")
    report = {
        "manufacturer": "Unitree",
        "model": "G1",
        "variant": contract.variant,
        "asset_source": str(URDF_PATH),
        "mujoco_source": str(MUJOCO_MODEL_PATH),
        "robot_prim_path": "{ENV_REGEX_NS}/Robot",
        "joint_count": contract.action_dim,
        "controlled_joint_count": contract.action_dim,
        "joint_names_policy_order": contract.policy_names,
        "joint_names_articulation_order": contract.sdk_names,
        "rigid_body_count_from_urdf": len(links),
        "rigid_body_names_from_urdf": [item.attrib["name"] for item in links],
        "actuator_groups": ["N7520-14.3", "N7520-22.5", "N5020-16-arms", "N5020-16-ankle-waist", "W4010-25"],
        "self_collisions": True,
        "contact_sensor_pattern": "{ENV_REGEX_NS}/Robot/.*",
        "physics_dt_s": contract.physics_dt,
        "control_decimation": contract.decimation,
        "policy_frequency_hz": 1.0 / contract.policy_dt,
        "joints": contract.to_dict()["joints"],
    }
    output = ARTIFACTS / "contracts" / "g1_asset_inspection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (ARTIFACTS / "reports" / "G1_ASSET_REPORT.md").write_text(
        "# Unitree G1 asset report\n\n"
        f"- Variant: {contract.variant}\n"
        f"- URDF: {URDF_PATH}\n"
        f"- MuJoCo: {MUJOCO_MODEL_PATH}\n"
        f"- Rigid bodies (URDF): {len(links)}\n"
        f"- Controlled body joints: {contract.action_dim}\n"
        f"- Physics/policy frequency: {1 / contract.physics_dt:.0f}/{1 / contract.policy_dt:.0f} Hz\n"
        "- Hands: optional separate subsystem; excluded from 29-DoF policy\n",
        encoding="utf-8",
    )
    print("확인 결과: 공식 Unitree G1 29-DoF rev_1_0 모델입니다.")
    print(f"URDF          : {URDF_PATH}")
    print(f"MuJoCo model  : {MUJOCO_MODEL_PATH}")
    print(f"links/joints  : {len(links)}/{contract.action_dim} revolute body joints")
    print(f"visual/collision elements: {len(visuals)}/{len(collisions)}")
    print("hands          : body policy에서 제외; 별도 optional subsystem")
    print(f"action order   : {', '.join(contract.policy_names)}")
    print(f"policy -> SDK  : {contract.policy_to_sdk}")
    print(f"inspection JSON: {output}")


if __name__ == "__main__":
    main()
