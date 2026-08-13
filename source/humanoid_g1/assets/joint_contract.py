"""Authoritative 29-DoF ordering and cross-engine mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import math
import xml.etree.ElementTree as ET

import yaml

from humanoid_g1.utils.paths import CONTRACT_PATH, MUJOCO_MODEL_PATH, URDF_PATH


class ContractError(ValueError):
    """Raised when an asset or runtime vector violates the joint contract."""


@dataclass(frozen=True)
class Joint:
    policy_index: int
    sdk_index: int
    name: str
    lower: float
    upper: float
    velocity: float
    effort: float
    default: float
    kp: float
    kd: float


@dataclass(frozen=True)
class JointContract:
    variant: str
    joints: tuple[Joint, ...]
    action_scale: float
    physics_dt: float
    decimation: int

    @property
    def action_dim(self) -> int:
        return len(self.joints)

    @property
    def policy_dt(self) -> float:
        return self.physics_dt * self.decimation

    @property
    def policy_names(self) -> list[str]:
        return [joint.name for joint in self.joints]

    @property
    def sdk_names(self) -> list[str]:
        return [joint.name for joint in sorted(self.joints, key=lambda item: item.sdk_index)]

    @property
    def policy_to_sdk(self) -> list[int]:
        return [joint.sdk_index for joint in self.joints]

    @property
    def sdk_to_policy(self) -> list[int]:
        inverse = [0] * self.action_dim
        for joint in self.joints:
            inverse[joint.sdk_index] = joint.policy_index
        return inverse

    def reorder_policy_to_sdk(self, values: Iterable[float]) -> list[float]:
        source = list(values)
        self.validate_vector(source, "policy vector")
        result = [0.0] * self.action_dim
        for policy_index, sdk_index in enumerate(self.policy_to_sdk):
            result[sdk_index] = source[policy_index]
        return result

    def reorder_sdk_to_policy(self, values: Iterable[float]) -> list[float]:
        source = list(values)
        self.validate_vector(source, "SDK vector")
        return [source[index] for index in self.policy_to_sdk]

    def validate_vector(self, values: Iterable[float], label: str) -> None:
        vector = list(values)
        if len(vector) != self.action_dim:
            raise ContractError(f"{label} must contain {self.action_dim} values, got {len(vector)}")
        if not all(math.isfinite(float(value)) for value in vector):
            raise ContractError(f"{label} contains NaN or Inf")

    def validate(self) -> None:
        if self.variant != "g1_29dof_rev_1_0":
            raise ContractError(f"unsupported G1 variant: {self.variant}")
        if self.action_dim != 29:
            raise ContractError(f"G1 body action dimension must be 29, got {self.action_dim}")
        if [item.policy_index for item in self.joints] != list(range(29)):
            raise ContractError("policy indices must be contiguous 0..28")
        if sorted(item.sdk_index for item in self.joints) != list(range(29)):
            raise ContractError("SDK indices must be a permutation of 0..28")
        if len(set(self.policy_names)) != 29:
            raise ContractError("joint names must be unique")
        for item in self.joints:
            if not item.lower <= item.default <= item.upper:
                raise ContractError(f"default pose outside limit for {item.name}")
            if min(item.velocity, item.effort, item.kp, item.kd) < 0:
                raise ContractError(f"negative motor parameter for {item.name}")
        if self.physics_dt <= 0 or self.decimation <= 0:
            raise ContractError("physics_dt and decimation must be positive")

    def validate_urdf(self, path: Path = URDF_PATH) -> None:
        root = ET.parse(path).getroot()
        urdf_order = [
            element.attrib["name"]
            for element in root.findall("joint")
            if element.attrib.get("type") in {"revolute", "continuous"}
        ]
        if urdf_order != self.sdk_names:
            raise ContractError("URDF/Isaac source joint order does not match SDK joint order")
        urdf = {
            element.attrib["name"]: element
            for element in root.findall("joint")
            if element.attrib.get("type") in {"revolute", "continuous"}
        }
        missing = sorted(set(self.policy_names) - set(urdf))
        extra = sorted(set(urdf) - set(self.policy_names))
        if missing or extra:
            raise ContractError(f"URDF mismatch; missing={missing}, extra={extra}")
        for joint in self.joints:
            limit = urdf[joint.name].find("limit")
            if limit is None:
                raise ContractError(f"URDF joint {joint.name} has no limit")
            for key, expected in (
                ("lower", joint.lower),
                ("upper", joint.upper),
                ("velocity", joint.velocity),
                ("effort", joint.effort),
            ):
                actual = float(limit.attrib[key])
                if abs(actual - expected) > 1.0e-6:
                    raise ContractError(f"{joint.name} {key}: contract={expected}, URDF={actual}")

    def validate_mujoco(self, path: Path = MUJOCO_MODEL_PATH) -> None:
        root = ET.parse(path).getroot()
        actuators = [element.attrib["joint"] for element in root.findall("./actuator/*")]
        if actuators != self.sdk_names:
            raise ContractError("MuJoCo actuator order does not match SDK joint order")

    def to_dict(self) -> dict[str, Any]:
        joints = []
        for item in self.joints:
            joints.append(
                {
                    "policy_index": item.policy_index,
                    "joint_name": item.name,
                    "isaac_index": item.sdk_index,
                    "mujoco_index": item.sdk_index,
                    "sdk_motor_index": item.sdk_index,
                    "default_position_rad": item.default,
                    "lower_limit_rad": item.lower,
                    "upper_limit_rad": item.upper,
                    "velocity_limit_rad_s": item.velocity,
                    "effort_limit_nm": item.effort,
                    "kp": item.kp,
                    "kd": item.kd,
                    "action_scale": self.action_scale,
                }
            )
        return {
            "variant": self.variant,
            "action_dim": self.action_dim,
            "action_scale": self.action_scale,
            "physics_dt": self.physics_dt,
            "decimation": self.decimation,
            "policy_dt": self.policy_dt,
            "policy_to_sdk": self.policy_to_sdk,
            "sdk_to_policy": self.sdk_to_policy,
            "joints": joints,
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_joint_contract(path: Path | str = CONTRACT_PATH) -> JointContract:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    timing = payload["timing"]
    joints = tuple(Joint(**item) for item in payload["joints"])
    contract = JointContract(
        variant=payload["robot"]["variant"],
        joints=joints,
        action_scale=float(payload["control"]["action_scale"]),
        physics_dt=float(timing["physics_dt"]),
        decimation=int(timing["decimation"]),
    )
    contract.validate()
    return contract
