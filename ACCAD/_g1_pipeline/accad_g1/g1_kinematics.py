"""Differentiable Unitree G1 29-DoF forward kinematics from the project URDF.

The public trajectory contract uses the interleaved ``policy_index`` order in
``configs/robot/g1_29dof.yaml``.  The URDF and Isaac articulation instead use
SDK/URDF order.  This module resolves every revolute joint by *name*, so those
orders can never be confused silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from .paths import PROJECT_ROOT


G1_CONTRACT_PATH = PROJECT_ROOT / "configs" / "robot" / "g1_29dof.yaml"
G1_URDF_PATH = (
    PROJECT_ROOT
    / "third_party"
    / "unitree_ros"
    / "robots"
    / "g1_description"
    / "g1_29dof_rev_1_0.urdf"
)
FOOT_SOLE_OFFSET = {
    "left_ankle_roll_link": np.asarray([0.035, 0.0, -0.03], dtype=np.float64),
    "right_ankle_roll_link": np.asarray([0.035, 0.0, -0.03], dtype=np.float64),
}
FOOT_SUPPORT_OFFSETS = {
    "left_ankle_roll_link": np.asarray(
        [
            [-0.05, 0.025, -0.03],
            [-0.05, -0.025, -0.03],
            [0.12, 0.030, -0.03],
            [0.12, -0.030, -0.03],
        ],
        dtype=np.float64,
    ),
    "right_ankle_roll_link": np.asarray(
        [
            [-0.05, 0.025, -0.03],
            [-0.05, -0.025, -0.03],
            [0.12, 0.030, -0.03],
            [0.12, -0.030, -0.03],
        ],
        dtype=np.float64,
    ),
}


@dataclass(frozen=True)
class G1Contract:
    policy_names: tuple[str, ...]
    sdk_names: tuple[str, ...]
    policy_to_sdk: tuple[int, ...]
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    default: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    action_scale: float
    physics_dt: float
    decimation: int
    variant: str

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.decimation


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray | None


def load_g1_contract(path: str | Path = G1_CONTRACT_PATH) -> G1Contract:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    joints = sorted(payload["joints"], key=lambda item: int(item["policy_index"]))
    if [int(item["policy_index"]) for item in joints] != list(range(29)):
        raise ValueError("G1 policy indices must be contiguous 0..28")
    policy_names = tuple(str(item["name"]) for item in joints)
    policy_to_sdk = tuple(int(item["sdk_index"]) for item in joints)
    if sorted(policy_to_sdk) != list(range(29)) or len(set(policy_names)) != 29:
        raise ValueError("G1 SDK indices/names must be unique permutations")
    sdk_names_list = [""] * 29
    for policy_name, sdk_index in zip(policy_names, policy_to_sdk, strict=True):
        sdk_names_list[sdk_index] = policy_name

    def values(name: str) -> np.ndarray:
        return np.asarray([float(item[name]) for item in joints], dtype=np.float64)

    timing = payload["timing"]
    contract = G1Contract(
        policy_names=policy_names,
        sdk_names=tuple(sdk_names_list),
        policy_to_sdk=policy_to_sdk,
        lower=values("lower"),
        upper=values("upper"),
        velocity=values("velocity"),
        effort=values("effort"),
        default=values("default"),
        kp=values("kp"),
        kd=values("kd"),
        action_scale=float(payload["control"]["action_scale"]),
        physics_dt=float(timing["physics_dt"]),
        decimation=int(timing["decimation"]),
        variant=str(payload["robot"]["variant"]),
    )
    if contract.variant != "g1_29dof_rev_1_0":
        raise ValueError(f"unsupported G1 variant: {contract.variant}")
    if not np.all((contract.lower <= contract.default) & (contract.default <= contract.upper)):
        raise ValueError("G1 default pose violates a hard joint limit")
    return contract


def policy_to_sdk(values: np.ndarray, contract: G1Contract) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1] != 29:
        raise ValueError(f"policy vector must end in 29 values; got {array.shape}")
    result = np.empty_like(array)
    result[..., np.asarray(contract.policy_to_sdk)] = array
    return result


def sdk_to_policy(values: np.ndarray, contract: G1Contract) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1] != 29:
        raise ValueError(f"SDK vector must end in 29 values; got {array.shape}")
    return array[..., np.asarray(contract.policy_to_sdk)]


def _xyz(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(
        [float(value) for value in text.split()] if text else default,
        dtype=np.float64,
    )


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def parse_urdf_joints(path: str | Path = G1_URDF_PATH) -> tuple[str, tuple[str, ...], tuple[UrdfJoint, ...]]:
    root = ET.parse(path).getroot()
    link_names = tuple(str(element.attrib["name"]) for element in root.findall("link"))
    parsed: list[UrdfJoint] = []
    child_links: set[str] = set()
    for element in root.findall("joint"):
        kind = str(element.attrib["type"])
        if kind not in {"fixed", "revolute", "continuous"}:
            raise ValueError(f"unsupported URDF joint type {kind!r}")
        parent = str(element.find("parent").attrib["link"])
        child = str(element.find("child").attrib["link"])
        child_links.add(child)
        origin_element = element.find("origin")
        xyz = _xyz(
            origin_element.attrib.get("xyz") if origin_element is not None else None,
            (0.0, 0.0, 0.0),
        )
        rpy = _xyz(
            origin_element.attrib.get("rpy") if origin_element is not None else None,
            (0.0, 0.0, 0.0),
        )
        rotation = Rotation.from_euler("xyz", rpy).as_matrix()
        axis_element = element.find("axis")
        axis = None
        if kind != "fixed":
            axis = _xyz(
                axis_element.attrib.get("xyz") if axis_element is not None else None,
                (1.0, 0.0, 0.0),
            )
            norm = float(np.linalg.norm(axis))
            if norm <= 0.0:
                raise ValueError(f"zero axis for URDF joint {element.attrib['name']}")
            axis = axis / norm
        parsed.append(
            UrdfJoint(
                name=str(element.attrib["name"]),
                kind=kind,
                parent=parent,
                child=child,
                origin=_homogeneous(rotation, xyz),
                axis=axis,
            )
        )
    root_links = sorted(set(link_names) - child_links)
    if root_links != ["pelvis"]:
        raise ValueError(f"expected pelvis as the only URDF root; got {root_links}")

    # Convert document order into an explicit parent-before-child traversal.
    remaining = list(parsed)
    ordered: list[UrdfJoint] = []
    available = {"pelvis"}
    while remaining:
        progressed = False
        for joint in list(remaining):
            if joint.parent in available:
                ordered.append(joint)
                available.add(joint.child)
                remaining.remove(joint)
                progressed = True
        if not progressed:
            raise ValueError("URDF joint graph is disconnected or cyclic")
    if available != set(link_names):
        raise ValueError("URDF traversal did not cover every link")
    return "pelvis", link_names, tuple(ordered)


def _torch_transform(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    leading = torch.broadcast_shapes(rotation.shape[:-2], translation.shape[:-1])
    rotation = rotation.expand(leading + (3, 3))
    translation = translation.expand(leading + (3,))
    top = torch.cat((rotation, translation.unsqueeze(-1)), dim=-1)
    bottom = torch.zeros(leading + (1, 4), dtype=rotation.dtype, device=rotation.device)
    bottom[..., 0, 3] = 1.0
    return torch.cat((top, bottom), dim=-2)


def quaternion_wxyz_to_matrix_torch(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError(f"quaternion must end in 4 values; got {quaternion.shape}")
    q = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-12)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion_wxyz(matrices: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrices, dtype=np.float64)
    shape = matrix.shape[:-2]
    xyzw = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_quat().reshape(shape + (4,))
    result = xyzw[..., [3, 0, 1, 2]]
    for frame in range(1, result.shape[0] if result.ndim >= 2 else 1):
        flip = np.sum(result[frame - 1] * result[frame], axis=-1, keepdims=True) < 0.0
        result[frame] = np.where(flip, -result[frame], result[frame])
    return np.asarray(result, dtype=np.float32)


class TorchG1Kinematics:
    """Name-resolved batched FK with gradients with respect to policy-order q."""

    def __init__(
        self,
        contract_path: str | Path = G1_CONTRACT_PATH,
        urdf_path: str | Path = G1_URDF_PATH,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.contract = load_g1_contract(contract_path)
        self.root_link, self.link_names, self.joints = parse_urdf_joints(urdf_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.policy_index = {
            name: index for index, name in enumerate(self.contract.policy_names)
        }
        urdf_revolute = tuple(joint.name for joint in self.joints if joint.kind != "fixed")
        if urdf_revolute != self.contract.sdk_names:
            raise ValueError("URDF revolute order does not match the G1 SDK contract")
        self._origins = {
            joint.name: torch.as_tensor(
                joint.origin, dtype=dtype, device=self.device
            )
            for joint in self.joints
        }
        self._axes = {
            joint.name: (
                torch.as_tensor(joint.axis, dtype=dtype, device=self.device)
                if joint.axis is not None
                else None
            )
            for joint in self.joints
        }

    def _axis_rotation(self, angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        x, y, z = axis.unbind(dim=-1)
        zero = torch.zeros((), dtype=self.dtype, device=self.device)
        skew = torch.stack(
            (zero, -z, y, z, zero, -x, -y, x, zero)
        ).reshape(3, 3)
        identity = torch.eye(3, dtype=self.dtype, device=self.device)
        return (
            identity
            + torch.sin(angle)[..., None, None] * skew
            + (1.0 - torch.cos(angle))[..., None, None] * (skew @ skew)
        )

    def forward(
        self,
        joint_position_policy: torch.Tensor,
        root_position_w: torch.Tensor,
        root_quaternion_wxyz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        q = joint_position_policy.to(device=self.device, dtype=self.dtype)
        root_pos = root_position_w.to(device=self.device, dtype=self.dtype)
        root_quat = root_quaternion_wxyz.to(device=self.device, dtype=self.dtype)
        if q.shape[-1] != 29 or root_pos.shape[-1] != 3 or root_quat.shape[-1] != 4:
            raise ValueError("FK expects q[...,29], root position[...,3], quaternion[...,4]")
        leading = torch.broadcast_shapes(q.shape[:-1], root_pos.shape[:-1], root_quat.shape[:-1])
        q = q.expand(leading + (29,))
        root_pos = root_pos.expand(leading + (3,))
        root_quat = root_quat.expand(leading + (4,))
        transforms: dict[str, torch.Tensor] = {
            self.root_link: _torch_transform(
                quaternion_wxyz_to_matrix_torch(root_quat), root_pos
            )
        }
        for joint in self.joints:
            parent = transforms[joint.parent]
            origin = self._origins[joint.name].expand(leading + (4, 4))
            if joint.kind == "fixed":
                motion = torch.eye(4, dtype=self.dtype, device=self.device).expand(
                    leading + (4, 4)
                )
            else:
                angle = q[..., self.policy_index[joint.name]]
                rotation = self._axis_rotation(angle, self._axes[joint.name])
                translation = torch.zeros(leading + (3,), dtype=self.dtype, device=self.device)
                motion = _torch_transform(rotation, translation)
            transforms[joint.child] = parent @ origin @ motion
        return transforms

    def link_positions(
        self, transforms: dict[str, torch.Tensor], names: list[str] | tuple[str, ...]
    ) -> torch.Tensor:
        missing = [name for name in names if name not in transforms]
        if missing:
            raise KeyError(f"unknown G1 link names: {missing}")
        return torch.stack([transforms[name][..., :3, 3] for name in names], dim=-2)

    def link_rotations(
        self, transforms: dict[str, torch.Tensor], names: list[str] | tuple[str, ...]
    ) -> torch.Tensor:
        missing = [name for name in names if name not in transforms]
        if missing:
            raise KeyError(f"unknown G1 link names: {missing}")
        return torch.stack([transforms[name][..., :3, :3] for name in names], dim=-3)

    def sole_positions(self, transforms: dict[str, torch.Tensor]) -> torch.Tensor:
        positions = []
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            offset = torch.as_tensor(
                FOOT_SOLE_OFFSET[name], dtype=self.dtype, device=self.device
            )
            transform = transforms[name]
            positions.append(
                transform[..., :3, 3] + transform[..., :3, :3] @ offset
            )
        return torch.stack(positions, dim=-2)

    def foot_support_positions(self, transforms: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return collision-sphere centers as ``[..., foot=2, point=4, xyz=3]``."""

        positions = []
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            offsets = torch.as_tensor(
                FOOT_SUPPORT_OFFSETS[name], dtype=self.dtype, device=self.device
            )
            transform = transforms[name]
            rotated = torch.einsum("...ij,pj->...pi", transform[..., :3, :3], offsets)
            positions.append(transform[..., None, :3, 3] + rotated)
        return torch.stack(positions, dim=-3)

    def stacked(
        self,
        transforms: dict[str, torch.Tensor],
        names: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
        selected = names or self.link_names
        return (
            selected,
            self.link_positions(transforms, selected),
            self.link_rotations(transforms, selected),
        )


def fk_numpy(
    joint_position_policy: np.ndarray,
    root_position_w: np.ndarray,
    root_quaternion_wxyz: np.ndarray,
    *,
    link_names: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    solver = TorchG1Kinematics(device="cpu", dtype=torch.float64)
    with torch.inference_mode():
        transforms = solver.forward(
            torch.as_tensor(joint_position_policy, dtype=torch.float64),
            torch.as_tensor(root_position_w, dtype=torch.float64),
            torch.as_tensor(root_quaternion_wxyz, dtype=torch.float64),
        )
        names, positions, rotations = solver.stacked(transforms, link_names)
    return names, positions.numpy(), rotations.numpy()


def contract_to_dict(contract: G1Contract) -> dict[str, Any]:
    return {
        "variant": contract.variant,
        "policy_names": list(contract.policy_names),
        "sdk_names": list(contract.sdk_names),
        "policy_to_sdk": list(contract.policy_to_sdk),
        "lower": contract.lower.tolist(),
        "upper": contract.upper.tolist(),
        "velocity": contract.velocity.tolist(),
        "effort": contract.effort.tolist(),
        "default": contract.default.tolist(),
        "kp": contract.kp.tolist(),
        "kd": contract.kd.tolist(),
        "action_scale": contract.action_scale,
        "physics_dt": contract.physics_dt,
        "decimation": contract.decimation,
        "control_dt": contract.control_dt,
    }
