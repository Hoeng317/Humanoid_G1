"""Build the exact actor observation used by the Isaac Lab task."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable
import math

import numpy as np

from humanoid_g1.assets.joint_contract import JointContract, load_joint_contract


@dataclass(frozen=True)
class LowState:
    """Hardware-neutral subset of Unitree ``LowState_`` in SDK ordering."""

    timestamp_s: float
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    quaternion_wxyz: np.ndarray
    angular_velocity: np.ndarray
    remote_estop: bool = False

    @classmethod
    def from_arrays(
        cls,
        timestamp_s: float,
        joint_position: Iterable[float],
        joint_velocity: Iterable[float],
        quaternion_wxyz: Iterable[float],
        angular_velocity: Iterable[float],
        remote_estop: bool = False,
    ) -> "LowState":
        return cls(
            timestamp_s=float(timestamp_s),
            joint_position=np.asarray(list(joint_position), dtype=np.float32),
            joint_velocity=np.asarray(list(joint_velocity), dtype=np.float32),
            quaternion_wxyz=np.asarray(list(quaternion_wxyz), dtype=np.float32),
            angular_velocity=np.asarray(list(angular_velocity), dtype=np.float32),
            remote_estop=bool(remote_estop),
        )


def projected_gravity_from_quaternion(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Rotate world gravity ``[0, 0, -1]`` into the robot body frame."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must contain four finite values in wxyz order")
    norm = np.linalg.norm(quaternion)
    if norm < 1.0e-8:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = quaternion / norm
    # Third row of body-to-world rotation dotted with world gravity.
    return np.asarray(
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), -(1.0 - 2.0 * (x * x + y * y))],
        dtype=np.float32,
    )


class ObservationAdapter:
    """Stateful 5-frame actor observation adapter (96 × 5 = 480)."""

    single_step_dim = 96
    observation_dim = 480

    def __init__(self, contract: JointContract | None = None, history_length: int = 5):
        self.contract = contract or load_joint_contract()
        if history_length != 5:
            raise ValueError("deployment contract requires history_length=5")
        self.history_length = history_length
        self._history: deque[np.ndarray] = deque(maxlen=history_length)

    def reset(self) -> None:
        self._history.clear()

    def build(
        self,
        state: LowState,
        command_xyz: Iterable[float],
        last_action_policy: Iterable[float],
    ) -> np.ndarray:
        self.contract.validate_vector(state.joint_position, "LowState joint_position")
        self.contract.validate_vector(state.joint_velocity, "LowState joint_velocity")
        command = np.asarray(list(command_xyz), dtype=np.float32)
        last_action = np.asarray(list(last_action_policy), dtype=np.float32)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("command must be three finite values")
        self.contract.validate_vector(last_action, "last_action")
        if state.angular_velocity.shape != (3,) or not np.isfinite(state.angular_velocity).all():
            raise ValueError("angular_velocity must be three finite values")
        q_policy = np.asarray(self.contract.reorder_sdk_to_policy(state.joint_position), dtype=np.float32)
        dq_policy = np.asarray(self.contract.reorder_sdk_to_policy(state.joint_velocity), dtype=np.float32)
        defaults = np.asarray([joint.default for joint in self.contract.joints], dtype=np.float32)
        frame = np.concatenate(
            (
                0.2 * state.angular_velocity,
                projected_gravity_from_quaternion(state.quaternion_wxyz),
                command,
                q_policy - defaults,
                0.05 * dq_policy,
                last_action,
            )
        ).astype(np.float32, copy=False)
        if frame.shape != (self.single_step_dim,) or not np.isfinite(frame).all():
            raise ValueError("actor observation frame is invalid")
        if not self._history:
            for _ in range(self.history_length):
                self._history.append(frame.copy())
        else:
            self._history.append(frame)
        observation = np.concatenate(tuple(self._history))
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(f"observation dimension mismatch: {observation.shape}")
        return observation

