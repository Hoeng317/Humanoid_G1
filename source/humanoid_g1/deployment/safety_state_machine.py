"""Explicit controller states and safety validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math
import time

import numpy as np

from humanoid_g1.assets.joint_contract import JointContract, load_joint_contract
from humanoid_g1.deployment.observation_adapter import LowState, projected_gravity_from_quaternion


class ControllerState(Enum):
    DISCONNECTED = auto()
    IDLE = auto()
    DAMPING = auto()
    ZERO_TORQUE = auto()
    MOVE_TO_DEFAULT = auto()
    STAND_READY = auto()
    POLICY_READY = auto()
    POLICY_RUNNING = auto()
    STOPPING = auto()
    FAULT = auto()


class SafetyFault(RuntimeError):
    """A controller input or timing condition requires an immediate FAULT."""


_TRANSITIONS = {
    ControllerState.DISCONNECTED: {ControllerState.IDLE, ControllerState.FAULT},
    ControllerState.IDLE: {ControllerState.DAMPING, ControllerState.ZERO_TORQUE, ControllerState.FAULT},
    ControllerState.DAMPING: {ControllerState.MOVE_TO_DEFAULT, ControllerState.STOPPING, ControllerState.FAULT},
    ControllerState.ZERO_TORQUE: {ControllerState.DAMPING, ControllerState.STOPPING, ControllerState.FAULT},
    ControllerState.MOVE_TO_DEFAULT: {ControllerState.STAND_READY, ControllerState.STOPPING, ControllerState.FAULT},
    ControllerState.STAND_READY: {ControllerState.POLICY_READY, ControllerState.STOPPING, ControllerState.FAULT},
    ControllerState.POLICY_READY: {ControllerState.POLICY_RUNNING, ControllerState.STOPPING, ControllerState.FAULT},
    ControllerState.POLICY_RUNNING: {ControllerState.STOPPING, ControllerState.FAULT},
    ControllerState.STOPPING: {ControllerState.DAMPING, ControllerState.IDLE, ControllerState.FAULT},
    ControllerState.FAULT: {ControllerState.DAMPING, ControllerState.IDLE},
}


class SafetyStateMachine:
    def __init__(self):
        self.state = ControllerState.DISCONNECTED
        self.reason = ""

    def transition(self, target: ControllerState, explicit: bool = False) -> None:
        if target not in _TRANSITIONS[self.state]:
            raise SafetyFault(f"invalid transition {self.state.name} -> {target.name}")
        if target == ControllerState.POLICY_RUNNING and not explicit:
            raise SafetyFault("POLICY_RUNNING always requires explicit operator input")
        self.state = target
        self.reason = ""

    def fault(self, reason: str) -> None:
        self.state = ControllerState.FAULT
        self.reason = reason


@dataclass
class SafetyLimits:
    low_state_timeout_s: float = 0.1
    policy_timeout_s: float = 0.01
    loop_overrun_s: float = 0.03
    max_roll_rad: float = 0.7
    max_pitch_rad: float = 0.7
    target_margin_rad: float = 0.05


class SafetyGate:
    def __init__(self, contract: JointContract | None = None, limits: SafetyLimits | None = None):
        self.contract = contract or load_joint_contract()
        self.limits = limits or SafetyLimits()

    def validate_state(self, state: LowState, now_s: float | None = None, network_connected: bool = True) -> None:
        now_s = time.monotonic() if now_s is None else now_s
        if not network_connected:
            raise SafetyFault("network disconnect")
        if state.remote_estop:
            raise SafetyFault("remote emergency stop")
        if now_s - state.timestamp_s > self.limits.low_state_timeout_s:
            raise SafetyFault("LowState timeout or stale DDS packet")
        self.contract.validate_vector(state.joint_position, "joint position")
        self.contract.validate_vector(state.joint_velocity, "joint velocity")
        quaternion = np.asarray(state.quaternion_wxyz)
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise SafetyFault("invalid IMU quaternion")
        norm = float(np.linalg.norm(quaternion))
        if not 0.8 <= norm <= 1.2:
            raise SafetyFault("invalid IMU quaternion norm")
        for sdk_index, joint in enumerate(sorted(self.contract.joints, key=lambda item: item.sdk_index)):
            q = float(state.joint_position[sdk_index])
            dq = float(state.joint_velocity[sdk_index])
            if q < joint.lower or q > joint.upper:
                raise SafetyFault(f"joint position limit: {joint.name}")
            if abs(dq) > joint.velocity:
                raise SafetyFault(f"joint velocity limit: {joint.name}")
        gravity = projected_gravity_from_quaternion(quaternion)
        roll = math.atan2(float(gravity[1]), -float(gravity[2]))
        pitch = math.atan2(-float(gravity[0]), math.hypot(float(gravity[1]), float(gravity[2])))
        if abs(roll) > self.limits.max_roll_rad or abs(pitch) > self.limits.max_pitch_rad:
            raise SafetyFault("body orientation limit")

    def validate_action(self, action: np.ndarray, target_policy: np.ndarray, inference_s: float) -> None:
        self.contract.validate_vector(action, "action")
        self.contract.validate_vector(target_policy, "target position")
        if inference_s > self.limits.policy_timeout_s:
            raise SafetyFault("policy inference timeout")
        for value, joint in zip(target_policy, self.contract.joints):
            if not joint.lower + self.limits.target_margin_rad <= value <= joint.upper - self.limits.target_margin_rad:
                raise SafetyFault(f"target position limit: {joint.name}")

    def validate_loop_time(self, elapsed_s: float) -> None:
        if elapsed_s > self.limits.loop_overrun_s:
            raise SafetyFault("control loop overrun")
