import numpy as np
import pytest

from humanoid_g1.assets.joint_contract import load_joint_contract
from humanoid_g1.deployment.observation_adapter import LowState
from humanoid_g1.deployment.safety_state_machine import (
    ControllerState,
    SafetyFault,
    SafetyGate,
    SafetyStateMachine,
)
from humanoid_g1.deployment.watchdog import Watchdog


def safe_state(timestamp=1.0):
    contract = load_joint_contract()
    q = contract.reorder_policy_to_sdk([joint.default for joint in contract.joints])
    return LowState.from_arrays(timestamp, q, [0.0] * 29, [1.0, 0.0, 0.0, 0.0], [0.0] * 3)


def test_policy_running_needs_explicit_operator_transition():
    machine = SafetyStateMachine()
    machine.transition(ControllerState.IDLE)
    machine.transition(ControllerState.DAMPING)
    machine.transition(ControllerState.MOVE_TO_DEFAULT)
    machine.transition(ControllerState.STAND_READY)
    machine.transition(ControllerState.POLICY_READY)
    with pytest.raises(SafetyFault, match="explicit"):
        machine.transition(ControllerState.POLICY_RUNNING)
    machine.transition(ControllerState.POLICY_RUNNING, explicit=True)


def test_stale_state_and_estop_fault():
    gate = SafetyGate()
    with pytest.raises(SafetyFault, match="timeout"):
        gate.validate_state(safe_state(1.0), now_s=2.0)
    state = safe_state(1.0)
    estop = LowState(
        state.timestamp_s,
        state.joint_position,
        state.joint_velocity,
        state.quaternion_wxyz,
        state.angular_velocity,
        True,
    )
    with pytest.raises(SafetyFault, match="emergency"):
        gate.validate_state(estop, now_s=1.0)


def test_watchdog():
    watchdog = Watchdog(0.1)
    assert watchdog.expired(1.0)
    watchdog.kick(1.0)
    assert not watchdog.expired(1.05)
    assert watchdog.expired(1.2)

