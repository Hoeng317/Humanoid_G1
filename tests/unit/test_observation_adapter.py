import numpy as np

from humanoid_g1.assets.joint_contract import load_joint_contract
from humanoid_g1.deployment.observation_adapter import (
    LowState,
    ObservationAdapter,
    projected_gravity_from_quaternion,
)


def safe_state():
    contract = load_joint_contract()
    positions = contract.reorder_policy_to_sdk([joint.default for joint in contract.joints])
    return LowState.from_arrays(
        1.0, positions, [0.0] * 29, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    )


def test_identity_orientation_projects_downward_gravity():
    np.testing.assert_allclose(projected_gravity_from_quaternion(np.array([1, 0, 0, 0])), [0, 0, -1])


def test_observation_schema_and_initial_history():
    adapter = ObservationAdapter()
    observation = adapter.build(safe_state(), [0.1, 0.0, -0.1], [0.0] * 29)
    assert observation.shape == (480,)
    frames = observation.reshape(5, 96)
    for frame in frames[1:]:
        np.testing.assert_array_equal(frame, frames[0])
    # actor excludes base linear velocity: first terms are gyro, gravity, command.
    np.testing.assert_allclose(frames[0][0:3], 0.0)
    np.testing.assert_allclose(frames[0][3:6], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(frames[0][6:9], [0.1, 0.0, -0.1])

