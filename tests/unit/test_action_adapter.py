import numpy as np

from humanoid_g1.deployment.action_adapter import ActionAdapter


def test_action_clipping_rate_limit_and_sdk_order():
    adapter = ActionAdapter(action_clip=1.0, delta_limit=0.15)
    result = adapter.apply(np.full(29, 2.0))
    np.testing.assert_allclose(result.clipped_policy, 0.15)
    assert result.saturated_count == 29
    assert result.target_policy.shape == (29,)
    assert result.target_sdk.shape == (29,)


def test_target_stays_inside_soft_joint_limits():
    adapter = ActionAdapter(delta_limit=100.0)
    result = adapter.apply(np.linspace(-10, 10, 29))
    for value, joint in zip(result.target_policy, adapter.contract.joints):
        assert joint.lower + 0.05 <= value <= joint.upper - 0.05

