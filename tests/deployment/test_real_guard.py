from pathlib import Path

import pytest

from humanoid_g1.deployment.real_guard import RealModeDenied, assert_real_mode_allowed


def test_real_guard_rejects_without_flags_before_dds_import(tmp_path: Path):
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"not-a-policy")
    profile = tmp_path / "hardware.yaml"
    profile.write_text("hardware:\n  verified: false\n  motor_index_verified: false\n", encoding="utf-8")
    with pytest.raises(RealModeDenied, match="publisher NOT created"):
        assert_real_mode_allowed(
            real=False,
            acknowledge_hardware_risk=False,
            interface="lo",
            policy_path=policy,
            hardware_profile_path=profile,
        )

