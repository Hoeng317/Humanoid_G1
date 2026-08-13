from pathlib import Path
import hashlib

import pytest

from humanoid_g1.deployment.policy_runtime import PolicyError, verify_bundle_checksums


def test_policy_bundle_checksum_detects_corruption(tmp_path: Path):
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"verified")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  policy.pt\n", encoding="utf-8")
    verify_bundle_checksums(tmp_path)
    policy.write_bytes(b"corrupt")
    with pytest.raises(PolicyError, match="checksum mismatch"):
        verify_bundle_checksums(tmp_path)
