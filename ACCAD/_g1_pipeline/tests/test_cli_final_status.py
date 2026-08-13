from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from accad_g1.cli import _verify_final_artifact_snapshot


def _record(path: Path, root: Path, scope: str) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "scope": scope,
        "bytes": path.stat().st_size,
        "checksum_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_gate5_snapshot_rehashes_unique_artifacts_and_detects_mutation(
    tmp_path: Path,
) -> None:
    project = tmp_path / "humanoid_G1"
    accad = project / "ACCAD"
    local = accad / "_g1_pipeline" / "work" / "policy.pt"
    dependency = project / "configs" / "robot" / "g1.yaml"
    local.parent.mkdir(parents=True)
    dependency.parent.mkdir(parents=True)
    local.write_bytes(b"policy-v1")
    dependency.write_bytes(b"joints: 29\n")
    local_record = _record(local, accad, "accad")
    dependency_record = _record(dependency, project, "project_read_only")
    manifest = {
        "artifact_root": ".",
        "artifact_unique_count": 2,
        "artifacts": {
            "exports": [local_record],
            "reports": [dict(local_record)],
            "external_read_only_dependencies": [dependency_record],
        },
    }

    valid = _verify_final_artifact_snapshot(
        manifest, accad_root=accad, project_root=project
    )
    assert valid == {
        "current": True,
        "verified_unique_artifact_count": 2,
        "declared_unique_artifact_count": 2,
        "errors": [],
    }

    local.write_bytes(b"policy-v2-is-different")
    stale = _verify_final_artifact_snapshot(
        manifest, accad_root=accad, project_root=project
    )
    assert stale["current"] is False
    assert any("size changed" in error for error in stale["errors"])
    assert any("checksum changed" in error for error in stale["errors"])


def test_gate5_snapshot_rejects_path_escape(tmp_path: Path) -> None:
    project = tmp_path / "humanoid_G1"
    accad = project / "ACCAD"
    accad.mkdir(parents=True)
    outside = project / "outside.bin"
    outside.write_bytes(b"outside")
    manifest = {
        "artifact_root": ".",
        "artifact_unique_count": 1,
        "artifacts": {
            "bad": [
                {
                    "path": "../outside.bin",
                    "scope": "accad",
                    "bytes": outside.stat().st_size,
                    "checksum_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                }
            ]
        },
    }

    result = _verify_final_artifact_snapshot(
        manifest, accad_root=accad, project_root=project
    )
    assert result["current"] is False
    assert any("escapes" in error for error in result["errors"])

