from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import accad_g1.gate3_manifests as manifests  # noqa: E402
from accad_g1.motion_library import motion_paths_from_manifest  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_npz(
    path: Path,
    *,
    source_file: str,
    source_checksum: str,
    human_checksum: str,
    version: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        retarget_version=np.asarray(version),
        fps=np.asarray(50.0),
        frame_count=np.asarray(11),
        source_stageii_file=np.asarray(source_file),
        source_stageii_checksum_sha256=np.asarray(source_checksum),
        source_checksum_sha256=np.asarray(human_checksum),
        g1_contract_checksum_sha256=np.asarray("1" * 64),
        g1_urdf_checksum_sha256=np.asarray("2" * 64),
        correspondence_checksum_sha256=np.asarray("3" * 64),
    )


def _install_local_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(manifests, "PIPELINE_ROOT", root)

    def local_output(path: str | Path) -> Path:
        resolved = Path(path).resolve()
        resolved.relative_to(root.resolve())
        return resolved

    monkeypatch.setattr(manifests, "assert_output_is_local", local_output)


def _snapshot(
    root: Path,
    *,
    statuses: dict[str, str] | None = None,
    version: str = "retarget-test-v1",
) -> dict[str, Path]:
    statuses = statuses or {split: "completed" for split in manifests.SPLIT_ORDER}
    reference_root = root / "work" / "g1"
    human_root = root / "work" / "human"
    reports = root / "work" / "reports" / "batch"
    items: list[dict[str, Any]] = []
    report_checksums: dict[str, str] = {}

    for ordinal, split in enumerate(manifests.SPLIT_ORDER):
        source_file = f"Subject_{split}/{split}_motion_stageii.npz"
        source_checksum = hashlib.sha256(source_file.encode()).hexdigest()
        human = human_root / f"Subject_{split}/{split}_motion_human_50hz.npz"
        human.parent.mkdir(parents=True, exist_ok=True)
        human.write_bytes(f"human:{split}".encode())
        human_checksum = _sha256(human)
        g1 = reference_root / f"Subject_{split}/{split}_motion_g1_50hz.npz"
        _write_npz(
            g1,
            source_file=source_file,
            source_checksum=source_checksum,
            human_checksum=human_checksum,
            version=version,
        )
        item_id = f"{split}:{source_file}"
        status = statuses[split]
        item_report = reports / "items" / split / f"{split}_motion.json"
        item_report.parent.mkdir(parents=True, exist_ok=True)
        report_document = {
            "schema_version": manifests.BATCH_SCHEMA_VERSION,
            "item_id": item_id,
            "split": split,
            "source_file": source_file,
            "status": status,
            "error": None if status == "completed" else {"type": "FakeFailure"},
        }
        item_report.write_text(json.dumps(report_document), encoding="utf-8")
        report_checksum = _sha256(item_report)
        report_checksums[item_id] = report_checksum
        item = {
            **report_document,
            "ordinal": ordinal,
            "source_manifest_checksum_sha256": source_checksum,
            "source_actual_checksum_sha256": source_checksum,
            "subject": f"Subject_{split}",
            "category": "test",
            "motion_name": f"{split}_motion",
            "item_report_path": str(item_report),
            "item_report_checksum_sha256": report_checksum,
            "stages": {
                "g1": {
                    "action": "generated",
                    "validation": {
                        "gate3_kinematic_complete": status == "completed",
                        "gate3_complete": False,
                    },
                }
            },
            "artifacts": {
                "human": {
                    "path": str(human),
                    "exists": True,
                    "size_bytes": human.stat().st_size,
                    "checksum_sha256": human_checksum,
                },
                "g1": {
                    "path": str(g1),
                    "exists": True,
                    "size_bytes": g1.stat().st_size,
                    "checksum_sha256": _sha256(g1),
                },
            },
        }
        items.append(item)

    items_path = reports / "batch_items.jsonl"
    items_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in items),
        encoding="utf-8",
    )
    completed = sum(item["status"] == "completed" for item in items)
    failed = len(items) - completed
    summary = {
        "schema_version": manifests.BATCH_SCHEMA_VERSION,
        "finished_at_utc": "2026-08-08T00:00:00+00:00",
        "settings": {
            "splits": list(manifests.SPLIT_ORDER),
            "max_items": None,
        },
        "counts": {
            "candidate_total": len(items),
            "candidate_by_split": {split: 1 for split in manifests.SPLIT_ORDER},
            "selected_total": len(items),
            "selected_by_split": {split: 1 for split in manifests.SPLIT_ORDER},
            "completed": completed,
            "failed": failed,
        },
        "failures": [] if not failed else [{"item_id": "failure"}],
        "all_selected_items_completed": failed == 0,
        "artifacts": {
            "items_jsonl": {
                "path": str(items_path),
                "checksum_sha256": _sha256(items_path),
                "records": len(items),
            },
            "item_report_checksums_sha256": report_checksums,
        },
    }
    summary_path = reports / "batch_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return {
        "items": items_path,
        "summary": summary_path,
        "references": reference_root,
        "output": root / "work" / "manifests",
    }


def test_complete_snapshot_builds_checksummed_split_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pipeline"
    root.mkdir()
    _install_local_root(monkeypatch, root)
    paths = _snapshot(root)
    validation_calls: list[str] = []

    def validate(path, expected_source_path=None):
        archive = Path(path)
        validation_calls.append(archive.name)
        assert Path(expected_source_path).is_file()
        return {
            "gate3_kinematic_complete": True,
            "archive_checksum_sha256": _sha256(archive),
        }

    monkeypatch.setattr(manifests, "validate_g1_archive", validate)
    result = manifests.build_gate3_manifests(
        items_jsonl=paths["items"],
        batch_summary=paths["summary"],
        output_dir=paths["output"],
        reference_root=paths["references"],
        required_retarget_version="retarget-test-v1",
    )

    assert result["counts"]["promoted_total"] == 3
    assert result["counts"]["promoted_by_split"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert len(validation_calls) == 3
    train = json.loads((paths["output"] / "g1_train.json").read_text())
    assert train["kind"] == "split:train"
    assert train["motions"][0]["gate3_kinematic_complete"] is True
    assert train["motions"][0]["g1_file"] == "Subject_train/train_motion_g1_50hz.npz"
    assert len(train["motions"][0]["g1_checksum_sha256"]) == 64

    combined = paths["output"] / "g1_splits.json"
    selection = motion_paths_from_manifest(
        combined, paths["references"], split="validation", require_all=True
    )
    assert selection.paths[0].name == "validation_motion_g1_50hz.npz"
    build_report = json.loads((paths["output"] / "g1_manifest_build.json").read_text())
    assert build_report["status"] == "completed"
    assert set(build_report["manifests"]) == {"train", "validation", "test", "combined"}
    assert not list(paths["output"].glob("*.tmp"))


def test_archive_tampering_is_rejected_before_any_manifest_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pipeline"
    root.mkdir()
    _install_local_root(monkeypatch, root)
    paths = _snapshot(root)
    victim = paths["references"] / "Subject_train/train_motion_g1_50hz.npz"
    victim.write_bytes(victim.read_bytes() + b"tampered")

    with pytest.raises(manifests.Gate3ManifestError, match="checksum changed"):
        manifests.build_gate3_manifests(
            items_jsonl=paths["items"],
            batch_summary=paths["summary"],
            output_dir=paths["output"],
            reference_root=paths["references"],
            required_retarget_version="retarget-test-v1",
        )
    assert not paths["output"].exists()


def test_incomplete_batch_requires_explicit_partial_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pipeline"
    root.mkdir()
    _install_local_root(monkeypatch, root)
    paths = _snapshot(root, statuses={"train": "completed", "validation": "failed", "test": "completed"})

    with pytest.raises(manifests.Gate3ManifestError, match="complete train/validation/test"):
        manifests.build_gate3_manifests(
            items_jsonl=paths["items"],
            batch_summary=paths["summary"],
            output_dir=paths["output"],
            reference_root=paths["references"],
            required_retarget_version="retarget-test-v1",
            revalidate=False,
        )

    result = manifests.build_gate3_manifests(
        items_jsonl=paths["items"],
        batch_summary=paths["summary"],
        output_dir=paths["output"],
        reference_root=paths["references"],
        required_retarget_version="retarget-test-v1",
        require_complete_batch=False,
        revalidate=False,
    )
    assert result["counts"]["promoted_total"] == 2
    validation = json.loads((paths["output"] / "g1_validation.json").read_text())
    assert validation["num_motions"] == 0


def test_jsonl_summary_checksum_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pipeline"
    root.mkdir()
    _install_local_root(monkeypatch, root)
    paths = _snapshot(root)
    paths["items"].write_text(paths["items"].read_text() + "\n", encoding="utf-8")

    with pytest.raises(manifests.Gate3ManifestError, match="blank line|checksum"):
        manifests.build_gate3_manifests(
            items_jsonl=paths["items"],
            batch_summary=paths["summary"],
            output_dir=paths["output"],
            reference_root=paths["references"],
            required_retarget_version="retarget-test-v1",
            revalidate=False,
        )
