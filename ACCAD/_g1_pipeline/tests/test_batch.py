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

import accad_g1.batch as batch  # noqa: E402


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(source_root: Path, source_file: str, *, checksum: str | None = None) -> dict[str, Any]:
    source = source_root / source_file
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        source.write_bytes(f"raw:{source_file}".encode())
    return {
        "source_file": source_file,
        "checksum_sha256": checksum or _checksum(source),
        "valid": True,
        "subject": "subject",
        "category": source.parent.name,
        "motion_name": source.stem.removesuffix("_stageii"),
        "num_frames": 120,
        "fps": 120.0,
        "duration_sec": 1.0,
    }


def _write_manifests(
    root: Path,
    records: dict[str, list[dict[str, Any]]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for split in batch.SPLIT_ORDER:
        motions = records.get(split, [])
        (root / f"accad_{split}.json").write_text(
            json.dumps(
                {
                    "dataset": "ACCAD",
                    "kind": f"split:{split}",
                    "num_motions": len(motions),
                    "duration_sec": float(len(motions)),
                    "motions": motions,
                }
            ),
            encoding="utf-8",
        )
    return root


def _workspace(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "source": tmp_path / "raw",
        "manifests": tmp_path / "manifests",
        "outputs": tmp_path / "outputs",
        "reports": tmp_path / "reports",
        "models": tmp_path / "models",
        "correspondence": tmp_path / "correspondence.yaml",
    }
    paths["source"].mkdir(parents=True)
    paths["models"].mkdir(parents=True)
    paths["correspondence"].write_text("schema_version: test\n", encoding="utf-8")
    return paths


def _install_fake_stages(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    calls: dict[str, list[str]] = {
        "reconstruct": [],
        "validate_human": [],
        "retarget": [],
        "validate_g1": [],
    }

    def reconstruct(source, model_root, output, target_fps=50.0, chunk_size=128):
        source_path = Path(source)
        output_path = Path(output)
        calls["reconstruct"].append(source_path.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"human:" + source_path.read_bytes())
        return {
            "ok": True,
            "output_path": str(output_path),
            "target_fps": target_fps,
            "chunk_size": chunk_size,
            "model_root": str(model_root),
        }

    def validate_human(output, source_root, profile="generic"):
        path = Path(output)
        calls["validate_human"].append(path.name)
        passed = path.is_file() and path.read_bytes().startswith(b"human:")
        return {
            "ok": passed,
            "gate2_complete": passed,
            "archive_path": str(path),
            "archive_checksum_sha256": _checksum(path) if path.is_file() else None,
            "profile": profile,
            "source_root": str(source_root),
            "errors": [] if passed else ["invalid fake human"],
        }

    def retarget(human, output, correspondence_path):
        human_path = Path(human)
        output_path = Path(output)
        calls["retarget"].append(human_path.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = np.frombuffer(b"g1:" + human_path.read_bytes(), dtype=np.uint8)
        with output_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                fake_payload=payload,
                retarget_version=np.asarray(batch.RETARGET_VERSION),
            )
        return {
            "ok": True,
            "gate3_complete": False,
            "output_path": str(output_path),
            "archive_checksum_sha256": _checksum(output_path),
            "correspondence_path": str(correspondence_path),
        }

    def validate_g1(output, expected_source_path=None):
        path = Path(output)
        calls["validate_g1"].append(path.name)
        expected = Path(expected_source_path) if expected_source_path else None
        try:
            with np.load(path, allow_pickle=False) as archive:
                payload = archive["fake_payload"].tobytes()
                retarget_version = str(archive["retarget_version"].item())
        except Exception:
            payload = b""
            retarget_version = ""
        passed = bool(
            payload.startswith(b"g1:human:")
            and retarget_version == batch.RETARGET_VERSION
            and expected is not None
            and expected.is_file()
        )
        return {
            "ok": passed,
            "gate3_kinematic_complete": passed,
            # This must remain false and must not cause batch failure.
            "gate3_complete": False,
            "archive_path": str(path),
            "archive_checksum_sha256": _checksum(path) if path.is_file() else None,
            "required_failures": [] if passed else ["fake.invalid"],
            "isaac_replay": {"required": True, "status": "pending"},
        }

    monkeypatch.setattr(batch, "reconstruct_human_motion", reconstruct)
    monkeypatch.setattr(batch, "validate_human_archive", validate_human)
    monkeypatch.setattr(batch, "retarget_human_to_g1", retarget)
    monkeypatch.setattr(batch, "validate_g1_archive", validate_g1)
    return calls


def _run(paths: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    return batch.run_batch(
        manifest_root=paths["manifests"],
        output_root=paths["outputs"],
        report_root=paths["reports"],
        model_root=paths["models"],
        source_root=paths["source"],
        correspondence_path=paths["correspondence"],
        **kwargs,
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_deterministic_split_order_max_items_and_atomic_reports(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _workspace(pipeline_tmp_path)
    records = {
        "train": [
            _record(paths["source"], "Train/Z_stageii.npz"),
            _record(paths["source"], "Train/A_stageii.npz"),
        ],
        "validation": [_record(paths["source"], "Validation/C_stageii.npz")],
        "test": [_record(paths["source"], "Test/D_stageii.npz")],
    }
    _write_manifests(paths["manifests"], records)
    calls = _install_fake_stages(monkeypatch)

    result = _run(paths, max_items=3)
    assert calls["reconstruct"] == [
        "A_stageii.npz",
        "Z_stageii.npz",
        "C_stageii.npz",
    ]
    assert result["counts"]["candidate_total"] == 4
    assert result["counts"]["selected_total"] == 3
    assert result["counts"]["selected_by_split"] == {
        "train": 2,
        "validation": 1,
        "test": 0,
    }
    assert result["counts"]["completed"] == 3
    assert result["counts"]["failed"] == 0
    assert result["all_selected_items_completed"] is True
    assert result["live_isaac_replay_complete"] is False

    jsonl_path = Path(result["artifacts"]["items_jsonl"]["path"])
    items = _jsonl(jsonl_path)
    assert [item["item_id"] for item in items] == [
        "train:Train/A_stageii.npz",
        "train:Train/Z_stageii.npz",
        "validation:Validation/C_stageii.npz",
    ]
    assert all(item["stages"]["g1"]["validation"]["gate3_complete"] is False for item in items)
    assert all(item["status"] == "completed" for item in items)
    assert Path(result["artifacts"]["summary_json"]["path"]).is_file()
    assert result["artifacts"]["summary_json"]["checksum_sha256"] == _checksum(
        Path(result["artifacts"]["summary_json"]["path"])
    )
    assert not list(paths["reports"].rglob("*.tmp"))


def test_resume_reuses_only_valid_artifacts_and_force_regenerates(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _workspace(pipeline_tmp_path)
    _write_manifests(
        paths["manifests"],
        {"train": [_record(paths["source"], "Train/A_stageii.npz")]},
    )
    calls = _install_fake_stages(monkeypatch)
    first = _run(paths)
    assert first["counts"]["human_actions"] == {"generated": 1}
    assert first["counts"]["g1_actions"] == {"generated": 1}

    for values in calls.values():
        values.clear()
    resumed = _run(paths, resume=True)
    assert calls["reconstruct"] == []
    assert calls["retarget"] == []
    assert len(calls["validate_human"]) == 1
    assert len(calls["validate_g1"]) == 1
    assert resumed["counts"]["human_actions"] == {"reused": 1}
    assert resumed["counts"]["g1_actions"] == {"reused": 1}

    for values in calls.values():
        values.clear()
    forced = _run(paths, force=True)
    assert calls["reconstruct"] == ["A_stageii.npz"]
    assert calls["retarget"] == ["A_human_50hz.npz"]
    assert forced["counts"]["human_actions"] == {"generated": 1}
    assert forced["counts"]["g1_actions"] == {"generated": 1}


def test_resume_false_existing_output_is_an_isolated_item_failure(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _workspace(pipeline_tmp_path)
    _write_manifests(
        paths["manifests"],
        {"train": [_record(paths["source"], "Train/A_stageii.npz")]},
    )
    calls = _install_fake_stages(monkeypatch)
    _run(paths)
    for values in calls.values():
        values.clear()

    result = _run(paths, resume=False)
    assert result["counts"]["completed"] == 0
    assert result["counts"]["failed"] == 1
    assert result["failures"][0]["error"]["type"] == "FileExistsError"
    assert calls["reconstruct"] == []
    assert calls["retarget"] == []


def test_source_checksum_failure_does_not_stop_later_items(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _workspace(pipeline_tmp_path)
    bad = _record(paths["source"], "Train/A_bad_stageii.npz")
    bad["checksum_sha256"] = "0" * 64
    good = _record(paths["source"], "Train/B_good_stageii.npz")
    _write_manifests(paths["manifests"], {"train": [good, bad]})
    calls = _install_fake_stages(monkeypatch)

    result = _run(paths)
    assert result["counts"]["selected_total"] == 2
    assert result["counts"]["completed"] == 1
    assert result["counts"]["failed"] == 1
    assert result["failures"][0]["source_file"] == "Train/A_bad_stageii.npz"
    assert "checksum mismatch" in result["failures"][0]["error"]["message"]
    assert calls["reconstruct"] == ["B_good_stageii.npz"]
    failed_report = Path(result["failures"][0]["item_report_path"])
    assert failed_report.is_file()
    assert json.loads(failed_report.read_text(encoding="utf-8"))["status"] == "failed"


def test_manifest_leakage_and_nonlocal_output_are_rejected_before_work(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _workspace(pipeline_tmp_path)
    shared = _record(paths["source"], "Shared/A_stageii.npz")
    duplicate = _record(paths["source"], "Other/B_stageii.npz", checksum=shared["checksum_sha256"])
    _write_manifests(
        paths["manifests"],
        {"train": [shared], "validation": [duplicate]},
    )
    _install_fake_stages(monkeypatch)
    with pytest.raises(batch.BatchManifestError, match="checksum"):
        _run(paths)

    _write_manifests(paths["manifests"], {"train": [shared]})
    with pytest.raises(ValueError, match="inside"):
        batch.run_batch(
            manifest_root=paths["manifests"],
            output_root=Path("/tmp/nonlocal_accad_batch_output"),
            report_root=paths["reports"],
            model_root=paths["models"],
            source_root=paths["source"],
            correspondence_path=paths["correspondence"],
        )
    with pytest.raises(ValueError, match="max_items"):
        _run(paths, max_items=-1)
    with pytest.raises(ValueError, match="max_items"):
        _run(paths, max_items=True)


def test_max_items_zero_writes_empty_aggregate_without_stage_calls(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _workspace(pipeline_tmp_path)
    _write_manifests(
        paths["manifests"],
        {"train": [_record(paths["source"], "Train/A_stageii.npz")]},
    )
    calls = _install_fake_stages(monkeypatch)
    result = _run(paths, max_items=0)
    assert result["counts"]["candidate_total"] == 1
    assert result["counts"]["selected_total"] == 0
    assert result["counts"]["completed"] == 0
    assert result["all_selected_items_completed"] is False
    assert all(not values for values in calls.values())
    assert Path(result["artifacts"]["items_jsonl"]["path"]).read_text(encoding="utf-8") == ""
