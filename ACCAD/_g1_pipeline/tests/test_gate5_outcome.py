from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


PIPELINE_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_PACKAGE_ROOT))

from accad_g1.gate5_outcome import (
    Gate5OutcomeError,
    build_gate5_terminal_outcome,
    verify_gate5_terminal_outcome,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _evidence(tmp_path: Path) -> dict[str, Path]:
    accad = tmp_path / "ACCAD"
    pipeline = accad / "_g1_pipeline"
    work = pipeline / "work"
    run_dir = work / "training" / "runs" / "production"
    evaluation_dir = work / "training" / "evaluations" / "validation"
    checkpoint = run_dir / "model_9.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"immutable-checkpoint")
    train_manifest = work / "dynamic" / "g1_train.json"
    validation_manifest = work / "dynamic" / "g1_validation.json"
    _write_json(train_manifest, {"split": "train", "motions": ["train-motion"]})
    _write_json(
        validation_manifest,
        {"split": "validation", "motions": ["held-out-a", "held-out-b"]},
    )
    checkpoint_record = {
        "path": str(checkpoint.resolve()),
        "bytes": checkpoint.stat().st_size,
        "checksum_sha256": _sha(checkpoint),
    }
    training_summary = run_dir / "run_summary.json"
    _write_json(
        training_summary,
        {
            "schema_version": "accad_g1_tracking_train_v2",
            "status": "completed",
            "run_dir": str(run_dir.resolve()),
            "training": {
                "learning_updates_requested": 10,
                "learning_updates_completed": 10,
                "final_checkpoint_iteration_index": 9,
                "total_timesteps": 20480,
            },
            "checkpoints": [checkpoint_record],
            "motion_input": {
                "manifest_promotion_verified": True,
                "manifests": [
                    {
                        "path": str(train_manifest.resolve()),
                        "checksum_sha256": _sha(train_manifest),
                        "split": "train",
                    }
                ],
            },
        },
    )
    validation_report = evaluation_dir / "suite_evaluation_summary.json"
    _write_json(
        validation_report,
        {
            "schema_version": "accad_g1_tracking_suite_evaluation_v2",
            "status": "completed",
            "passed": False,
            "checkpoint": checkpoint_record,
            "evaluation_contract": {
                "action_source": "checkpoint",
                "official_acceptance_eligible": True,
                "deterministic_policy_inference": True,
                "first_episode_only": True,
            },
            "motion_input": {
                "split": "validation",
                "motion_count": 2,
                "manifest_promotion_verified": True,
                "manifests": [
                    {
                        "path": str(validation_manifest.resolve()),
                        "checksum_sha256": _sha(validation_manifest),
                        "split": "validation",
                    }
                ],
            },
            "aggregate": {
                "motion_count": 2,
                "finished_first_episodes": 2,
                "successful_motion_end_episodes": 0,
                "completion_ratio_mean": 0.25,
                "simulation_app_stopped_early": False,
            },
            "acceptance": {
                "passed": False,
                "checks": {
                    "completion": {
                        "passed": False,
                        "comparison": ">=",
                        "threshold": 0.9,
                        "value": 0.25,
                    },
                    "finite": {
                        "passed": True,
                        "comparison": "==",
                        "expected": True,
                        "value": True,
                    },
                },
            },
        },
    )
    return {
        "accad": accad,
        "pipeline": pipeline,
        "work": work,
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "training_summary": training_summary,
        "validation_report": validation_report,
        "output": work / "reports" / "gate5_terminal_outcome.json",
    }


def _build(paths: dict[str, Path]) -> dict[str, object]:
    return build_gate5_terminal_outcome(
        training_run=paths["run_dir"],
        validation_report=paths["validation_report"],
        accad_root=paths["accad"],
        pipeline_root=paths["pipeline"],
        work_root=paths["work"],
    )


def test_terminal_failure_distinguishes_execution_from_acceptance(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    report = _build(paths)

    assert paths["output"].is_file()
    assert report["status"] == "failed"
    assert report["complete"] is False
    assert report["execution"] == {
        "complete": True,
        "training_complete": True,
        "validation_complete": True,
        "test_executed": False,
    }
    assert report["policy_acceptance"]["passed"] is False
    assert report["policy_acceptance"]["failed_checks"] == [
        {
            "name": "completion",
            "comparison": ">=",
            "threshold": 0.9,
            "value": 0.25,
        }
    ]
    assert report["test"] == {
        "status": "sealed_not_opened",
        "executed": False,
        "report_consumed": False,
        "reason": "validation acceptance failed before the held-out test gate",
    }
    assert report["success_publication"]["published"] is False
    assert not (paths["work"] / "reports" / "final_completion.json").exists()
    assert not (paths["work"] / "reports" / "final_artifact_manifest.json").exists()

    snapshot = verify_gate5_terminal_outcome(
        report,
        accad_root=paths["accad"],
        pipeline_root=paths["pipeline"],
        work_root=paths["work"],
    )
    assert snapshot == {
        "current": True,
        "verified_source_artifact_count": 5,
        "errors": [],
    }


def test_terminal_failure_snapshot_detects_source_mutation(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    report = _build(paths)
    paths["checkpoint"].write_bytes(b"mutated-checkpoint")

    snapshot = verify_gate5_terminal_outcome(
        report,
        accad_root=paths["accad"],
        pipeline_root=paths["pipeline"],
        work_root=paths["work"],
    )
    assert snapshot["current"] is False
    assert snapshot["verified_source_artifact_count"] == 0
    assert any("source artifact 'final_checkpoint'" in item for item in snapshot["errors"])


@pytest.mark.parametrize("forbidden_case", ["passed", "test", "diagnostic"])
def test_terminal_failure_rejects_non_official_validation_evidence(
    tmp_path: Path, forbidden_case: str
) -> None:
    paths = _evidence(tmp_path)
    report = json.loads(paths["validation_report"].read_text(encoding="utf-8"))
    if forbidden_case == "passed":
        report["passed"] = True
        report["acceptance"]["passed"] = True
        report["acceptance"]["checks"]["completion"]["passed"] = True
    elif forbidden_case == "test":
        report["motion_input"]["split"] = "test"
        report["motion_input"]["manifests"][0]["split"] = "test"
    else:
        report["evaluation_contract"]["action_source"] = "zero"
        report["evaluation_contract"]["official_acceptance_eligible"] = False
    _write_json(paths["validation_report"], report)

    with pytest.raises(Gate5OutcomeError):
        _build(paths)
    assert not paths["output"].exists()


def test_terminal_failure_requires_final_training_checkpoint(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    report = json.loads(paths["validation_report"].read_text(encoding="utf-8"))
    early = paths["run_dir"] / "model_8.pt"
    early.write_bytes(b"early-checkpoint")
    early_record = {
        "path": str(early.resolve()),
        "bytes": early.stat().st_size,
        "checksum_sha256": _sha(early),
    }
    report["checkpoint"] = early_record
    _write_json(paths["validation_report"], report)
    training = json.loads(paths["training_summary"].read_text(encoding="utf-8"))
    training["checkpoints"].append(early_record)
    _write_json(paths["training_summary"], training)

    with pytest.raises(Gate5OutcomeError, match="final training checkpoint"):
        _build(paths)


def test_terminal_failure_refuses_positive_final_artifacts(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    positive = paths["work"] / "reports" / "final_completion.json"
    _write_json(positive, {"status": "completed"})

    with pytest.raises(Gate5OutcomeError, match="positive final artifact"):
        _build(paths)
    assert not paths["output"].exists()
