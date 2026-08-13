from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import accad_g1.parallel_batch as parallel  # noqa: E402


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace(tmp_path: Path, count: int = 7) -> dict[str, Path]:
    paths = {
        "source": tmp_path / "raw",
        "manifests": tmp_path / "manifests",
        "outputs": tmp_path / "outputs",
        "reports": tmp_path / "reports" / "batch",
        "models": tmp_path / "models",
        "correspondence": tmp_path / "correspondence.yaml",
    }
    for name in ("source", "manifests", "models"):
        paths[name].mkdir(parents=True, exist_ok=True)
    paths["correspondence"].write_text("schema_version: test\n", encoding="utf-8")

    records: dict[str, list[dict[str, Any]]] = {
        split: [] for split in parallel.SPLIT_ORDER
    }
    split_cycle = ("train", "validation", "test")
    for index in range(count):
        split = split_cycle[index % len(split_cycle)]
        relative = Path(split.title()) / f"Motion_{index:02d}_stageii.npz"
        source = paths["source"] / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"raw-{index}".encode())
        records[split].append(
            {
                "source_file": relative.as_posix(),
                "checksum_sha256": _checksum(source),
                "valid": True,
                "subject": f"subject-{index}",
                "category": relative.parent.name,
                "motion_name": relative.stem.removesuffix("_stageii"),
                "num_frames": 100 + index,
                "fps": 100.0,
                "duration_sec": 1.0 + index / 10.0,
            }
        )
    for split in parallel.SPLIT_ORDER:
        motions = records[split]
        (paths["manifests"] / f"accad_{split}.json").write_text(
            json.dumps(
                {
                    "dataset": "ACCAD",
                    "kind": f"split:{split}",
                    "num_motions": len(motions),
                    "duration_sec": sum(item["duration_sec"] for item in motions),
                    "motions": motions,
                }
            ),
            encoding="utf-8",
        )
    return paths


def test_shard_plan_is_deterministic_disjoint_and_loadable(pipeline_tmp_path: Path):
    paths = _workspace(pipeline_tmp_path, count=7)
    first = parallel.build_shard_plan(
        manifest_root=paths["manifests"],
        report_root=paths["reports"],
        num_shards=3,
    )
    second = parallel.build_shard_plan(
        manifest_root=paths["manifests"],
        report_root=paths["reports"],
        num_shards=3,
    )

    assert first["run_id"] == second["run_id"]
    assert first["selected_total"] == 7
    assert [shard["ordinals"] for shard in first["shards"]] == [
        [0, 3, 6],
        [1, 4],
        [2, 5],
    ]
    all_ids = [item for shard in first["shards"] for item in shard["item_ids"]]
    assert len(all_ids) == len(set(all_ids)) == 7
    assert set(all_ids) == set(first["selected_item_ids"])

    # The canonical loader must accept every generated shard manifest and see
    # exactly the IDs declared in the plan.
    for shard in first["shards"]:
        records, _, _ = parallel.batch_module._select_records(
            Path(shard["manifest_root"]), parallel.SPLIT_ORDER, None
        )
        loaded_ids = [f"{record['split']}:{record['source_file']}" for record in records]
        assert set(loaded_ids) == set(shard["item_ids"])
        assert len(records) == shard["num_items"]


def test_plan_applies_global_prefix_before_sharding_and_rejects_bad_counts(
    pipeline_tmp_path: Path,
):
    paths = _workspace(pipeline_tmp_path, count=7)
    plan = parallel.build_shard_plan(
        manifest_root=paths["manifests"],
        report_root=paths["reports"],
        num_shards=2,
        max_items=4,
    )
    assert plan["selected_total"] == 4
    assert [shard["ordinals"] for shard in plan["shards"]] == [[0, 2], [1, 3]]
    with pytest.raises(ValueError, match="cannot exceed"):
        parallel.build_shard_plan(
            manifest_root=paths["manifests"],
            report_root=paths["reports"],
            num_shards=8,
        )
    with pytest.raises(ValueError, match="positive integer"):
        parallel.build_shard_plan(
            manifest_root=paths["manifests"],
            report_root=paths["reports"],
            num_shards=0,
        )
    with pytest.raises(ValueError, match="unknown batch splits"):
        parallel.run_parallel_batch(
            manifest_root=paths["manifests"],
            output_root=paths["outputs"],
            report_root=paths["reports"],
            model_root=paths["models"],
            source_root=paths["source"],
            correspondence_path=paths["correspondence"],
            splits=("train", "not-a-split"),
            num_shards=2,
            plan_only=True,
        )


def test_plan_only_never_launches_workers_or_canonical_batch(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _workspace(pipeline_tmp_path, count=4)

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("motion processing must not start in plan-only mode")

    monkeypatch.setattr(parallel, "_launch_workers", forbidden)
    monkeypatch.setattr(parallel.batch_module, "run_batch", forbidden)
    result = parallel.run_parallel_batch(
        manifest_root=paths["manifests"],
        output_root=paths["outputs"],
        report_root=paths["reports"],
        model_root=paths["models"],
        source_root=paths["source"],
        correspondence_path=paths["correspondence"],
        num_shards=2,
        plan_only=True,
    )
    assert result["status"] == "planned"
    assert result["motion_processing_started"] is False
    assert Path(result["plan"]["plan_artifact"]["path"]).is_file()


def test_orchestrator_canonicalizes_after_isolated_worker_problem(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _workspace(pipeline_tmp_path, count=4)
    calls: dict[str, Any] = {}

    def fake_launch(plan: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        calls["worker_plan"] = plan
        calls["worker_kwargs"] = kwargs
        return [
            {"index": 0, "name": "shard-0", "status": "completed", "returncode": 0},
            {"index": 1, "name": "shard-1", "status": "process_failed", "returncode": 1},
        ]

    def fake_canonical(**kwargs: Any) -> dict[str, Any]:
        calls["canonical_kwargs"] = kwargs
        return {
            "all_selected_items_completed": True,
            "counts": {"selected_total": 4, "completed": 4, "failed": 0},
            "failures": [],
            "artifacts": {},
        }

    monkeypatch.setattr(parallel, "_launch_workers", fake_launch)
    monkeypatch.setattr(parallel.batch_module, "run_batch", fake_canonical)
    result = parallel.run_parallel_batch(
        manifest_root=paths["manifests"],
        output_root=paths["outputs"],
        report_root=paths["reports"],
        model_root=paths["models"],
        source_root=paths["source"],
        correspondence_path=paths["correspondence"],
        num_shards=2,
        max_workers=2,
        force=True,
    )

    assert result["success"] is True
    assert result["worker_problem_count"] == 1
    assert result["worker_problems_recovered_by_canonical"] is True
    canonical = calls["canonical_kwargs"]
    assert canonical["report_root"] == paths["reports"].resolve()
    assert canonical["output_root"] == paths["outputs"].resolve()
    assert canonical["resume"] is True
    assert canonical["force"] is False
    shard_reports = {
        shard["report_root"] for shard in calls["worker_plan"]["shards"]
    }
    assert len(shard_reports) == 2
    assert str(paths["reports"].resolve()) not in shard_reports
    assert Path(result["summary_artifact"]["path"]).is_file()


def test_worker_status_records_success_and_version_mismatch(pipeline_tmp_path: Path):
    paths = _workspace(pipeline_tmp_path, count=1)
    status_path = paths["reports"] / "worker.json"
    received: dict[str, Any] = {}

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {
            "all_selected_items_completed": True,
            "counts": {"selected_total": 1, "completed": 1, "failed": 0},
            "failures": [],
            "artifacts": {"summary_json": {"path": "fake"}},
        }

    exit_code = parallel._execute_worker(
        manifest_root=paths["manifests"],
        output_root=paths["outputs"],
        report_root=paths["reports"] / "shard",
        model_root=paths["models"],
        source_root=paths["source"],
        correspondence_path=paths["correspondence"],
        worker_status_path=status_path,
        required_retarget_version=parallel.RETARGET_VERSION,
        splits=parallel.SPLIT_ORDER,
        target_fps=50.0,
        chunk_size=128,
        force=False,
        runner=fake_runner,
    )
    assert exit_code == 0
    assert json.loads(status_path.read_text(encoding="utf-8"))["success"] is True
    assert received["resume"] is True
    assert received["force"] is False

    mismatch_path = paths["reports"] / "worker_mismatch.json"
    exit_code = parallel._execute_worker(
        manifest_root=paths["manifests"],
        output_root=paths["outputs"],
        report_root=paths["reports"] / "shard",
        model_root=paths["models"],
        source_root=paths["source"],
        correspondence_path=paths["correspondence"],
        worker_status_path=mismatch_path,
        required_retarget_version="outdated-version",
        splits=parallel.SPLIT_ORDER,
        target_fps=50.0,
        chunk_size=128,
        force=False,
        runner=lambda **_: pytest.fail("runner must not be called on version mismatch"),
    )
    mismatch = json.loads(mismatch_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert mismatch["status"] == "failed"
    assert "changed" in mismatch["error"]["message"]
