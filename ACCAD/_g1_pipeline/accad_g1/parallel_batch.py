"""Safe process-level sharding for the canonical ACCAD -> G1 batch.

The canonical :func:`accad_g1.batch.run_batch` intentionally stays sequential.
This module accelerates it without changing its item semantics:

1. Load the same deterministic Gate-1 record sequence.
2. Assign global ordinal ``i`` to shard ``i % num_shards``.
3. Give every shard a private manifest and report directory, while all shards
   use the canonical output root.  Source records are disjoint, so artifact
   destinations are disjoint too.
4. Run each shard in an isolated Python subprocess.
5. Run the canonical batch once with ``resume=True``.  This validates/reuses
   every artifact, retries anything missing or invalid, and writes the official
   ``batch_items.jsonl`` and ``batch_summary.json`` at the requested report root.

Generated paths are checked by the same ACCAD-local guards as the canonical
batch.  Motion archives are written atomically by the existing stage code.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

from . import batch as batch_module
from .batch import SPLIT_ORDER
from .paths import (
    ACCAD_ROOT,
    DEFAULT_CONFIG,
    PIPELINE_ROOT,
    assert_output_is_local,
    load_config,
    local_path,
)
from .retarget import DEFAULT_CORRESPONDENCE, RETARGET_VERSION


PARALLEL_BATCH_SCHEMA_VERSION = "accad_g1_parallel_batch_v1"
SHARD_ASSIGNMENT = "global_deterministic_ordinal_modulo_num_shards"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = assert_output_is_local(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                _json_ready(payload),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _normalized_manifest_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert batch's normalized record back to Gate-1 manifest vocabulary."""

    return {
        "source_file": record["source_file"],
        "checksum_sha256": record["checksum_sha256"],
        "valid": True,
        "subject": record.get("subject"),
        "category": record.get("category"),
        "motion_name": record.get("motion_name"),
        "num_frames": record.get("source_num_frames"),
        "fps": record.get("source_fps"),
        "duration_sec": record.get("source_duration_sec"),
    }


def build_shard_plan(
    *,
    manifest_root: str | Path,
    report_root: str | Path,
    num_shards: int,
    splits: Sequence[str] = SPLIT_ORDER,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Write deterministic, disjoint shard manifests and return their plan.

    No motion processing occurs here.  The generated manifest tree is keyed by
    a content-derived run id, so a changed selection or shard count cannot mix
    reports from an earlier plan.
    """

    num_shards = _positive_integer(num_shards, "num_shards")
    if max_items is not None and (
        isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 0
    ):
        raise ValueError("max_items must be None or a non-negative integer")
    if isinstance(splits, (str, bytes)):
        raise ValueError("splits must be a sequence of split names, not a string")

    manifests = batch_module._local_directory(manifest_root, "manifest_root")
    reports = batch_module._local_directory(report_root, "report_root")
    records, manifest_info, candidate_counts = batch_module._select_records(
        manifests, tuple(splits), max_items
    )
    if not records:
        raise ValueError("parallel batch selection is empty")
    if num_shards > len(records):
        raise ValueError(
            f"num_shards ({num_shards}) cannot exceed selected records ({len(records)})"
        )

    selected_splits = tuple(split for split in SPLIT_ORDER if split in set(splits))
    plan_identity = {
        "schema_version": PARALLEL_BATCH_SCHEMA_VERSION,
        "retarget_version": RETARGET_VERSION,
        "assignment": SHARD_ASSIGNMENT,
        "num_shards": num_shards,
        "splits": list(selected_splits),
        "max_items": max_items,
        "manifest_checksums": {
            split: manifest_info[split]["checksum_sha256"] for split in selected_splits
        },
        "selected_item_ids": [
            f"{record['split']}:{record['source_file']}" for record in records
        ],
    }
    identity_bytes = json.dumps(
        plan_identity, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    run_id = hashlib.sha256(identity_bytes).hexdigest()[:16]
    parallel_root = assert_output_is_local(reports / "_parallel" / run_id)

    shard_records: list[list[tuple[int, dict[str, Any]]]] = [
        [] for _ in range(num_shards)
    ]
    for ordinal, record in enumerate(records):
        shard_records[ordinal % num_shards].append((ordinal, record))

    shards: list[dict[str, Any]] = []
    observed_ids: list[str] = []
    for shard_index, assignments in enumerate(shard_records):
        shard_name = f"shard-{shard_index:03d}-of-{num_shards:03d}"
        shard_root = assert_output_is_local(parallel_root / shard_name)
        shard_manifest_root = assert_output_is_local(shard_root / "manifests")
        shard_report_root = assert_output_is_local(shard_root / "reports")
        by_split: dict[str, list[dict[str, Any]]] = {
            split: [] for split in selected_splits
        }
        ordinals: list[int] = []
        item_ids: list[str] = []
        duration_by_split = {split: 0.0 for split in selected_splits}
        for ordinal, record in assignments:
            split = str(record["split"])
            by_split[split].append(_normalized_manifest_record(record))
            ordinals.append(ordinal)
            item_id = f"{split}:{record['source_file']}"
            item_ids.append(item_id)
            observed_ids.append(item_id)
            duration_by_split[split] += float(record.get("source_duration_sec") or 0.0)

        manifest_artifacts: dict[str, Any] = {}
        for split in selected_splits:
            manifest_path = shard_manifest_root / f"accad_{split}.json"
            document = {
                "dataset": "ACCAD",
                "kind": f"split:{split}",
                "num_motions": len(by_split[split]),
                "duration_sec": duration_by_split[split],
                "parallel_shard": {
                    "run_id": run_id,
                    "index": shard_index,
                    "count": num_shards,
                    "assignment": SHARD_ASSIGNMENT,
                    "retarget_version": RETARGET_VERSION,
                },
                "motions": by_split[split],
            }
            _atomic_json(manifest_path, document)
            manifest_artifacts[split] = {
                "path": str(manifest_path),
                "checksum_sha256": _sha256(manifest_path),
                "count": len(by_split[split]),
            }

        shards.append(
            {
                "index": shard_index,
                "name": shard_name,
                "manifest_root": str(shard_manifest_root),
                "report_root": str(shard_report_root),
                "worker_status_path": str(shard_root / "worker_status.json"),
                "log_path": str(shard_root / "worker.log"),
                "num_items": len(assignments),
                "ordinals": ordinals,
                "item_ids": item_ids,
                "manifests": manifest_artifacts,
            }
        )

    expected_ids = plan_identity["selected_item_ids"]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(expected_ids):
        raise RuntimeError("internal shard partition is not a disjoint exact cover")

    plan: dict[str, Any] = {
        **plan_identity,
        "created_at_utc": _utc_now(),
        "run_id": run_id,
        "parallel_root": str(parallel_root),
        "source_manifest_root": str(manifests),
        "canonical_report_root": str(reports),
        "candidate_counts": candidate_counts,
        "selected_total": len(records),
        "shards": shards,
    }
    plan_path = assert_output_is_local(parallel_root / "shard_plan.json")
    _atomic_json(plan_path, plan)
    plan["plan_artifact"] = {
        "path": str(plan_path),
        "checksum_sha256": _sha256(plan_path),
    }
    return plan


def _worker_command(
    shard: Mapping[str, Any],
    *,
    output_root: Path,
    model_root: Path,
    source_root: Path,
    correspondence_path: Path,
    splits: Sequence[str],
    target_fps: float,
    chunk_size: int,
    force: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(PIPELINE_ROOT / "run_parallel_batch.py"),
        "_worker",
        "--manifest-root",
        str(shard["manifest_root"]),
        "--output-root",
        str(output_root),
        "--report-root",
        str(shard["report_root"]),
        "--model-root",
        str(model_root),
        "--source-root",
        str(source_root),
        "--correspondence",
        str(correspondence_path),
        "--worker-status",
        str(shard["worker_status_path"]),
        "--required-retarget-version",
        RETARGET_VERSION,
        "--target-fps",
        str(target_fps),
        "--chunk-size",
        str(chunk_size),
        "--splits",
        *splits,
    ]
    if force:
        command.append("--force")
    return command


def _load_worker_status(path: Path) -> dict[str, Any] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _launch_workers(
    plan: Mapping[str, Any],
    *,
    max_workers: int,
    output_root: Path,
    model_root: Path,
    source_root: Path,
    correspondence_path: Path,
    splits: Sequence[str],
    target_fps: float,
    chunk_size: int,
    force: bool,
) -> list[dict[str, Any]]:
    """Launch isolated shard subprocesses; never cancel siblings on one failure."""

    max_workers = _positive_integer(max_workers, "max_workers")
    pending = deque(dict(shard) for shard in plan["shards"])
    active: dict[int, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    child_env.setdefault("OMP_NUM_THREADS", "1")

    try:
        while pending or active:
            while pending and len(active) < max_workers:
                shard = pending.popleft()
                log_path = assert_output_is_local(shard["log_path"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                command = _worker_command(
                    shard,
                    output_root=output_root,
                    model_root=model_root,
                    source_root=source_root,
                    correspondence_path=correspondence_path,
                    splits=splits,
                    target_fps=target_fps,
                    chunk_size=chunk_size,
                    force=force,
                )
                started = time.perf_counter()
                started_at = _utc_now()
                log_stream = log_path.open("w", encoding="utf-8")
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=PIPELINE_ROOT,
                        env=child_env,
                        stdin=subprocess.DEVNULL,
                        stdout=log_stream,
                        stderr=subprocess.STDOUT,
                    )
                except OSError as exc:
                    log_stream.close()
                    completed.append(
                        {
                            "index": shard["index"],
                            "name": shard["name"],
                            "status": "launch_failed",
                            "returncode": None,
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                            "started_at_utc": started_at,
                            "finished_at_utc": _utc_now(),
                            "wall_seconds": time.perf_counter() - started,
                            "log_path": str(log_path),
                        }
                    )
                    continue
                active[process.pid] = {
                    "process": process,
                    "shard": shard,
                    "command": command,
                    "log_stream": log_stream,
                    "started": started,
                    "started_at_utc": started_at,
                }

            finished_pids: list[int] = []
            for pid, state in active.items():
                process = state["process"]
                returncode = process.poll()
                if returncode is None:
                    continue
                state["log_stream"].close()
                shard = state["shard"]
                status_path = Path(shard["worker_status_path"])
                worker_status = _load_worker_status(status_path)
                log_path = Path(shard["log_path"])
                if returncode == 0:
                    process_status = "completed"
                elif worker_status is None:
                    process_status = "process_failed"
                elif worker_status.get("status") == "completed_with_failures":
                    process_status = "completed_with_failures"
                else:
                    process_status = "worker_failed"
                completed.append(
                    {
                        "index": shard["index"],
                        "name": shard["name"],
                        "status": process_status,
                        "returncode": returncode,
                        "started_at_utc": state["started_at_utc"],
                        "finished_at_utc": _utc_now(),
                        "wall_seconds": time.perf_counter() - state["started"],
                        "worker_status": worker_status,
                        "worker_status_path": str(status_path),
                        "log_path": str(log_path),
                        "log_checksum_sha256": _sha256(log_path) if log_path.is_file() else None,
                    }
                )
                finished_pids.append(pid)
            for pid in finished_pids:
                del active[pid]
            if active and not finished_pids:
                time.sleep(0.05)
    except BaseException:
        for state in active.values():
            process = state["process"]
            if process.poll() is None:
                process.terminate()
        for state in active.values():
            try:
                state["process"].wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                state["process"].kill()
                state["process"].wait()
            state["log_stream"].close()
        raise

    completed.sort(key=lambda item: int(item["index"]))
    return completed


def _execute_worker(
    *,
    manifest_root: Path,
    output_root: Path,
    report_root: Path,
    model_root: Path,
    source_root: Path,
    correspondence_path: Path,
    worker_status_path: Path,
    required_retarget_version: str,
    splits: Sequence[str],
    target_fps: float,
    chunk_size: int,
    force: bool,
    runner: Callable[..., dict[str, Any]] = batch_module.run_batch,
) -> int:
    """Execute one shard and always leave a small machine-readable status."""

    started_at = _utc_now()
    started = time.perf_counter()
    status: dict[str, Any] = {
        "schema_version": PARALLEL_BATCH_SCHEMA_VERSION,
        "started_at_utc": started_at,
        "required_retarget_version": required_retarget_version,
        "runtime_retarget_version": RETARGET_VERSION,
    }
    exit_code = 1
    try:
        if required_retarget_version != RETARGET_VERSION:
            raise RuntimeError(
                "retarget implementation changed after the shard plan was created: "
                f"required {required_retarget_version}, runtime {RETARGET_VERSION}"
            )
        result = runner(
            manifest_root=manifest_root,
            output_root=output_root,
            report_root=report_root,
            model_root=model_root,
            source_root=source_root,
            correspondence_path=correspondence_path,
            splits=tuple(splits),
            target_fps=target_fps,
            chunk_size=chunk_size,
            resume=not force,
            force=force,
            max_items=None,
        )
        success = bool(result.get("all_selected_items_completed", False))
        status.update(
            {
                "status": "completed" if success else "completed_with_failures",
                "success": success,
                "counts": result.get("counts"),
                "failures": result.get("failures"),
                "batch_summary": result.get("artifacts", {}).get("summary_json"),
            }
        )
        exit_code = 0 if success else 2
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "success": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "traceback": traceback.format_exc(),
            }
        )
        exit_code = 1
    status["finished_at_utc"] = _utc_now()
    status["wall_seconds"] = time.perf_counter() - started
    _atomic_json(worker_status_path, status)
    return exit_code


def run_parallel_batch(
    *,
    manifest_root: str | Path,
    output_root: str | Path,
    report_root: str | Path,
    model_root: str | Path,
    source_root: str | Path = ACCAD_ROOT,
    correspondence_path: str | Path = DEFAULT_CORRESPONDENCE,
    splits: Sequence[str] = SPLIT_ORDER,
    target_fps: float = 50.0,
    chunk_size: int = 128,
    num_shards: int = 4,
    max_workers: int | None = None,
    force: bool = False,
    max_items: int | None = None,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Run disjoint shard subprocesses and canonicalize with ``run_batch``.

    ``plan_only=True`` only writes and returns deterministic shard manifests.
    It is safe for inspection and never starts a worker or canonical batch.
    """

    num_shards = _positive_integer(num_shards, "num_shards")
    max_workers = num_shards if max_workers is None else _positive_integer(
        max_workers, "max_workers"
    )
    if not isinstance(force, bool) or not isinstance(plan_only, bool):
        raise ValueError("force and plan_only must be booleans")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if (
        isinstance(target_fps, bool)
        or not isinstance(target_fps, (int, float))
        or not math.isfinite(float(target_fps))
        or target_fps <= 0
    ):
        raise ValueError("target_fps must be finite and positive")
    if isinstance(splits, (str, bytes)):
        raise ValueError("splits must be a sequence of split names, not a string")
    unknown_splits = sorted(set(splits) - set(SPLIT_ORDER))
    if unknown_splits:
        raise ValueError(f"unknown batch splits: {unknown_splits}")

    manifests = batch_module._local_directory(manifest_root, "manifest_root")
    outputs = batch_module._local_directory(output_root, "output_root")
    reports = batch_module._local_directory(report_root, "report_root")
    models = batch_module._local_directory(model_root, "model_root")
    raw_root = batch_module._source_root(source_root)
    correspondence = batch_module._local_file(correspondence_path, "correspondence_path")
    selected_splits = tuple(split for split in SPLIT_ORDER if split in set(splits))

    plan = build_shard_plan(
        manifest_root=manifests,
        report_root=reports,
        num_shards=num_shards,
        splits=selected_splits,
        max_items=max_items,
    )
    if plan_only:
        return {
            "schema_version": PARALLEL_BATCH_SCHEMA_VERSION,
            "status": "planned",
            "plan": plan,
            "motion_processing_started": False,
        }

    reports.mkdir(parents=True, exist_ok=True)
    lock_path = assert_output_is_local(reports / ".parallel_batch.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another parallel batch owns the report-root lock: {lock_path}"
            ) from exc

        started_at = _utc_now()
        started = time.perf_counter()
        worker_results = _launch_workers(
            plan,
            max_workers=min(max_workers, num_shards),
            output_root=outputs,
            model_root=models,
            source_root=raw_root,
            correspondence_path=correspondence,
            splits=selected_splits,
            target_fps=float(target_fps),
            chunk_size=chunk_size,
            force=force,
        )

        # This is deliberately the unmodified canonical runner.  It verifies
        # current-version artifacts, retries invalid/missing worker outputs, and
        # writes the official report directly in report_root.
        canonical = batch_module.run_batch(
            manifest_root=manifests,
            output_root=outputs,
            report_root=reports,
            model_root=models,
            source_root=raw_root,
            correspondence_path=correspondence,
            splits=selected_splits,
            target_fps=float(target_fps),
            chunk_size=chunk_size,
            resume=True,
            force=False,
            max_items=max_items,
        )

        worker_problem_count = sum(
            result.get("status") != "completed" for result in worker_results
        )
        canonical_success = bool(canonical.get("all_selected_items_completed", False))
        summary = {
            "schema_version": PARALLEL_BATCH_SCHEMA_VERSION,
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "retarget_version": RETARGET_VERSION,
            "status": "completed" if canonical_success else "failed",
            "success": canonical_success,
            "plan": plan,
            "settings": {
                "num_shards": num_shards,
                "max_workers": min(max_workers, num_shards),
                "splits": list(selected_splits),
                "max_items": max_items,
                "worker_force": force,
                "canonical_resume": True,
                "canonical_force": False,
                "output_root": str(outputs),
                "canonical_report_root": str(reports),
            },
            "workers": worker_results,
            "worker_problem_count": worker_problem_count,
            "worker_problems_recovered_by_canonical": bool(
                worker_problem_count and canonical_success
            ),
            "canonical": canonical,
        }
        summary_path = assert_output_is_local(reports / "parallel_batch_summary.json")
        _atomic_json(summary_path, summary)
        summary["summary_artifact"] = {
            "path": str(summary_path),
            "checksum_sha256": _sha256(summary_path),
        }
        return summary


def _add_common_worker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--correspondence", required=True)
    parser.add_argument("--worker-status", required=True)
    parser.add_argument("--required-retarget-version", required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLIT_ORDER, required=True)
    parser.add_argument("--target-fps", type=float, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the canonical ACCAD-to-G1 batch with disjoint worker shards."
    )
    subparsers = parser.add_subparsers(dest="command")
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    _add_common_worker_arguments(worker)

    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--splits", nargs="+", choices=SPLIT_ORDER, default=list(SPLIT_ORDER))
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write/inspect shard manifests without starting motion processing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "_worker":
        return _execute_worker(
            manifest_root=Path(args.manifest_root).expanduser().resolve(),
            output_root=Path(args.output_root).expanduser().resolve(),
            report_root=Path(args.report_root).expanduser().resolve(),
            model_root=Path(args.model_root).expanduser().resolve(),
            source_root=Path(args.source_root).expanduser().resolve(),
            correspondence_path=Path(args.correspondence).expanduser().resolve(),
            worker_status_path=Path(args.worker_status).expanduser().resolve(),
            required_retarget_version=args.required_retarget_version,
            splits=tuple(args.splits),
            target_fps=args.target_fps,
            chunk_size=args.chunk_size,
            force=args.force,
        )

    config = load_config(args.config)
    configured_paths = config.get("paths", {})
    work_root = assert_output_is_local(local_path(configured_paths.get("work", "work")))
    model_root = assert_output_is_local(local_path(configured_paths.get("models", "models")))
    human_config = config.get("human", {})
    try:
        result = run_parallel_batch(
            manifest_root=work_root / "manifests",
            output_root=work_root,
            report_root=work_root / "reports" / "batch",
            model_root=model_root,
            source_root=ACCAD_ROOT,
            correspondence_path=DEFAULT_CORRESPONDENCE,
            splits=tuple(args.splits),
            target_fps=float(human_config.get("target_fps", 50.0)),
            chunk_size=int(human_config.get("chunk_size", 128)),
            num_shards=args.num_shards,
            max_workers=args.max_workers,
            force=args.force,
            max_items=args.max_items,
            plan_only=args.plan_only,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2, sort_keys=True))
    if args.plan_only:
        return 0
    return 0 if result.get("success", False) else 2


__all__ = [
    "PARALLEL_BATCH_SCHEMA_VERSION",
    "SHARD_ASSIGNMENT",
    "build_shard_plan",
    "run_parallel_batch",
]
