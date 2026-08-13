"""Failure-isolated batch orchestration for the ACCAD → G1 gated pipeline.

The batch runner only keeps manifest records and small reports in memory.  Each
motion is reconstructed, independently validated, retargeted, and independently
validated before the next item begins.  Numeric motion arrays remain inside the
existing stage functions and are never accumulated here.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .g1_validation import validate_g1_archive
from .human import reconstruct_human_motion
from .paths import ACCAD_ROOT, PIPELINE_ROOT, assert_output_is_local
from .retarget import DEFAULT_CORRESPONDENCE, RETARGET_VERSION, retarget_human_to_g1
from .validation import validate_human_archive


BATCH_SCHEMA_VERSION = "accad_g1_batch_v1"
SPLIT_ORDER = ("train", "validation", "test")


class BatchManifestError(ValueError):
    """Raised before processing when Gate-1 split manifests are inconsistent."""


class BatchStageError(RuntimeError):
    """One item failed a generated/reused artifact gate."""


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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    destination = assert_output_is_local(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_jsonl(path: Path, payloads: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for payload in payloads
    )
    _atomic_write_text(path, text)


def _local_directory(path: str | Path, label: str) -> Path:
    resolved = assert_output_is_local(Path(path).expanduser().resolve())
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def _local_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(PIPELINE_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside {PIPELINE_ROOT}: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _source_root(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(ACCAD_ROOT)
    except ValueError as exc:
        raise ValueError(f"source_root must stay inside {ACCAD_ROOT}: {resolved}") from exc
    if not resolved.is_dir():
        raise FileNotFoundError(f"source_root does not exist: {resolved}")
    return resolved


def _load_split_manifest(path: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatchManifestError(f"cannot read {split} manifest {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise BatchManifestError(f"{split} manifest must be a JSON mapping")
    if document.get("kind") != f"split:{split}":
        raise BatchManifestError(
            f"{split} manifest kind must be 'split:{split}', got {document.get('kind')!r}"
        )
    motions = document.get("motions")
    if not isinstance(motions, list):
        raise BatchManifestError(f"{split} manifest motions must be a list")
    declared_count = document.get("num_motions")
    if declared_count is not None and int(declared_count) != len(motions):
        raise BatchManifestError(
            f"{split} manifest declares {declared_count} motions but stores {len(motions)}"
        )
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(motions):
        if not isinstance(record, dict):
            raise BatchManifestError(f"{split} motion {index} must be a mapping")
        source_file = record.get("source_file")
        checksum = record.get("checksum_sha256")
        if not isinstance(source_file, str) or not source_file.endswith("_stageii.npz"):
            raise BatchManifestError(
                f"{split} motion {index} has no canonical *_stageii.npz source_file"
            )
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise BatchManifestError(f"{split} motion {source_file} has no SHA-256")
        try:
            int(checksum, 16)
        except ValueError as exc:
            raise BatchManifestError(
                f"{split} motion {source_file} checksum is not hexadecimal"
            ) from exc
        if record.get("valid") is not True:
            raise BatchManifestError(f"invalid motion appears in {split} split: {source_file}")
        normalized.append(
            {
                "split": split,
                "source_file": source_file,
                "checksum_sha256": checksum.lower(),
                "subject": record.get("subject"),
                "category": record.get("category"),
                "motion_name": record.get("motion_name"),
                "source_num_frames": record.get("num_frames"),
                "source_fps": record.get("fps"),
                "source_duration_sec": record.get("duration_sec"),
            }
        )
    normalized.sort(key=lambda item: item["source_file"])
    return normalized, {
        "path": str(path),
        "checksum_sha256": _sha256(path),
        "count": len(normalized),
        "duration_sec": document.get("duration_sec"),
    }


def _select_records(
    manifest_root: Path,
    requested_splits: Sequence[str],
    max_items: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    unknown = sorted(set(requested_splits) - set(SPLIT_ORDER))
    if unknown:
        raise ValueError(f"unknown batch splits: {unknown}")
    selected_splits = tuple(split for split in SPLIT_ORDER if split in set(requested_splits))
    if not selected_splits:
        raise ValueError("at least one Gate-1 split must be selected")
    all_records: list[dict[str, Any]] = []
    manifest_info: dict[str, Any] = {}
    candidate_counts: dict[str, int] = {}
    for split in selected_splits:
        path = manifest_root / f"accad_{split}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Gate-1 {split} manifest is missing: {path}")
        records, info = _load_split_manifest(path, split)
        all_records.extend(records)
        manifest_info[split] = info
        candidate_counts[split] = len(records)

    source_files = [record["source_file"] for record in all_records]
    if len(source_files) != len(set(source_files)):
        raise BatchManifestError("a source_file appears in more than one Gate-1 split")
    checksums = [record["checksum_sha256"] for record in all_records]
    if len(checksums) != len(set(checksums)):
        raise BatchManifestError("a source checksum appears in more than one Gate-1 split")
    if max_items is not None:
        all_records = all_records[:max_items]
    return all_records, manifest_info, candidate_counts


def _canonical_paths(
    record: Mapping[str, Any],
    source_root: Path,
    output_root: Path,
    report_root: Path,
) -> dict[str, Path]:
    relative = Path(str(record["source_file"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise BatchManifestError(f"unsafe source_file in manifest: {relative}")
    source = (source_root / relative).resolve()
    try:
        source.relative_to(source_root)
    except ValueError as exc:
        raise BatchManifestError(f"source path escapes source_root: {relative}") from exc
    stem = relative.name.removesuffix("_stageii.npz")
    subdirectory = relative.parent
    human = assert_output_is_local(
        output_root / "human" / subdirectory / f"{stem}_human_50hz.npz"
    )
    g1 = assert_output_is_local(
        output_root / "g1" / subdirectory / f"{stem}_g1_50hz.npz"
    )
    item_report = assert_output_is_local(
        report_root / "items" / str(record["split"]) / subdirectory / f"{stem}.json"
    )
    return {"source": source, "human": human, "g1": g1, "item_report": item_report}


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "checksum_sha256": _sha256(path) if path.is_file() else None,
    }


def _require_gate(report: Mapping[str, Any], field: str, stage: str) -> None:
    if not bool(report.get(field, False)):
        details = report.get("required_failures") or report.get("errors") or report.get("statuses")
        raise BatchStageError(f"{stage} failed {field}: {details}")


def _human_stage(
    *,
    source: Path,
    output: Path,
    model_root: Path,
    target_fps: float,
    chunk_size: int,
    source_root: Path,
    resume: bool,
    force: bool,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, float]]:
    durations = {"generation_seconds": 0.0, "validation_seconds": 0.0}
    prior_validation: dict[str, Any] | None = None
    action = "generated"
    if output.exists() and not force:
        if not resume:
            raise FileExistsError(
                f"human output exists; use resume=True or force=True: {output}"
            )
        started = time.perf_counter()
        try:
            prior_validation = validate_human_archive(output, source_root, profile="generic")
        except Exception as exc:
            prior_validation = {
                "ok": False,
                "gate2_complete": False,
                "validation_exception": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        durations["validation_seconds"] += time.perf_counter() - started
        if bool(prior_validation.get("gate2_complete", False)):
            return "reused", {"ok": True, "output_path": str(output)}, prior_validation, durations
        action = "regenerated_invalid_existing"

    started = time.perf_counter()
    generation = reconstruct_human_motion(
        source,
        model_root,
        output,
        target_fps=target_fps,
        chunk_size=chunk_size,
    )
    durations["generation_seconds"] += time.perf_counter() - started
    started = time.perf_counter()
    validation = validate_human_archive(output, source_root, profile="generic")
    durations["validation_seconds"] += time.perf_counter() - started
    if prior_validation is not None:
        generation = dict(generation)
        generation["invalid_existing_validation"] = prior_validation
    _require_gate(validation, "gate2_complete", "Gate-2 validation")
    return action, generation, validation, durations


def _g1_stage(
    *,
    human: Path,
    output: Path,
    correspondence_path: Path,
    resume: bool,
    force: bool,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, float]]:
    durations = {"generation_seconds": 0.0, "validation_seconds": 0.0}
    prior_validation: dict[str, Any] | None = None
    action = "generated"
    if output.exists() and not force:
        if not resume:
            raise FileExistsError(
                f"G1 output exists; use resume=True or force=True: {output}"
            )
        started = time.perf_counter()
        try:
            prior_validation = validate_g1_archive(output, expected_source_path=human)
        except Exception as exc:
            prior_validation = {
                "ok": False,
                "gate3_kinematic_complete": False,
                "validation_exception": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        durations["validation_seconds"] += time.perf_counter() - started
        stored_retarget_version: str | None = None
        try:
            with np.load(output, allow_pickle=False) as archive:
                value = archive["retarget_version"]
                if value.shape == () and not value.dtype.hasobject:
                    stored_retarget_version = str(value.item())
        except (OSError, KeyError, ValueError):
            stored_retarget_version = None
        version_matches = stored_retarget_version == RETARGET_VERSION
        prior_validation["retarget_version_reuse_check"] = {
            "stored": stored_retarget_version,
            "required": RETARGET_VERSION,
            "matches": version_matches,
        }
        if bool(prior_validation.get("gate3_kinematic_complete", False)) and version_matches:
            return "reused", {"ok": True, "output_path": str(output)}, prior_validation, durations
        action = (
            "regenerated_outdated_existing"
            if bool(prior_validation.get("gate3_kinematic_complete", False))
            else "regenerated_invalid_existing"
        )

    started = time.perf_counter()
    generation = retarget_human_to_g1(
        human,
        output,
        correspondence_path=correspondence_path,
    )
    durations["generation_seconds"] += time.perf_counter() - started
    started = time.perf_counter()
    validation = validate_g1_archive(output, expected_source_path=human)
    durations["validation_seconds"] += time.perf_counter() - started
    if prior_validation is not None:
        generation = dict(generation)
        generation["invalid_existing_validation"] = prior_validation
    # The archive intentionally stores gate3_complete=false.  Offline batch
    # acceptance is the independent Pinocchio/contract result below; live
    # Isaac replay remains a separate aggregate gate.
    _require_gate(validation, "gate3_kinematic_complete", "Gate-3 kinematic validation")
    return action, generation, validation, durations


def _run_item(
    record: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
    model_root: Path,
    correspondence_path: Path,
    source_root: Path,
    target_fps: float,
    chunk_size: int,
    resume: bool,
    force: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    item: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "item_id": f"{record['split']}:{record['source_file']}",
        "split": record["split"],
        "source_file": record["source_file"],
        "source_manifest_checksum_sha256": record["checksum_sha256"],
        "subject": record.get("subject"),
        "category": record.get("category"),
        "motion_name": record.get("motion_name"),
        "status": "running",
        "started_at_utc": _utc_now(),
        "paths": {name: str(path) for name, path in paths.items()},
        "durations": {},
        "stages": {},
        "artifacts": {},
        "error": None,
    }
    try:
        source = paths["source"]
        verify_started = time.perf_counter()
        if not source.is_file():
            raise FileNotFoundError(f"raw source does not exist: {source}")
        actual_checksum = _sha256(source)
        item["durations"]["source_verification_seconds"] = time.perf_counter() - verify_started
        item["source_actual_checksum_sha256"] = actual_checksum
        if actual_checksum != record["checksum_sha256"]:
            raise BatchStageError(
                f"raw source checksum mismatch: expected {record['checksum_sha256']}, got {actual_checksum}"
            )

        paths["human"].parent.mkdir(parents=True, exist_ok=True)
        human_action, human_generation, human_validation, human_durations = _human_stage(
            source=source,
            output=paths["human"],
            model_root=model_root,
            target_fps=target_fps,
            chunk_size=chunk_size,
            source_root=source_root,
            resume=resume,
            force=force,
        )
        item["stages"]["human"] = {
            "action": human_action,
            "generation": human_generation,
            "validation": human_validation,
        }
        item["durations"]["human"] = human_durations
        item["artifacts"]["human"] = _artifact(paths["human"])

        paths["g1"].parent.mkdir(parents=True, exist_ok=True)
        g1_action, g1_generation, g1_validation, g1_durations = _g1_stage(
            human=paths["human"],
            output=paths["g1"],
            correspondence_path=correspondence_path,
            resume=resume,
            force=force,
        )
        item["stages"]["g1"] = {
            "action": g1_action,
            "generation": g1_generation,
            "validation": g1_validation,
        }
        item["durations"]["g1"] = g1_durations
        item["artifacts"]["g1"] = _artifact(paths["g1"])
        item["status"] = "completed"
    except Exception as exc:
        item["status"] = "failed"
        item["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if paths["human"].is_file() and "human" not in item["artifacts"]:
            item["artifacts"]["human"] = _artifact(paths["human"])
        if paths["g1"].is_file() and "g1" not in item["artifacts"]:
            item["artifacts"]["g1"] = _artifact(paths["g1"])
    item["finished_at_utc"] = _utc_now()
    item["durations"]["total_seconds"] = time.perf_counter() - started
    return item


def run_batch(
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
    resume: bool = True,
    force: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Run Gate 2 and offline Gate 3 for deterministic Gate-1 split records.

    ``max_items`` truncates the global deterministic sequence after manifests
    have been ordered as train → validation → test and each split has been
    sorted by ``source_file``.  It counts selected records, including resumed
    and failed records, so reruns always select the same prefix.

    ``resume=True`` reuses only artifacts that independently validate.  An
    invalid existing artifact is regenerated atomically.  ``resume=False``
    treats any existing destination as an item failure.  ``force=True`` always
    regenerates and takes precedence over resume behavior.
    """

    if max_items is not None and (
        isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 0
    ):
        raise ValueError("max_items must be None or a non-negative integer")
    if not isinstance(resume, bool) or not isinstance(force, bool):
        raise ValueError("resume and force must be booleans")
    if isinstance(splits, (str, bytes)):
        raise ValueError("splits must be a sequence of split names, not a string")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("target_fps must be finite and positive")

    manifests = _local_directory(manifest_root, "manifest_root")
    outputs = _local_directory(output_root, "output_root")
    reports = _local_directory(report_root, "report_root")
    models = _local_directory(model_root, "model_root")
    raw_root = _source_root(source_root)
    correspondence = _local_file(correspondence_path, "correspondence_path")
    records, manifest_info, candidate_counts = _select_records(
        manifests, splits, max_items
    )

    started_at = _utc_now()
    wall_started = time.perf_counter()
    item_results: list[dict[str, Any]] = []
    item_report_checksums: dict[str, str] = {}
    for ordinal, record in enumerate(records):
        paths = _canonical_paths(record, raw_root, outputs, reports)
        result = _run_item(
            record,
            paths=paths,
            model_root=models,
            correspondence_path=correspondence,
            source_root=raw_root,
            target_fps=float(target_fps),
            chunk_size=chunk_size,
            resume=bool(resume),
            force=bool(force),
        )
        result["ordinal"] = ordinal
        _atomic_json(paths["item_report"], result)
        report_checksum = _sha256(paths["item_report"])
        result["item_report_path"] = str(paths["item_report"])
        result["item_report_checksum_sha256"] = report_checksum
        item_report_checksums[result["item_id"]] = report_checksum
        item_results.append(result)

    jsonl_path = assert_output_is_local(reports / "batch_items.jsonl")
    _atomic_jsonl(jsonl_path, item_results)
    jsonl_checksum = _sha256(jsonl_path)
    status_counts = Counter(str(item["status"]) for item in item_results)
    split_selected = Counter(str(item["split"]) for item in item_results)
    human_actions = Counter(
        str(item["stages"].get("human", {}).get("action"))
        for item in item_results
        if "human" in item["stages"]
    )
    g1_actions = Counter(
        str(item["stages"].get("g1", {}).get("action"))
        for item in item_results
        if "g1" in item["stages"]
    )
    failures = [
        {
            "ordinal": item["ordinal"],
            "item_id": item["item_id"],
            "split": item["split"],
            "source_file": item["source_file"],
            "error": item["error"],
            "item_report_path": item["item_report_path"],
        }
        for item in item_results
        if item["status"] == "failed"
    ]
    total_item_seconds = sum(float(item["durations"]["total_seconds"]) for item in item_results)
    stage_duration_totals = {
        "source_verification_seconds": sum(
            float(item["durations"].get("source_verification_seconds", 0.0))
            for item in item_results
        ),
        "human_generation_seconds": sum(
            float(item["durations"].get("human", {}).get("generation_seconds", 0.0))
            for item in item_results
        ),
        "human_validation_seconds": sum(
            float(item["durations"].get("human", {}).get("validation_seconds", 0.0))
            for item in item_results
        ),
        "g1_generation_seconds": sum(
            float(item["durations"].get("g1", {}).get("generation_seconds", 0.0))
            for item in item_results
        ),
        "g1_validation_seconds": sum(
            float(item["durations"].get("g1", {}).get("validation_seconds", 0.0))
            for item in item_results
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "settings": {
            "source_root": str(raw_root),
            "manifest_root": str(manifests),
            "output_root": str(outputs),
            "report_root": str(reports),
            "model_root": str(models),
            "correspondence_path": str(correspondence),
            "correspondence_checksum_sha256": _sha256(correspondence),
            "splits": [split for split in SPLIT_ORDER if split in set(splits)],
            "target_fps": float(target_fps),
            "chunk_size": chunk_size,
            "resume": bool(resume),
            "force": bool(force),
            "max_items": max_items,
            "max_items_semantics": "global train-validation-test prefix; resumed and failed records count",
        },
        "manifests": manifest_info,
        "counts": {
            "candidate_total": sum(candidate_counts.values()),
            "candidate_by_split": candidate_counts,
            "selected_total": len(records),
            "selected_by_split": {
                split: int(split_selected.get(split, 0)) for split in SPLIT_ORDER
            },
            "completed": int(status_counts.get("completed", 0)),
            "failed": int(status_counts.get("failed", 0)),
            "human_actions": dict(sorted(human_actions.items())),
            "g1_actions": dict(sorted(g1_actions.items())),
        },
        "durations": {
            "wall_seconds": time.perf_counter() - wall_started,
            "sum_item_seconds": total_item_seconds,
            "stage_totals": stage_duration_totals,
            "source_duration_selected_seconds": sum(
                float(record.get("source_duration_sec") or 0.0) for record in records
            ),
        },
        "failures": failures,
        "all_selected_items_completed": bool(records) and not failures,
        "live_isaac_replay_complete": False,
        "live_isaac_replay_note": (
            "Batch Gate-3 acceptance is Pinocchio/contract kinematic validation; "
            "the serialized gate3_complete flag remains false until live Isaac replay."
        ),
        "artifacts": {
            "items_jsonl": {
                "path": str(jsonl_path),
                "checksum_sha256": jsonl_checksum,
                "records": len(item_results),
            },
            "item_report_checksums_sha256": item_report_checksums,
        },
    }
    summary_path = assert_output_is_local(reports / "batch_summary.json")
    _atomic_json(summary_path, summary)
    summary_checksum = _sha256(summary_path)
    result = dict(summary)
    result["artifacts"] = dict(summary["artifacts"])
    result["artifacts"]["summary_json"] = {
        "path": str(summary_path),
        "checksum_sha256": summary_checksum,
    }
    return result


__all__ = [
    "BATCH_SCHEMA_VERSION",
    "BatchManifestError",
    "BatchStageError",
    "SPLIT_ORDER",
    "run_batch",
]
