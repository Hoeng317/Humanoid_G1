"""Promote independently validated batch outputs into training manifests.

The batch runner intentionally does not mutate training manifests while it is
retargeting.  This module consumes the *finished* batch JSONL snapshot, checks
its provenance against ``batch_summary.json`` and the per-item reports, then
independently revalidates each G1 archive before publishing deterministic
train/validation/test manifests.

Only offline/kinematic Gate-3 is asserted here.  The serialized archives keep
``gate3_complete=false`` until a live Isaac replay has been performed; that is
orthogonal to whether an archive is safe to use as a tracking reference.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .batch import BATCH_SCHEMA_VERSION, SPLIT_ORDER
from .g1_validation import validate_g1_archive
from .paths import PIPELINE_ROOT, assert_output_is_local
from .retarget import RETARGET_VERSION


MANIFEST_SCHEMA_VERSION = "accad_g1_gate3_manifest_v1"
BUILD_SCHEMA_VERSION = "accad_g1_gate3_manifest_build_v1"
DEFAULT_ITEMS_JSONL = PIPELINE_ROOT / "work" / "reports" / "batch" / "batch_items.jsonl"
DEFAULT_BATCH_SUMMARY = PIPELINE_ROOT / "work" / "reports" / "batch" / "batch_summary.json"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "work" / "manifests"
DEFAULT_REFERENCE_ROOT = PIPELINE_ROOT / "work" / "g1"
_SHA256_LENGTH = 64


class Gate3ManifestError(ValueError):
    """Raised when a batch snapshot cannot be promoted without ambiguity."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checksum(value: Any, label: str) -> str:
    text = str(value).lower()
    if len(text) != _SHA256_LENGTH:
        raise Gate3ManifestError(f"{label} must be a 64-character SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise Gate3ManifestError(f"{label} is not hexadecimal") from exc
    return text


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Gate3ManifestError(f"{label} must be a JSON mapping")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise Gate3ManifestError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Gate3ManifestError(f"{label} must be an integer") from exc
    if parsed < minimum:
        raise Gate3ManifestError(f"{label} must be at least {minimum}")
    return parsed


def _scalar(arrays: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in arrays:
        raise Gate3ManifestError(f"G1 archive is missing scalar {name!r}")
    value = arrays[name]
    if value.shape != () or value.dtype.hasobject:
        raise Gate3ManifestError(f"G1 archive scalar {name!r} has unsafe shape/dtype")
    return value.item()


def _inside(path: str | Path, root: Path, label: str, *, file: bool = False) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise Gate3ManifestError(f"{label} must stay below {root.resolve()}: {resolved}") from exc
    if file and not resolved.is_file():
        raise Gate3ManifestError(f"{label} is not an existing file: {resolved}")
    return resolved


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Gate3ManifestError(f"cannot read {label} {path}: {exc}") from exc
    return _mapping(document, label)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise Gate3ManifestError(
                        f"batch items JSONL contains a blank line at {line_number}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise Gate3ManifestError(
                        f"invalid batch items JSONL line {line_number}: {exc}"
                    ) from exc
                records.append(_mapping(value, f"batch item line {line_number}"))
    except (OSError, UnicodeError) as exc:
        raise Gate3ManifestError(f"cannot read batch items JSONL {path}: {exc}") from exc
    if not records:
        raise Gate3ManifestError("batch items JSONL is empty")
    return records


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = assert_output_is_local(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _canonical_g1_relative(source_file: str) -> Path:
    source = Path(source_file)
    if source.is_absolute() or ".." in source.parts or source.name in {"", "."}:
        raise Gate3ManifestError(f"unsafe source_file in batch item: {source_file!r}")
    if not source.name.endswith("_stageii.npz"):
        raise Gate3ManifestError(f"non-canonical Stage-II source_file: {source_file!r}")
    return source.with_name(
        source.name.removesuffix("_stageii.npz") + "_g1_50hz.npz"
    )


def _verify_summary(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
    items_path: Path,
    records: Sequence[Mapping[str, Any]],
    require_complete_batch: bool,
) -> Mapping[str, str]:
    if summary.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise Gate3ManifestError(
            f"unsupported batch summary schema: {summary.get('schema_version')!r}"
        )
    artifacts = _mapping(summary.get("artifacts"), "batch summary artifacts")
    jsonl = _mapping(artifacts.get("items_jsonl"), "batch summary items_jsonl artifact")
    declared_path = _inside(jsonl.get("path", ""), PIPELINE_ROOT, "summary items JSONL", file=True)
    if declared_path != items_path:
        raise Gate3ManifestError(
            f"batch summary names a different JSONL: {declared_path} != {items_path}"
        )
    expected_jsonl_checksum = _checksum(
        jsonl.get("checksum_sha256"), "batch summary items JSONL checksum"
    )
    actual_jsonl_checksum = _sha256(items_path)
    if actual_jsonl_checksum != expected_jsonl_checksum:
        raise Gate3ManifestError("batch items JSONL checksum does not match batch summary")
    if _integer(jsonl.get("records"), "batch summary JSONL record count") != len(records):
        raise Gate3ManifestError("batch items JSONL record count does not match batch summary")

    counts = _mapping(summary.get("counts"), "batch summary counts")
    selected = _integer(counts.get("selected_total"), "selected_total")
    completed = _integer(counts.get("completed"), "completed")
    failed = _integer(counts.get("failed"), "failed")
    if selected != len(records) or completed + failed != selected:
        raise Gate3ManifestError("batch summary status counts are inconsistent with JSONL")
    observed = Counter(str(record.get("status")) for record in records)
    if observed.get("completed", 0) != completed or observed.get("failed", 0) != failed:
        raise Gate3ManifestError("batch item statuses do not match batch summary")

    settings = _mapping(summary.get("settings"), "batch summary settings")
    if require_complete_batch:
        candidate = _integer(counts.get("candidate_total"), "candidate_total")
        candidate_by_split = _mapping(
            counts.get("candidate_by_split"), "candidate_by_split"
        )
        selected_by_split = _mapping(
            counts.get("selected_by_split"), "selected_by_split"
        )
        if (
            not bool(summary.get("all_selected_items_completed", False))
            or failed != 0
            or selected != completed
            or selected != candidate
            or settings.get("max_items") is not None
            or tuple(settings.get("splits", ())) != SPLIT_ORDER
        ):
            raise Gate3ManifestError(
                "a complete train/validation/test batch is required for manifest promotion"
            )
        for split in SPLIT_ORDER:
            if _integer(candidate_by_split.get(split), f"candidate_by_split.{split}") != _integer(
                selected_by_split.get(split), f"selected_by_split.{split}"
            ):
                raise Gate3ManifestError(f"batch did not select every {split} candidate")

    report_checksums = _mapping(
        artifacts.get("item_report_checksums_sha256"),
        "batch summary item-report checksums",
    )
    if set(report_checksums) != {str(record.get("item_id")) for record in records}:
        raise Gate3ManifestError("batch summary item-report set does not match JSONL item IDs")
    return {
        str(item_id): _checksum(value, f"item report checksum for {item_id}")
        for item_id, value in report_checksums.items()
    }


def _verify_item_report(
    item: Mapping[str, Any],
    summary_report_checksums: Mapping[str, str] | None,
) -> tuple[Path, str]:
    item_id = str(item.get("item_id", ""))
    path = _inside(
        item.get("item_report_path", ""), PIPELINE_ROOT, f"item report for {item_id}", file=True
    )
    expected = _checksum(
        item.get("item_report_checksum_sha256"), f"JSONL item report checksum for {item_id}"
    )
    actual = _sha256(path)
    if actual != expected:
        raise Gate3ManifestError(f"item report checksum mismatch for {item_id}")
    if summary_report_checksums is not None and summary_report_checksums.get(item_id) != actual:
        raise Gate3ManifestError(f"summary item report checksum mismatch for {item_id}")
    report = _read_json(path, f"item report for {item_id}")
    for key in ("item_id", "split", "source_file", "status"):
        if report.get(key) != item.get(key):
            raise Gate3ManifestError(f"item report field {key!r} disagrees for {item_id}")
    return path, actual


def _archive_record(
    item: Mapping[str, Any],
    *,
    reference_root: Path,
    required_retarget_version: str,
    revalidate: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    item_id = str(item.get("item_id", ""))
    split = str(item.get("split", ""))
    source_file = str(item.get("source_file", ""))
    if split not in SPLIT_ORDER or item_id != f"{split}:{source_file}":
        raise Gate3ManifestError(f"invalid split/item_id relation: {item_id!r}")
    if item.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise Gate3ManifestError(f"unsupported batch item schema for {item_id}")
    if item.get("status") != "completed" or item.get("error") is not None:
        raise Gate3ManifestError(f"cannot promote non-completed batch item {item_id}")

    stages = _mapping(item.get("stages"), f"stages for {item_id}")
    g1_stage = _mapping(stages.get("g1"), f"g1 stage for {item_id}")
    validation = _mapping(g1_stage.get("validation"), f"g1 validation for {item_id}")
    if validation.get("gate3_kinematic_complete") is not True:
        raise Gate3ManifestError(f"batch Gate-3 kinematic validation did not pass for {item_id}")

    artifacts = _mapping(item.get("artifacts"), f"artifacts for {item_id}")
    g1_artifact = _mapping(artifacts.get("g1"), f"G1 artifact for {item_id}")
    human_artifact = _mapping(artifacts.get("human"), f"human artifact for {item_id}")
    if g1_artifact.get("exists") is not True or human_artifact.get("exists") is not True:
        raise Gate3ManifestError(f"required artifact was absent when batch recorded {item_id}")
    g1_path = _inside(
        g1_artifact.get("path", ""), reference_root, f"G1 archive for {item_id}", file=True
    )
    human_path = _inside(
        human_artifact.get("path", ""), PIPELINE_ROOT, f"human archive for {item_id}", file=True
    )
    expected_relative = _canonical_g1_relative(source_file)
    actual_relative = g1_path.relative_to(reference_root)
    if actual_relative != expected_relative:
        raise Gate3ManifestError(
            f"G1 archive path is not canonical for {item_id}: {actual_relative}"
        )

    expected_g1_checksum = _checksum(
        g1_artifact.get("checksum_sha256"), f"G1 artifact checksum for {item_id}"
    )
    actual_g1_checksum = _sha256(g1_path)
    if actual_g1_checksum != expected_g1_checksum:
        raise Gate3ManifestError(f"G1 archive checksum changed after batch for {item_id}")
    expected_size = _integer(g1_artifact.get("size_bytes"), f"G1 artifact size for {item_id}")
    if g1_path.stat().st_size != expected_size:
        raise Gate3ManifestError(f"G1 archive size changed after batch for {item_id}")
    expected_human_checksum = _checksum(
        human_artifact.get("checksum_sha256"), f"human artifact checksum for {item_id}"
    )
    if _sha256(human_path) != expected_human_checksum:
        raise Gate3ManifestError(f"human archive checksum changed after batch for {item_id}")

    source_manifest_checksum = _checksum(
        item.get("source_manifest_checksum_sha256"), f"source checksum for {item_id}"
    )
    if _checksum(item.get("source_actual_checksum_sha256"), f"actual source checksum for {item_id}") != source_manifest_checksum:
        raise Gate3ManifestError(f"raw source checksum disagreed during batch for {item_id}")

    try:
        with np.load(g1_path, allow_pickle=False) as arrays:
            object_keys = [name for name in arrays.files if arrays[name].dtype.hasobject]
            if object_keys:
                raise Gate3ManifestError(
                    f"object arrays are forbidden in G1 archive {item_id}: {object_keys}"
                )
            retarget_version = str(_scalar(arrays, "retarget_version"))
            fps = float(_scalar(arrays, "fps"))
            frame_count = _integer(_scalar(arrays, "frame_count"), "frame_count", minimum=2)
            archive_source = str(_scalar(arrays, "source_stageii_file"))
            archive_source_checksum = _checksum(
                _scalar(arrays, "source_stageii_checksum_sha256"),
                f"archive Stage-II checksum for {item_id}",
            )
            archive_human_checksum = _checksum(
                _scalar(arrays, "source_checksum_sha256"),
                f"archive human checksum for {item_id}",
            )
            contract = {
                "retarget_version": retarget_version,
                "g1_contract_checksum_sha256": _checksum(
                    _scalar(arrays, "g1_contract_checksum_sha256"), "G1 contract checksum"
                ),
                "g1_urdf_checksum_sha256": _checksum(
                    _scalar(arrays, "g1_urdf_checksum_sha256"), "G1 URDF checksum"
                ),
                "correspondence_checksum_sha256": _checksum(
                    _scalar(arrays, "correspondence_checksum_sha256"),
                    "correspondence checksum",
                ),
            }
    except Gate3ManifestError:
        raise
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise Gate3ManifestError(f"cannot inspect G1 archive for {item_id}: {exc}") from exc

    if not np.isfinite(fps) or fps <= 0.0:
        raise Gate3ManifestError(f"invalid FPS in G1 archive for {item_id}: {fps}")
    if retarget_version != required_retarget_version:
        raise Gate3ManifestError(
            f"outdated retarget version for {item_id}: {retarget_version!r}; "
            f"required {required_retarget_version!r}"
        )
    if archive_source != source_file or archive_source_checksum != source_manifest_checksum:
        raise Gate3ManifestError(f"G1 archive Stage-II provenance mismatch for {item_id}")
    if archive_human_checksum != expected_human_checksum:
        raise Gate3ManifestError(f"G1 archive human provenance mismatch for {item_id}")

    if revalidate:
        try:
            fresh = validate_g1_archive(g1_path, expected_source_path=human_path)
        except Exception as exc:
            raise Gate3ManifestError(
                f"independent Gate-3 validation raised for {item_id}: {exc}"
            ) from exc
        if fresh.get("gate3_kinematic_complete") is not True:
            failures = fresh.get("required_failures") or fresh.get("errors")
            raise Gate3ManifestError(
                f"independent Gate-3 validation failed for {item_id}: {failures}"
            )
        fresh_checksum = fresh.get("archive_checksum_sha256")
        if fresh_checksum is not None and _checksum(
            fresh_checksum, f"fresh validation checksum for {item_id}"
        ) != actual_g1_checksum:
            raise Gate3ManifestError(f"fresh validation checksum mismatch for {item_id}")

    record = {
        "batch_item_id": item_id,
        "split": split,
        "source_file": source_file,
        "source_checksum_sha256": source_manifest_checksum,
        "human_file": human_path.relative_to(PIPELINE_ROOT).as_posix(),
        "human_checksum_sha256": expected_human_checksum,
        "g1_file": actual_relative.as_posix(),
        "g1_checksum_sha256": actual_g1_checksum,
        "g1_size_bytes": expected_size,
        "retarget_version": retarget_version,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": (frame_count - 1) / fps,
        "subject": item.get("subject"),
        "category": item.get("category"),
        "motion_name": item.get("motion_name"),
        "gate3_kinematic_complete": True,
        "live_isaac_replay_scope": "dataset-level separate gate",
    }
    return record, contract


def build_gate3_manifests(
    *,
    items_jsonl: str | Path = DEFAULT_ITEMS_JSONL,
    batch_summary: str | Path | None = DEFAULT_BATCH_SUMMARY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    reference_root: str | Path = DEFAULT_REFERENCE_ROOT,
    require_complete_batch: bool = True,
    revalidate: bool = True,
    required_retarget_version: str = RETARGET_VERSION,
) -> dict[str, Any]:
    """Build checksummed Gate-3 manifests from a stable batch snapshot.

    By default, publication is refused unless the batch covered every candidate
    in all three splits and every item completed.  ``require_complete_batch``
    may be disabled only for explicitly labelled smoke/debug subsets.
    """

    items_path = _inside(items_jsonl, PIPELINE_ROOT, "batch items JSONL", file=True)
    reference = _inside(reference_root, PIPELINE_ROOT, "G1 reference root")
    if not reference.is_dir():
        raise Gate3ManifestError(f"G1 reference root is not a directory: {reference}")
    outputs = _inside(output_dir, PIPELINE_ROOT, "manifest output directory")
    if outputs.exists() and not outputs.is_dir():
        raise Gate3ManifestError(f"manifest output is not a directory: {outputs}")
    if not isinstance(require_complete_batch, bool) or not isinstance(revalidate, bool):
        raise Gate3ManifestError("require_complete_batch and revalidate must be booleans")
    if not required_retarget_version:
        raise Gate3ManifestError("required_retarget_version cannot be empty")

    records = _read_jsonl(items_path)
    summary: Mapping[str, Any] | None = None
    summary_path: Path | None = None
    report_checksums: Mapping[str, str] | None = None
    if batch_summary is not None:
        summary_path = _inside(batch_summary, PIPELINE_ROOT, "batch summary", file=True)
        summary = _read_json(summary_path, "batch summary")
        report_checksums = _verify_summary(
            summary,
            summary_path=summary_path,
            items_path=items_path,
            records=records,
            require_complete_batch=require_complete_batch,
        )
    elif require_complete_batch:
        raise Gate3ManifestError(
            "batch_summary is required when require_complete_batch is enabled"
        )

    completed_items = [record for record in records if record.get("status") == "completed"]
    if not completed_items:
        raise Gate3ManifestError("batch snapshot has no completed items to promote")
    item_ids = [str(item.get("item_id", "")) for item in records]
    if len(item_ids) != len(set(item_ids)) or any(not value for value in item_ids):
        raise Gate3ManifestError("batch snapshot contains missing or duplicate item IDs")
    if require_complete_batch and len(completed_items) != len(records):
        raise Gate3ManifestError("complete promotion cannot filter failed batch records")

    promoted: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    contracts: list[dict[str, str]] = []
    item_report_provenance: dict[str, dict[str, str]] = {}
    for item in completed_items:
        report_path, report_checksum = _verify_item_report(item, report_checksums)
        record, contract = _archive_record(
            item,
            reference_root=reference,
            required_retarget_version=required_retarget_version,
            revalidate=revalidate,
        )
        promoted[record["split"]].append(record)
        contracts.append(contract)
        item_report_provenance[record["batch_item_id"]] = {
            "path": report_path.relative_to(PIPELINE_ROOT).as_posix(),
            "checksum_sha256": report_checksum,
        }

    for split in SPLIT_ORDER:
        promoted[split].sort(key=lambda record: record["g1_file"])
    flat = [record for split in SPLIT_ORDER for record in promoted[split]]
    for key in ("batch_item_id", "source_file", "g1_file", "g1_checksum_sha256"):
        values = [str(record[key]) for record in flat]
        if len(values) != len(set(values)):
            raise Gate3ManifestError(f"promoted records contain duplicate {key}")
    canonical_contracts = {
        json.dumps(contract, sort_keys=True, separators=(",", ":")) for contract in contracts
    }
    if len(canonical_contracts) != 1:
        raise Gate3ManifestError("promoted archives do not share one G1/URDF/correspondence contract")
    contract = contracts[0]

    items_checksum = _sha256(items_path)
    batch_finished_at = summary.get("finished_at_utc") if summary is not None else None
    source = {
        "batch_schema_version": BATCH_SCHEMA_VERSION,
        "batch_items_file": items_path.relative_to(PIPELINE_ROOT).as_posix(),
        "batch_items_checksum_sha256": items_checksum,
        "batch_summary_file": (
            summary_path.relative_to(PIPELINE_ROOT).as_posix() if summary_path else None
        ),
        "batch_summary_checksum_sha256": _sha256(summary_path) if summary_path else None,
        "batch_finished_at_utc": batch_finished_at,
        "complete_batch_required": require_complete_batch,
        "independent_revalidation_performed": revalidate,
    }
    common = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "ACCAD",
        "gate": "gate3_kinematic_complete",
        "live_isaac_replay": "separate dataset-level gate; not asserted per archive",
        "reference_root": reference.relative_to(PIPELINE_ROOT).as_posix(),
        "retarget_version": required_retarget_version,
        "contract": contract,
        "source_batch": source,
    }
    split_documents: dict[str, dict[str, Any]] = {}
    for split in SPLIT_ORDER:
        motions = promoted[split]
        split_documents[split] = {
            **common,
            "kind": f"split:{split}",
            "split": split,
            "num_motions": len(motions),
            "num_frames": sum(int(record["frame_count"]) for record in motions),
            "duration_sec": sum(float(record["duration_sec"]) for record in motions),
            "motions": motions,
        }

    combined = {
        **common,
        "kind": "splits",
        "num_motions": len(flat),
        "num_motions_by_split": {split: len(promoted[split]) for split in SPLIT_ORDER},
        "num_frames": sum(int(record["frame_count"]) for record in flat),
        "duration_sec": sum(float(record["duration_sec"]) for record in flat),
        "splits": promoted,
    }
    motion_set_digest = hashlib.sha256(
        json.dumps(
            [
                [record["split"], record["g1_file"], record["g1_checksum_sha256"]]
                for record in flat
            ],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    combined["motion_set_checksum_sha256"] = motion_set_digest
    for document in split_documents.values():
        document["motion_set_checksum_sha256"] = motion_set_digest

    # Validate every input and construct all documents before publishing any of
    # them.  Each individual replace is atomic; the build summary is written
    # last and acts as the commit witness for the complete set.
    output_paths: dict[str, Path] = {}
    for split in SPLIT_ORDER:
        path = outputs / f"g1_{split}.json"
        _atomic_json(path, split_documents[split])
        output_paths[split] = path
    combined_path = outputs / "g1_splits.json"
    _atomic_json(combined_path, combined)
    output_paths["combined"] = combined_path

    build_report: dict[str, Any] = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "status": "completed",
        "source_batch": source,
        "reference_root": reference.relative_to(PIPELINE_ROOT).as_posix(),
        "retarget_version": required_retarget_version,
        "contract": contract,
        "counts": {
            "promoted_total": len(flat),
            "promoted_by_split": {split: len(promoted[split]) for split in SPLIT_ORDER},
            "source_records": len(records),
            "source_completed_records": len(completed_items),
        },
        "num_frames": combined["num_frames"],
        "duration_sec": combined["duration_sec"],
        "motion_set_checksum_sha256": motion_set_digest,
        "manifests": {
            name: {
                "path": path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": _sha256(path),
                "num_motions": (
                    len(flat) if name == "combined" else len(promoted[name])
                ),
            }
            for name, path in output_paths.items()
        },
        "item_reports": item_report_provenance,
    }
    build_report_path = outputs / "g1_manifest_build.json"
    _atomic_json(build_report_path, build_report)
    result = dict(build_report)
    result["build_report"] = {
        "path": str(build_report_path),
        "checksum_sha256": _sha256(build_report_path),
    }
    result["manifest_paths"] = {name: str(path) for name, path in output_paths.items()}
    return result


__all__ = [
    "BUILD_SCHEMA_VERSION",
    "DEFAULT_BATCH_SUMMARY",
    "DEFAULT_ITEMS_JSONL",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REFERENCE_ROOT",
    "Gate3ManifestError",
    "MANIFEST_SCHEMA_VERSION",
    "build_gate3_manifests",
]
