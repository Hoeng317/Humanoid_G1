"""Command-line orchestration for the gated ACCAD-to-G1 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import (
    ACCAD_ROOT,
    DEFAULT_B3_INPUT,
    DEFAULT_CONFIG,
    PIPELINE_ROOT,
    assert_output_is_local,
    load_config,
    local_path,
)


def _resolved_roots(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    paths = config.get("paths", {})
    work_root = assert_output_is_local(local_path(paths.get("work", "work")))
    model_root = assert_output_is_local(local_path(paths.get("models", "models")))
    vendor_root = assert_output_is_local(local_path(paths.get("vendor", "vendor")))
    return work_root, model_root, vendor_root


def _enable_local_vendor(vendor_root: Path) -> None:
    if vendor_root.is_dir() and str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = assert_output_is_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_json_ready(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    """Load a JSON object for status reporting without mutating any artifact."""

    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_final_artifact_snapshot(
    manifest: dict[str, Any] | None,
    *,
    accad_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Re-hash every unique artifact in a final manifest for live Gate 5 status.

    The final manifest groups the same file under multiple categories.  Records
    are therefore deduplicated by ``(scope, path)`` before hashing.  Invalid or
    escaping paths fail closed and are reported instead of raising from the
    read-only ``status`` command.
    """

    errors: list[str] = []
    checked: dict[tuple[str, str], tuple[int, str]] = {}
    if not isinstance(manifest, dict):
        return {
            "current": False,
            "verified_unique_artifact_count": 0,
            "declared_unique_artifact_count": None,
            "errors": ["final artifact manifest is absent or invalid"],
        }
    categories = manifest.get("artifacts")
    if not isinstance(categories, dict) or not categories:
        errors.append("artifact categories are absent or invalid")
        categories = {}
    if manifest.get("artifact_root") != ".":
        errors.append("artifact_root must be '.'")

    roots = {
        "accad": accad_root.resolve(),
        "project_read_only": project_root.resolve(),
    }
    for category, records in categories.items():
        if not isinstance(category, str) or not isinstance(records, list):
            errors.append(f"invalid artifact category {category!r}")
            continue
        for index, value in enumerate(records):
            label = f"{category}[{index}]"
            if not isinstance(value, dict):
                errors.append(f"{label} is not an object")
                continue
            scope = value.get("scope")
            relative_value = value.get("path")
            if scope not in roots or not isinstance(relative_value, str) or not relative_value:
                errors.append(f"{label} has an invalid scope/path")
                continue
            relative = Path(relative_value)
            if relative.is_absolute():
                errors.append(f"{label} path must be relative")
                continue
            root = roots[scope]
            resolved = (root / relative).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                errors.append(f"{label} path escapes its {scope} root")
                continue
            size = value.get("bytes")
            checksum = value.get("checksum_sha256")
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or not isinstance(checksum, str)
                or len(checksum) != 64
            ):
                errors.append(f"{label} has invalid size/checksum metadata")
                continue
            try:
                int(checksum, 16)
            except ValueError:
                errors.append(f"{label} checksum is not hexadecimal")
                continue
            declared = (size, checksum.lower())
            key = (scope, relative.as_posix())
            previous = checked.get(key)
            if previous is not None:
                if previous != declared:
                    errors.append(f"{label} conflicts with a duplicate artifact record")
                continue
            checked[key] = declared
            if not resolved.is_file():
                errors.append(f"{label} artifact is missing")
                continue
            try:
                actual_size = resolved.stat().st_size
                actual_checksum = _sha256_file(resolved)
            except OSError as exc:
                errors.append(f"{label} cannot be read: {exc}")
                continue
            if actual_size != size:
                errors.append(f"{label} size changed")
            if actual_checksum != checksum.lower():
                errors.append(f"{label} checksum changed")

    declared_count = manifest.get("artifact_unique_count")
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count <= 0
    ):
        errors.append("artifact_unique_count must be a positive integer")
    elif declared_count != len(checked):
        errors.append(
            "artifact_unique_count disagrees with deduplicated manifest records"
        )
    return {
        "current": not errors,
        "verified_unique_artifact_count": len(checked),
        "declared_unique_artifact_count": declared_count,
        "errors": errors,
    }


def _run_audit(config: dict[str, Any]) -> dict[str, Any]:
    from .data import audit_dataset

    work_root, _, _ = _resolved_roots(config)
    manifest_root = work_root / "manifests"
    summary = audit_dataset(ACCAD_ROOT, config, manifest_root)
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def _run_preflight(config: dict[str, Any], *, quiet: bool = False) -> dict[str, Any]:
    work_root, model_root, vendor_root = _resolved_roots(config)
    _enable_local_vendor(vendor_root)
    from .human import check_human_prerequisites

    report = check_human_prerequisites(PIPELINE_ROOT, model_root, vendor_root)
    report["ready"] = bool(report.get("ready", report.get("ok", False)))
    report.setdefault("checked_at_utc", datetime.now(timezone.utc).isoformat())
    report.setdefault("gate", 2)
    _write_json_atomic(work_root / "reports" / "gate2_prerequisites.json", report)
    if not quiet:
        print(json.dumps(_json_ready(report), ensure_ascii=False, indent=2, sort_keys=True))
    return report


def _default_human_output(input_path: Path, work_root: Path) -> Path:
    try:
        category = input_path.resolve().relative_to(ACCAD_ROOT).parts[0]
    except (ValueError, IndexError):
        category = "external"
    stem = input_path.name.removesuffix("_stageii.npz")
    return work_root / "human" / category / f"{stem}_human_50hz.npz"


def _default_g1_output(human_path: Path, work_root: Path) -> Path:
    try:
        category = human_path.resolve().relative_to(work_root / "human").parts[0]
    except (ValueError, IndexError):
        category = "external"
    stem = human_path.name.removesuffix("_human_50hz.npz")
    return work_root / "g1" / category / f"{stem}_g1_50hz.npz"


def _run_reconstruct(
    config: dict[str, Any], input_value: str | Path | None, output_value: str | Path | None
) -> dict[str, Any]:
    work_root, model_root, vendor_root = _resolved_roots(config)
    _enable_local_vendor(vendor_root)
    from .human import reconstruct_human_motion

    input_path = Path(input_value).expanduser().resolve() if input_value else DEFAULT_B3_INPUT
    output_path = (
        assert_output_is_local(Path(output_value).expanduser().resolve())
        if output_value
        else _default_human_output(input_path, work_root)
    )
    human_cfg = config.get("human", {})
    report = reconstruct_human_motion(
        input_path,
        model_root,
        output_path,
        target_fps=float(human_cfg.get("target_fps", 50.0)),
        chunk_size=int(human_cfg.get("chunk_size", 128)),
    )
    report_path = work_root / "reports" / f"{output_path.stem}_gate2.json"
    _write_json_atomic(report_path, report)
    print(json.dumps(_json_ready(report), ensure_ascii=False, indent=2, sort_keys=True))
    return report


def _run_validate_human(
    config: dict[str, Any], input_value: str | Path | None, profile: str
) -> dict[str, Any]:
    work_root, _, vendor_root = _resolved_roots(config)
    _enable_local_vendor(vendor_root)
    from .validation import validate_human_archive

    input_path = (
        Path(input_value).expanduser().resolve()
        if input_value
        else _default_human_output(DEFAULT_B3_INPUT, work_root)
    )
    try:
        input_path.relative_to(PIPELINE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"human archive must stay inside {PIPELINE_ROOT}; got {input_path}"
        ) from exc
    report = validate_human_archive(input_path, ACCAD_ROOT, profile=profile)
    report_path = work_root / "reports" / f"{input_path.stem}_validation.json"
    _write_json_atomic(report_path, report)
    print(json.dumps(_json_ready(report), ensure_ascii=False, indent=2, sort_keys=True))
    return report


def _run_retarget(
    config: dict[str, Any],
    input_value: str | Path | None,
    output_value: str | Path | None,
    correspondence_value: str | Path | None,
) -> dict[str, Any]:
    work_root, _, vendor_root = _resolved_roots(config)
    _enable_local_vendor(vendor_root)
    from .retarget import DEFAULT_CORRESPONDENCE, retarget_human_to_g1

    human_path = (
        Path(input_value).expanduser().resolve()
        if input_value
        else _default_human_output(DEFAULT_B3_INPUT, work_root)
    )
    output_path = (
        assert_output_is_local(Path(output_value).expanduser().resolve())
        if output_value
        else _default_g1_output(human_path, work_root)
    )
    correspondence_path = (
        Path(correspondence_value).expanduser().resolve()
        if correspondence_value
        else DEFAULT_CORRESPONDENCE
    )
    report = retarget_human_to_g1(
        human_path,
        output_path,
        correspondence_path=correspondence_path,
    )
    report_path = work_root / "reports" / f"{output_path.stem}_retarget.json"
    _write_json_atomic(report_path, report)
    print(json.dumps(_json_ready(report), ensure_ascii=False, indent=2, sort_keys=True))
    return report


def _run_validate_g1(
    config: dict[str, Any], input_value: str | Path | None
) -> dict[str, Any]:
    work_root, _, vendor_root = _resolved_roots(config)
    _enable_local_vendor(vendor_root)
    from .g1_validation import validate_g1_archive

    default_human = _default_human_output(DEFAULT_B3_INPUT, work_root)
    input_path = (
        Path(input_value).expanduser().resolve()
        if input_value
        else _default_g1_output(default_human, work_root)
    )
    try:
        input_path.relative_to(PIPELINE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"G1 archive must stay inside {PIPELINE_ROOT}; got {input_path}"
        ) from exc
    report = validate_g1_archive(input_path)
    report_path = work_root / "reports" / f"{input_path.stem}_validation.json"
    _write_json_atomic(report_path, report)
    print(json.dumps(_json_ready(report), ensure_ascii=False, indent=2, sort_keys=True))
    return report


def _run_tensorboard_dashboard(
    config: dict[str, Any],
    output_value: str | Path | None,
    *,
    force: bool,
) -> dict[str, Any]:
    """Export a compact, read-only dashboard from the existing gate artifacts."""

    work_root, _, _ = _resolved_roots(config)
    from .tensorboard_dashboard import export_tensorboard_dashboard

    output_path = (
        local_path(output_value) if output_value else None
    )
    report = export_tensorboard_dashboard(
        work_root=work_root,
        output_dir=output_path,
        force=force,
    )
    concise = {
        "status": "completed",
        "schema_version": report.get("schema_version"),
        "output_dir": report.get("output_dir"),
        "runs": sorted(report.get("sections", {})),
        "launch_command": report.get("launch_command"),
        "manifest": str(Path(report["output_dir"]) / "dashboard_manifest.json"),
    }
    print(json.dumps(_json_ready(concise), ensure_ascii=False, indent=2, sort_keys=True))
    return report


def _run_batch(
    config: dict[str, Any],
    *,
    splits: list[str],
    max_items: int | None,
    force: bool,
) -> dict[str, Any]:
    work_root, model_root, vendor_root = _resolved_roots(config)
    _enable_local_vendor(vendor_root)
    from .batch import run_batch

    human_cfg = config.get("human", {})
    report = run_batch(
        manifest_root=work_root / "manifests",
        output_root=work_root,
        report_root=work_root / "reports" / "batch",
        model_root=model_root,
        splits=tuple(splits),
        target_fps=float(human_cfg.get("target_fps", 50.0)),
        chunk_size=int(human_cfg.get("chunk_size", 128)),
        resume=not force,
        force=force,
        max_items=max_items,
    )
    print(json.dumps(_json_ready(report), ensure_ascii=False, indent=2, sort_keys=True))
    return report


def _run_status(config: dict[str, Any]) -> dict[str, Any]:
    work_root, model_root, vendor_root = _resolved_roots(config)
    manifest_root = work_root / "manifests"
    prerequisite_file = work_root / "reports" / "gate2_prerequisites.json"
    prerequisites = _load_json_if_present(prerequisite_file)
    gate1_required = [
        "accad_inventory.json",
        "accad_quality.json",
        "accad_train.json",
        "accad_validation.json",
        "accad_test.json",
        "accad_excluded.json",
        "gate1_summary.json",
    ]
    gate1_summary_path = manifest_root / "gate1_summary.json"
    gate1_summary = _load_json_if_present(gate1_summary_path)
    gate1_files_present = all((manifest_root / name).is_file() for name in gate1_required)
    default_human = _default_human_output(DEFAULT_B3_INPUT, work_root)
    gate2_validation_file = (
        work_root / "reports" / f"{default_human.stem}_validation.json"
    )
    gate2_validation = _load_json_if_present(gate2_validation_file)
    gate2_complete = bool(
        gate2_validation and gate2_validation.get("gate2_complete", False)
    )

    batch_summary_file = work_root / "reports" / "batch" / "batch_summary.json"
    parallel_summary_file = work_root / "reports" / "batch" / "parallel_batch_summary.json"
    gate3_manifest_file = manifest_root / "g1_manifest_build.json"
    batch_summary = _load_json_if_present(batch_summary_file)
    parallel_summary = _load_json_if_present(parallel_summary_file)
    gate3_manifest = _load_json_if_present(gate3_manifest_file)
    batch_counts = batch_summary.get("counts", {}) if batch_summary else {}
    promoted_counts = gate3_manifest.get("counts", {}) if gate3_manifest else {}
    gate3_manifests_present = all(
        (manifest_root / f"g1_{split}.json").is_file()
        for split in ("train", "validation", "test")
    )
    gate3_complete = bool(
        batch_summary
        and batch_summary.get("all_selected_items_completed") is True
        and batch_counts.get("selected_total") == 249
        and batch_counts.get("completed") == 249
        and batch_counts.get("failed") == 0
        and parallel_summary
        and parallel_summary.get("status") == "completed"
        and parallel_summary.get("success") is True
        and parallel_summary.get("worker_problem_count") == 0
        and gate3_manifest
        and gate3_manifest.get("status") == "completed"
        and promoted_counts.get("promoted_total") == 249
        and promoted_counts.get("source_completed_records") == 249
        and gate3_manifests_present
    )

    b3_g1_file = work_root / "g1" / "Female1Walking_c3d" / "B3_-_walk1_g1_50hz.npz"
    isaac_kinematic_file = work_root / "reports" / "B3_-_walk1_isaac_kinematic.json"
    isaac_pd_file = work_root / "reports" / "B3_-_walk1_isaac_pd_replay.json"
    isaac_kinematic = _load_json_if_present(isaac_kinematic_file)
    isaac_pd = _load_json_if_present(isaac_pd_file)
    b3_checksum = _sha256_file(b3_g1_file) if b3_g1_file.is_file() else None
    live_input = isaac_kinematic.get("input", {}) if isaac_kinematic else {}
    gate4_complete = bool(
        gate3_complete
        and isaac_kinematic
        and isaac_kinematic.get("execution_complete") is True
        and isaac_kinematic.get("passed") is True
        and live_input.get("full_clip") is True
        and live_input.get("validated_frames") == live_input.get("total_frames")
        and b3_checksum is not None
        and live_input.get("sha256") == b3_checksum
    )

    final_completion_file = work_root / "reports" / "final_completion.json"
    final_artifact_file = work_root / "reports" / "final_artifact_manifest.json"
    terminal_outcome_file = work_root / "reports" / "gate5_terminal_outcome.json"
    final_completion = _load_json_if_present(final_completion_file)
    final_artifact = _load_json_if_present(final_artifact_file)
    terminal_outcome = _load_json_if_present(terminal_outcome_file)
    from .gate5_outcome import verify_gate5_terminal_outcome

    terminal_outcome_snapshot = verify_gate5_terminal_outcome(
        terminal_outcome,
        accad_root=ACCAD_ROOT,
        pipeline_root=PIPELINE_ROOT,
        work_root=work_root,
    )
    final_artifact_record = (
        final_completion.get("artifact_manifest", {}) if final_completion else {}
    )
    final_training = final_completion.get("training", {}) if final_completion else {}
    final_validation = final_completion.get("validation", {}) if final_completion else {}
    final_test = final_completion.get("test", {}) if final_completion else {}
    final_representative = (
        final_completion.get("representative_evaluation", {})
        if final_completion
        else {}
    )
    final_artifact_checksum = (
        _sha256_file(final_artifact_file) if final_artifact_file.is_file() else None
    )
    final_artifact_snapshot = _verify_final_artifact_snapshot(
        final_artifact,
        accad_root=ACCAD_ROOT,
        project_root=ACCAD_ROOT.parent,
    )
    gate5_complete = bool(
        final_completion
        and final_completion.get("schema_version") == "accad_g1_final_completion_v1"
        and final_completion.get("status") == "completed"
        and final_training.get("complete") is True
        and final_training.get("status") == "completed"
        and final_training.get("independent_export_validation", {}).get("status")
        == "passed"
        and final_validation.get("complete") is True
        and final_validation.get("status") == "completed"
        and final_test.get("complete") is True
        and final_test.get("status") == "completed"
        and final_representative.get("complete") is True
        and final_representative.get("status") == "completed"
        and bool(final_representative.get("video", {}).get("path"))
        and bool(final_representative.get("video", {}).get("checksum_sha256"))
        and final_artifact
        and final_artifact.get("schema_version") == "accad_g1_final_artifact_manifest_v1"
        and final_artifact.get("status") == "completed"
        and final_artifact.get("independent_policy_export_validation", {}).get("status")
        == "passed"
        and final_artifact_snapshot["current"] is True
        and final_artifact_checksum is not None
        and final_artifact_record.get("path")
        == "_g1_pipeline/work/reports/final_artifact_manifest.json"
        and final_artifact_record.get("bytes") == final_artifact_file.stat().st_size
        and final_artifact_record.get("checksum_sha256") == final_artifact_checksum
    )
    terminal_failure_current = bool(
        not gate5_complete
        and terminal_outcome
        and terminal_outcome_snapshot["current"] is True
        and terminal_outcome.get("status") == "failed"
        and terminal_outcome.get("complete") is False
        and terminal_outcome.get("execution", {}).get("complete") is True
        and terminal_outcome.get("policy_acceptance", {}).get("passed") is False
        and terminal_outcome.get("test", {}).get("status") == "sealed_not_opened"
    )
    if gate5_complete:
        gate5_status = "completed"
        gate5_execution_complete = True
    elif terminal_failure_current:
        gate5_status = "failed"
        gate5_execution_complete = True
    elif terminal_outcome_file.exists():
        gate5_status = "invalid_terminal_outcome"
        gate5_execution_complete = False
    else:
        gate5_status = "pending"
        gate5_execution_complete = False
    report = {
        "pipeline_root": str(PIPELINE_ROOT),
        "raw_root": str(ACCAD_ROOT),
        "configured_outputs_confined_to_pipeline": True,
        "gate_1": {
            "complete": bool(
                gate1_files_present
                and gate1_summary
                and gate1_summary.get("passed", False)
            ),
            "manifest_root": str(manifest_root),
            "counts": gate1_summary.get("counts") if gate1_summary else None,
            "quality_flag_counts": (
                gate1_summary.get("quality_flag_counts") if gate1_summary else None
            ),
        },
        "gate_2": {
            "ready": bool(
                prerequisites
                and prerequisites.get("ready", prerequisites.get("ok", False))
            ),
            "complete": gate2_complete,
            "model_expected": str(model_root / "smplx" / "SMPLX_NEUTRAL.npz"),
            "prerequisite_report": str(prerequisite_file),
            "human_reference": str(default_human),
            "validation_report": str(gate2_validation_file),
            "validation_statuses": (
                gate2_validation.get("statuses") if gate2_validation else None
            ),
        },
        "gate_3": {
            "complete": gate3_complete,
            "batch_summary": str(batch_summary_file),
            "parallel_summary": str(parallel_summary_file),
            "manifest_build_report": str(gate3_manifest_file),
            "counts": promoted_counts or None,
            "num_frames": gate3_manifest.get("num_frames") if gate3_manifest else None,
            "duration_sec": gate3_manifest.get("duration_sec") if gate3_manifest else None,
            "retarget_version": (
                gate3_manifest.get("retarget_version") if gate3_manifest else None
            ),
        },
        "gate_4": {
            "complete": gate4_complete,
            "kinematic_report": str(isaac_kinematic_file),
            "kinematic_passed": (
                isaac_kinematic.get("passed") if isaac_kinematic else None
            ),
            "current_b3_checksum_sha256": b3_checksum,
            "validated_b3_checksum_sha256": live_input.get("sha256"),
            "pd_baseline_report": str(isaac_pd_file),
            "pd_baseline_dynamics_passed": (
                isaac_pd.get("dynamics_passed") if isaac_pd else None
            ),
            "interpretation": (
                "Live kinematic replay is the Gate-4 contract check; open-loop PD is a "
                "diagnostic baseline and is not expected to replace the learned tracker."
            ),
        },
        "gate_5": {
            "complete": gate5_complete,
            "execution_complete": gate5_execution_complete,
            "final_completion_report": str(final_completion_file),
            "final_artifact_manifest": str(final_artifact_file),
            "artifact_snapshot": final_artifact_snapshot,
            "status": gate5_status,
            "terminal_outcome_report": str(terminal_outcome_file),
            "terminal_outcome_snapshot": terminal_outcome_snapshot,
            "terminal_outcome": terminal_outcome if terminal_failure_current else None,
            "training": final_training or None,
            "validation": final_validation or None,
            "test": (
                final_test
                or (
                    terminal_outcome.get("test")
                    if terminal_failure_current and terminal_outcome
                    else None
                )
            ),
            "representative_evaluation": final_representative or None,
            "policy_acceptance": (
                final_completion.get("policy_acceptance")
                if final_completion
                else (
                    terminal_outcome.get("policy_acceptance")
                    if terminal_failure_current and terminal_outcome
                    else None
                )
            ),
        },
        "vendor_root": str(vendor_root),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="accad-g1",
        description="Run the gated ACCAD-to-Unitree-G1 preprocessing pipeline.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Pipeline YAML path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="Run Gate 1 on all stageii files")
    subparsers.add_parser("preflight", help="Check Gate 2 package/model prerequisites")
    subparsers.add_parser("status", help="Show current gate state without changing it")
    subparsers.add_parser("all", help="Run each gate in order until one cannot proceed")
    reconstruct = subparsers.add_parser("reconstruct", help="Run SMPL-X reconstruction")
    reconstruct.add_argument("--input", default=None, help="stageii NPZ; defaults to B3 walk1")
    reconstruct.add_argument(
        "--output", default=None, help="Output NPZ; must be below ACCAD/_g1_pipeline"
    )
    validate_human = subparsers.add_parser(
        "validate-human", help="Validate a generated Gate 2 human NPZ"
    )
    validate_human.add_argument(
        "--input", default=None, help="Human NPZ; defaults to the generated B3 walk"
    )
    validate_human.add_argument(
        "--profile",
        choices=("generic", "b3-walk"),
        default="b3-walk",
        help="Use walking-only checks only for the B3 reference clip",
    )
    retarget = subparsers.add_parser(
        "retarget", help="Retarget a canonical human NPZ to G1 29-DoF"
    )
    retarget.add_argument(
        "--input", default=None, help="Human NPZ; defaults to the generated B3 walk"
    )
    retarget.add_argument(
        "--output", default=None, help="G1 NPZ; must stay below ACCAD/_g1_pipeline"
    )
    retarget.add_argument(
        "--correspondence", default=None, help="Human↔G1 YAML; defaults to correspondence.yaml"
    )
    validate_g1 = subparsers.add_parser(
        "validate-g1", help="Validate a G1 NPZ with Pinocchio and the G1 contracts"
    )
    validate_g1.add_argument(
        "--input", default=None, help="G1 NPZ; defaults to the generated B3 reference"
    )
    batch = subparsers.add_parser(
        "batch", help="Run Gate 2 and offline Gate 3 for the selected split manifests"
    )
    batch.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=["train", "validation", "test"],
    )
    batch.add_argument(
        "--max-items", type=int, default=None, help="Process only a deterministic global prefix"
    )
    batch.add_argument(
        "--force", action="store_true", help="Regenerate even independently valid artifacts"
    )
    tensorboard = subparsers.add_parser(
        "tensorboard",
        help="Build a compact 3-part TensorBoard dashboard from existing artifacts",
    )
    tensorboard.add_argument(
        "--output",
        default=None,
        help=(
            "Dashboard directory below work/tensorboard; defaults to "
            "work/tensorboard/accad_g1_overview"
        ),
    )
    tensorboard.add_argument(
        "--force",
        action="store_true",
        help="Replace only the selected generated dashboard directory if it exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "audit":
            _run_audit(config)
            return 0
        if args.command == "preflight":
            return 0 if _run_preflight(config).get("ready", False) else 2
        if args.command == "status":
            _run_status(config)
            return 0
        if args.command == "reconstruct":
            _run_reconstruct(config, args.input, args.output)
            return 0
        if args.command == "validate-human":
            return (
                0
                if _run_validate_human(config, args.input, args.profile).get(
                    "gate2_complete", False
                )
                else 2
            )
        if args.command == "retarget":
            _run_retarget(config, args.input, args.output, args.correspondence)
            return 0
        if args.command == "validate-g1":
            return (
                0
                if _run_validate_g1(config, args.input).get(
                    "gate3_kinematic_complete", False
                )
                else 2
            )
        if args.command == "batch":
            report = _run_batch(
                config,
                splits=args.splits,
                max_items=args.max_items,
                force=args.force,
            )
            return 0 if report.get("all_selected_items_completed", False) else 2
        if args.command == "tensorboard":
            _run_tensorboard_dashboard(config, args.output, force=args.force)
            return 0
        if args.command == "all":
            _run_audit(config)
            prerequisites = _run_preflight(config)
            if not prerequisites.get("ready", False):
                print(
                    "Pipeline stopped at Gate 2: supply the licensed SMPL-X neutral model "
                    "shown in the prerequisite report.",
                    file=sys.stderr,
                )
                return 2
            reconstruction = _run_reconstruct(config, None, None)
            validation = _run_validate_human(config, None, "b3-walk")
            if not validation.get("gate2_complete", False):
                print(
                    "Pipeline stopped inside Gate 2: inspect the generated validation "
                    "report before starting G1 retargeting.",
                    file=sys.stderr,
                )
                return 2
            return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
