"""Record and verify a terminal, unsuccessful Gate-5 outcome.

The success publisher in :mod:`accad_g1.final_artifacts` is deliberately
fail-closed: it cannot publish final artifacts until validation, test, and the
representative replay all pass.  That leaves a completed training run followed
by a failed validation looking indistinguishable from work that was never
executed.  This module records that negative result without relaxing the
success contract and, importantly, without accepting or reading a test report.

Only a completed production training summary and a completed, official,
deterministic *validation* report whose acceptance result is false are valid
inputs.  Every source used by the report is checksummed, and live verification
re-derives the semantic result from those sources.  The positive final
completion files must remain absent while this negative witness is current.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


SCHEMA_VERSION = "accad_g1_gate5_terminal_outcome_v1"
TRAINING_SCHEMA_VERSION = "accad_g1_tracking_train_v2"
VALIDATION_SCHEMA_VERSION = "accad_g1_tracking_suite_evaluation_v2"
DEFAULT_REPORT_NAME = "gate5_terminal_outcome.json"
FINAL_SUCCESS_REPORT_NAMES = (
    "final_completion.json",
    "final_artifact_manifest.json",
)

_CHECKPOINT_NAME = re.compile(r"model_(\d+)\.pt\Z")


class Gate5OutcomeError(RuntimeError):
    """Raised when the supplied evidence cannot prove a terminal failure."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise Gate5OutcomeError(f"{label} is not readable JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Gate5OutcomeError(f"{label} must contain a JSON object: {path}")
    return value


def _path_below(path: str | Path, root: Path, label: str, *, file: bool = True) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Gate5OutcomeError(f"{label} must stay below {root}: {resolved}") from exc
    if file and not resolved.is_file():
        raise Gate5OutcomeError(f"{label} is not a file: {resolved}")
    return resolved


def _relative(path: Path, accad_root: Path) -> str:
    try:
        return path.resolve().relative_to(accad_root.resolve()).as_posix()
    except ValueError as exc:
        raise Gate5OutcomeError(
            f"evidence must stay below ACCAD root {accad_root}: {path}"
        ) from exc


def _artifact(role: str, path: Path, accad_root: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": _relative(path, accad_root),
        "bytes": path.stat().st_size,
        "checksum_sha256": _sha256_file(path),
    }


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Gate5OutcomeError(f"{label} must be a non-negative integer")
    return value


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Gate5OutcomeError(f"{label} must be numeric")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise Gate5OutcomeError(f"{label} must be finite")
    return number


def _validate_declared_file(
    declaration: Mapping[str, Any],
    *,
    pipeline_root: Path,
    label: str,
) -> Path:
    path_value = declaration.get("path")
    checksum = declaration.get("checksum_sha256")
    if not isinstance(path_value, str) or not path_value:
        raise Gate5OutcomeError(f"{label}.path is missing")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise Gate5OutcomeError(f"{label}.checksum_sha256 is invalid")
    try:
        int(checksum, 16)
    except ValueError as exc:
        raise Gate5OutcomeError(f"{label}.checksum_sha256 is not hexadecimal") from exc
    path = _path_below(path_value, pipeline_root, label)
    actual_checksum = _sha256_file(path)
    if actual_checksum != checksum.lower():
        raise Gate5OutcomeError(f"{label} checksum does not match its live file")
    if "bytes" in declaration:
        size = declaration.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise Gate5OutcomeError(f"{label}.bytes is invalid")
        if path.stat().st_size != size:
            raise Gate5OutcomeError(f"{label} size does not match its live file")
    return path


def _resolve_training_summary(value: str | Path, pipeline_root: Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "run_summary.json"
    return _path_below(path, pipeline_root, "training summary")


def _failure_checks(acceptance: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = acceptance.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise Gate5OutcomeError("validation acceptance checks are absent")
    failed: list[dict[str, Any]] = []
    for name, value in sorted(checks.items()):
        if not isinstance(name, str) or not isinstance(value, dict):
            raise Gate5OutcomeError("validation acceptance checks are malformed")
        if not isinstance(value.get("passed"), bool):
            raise Gate5OutcomeError(f"validation acceptance check {name!r} lacks passed")
        if value["passed"]:
            continue
        record: dict[str, Any] = {
            "name": name,
            "comparison": value.get("comparison"),
            "value": value.get("value"),
        }
        if "threshold" in value:
            record["threshold"] = value["threshold"]
        if "expected" in value:
            record["expected"] = value["expected"]
        failed.append(record)
    if not failed:
        raise Gate5OutcomeError(
            "validation says acceptance failed but contains no failed acceptance check"
        )
    return failed


def _derive_payload(
    *,
    training_summary_path: Path,
    validation_report_path: Path,
    accad_root: Path,
    pipeline_root: Path,
    work_root: Path,
    generated_at_utc: str,
) -> dict[str, Any]:
    """Validate evidence and return the canonical terminal-failure payload."""

    training = _load_json_object(training_summary_path, "training summary")
    validation = _load_json_object(validation_report_path, "validation report")

    if training.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise Gate5OutcomeError("training summary schema is not production v2")
    if training.get("status") != "completed":
        raise Gate5OutcomeError("training execution did not complete")
    training_values = training.get("training")
    if not isinstance(training_values, dict):
        raise Gate5OutcomeError("training summary lacks training counters")
    requested = _require_nonnegative_int(
        training_values.get("learning_updates_requested"),
        "training.learning_updates_requested",
    )
    completed = _require_nonnegative_int(
        training_values.get("learning_updates_completed"),
        "training.learning_updates_completed",
    )
    final_iteration = _require_nonnegative_int(
        training_values.get("final_checkpoint_iteration_index"),
        "training.final_checkpoint_iteration_index",
    )
    if requested <= 0 or completed != requested or final_iteration != completed - 1:
        raise Gate5OutcomeError("training counters do not prove a complete final checkpoint")
    total_timesteps = _require_nonnegative_int(
        training_values.get("total_timesteps"), "training.total_timesteps"
    )
    if total_timesteps <= 0:
        raise Gate5OutcomeError("training.total_timesteps must be positive")

    declared_run_dir = training.get("run_dir")
    if not isinstance(declared_run_dir, str):
        raise Gate5OutcomeError("training run_dir is missing")
    run_dir = _path_below(declared_run_dir, pipeline_root, "training run_dir", file=False)
    if not run_dir.is_dir() or training_summary_path.parent != run_dir:
        raise Gate5OutcomeError("training summary does not belong to its declared run_dir")

    if validation.get("schema_version") != VALIDATION_SCHEMA_VERSION:
        raise Gate5OutcomeError("validation report schema is not suite-evaluation v2")
    if validation.get("status") != "completed":
        raise Gate5OutcomeError("validation execution did not complete")
    if validation.get("passed") is not False:
        raise Gate5OutcomeError("terminal failure requires validation passed=false")
    acceptance = validation.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("passed") is not False:
        raise Gate5OutcomeError("terminal failure requires acceptance.passed=false")
    failed_checks = _failure_checks(acceptance)

    contract = validation.get("evaluation_contract")
    if not isinstance(contract, dict):
        raise Gate5OutcomeError("validation evaluation_contract is absent")
    if contract.get("action_source") != "checkpoint":
        raise Gate5OutcomeError("diagnostic/zero-action reports cannot close Gate 5")
    if contract.get("official_acceptance_eligible") is not True:
        raise Gate5OutcomeError("validation report is not official-acceptance eligible")
    if contract.get("deterministic_policy_inference") is not True:
        raise Gate5OutcomeError("validation policy inference was not deterministic")
    if contract.get("first_episode_only") is not True:
        raise Gate5OutcomeError("validation did not use the first-episode-only contract")

    motion_input = validation.get("motion_input")
    if not isinstance(motion_input, dict) or motion_input.get("split") != "validation":
        raise Gate5OutcomeError("only the validation split may produce this outcome")
    if motion_input.get("manifest_promotion_verified") is not True:
        raise Gate5OutcomeError("validation manifest promotion was not checksum verified")
    motion_count = _require_nonnegative_int(
        motion_input.get("motion_count"), "validation motion_input.motion_count"
    )
    if motion_count <= 0:
        raise Gate5OutcomeError("validation selected no motions")
    validation_manifests = motion_input.get("manifests")
    if not isinstance(validation_manifests, list) or not validation_manifests:
        raise Gate5OutcomeError("validation manifest evidence is absent")
    if any(
        not isinstance(item, dict) or item.get("split") != "validation"
        for item in validation_manifests
    ):
        raise Gate5OutcomeError("validation evidence contains a non-validation manifest")

    aggregate = validation.get("aggregate")
    if not isinstance(aggregate, dict):
        raise Gate5OutcomeError("validation aggregate is absent")
    aggregate_motion_count = _require_nonnegative_int(
        aggregate.get("motion_count"), "validation aggregate.motion_count"
    )
    finished = _require_nonnegative_int(
        aggregate.get("finished_first_episodes"),
        "validation aggregate.finished_first_episodes",
    )
    successful = _require_nonnegative_int(
        aggregate.get("successful_motion_end_episodes"),
        "validation aggregate.successful_motion_end_episodes",
    )
    if aggregate_motion_count != motion_count or finished != motion_count:
        raise Gate5OutcomeError("validation did not finish one first episode per motion")
    if aggregate.get("simulation_app_stopped_early") is not False:
        raise Gate5OutcomeError("validation simulator stopped early")
    completion_ratio = _require_finite_number(
        aggregate.get("completion_ratio_mean"),
        "validation aggregate.completion_ratio_mean",
    )

    validation_checkpoint = validation.get("checkpoint")
    if not isinstance(validation_checkpoint, dict):
        raise Gate5OutcomeError("validation checkpoint evidence is absent")
    checkpoint_path = _validate_declared_file(
        validation_checkpoint,
        pipeline_root=pipeline_root,
        label="validation checkpoint",
    )
    match = _CHECKPOINT_NAME.fullmatch(checkpoint_path.name)
    if not match or int(match.group(1)) != final_iteration:
        raise Gate5OutcomeError("validation did not evaluate the final training checkpoint")
    expected_checkpoint = run_dir / f"model_{final_iteration}.pt"
    if checkpoint_path != expected_checkpoint:
        raise Gate5OutcomeError("validation checkpoint is outside the declared training run")
    checkpoint_records = training.get("checkpoints")
    if not isinstance(checkpoint_records, list):
        raise Gate5OutcomeError("training checkpoint inventory is absent")
    matching_records = [
        item
        for item in checkpoint_records
        if isinstance(item, dict)
        and item.get("path") == validation_checkpoint.get("path")
        and item.get("bytes") == validation_checkpoint.get("bytes")
        and item.get("checksum_sha256") == validation_checkpoint.get("checksum_sha256")
    ]
    if len(matching_records) != 1:
        raise Gate5OutcomeError(
            "validation checkpoint does not exactly match the training inventory"
        )

    source_artifacts = [
        _artifact("training_summary", training_summary_path, accad_root),
        _artifact("final_checkpoint", checkpoint_path, accad_root),
        _artifact("validation_summary", validation_report_path, accad_root),
    ]

    training_motion_input = training.get("motion_input")
    if not isinstance(training_motion_input, dict):
        raise Gate5OutcomeError("training motion_input evidence is absent")
    if training_motion_input.get("manifest_promotion_verified") is not True:
        raise Gate5OutcomeError("training manifest promotion was not checksum verified")
    training_manifests = training_motion_input.get("manifests")
    if not isinstance(training_manifests, list) or not training_manifests:
        raise Gate5OutcomeError("training manifest evidence is absent")
    if any(
        not isinstance(item, dict) or item.get("split") != "train"
        for item in training_manifests
    ):
        raise Gate5OutcomeError("training evidence contains a non-train manifest")
    for index, declaration in enumerate(training_manifests):
        manifest_path = _validate_declared_file(
            declaration,
            pipeline_root=pipeline_root,
            label=f"training manifest[{index}]",
        )
        source_artifacts.append(
            _artifact(f"training_manifest[{index}]", manifest_path, accad_root)
        )
    for index, declaration in enumerate(validation_manifests):
        manifest_path = _validate_declared_file(
            declaration,
            pipeline_root=pipeline_root,
            label=f"validation manifest[{index}]",
        )
        source_artifacts.append(
            _artifact(f"validation_manifest[{index}]", manifest_path, accad_root)
        )

    expected_success_paths = [
        (work_root / "reports" / name).resolve() for name in FINAL_SUCCESS_REPORT_NAMES
    ]
    for path in expected_success_paths:
        _path_below(path, pipeline_root, "positive final artifact", file=False)
        if path.exists():
            raise Gate5OutcomeError(
                f"cannot publish a negative outcome beside positive final artifact: {path}"
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "complete": False,
        "terminal": True,
        "outcome": "validation_policy_acceptance_failed",
        "generated_at_utc": generated_at_utc,
        "execution": {
            "complete": True,
            "training_complete": True,
            "validation_complete": True,
            "test_executed": False,
        },
        "policy_acceptance": {
            "passed": False,
            "failed_stage": "validation",
            "failed_check_count": len(failed_checks),
            "failed_checks": failed_checks,
        },
        "training": {
            "status": "completed",
            "execution_complete": True,
            "learning_updates_requested": requested,
            "learning_updates_completed": completed,
            "total_timesteps": total_timesteps,
            "final_checkpoint_iteration_index": final_iteration,
            "summary_source_role": "training_summary",
            "checkpoint_source_role": "final_checkpoint",
        },
        "validation": {
            "status": "completed",
            "execution_complete": True,
            "split": "validation",
            "official_acceptance_eligible": True,
            "acceptance_passed": False,
            "motion_count": motion_count,
            "finished_first_episodes": finished,
            "successful_motion_end_episodes": successful,
            "completion_ratio_mean": completion_ratio,
            "summary_source_role": "validation_summary",
        },
        "test": {
            "status": "sealed_not_opened",
            "executed": False,
            "report_consumed": False,
            "reason": "validation acceptance failed before the held-out test gate",
        },
        "success_publication": {
            "status": "blocked_by_validation_failure",
            "published": False,
            "expected_absent_paths": [
                _relative(path, accad_root) for path in expected_success_paths
            ],
        },
        "builder_contract": {
            "accepted_input_splits": ["train", "validation"],
            "test_report_argument_supported": False,
            "test_artifacts_read": False,
            "positive_acceptance_thresholds_changed": False,
        },
        "source_artifacts": source_artifacts,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def build_gate5_terminal_outcome(
    *,
    training_run: str | Path,
    validation_report: str | Path,
    output: str | Path | None = None,
    accad_root: str | Path,
    pipeline_root: str | Path,
    work_root: str | Path,
) -> dict[str, Any]:
    """Write a checksummed terminal-failure witness and return its payload."""

    accad = Path(accad_root).expanduser().resolve()
    pipeline = _path_below(pipeline_root, accad, "pipeline root", file=False)
    work = _path_below(work_root, pipeline, "work root", file=False)
    training_path = _resolve_training_summary(training_run, pipeline)
    validation_path = _path_below(validation_report, pipeline, "validation report")
    output_path = _path_below(
        output or (work / "reports" / DEFAULT_REPORT_NAME),
        pipeline,
        "Gate-5 terminal outcome",
        file=False,
    )
    if output_path.parent != (work / "reports").resolve():
        raise Gate5OutcomeError("Gate-5 terminal outcome must be in work/reports")
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = _derive_payload(
        training_summary_path=training_path,
        validation_report_path=validation_path,
        accad_root=accad,
        pipeline_root=pipeline,
        work_root=work,
        generated_at_utc=generated_at,
    )
    _write_json_atomic(output_path, payload)
    return payload


def verify_gate5_terminal_outcome(
    report: Mapping[str, Any] | None,
    *,
    accad_root: str | Path,
    pipeline_root: str | Path,
    work_root: str | Path,
) -> dict[str, Any]:
    """Re-hash and semantically re-derive a terminal-failure report."""

    errors: list[str] = []
    if not isinstance(report, Mapping):
        return {
            "current": False,
            "verified_source_artifact_count": 0,
            "errors": ["terminal Gate-5 outcome is absent or invalid"],
        }
    try:
        accad = Path(accad_root).expanduser().resolve()
        pipeline = _path_below(pipeline_root, accad, "pipeline root", file=False)
        work = _path_below(work_root, pipeline, "work root", file=False)
        source_values = report.get("source_artifacts")
        if not isinstance(source_values, list):
            raise Gate5OutcomeError("source_artifacts is absent")
        records: dict[str, Mapping[str, Any]] = {}
        for index, value in enumerate(source_values):
            if not isinstance(value, Mapping):
                raise Gate5OutcomeError(f"source_artifacts[{index}] is not an object")
            role = value.get("role")
            if not isinstance(role, str) or not role or role in records:
                raise Gate5OutcomeError(f"source_artifacts[{index}] has invalid role")
            records[role] = value
            relative = value.get("path")
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                raise Gate5OutcomeError(f"source artifact {role!r} has invalid path")
            path = _path_below(accad / relative, accad, f"source artifact {role!r}")
            size = value.get("bytes")
            checksum = value.get("checksum_sha256")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise Gate5OutcomeError(f"source artifact {role!r} has invalid bytes")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise Gate5OutcomeError(f"source artifact {role!r} has invalid checksum")
            try:
                int(checksum, 16)
            except ValueError as exc:
                raise Gate5OutcomeError(
                    f"source artifact {role!r} checksum is not hexadecimal"
                ) from exc
            if path.stat().st_size != size:
                raise Gate5OutcomeError(f"source artifact {role!r} size changed")
            if _sha256_file(path) != checksum.lower():
                raise Gate5OutcomeError(f"source artifact {role!r} checksum changed")

        required_roles = {"training_summary", "final_checkpoint", "validation_summary"}
        if not required_roles.issubset(records):
            raise Gate5OutcomeError("required source artifact roles are absent")
        timestamp = report.get("generated_at_utc")
        if not isinstance(timestamp, str):
            raise Gate5OutcomeError("generated_at_utc is invalid")
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise Gate5OutcomeError("generated_at_utc is invalid") from exc
        if parsed_timestamp.tzinfo is None:
            raise Gate5OutcomeError("generated_at_utc must include a timezone")

        training_path = accad / str(records["training_summary"]["path"])
        validation_path = accad / str(records["validation_summary"]["path"])
        derived = _derive_payload(
            training_summary_path=training_path.resolve(),
            validation_report_path=validation_path.resolve(),
            accad_root=accad,
            pipeline_root=pipeline,
            work_root=work,
            generated_at_utc=timestamp,
        )
        if dict(report) != derived:
            raise Gate5OutcomeError(
                "terminal Gate-5 outcome differs from its canonically derived evidence"
            )
    except (Gate5OutcomeError, OSError) as exc:
        errors.append(str(exc))
    source_count = len(report.get("source_artifacts", [])) if isinstance(
        report.get("source_artifacts"), list
    ) else 0
    return {
        "current": not errors,
        "verified_source_artifact_count": source_count if not errors else 0,
        "errors": errors,
    }


__all__ = [
    "DEFAULT_REPORT_NAME",
    "Gate5OutcomeError",
    "SCHEMA_VERSION",
    "build_gate5_terminal_outcome",
    "verify_gate5_terminal_outcome",
]
