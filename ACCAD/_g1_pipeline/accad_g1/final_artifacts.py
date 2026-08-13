"""Build the checksummed, final ACCAD-to-G1 completion snapshot.

This module never launches Isaac Sim and it never writes outside
``ACCAD/_g1_pipeline/work/reports``.  It uses CPU-only TorchScript and ONNX
loaders to independently execute the deployment exports.  A final
snapshot means that dynamic promotion, training, both deterministic held-out
suites, and a representative test video completed and that every referenced
artifact still matches its recorded checksum.  Both suites must pass the
separately versioned dynamic acceptance policy.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .evaluation_metrics import (
    DEFAULT_SUITE_THRESHOLDS,
    SUITE_ACCEPTANCE_SCHEMA_VERSION,
    build_suite_acceptance,
)
from .dynamic_retime import (
    DYNAMIC_RETARGET_VERSION,
    MANIFEST_SCHEMA_VERSION as DYNAMIC_MANIFEST_SCHEMA,
    PLAN_SCHEMA_VERSION as DYNAMIC_PLAN_SCHEMA,
    POLICY_ALGORITHM as DYNAMIC_POLICY_ALGORITHM,
    POLICY_SCHEMA_VERSION as DYNAMIC_POLICY_SCHEMA,
    REPORT_SCHEMA_VERSION as DYNAMIC_REPORT_SCHEMA,
)
from .physics_warmup import (
    DEFAULT_PHYSICS_WARMUP_STEPS,
    PHYSICS_WARMUP_SCHEMA_VERSION,
)
from .tracking_task import (
    DEFAULT_TRAINING_MOTION_ASSIGNMENT,
    EVALUATION_ISOLATION_SCHEMA_VERSION,
    TRAINING_OBJECTIVE_SCHEMA_VERSION,
    training_motion_assignment_contract,
    tracking_observation_contract,
)
from .video_validation import VideoValidationError, inspect_video


PACKAGE_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = PACKAGE_ROOT.parent
ACCAD_ROOT = PIPELINE_ROOT.parent
PROJECT_ROOT = ACCAD_ROOT.parent
WORK_ROOT = PIPELINE_ROOT / "work"
REPORT_ROOT = WORK_ROOT / "reports"

ARTIFACT_SCHEMA = "accad_g1_final_artifact_manifest_v1"
COMPLETION_SCHEMA = "accad_g1_final_completion_v1"
ACCEPTANCE_SCHEMA = SUITE_ACCEPTANCE_SCHEMA_VERSION
TRAINING_SCHEMA = "accad_g1_tracking_train_v2"
SUITE_SCHEMA = "accad_g1_tracking_suite_evaluation_v2"
PLAY_SCHEMA = "accad_g1_tracking_evaluation_v2"
GATE3_MANIFEST_SCHEMA = "accad_g1_gate3_manifest_v1"
RETARGET_VERSION = "g1-sequence-retarget-v27-ground-feasible-output-stance-float64-fk"

# The immutable v27 manifests remain the complete source population.  The
# v28 manifests contain only clips promoted by the frozen dynamic policy.
EXPECTED_SOURCE_SPLIT_COUNTS = {"train": 215, "validation": 5, "test": 29}
EXPECTED_CURRENT_DYNAMIC_COUNTS = {"train": 103, "validation": 2}
# Kept as a compatibility alias for callers which display source inventory.
EXPECTED_SPLIT_COUNTS = EXPECTED_SOURCE_SPLIT_COUNTS
EXPECTED_RUNTIME = {
    "action_dim": 29,
    "actor_observation_dim": 293,
    "critic_observation_dim": 505,
}
EXPECTED_TRACKING_POLICY_CONTRACT = (
    "accad_g1_tracking_v7_zero_velocity_target_train_selected"
)
EXPECTED_OBSERVATION_CONTRACT: Mapping[str, Any] = tracking_observation_contract()
EXPECTED_RECOVERY_PPO_LEARNING_RATE = 1.0e-4
EXPECTED_RECOVERY_PPO_SCHEDULE = "fixed"
EXPECTED_RECOVERY_PPO_DESIRED_KL = 0.01
EXPECTED_RESET_STATE_NOISE_SCALE = 0.0
EXPECTED_ALIVE_REWARD_WEIGHT = 0.25
EXPECTED_FAILURE_REWARD_WEIGHT = -10.0
EXPECTED_DIVERGENCE_RISK_REWARD_WEIGHT = -20.0
EXPECTED_ACTION_BOUNDARY_REWARD_WEIGHT = -2.0
EXPECTED_ACTION_LIMITER_GAP_REWARD_WEIGHT = -1.0
EXPECTED_DIVERGENCE_RISK_ONSET_FRACTION = 0.60
EXPECTED_ACTION_BOUNDARY_ONSET = 0.90
EXPECTED_DIVERGENCE_MAX_ROOT_DISTANCE_M = 0.80
EXPECTED_DIVERGENCE_MAX_MEAN_JOINT_ERROR_RAD = 1.0
EXPECTED_DIVERGENCE_MAX_MEAN_BODY_DISTANCE_M = 0.50
EXPECTED_PHYSICS_DT = 0.005
EXPECTED_CONTROL_DT = 0.02
EXPECTED_PHYSICS_WARMUP_STEPS = DEFAULT_PHYSICS_WARMUP_STEPS
EXPECTED_PHYSICS_WARMUP_TOLERANCE = 1.0e-5
# Isaac-side MotionLibrary tensors are float32.  Long clips can therefore
# round an otherwise exact manifest duration by several 1e-7 seconds when the
# evaluator copies the device value into JSON.  This tolerance is deliberately
# much smaller than one 50 Hz control step (0.02 s), while accepting that
# representational rounding.
RUNTIME_DURATION_ABS_TOL_S = 1.0e-6
EXPECTED_TRAINING_NUM_ENVS = 2048
EXPECTED_TRAINING_STEPS_PER_ENV = 32
EXPECTED_TRAINING_UPDATES = 5000
EXPECTED_TRAINING_MOTION_ASSIGNMENT = DEFAULT_TRAINING_MOTION_ASSIGNMENT
EXPECTED_TRAINING_TIMESTEPS = (
    EXPECTED_TRAINING_NUM_ENVS
    * EXPECTED_TRAINING_STEPS_PER_ENV
    * EXPECTED_TRAINING_UPDATES
)
EXPECTED_FINAL_CHECKPOINT_ITERATION = EXPECTED_TRAINING_UPDATES - 1
EXPECTED_CHECKPOINT_ITERATIONS = tuple(range(0, EXPECTED_TRAINING_UPDATES, 500)) + (
    EXPECTED_FINAL_CHECKPOINT_ITERATION,
)
EXPECTED_POLICY_INITIALIZATION_ITERATION = 599
EXPECTED_POLICY_INITIALIZATION_FILENAME = "model_599.pt"
UNFINISHED_DOCUMENT_MARKERS = (
    "[PLACEHOLDER",
    "<PLACEHOLDER>",
    "[TODO]",
    "[TBD]",
    "FINAL_RESULTS_PENDING",
    "<FINAL_CHECKPOINT>",
)

# These thresholds are imported from the evaluator's versioned CPU-only
# acceptance implementation and are fixed before a report is inspected.
# The final publication is intentionally stricter than an intermediate
# experiment report: both exhaustive promoted-motion suites must pass.
DYNAMIC_ACCEPTANCE_POLICY: Mapping[str, Any] = {
    "schema_version": ACCEPTANCE_SCHEMA,
    "kind": "predefined_tracking_suite_acceptance",
    "implementation": "accad_g1.evaluation_metrics.build_suite_acceptance",
    "thresholds": dict(DEFAULT_SUITE_THRESHOLDS),
    "interpretation": (
        "This verdict is recomputed from the versioned, predeclared suite thresholds. "
        "It remains separate from execution/integrity completion."
    ),
}

ENTRYPOINT_FILES: Sequence[str] = (
    "run.py",
    "run_isaac.py",
    "run_parallel_batch.py",
    "build_gate3_manifests.py",
    "build_dynamic_retime.py",
    "train_tracking.py",
    "play_tracking.py",
    "evaluate_tracking_suite.py",
    "build_final_artifacts.py",
)
LOCAL_CONFIG_FILES: Sequence[str] = (
    "config.yaml",
    "correspondence.yaml",
    "requirements-motion.txt",
    "README.md",
)


class FinalArtifactError(ValueError):
    """Raised when a claimed final snapshot is incomplete or inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_checksum(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FinalArtifactError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject Python's non-standard JSON NaN/Infinity extensions."""

    raise FinalArtifactError(f"non-finite JSON constant is forbidden: {value}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_no_duplicate_object,
                parse_constant=_reject_json_constant,
            )
    except FinalArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalArtifactError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalArtifactError(f"{label} must be a JSON object: {path}")
    return value


def _count_jsonl_objects(path: Path, label: str) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise FinalArtifactError(
                        f"{label} contains a blank record at line {line_number}"
                    )
                value = json.loads(
                    line,
                    object_pairs_hook=_no_duplicate_object,
                    # Historical per-item validation diagnostics use Infinity
                    # as a missing-value sentinel for absent contact channels.
                    # These records are counted/provenance-hashed only; final
                    # training and suite reports remain strictly finite JSON.
                    parse_constant=lambda _value: None,
                )
                if not isinstance(value, dict):
                    raise FinalArtifactError(
                        f"{label} line {line_number} must be a JSON object"
                    )
                count += 1
    except FinalArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalArtifactError(f"cannot read {label} {path}: {exc}") from exc
    return count


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalArtifactError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FinalArtifactError(f"{label} must be an array")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinalArtifactError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalArtifactError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise FinalArtifactError(f"{label} must be finite")
    return parsed


def _checksum(value: Any, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64:
        raise FinalArtifactError(f"{label} must be a 64-character SHA-256")
    try:
        int(text, 16)
    except ValueError as exc:
        raise FinalArtifactError(f"{label} must be hexadecimal") from exc
    return text


def _inside(path: str | Path, root: Path, label: str, *, kind: str = "file") -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise FinalArtifactError(f"{label} must remain below {root.resolve()}: {resolved}") from exc
    if kind == "file" and not resolved.is_file():
        raise FinalArtifactError(f"{label} is not an existing file: {resolved}")
    if kind == "directory" and not resolved.is_dir():
        raise FinalArtifactError(f"{label} is not an existing directory: {resolved}")
    return resolved


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ACCAD_ROOT.resolve()).as_posix()


def _artifact(path: Path, *, role: str, scope: str = "accad") -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FinalArtifactError(f"required artifact is missing: {resolved}")
    if scope == "accad":
        display = _relative(resolved)
    elif scope == "project_read_only":
        try:
            display = resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise FinalArtifactError(f"read-only dependency escaped project: {resolved}") from exc
    else:  # pragma: no cover - internal programming guard
        raise FinalArtifactError(f"unsupported artifact scope: {scope}")
    return {
        "path": display,
        "scope": scope,
        "role": role,
        "bytes": resolved.stat().st_size,
        "checksum_sha256": _sha256(resolved),
    }


class _Artifacts:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], dict[str, Any]] = {}
        self._paths: dict[tuple[str, str], Path] = {}

    def add(self, path: Path, category: str, role: str, *, scope: str = "accad") -> dict[str, Any]:
        record = _artifact(path, role=role, scope=scope)
        key = (scope, record["path"])
        previous = self._records.get(key)
        if previous is None:
            record["categories"] = [category]
            record["roles"] = [role]
            self._records[key] = record
            self._paths[key] = path.resolve()
            return record
        if previous["checksum_sha256"] != record["checksum_sha256"]:
            raise FinalArtifactError(f"artifact changed while snapshotting: {path}")
        if category not in previous["categories"]:
            previous["categories"].append(category)
        if role not in previous["roles"]:
            previous["roles"].append(role)
        return previous

    def grouped(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in self._records.values():
            for category in record["categories"]:
                view = {key: value for key, value in record.items() if key != "categories"}
                grouped.setdefault(category, []).append(view)
        for values in grouped.values():
            values.sort(key=lambda item: (item["scope"], item["path"], item["role"]))
        return dict(sorted(grouped.items()))

    @property
    def unique_count(self) -> int:
        return len(self._records)

    def verify_unchanged(self) -> None:
        """Fail if any collected file changed during snapshot construction."""

        for key, previous in self._records.items():
            current = _artifact(
                self._paths[key],
                role=str(previous["role"]),
                scope=str(previous["scope"]),
            )
            if (
                current["bytes"] != previous["bytes"]
                or current["checksum_sha256"] != previous["checksum_sha256"]
            ):
                raise FinalArtifactError(
                    f"artifact changed while final snapshot was built: {self._paths[key]}"
                )


def _verify_declared_artifact(
    record: Any,
    label: str,
    artifacts: _Artifacts,
    category: str,
    *,
    parent: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    item = _mapping(record, label)
    path = _inside(item.get("path", ""), ACCAD_ROOT, f"{label} path")
    if parent is not None:
        try:
            path.relative_to(parent.resolve())
        except ValueError as exc:
            raise FinalArtifactError(f"{label} must stay below {parent.resolve()}: {path}") from exc
    actual = _artifact(path, role=label)
    if _checksum(item.get("checksum_sha256"), f"{label} checksum") != actual["checksum_sha256"]:
        raise FinalArtifactError(f"{label} checksum mismatch: {path}")
    if "bytes" in item and _integer(item["bytes"], f"{label} bytes") != actual["bytes"]:
        raise FinalArtifactError(f"{label} size mismatch: {path}")
    artifacts.add(path, category, label)
    return path, actual


def _require_schema_status(
    report: Mapping[str, Any], schema: str, label: str, *, require_passed: bool = False
) -> None:
    if report.get("schema_version") != schema:
        raise FinalArtifactError(f"unsupported {label} schema: {report.get('schema_version')!r}")
    if report.get("status") != "completed":
        raise FinalArtifactError(f"{label} status is not completed")
    if require_passed and report.get("passed") is not True:
        raise FinalArtifactError(f"{label} did not pass")


def _manifest_path(split: str) -> Path:
    return WORK_ROOT / "manifests" / f"g1_{split}.json"


def _load_gate3_manifest(split: str, artifacts: _Artifacts) -> tuple[Path, dict[str, Any]]:
    manifest_path = _inside(_manifest_path(split), ACCAD_ROOT, f"Gate-3 {split} manifest")
    report = _read_json(manifest_path, f"Gate-3 {split} manifest")
    if report.get("schema_version") != GATE3_MANIFEST_SCHEMA or report.get("split") != split:
        raise FinalArtifactError(f"invalid Gate-3 {split} manifest contract")
    expected = EXPECTED_SOURCE_SPLIT_COUNTS[split]
    if report.get("num_motions") != expected:
        raise FinalArtifactError(f"Gate-3 {split} manifest must contain {expected} motions")
    if report.get("retarget_version") != RETARGET_VERSION:
        raise FinalArtifactError(f"Gate-3 {split} manifest retarget version mismatch")
    if (
        report.get("dataset") != "ACCAD"
        or report.get("kind") != f"split:{split}"
        or report.get("gate") != "gate3_kinematic_complete"
        or report.get("reference_root") != "work/g1"
    ):
        raise FinalArtifactError(f"Gate-3 {split} manifest provenance mismatch")
    _checksum(
        report.get("motion_set_checksum_sha256"),
        f"Gate-3 {split} motion-set checksum",
    )
    motions = _sequence(report.get("motions"), f"Gate-3 {split} motions")
    if len(motions) != expected:
        raise FinalArtifactError(f"Gate-3 {split} motion record count mismatch")
    reference_root = WORK_ROOT / "g1"
    paths: set[Path] = set()
    keys: set[str] = set()
    for index, value in enumerate(motions):
        motion = _mapping(value, f"Gate-3 {split} motion {index}")
        archive_path = _inside(
            reference_root / str(motion.get("g1_file", "")),
            ACCAD_ROOT,
            f"Gate-3 {split} archive {index}",
        )
        key = motion.get("source_file")
        if not isinstance(key, str) or not key or archive_path in paths or key in keys:
            raise FinalArtifactError(f"Gate-3 {split} motion identity is duplicated or empty")
        paths.add(archive_path)
        keys.add(key)
        expected_checksum = _checksum(
            motion.get("g1_checksum_sha256"),
            f"Gate-3 {split} archive checksum {index}",
        )
        expected_size = _integer(
            motion.get("g1_size_bytes"),
            f"Gate-3 {split} archive size {index}",
            minimum=1,
        )
        frame_count = _integer(
            motion.get("frame_count"),
            f"Gate-3 {split} frame count {index}",
            minimum=2,
        )
        fps = _number(motion.get("fps"), f"Gate-3 {split} fps {index}")
        duration = _number(
            motion.get("duration_sec"), f"Gate-3 {split} duration {index}"
        )
        if (
            archive_path.stat().st_size != expected_size
            or _sha256(archive_path) != expected_checksum
            or motion.get("split") != split
            or motion.get("retarget_version") != RETARGET_VERSION
            or motion.get("gate3_kinematic_complete") is not True
            or not math.isclose(fps, 50.0, abs_tol=1.0e-12)
            or not math.isclose(duration, (frame_count - 1) / fps, abs_tol=1.0e-9)
        ):
            raise FinalArtifactError(
                f"Gate-3 {split} archive metadata/checksum mismatch: {archive_path}"
            )
        artifacts.add(
            archive_path,
            "source_v27_motions",
            f"v27 Gate-3 {split} source motion",
        )
    artifacts.add(manifest_path, "manifests", f"Gate-3 {split} manifest")
    return manifest_path, report


def _dynamic_path(split: str, suffix: str) -> Path:
    return WORK_ROOT / "dynamic" / f"g1_{split}{suffix}"


def _pipeline_reference(value: Any, label: str, *, kind: str = "file") -> Path:
    raw = Path(str(value)).expanduser()
    candidate = raw if raw.is_absolute() else PIPELINE_ROOT / raw
    return _inside(candidate, ACCAD_ROOT, label, kind=kind)


def _expected_pipeline_relative(path: Path) -> str:
    return path.resolve().relative_to(PIPELINE_ROOT.resolve()).as_posix()


def _verify_dynamic_policy(
    path: Path,
    source_manifests: Mapping[str, tuple[Path, Mapping[str, Any]]],
    artifacts: _Artifacts,
) -> tuple[dict[str, Any], str]:
    policy = _read_json(path, "frozen dynamic policy")
    policy_checksum = _sha256(path)
    train_path, train_manifest = source_manifests["train"]
    expected_split_policy = {
        "train": "allowed_and_freezes_policy",
        "validation": "allowed_only_with_existing_frozen_train_policy",
        "test": "sealed_until_explicit_split_test_invocation",
    }
    train_source = _mapping(
        policy.get("train_source_manifest"), "dynamic policy train source"
    )
    if (
        policy.get("schema_version") != DYNAMIC_POLICY_SCHEMA
        or policy.get("status") != "frozen_from_train"
        or not _same_json_value(policy.get("algorithm"), DYNAMIC_POLICY_ALGORITHM)
        or _checksum(
            policy.get("algorithm_checksum_sha256"),
            "dynamic policy algorithm checksum",
        )
        != _canonical_checksum(DYNAMIC_POLICY_ALGORITHM)
        or policy.get("split_policy") != expected_split_policy
        or _pipeline_reference(
            train_source.get("path"), "dynamic policy train manifest"
        )
        != train_path
        or _checksum(
            train_source.get("checksum_sha256"),
            "dynamic policy train manifest checksum",
        )
        != _sha256(train_path)
        or train_source.get("motion_set_checksum_sha256")
        != train_manifest.get("motion_set_checksum_sha256")
        or train_source.get("num_motions") != EXPECTED_SOURCE_SPLIT_COUNTS["train"]
    ):
        raise FinalArtifactError("frozen dynamic scale policy contract mismatch")
    artifacts.add(path, "dynamic_retime", "frozen train-derived dynamic scale policy")
    return policy, policy_checksum


def _verify_dynamic_split(
    split: str,
    *,
    source_path: Path,
    source_manifest: Mapping[str, Any],
    policy_path: Path,
    policy: Mapping[str, Any],
    policy_checksum: str,
    artifacts: _Artifacts,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Verify the v27 -> v28 derivation without loading any NPZ numerically."""

    plan_path = _inside(
        _dynamic_path(split, "_scale_plan.json"),
        ACCAD_ROOT,
        f"dynamic {split} scale plan",
    )
    manifest_path = _inside(
        _dynamic_path(split, ".json"), ACCAD_ROOT, f"dynamic {split} manifest"
    )
    report_path = _inside(
        _dynamic_path(split, "_report.json"),
        ACCAD_ROOT,
        f"dynamic {split} report",
    )
    plan = _read_json(plan_path, f"dynamic {split} scale plan")
    manifest = _read_json(manifest_path, f"dynamic {split} manifest")
    report = _read_json(report_path, f"dynamic {split} report")
    plan_checksum = _sha256(plan_path)
    manifest_checksum = _sha256(manifest_path)
    source_checksum = _sha256(source_path)
    source_motion_set = source_manifest.get("motion_set_checksum_sha256")

    plan_source = _mapping(plan.get("source_manifest"), f"dynamic {split} plan source")
    plan_policy = _mapping(plan.get("frozen_policy"), f"dynamic {split} plan policy")
    if (
        plan.get("schema_version") != DYNAMIC_PLAN_SCHEMA
        or plan.get("status") != "completed"
        or plan.get("split") != split
        or _pipeline_reference(plan_source.get("path"), f"dynamic {split} source path")
        != source_path
        or _checksum(
            plan_source.get("checksum_sha256"), f"dynamic {split} source checksum"
        )
        != source_checksum
        or plan_source.get("motion_set_checksum_sha256") != source_motion_set
        or _pipeline_reference(plan_policy.get("path"), f"dynamic {split} policy path")
        != policy_path
        or _checksum(
            plan_policy.get("checksum_sha256"), f"dynamic {split} policy checksum"
        )
        != policy_checksum
        or _checksum(
            plan_policy.get("algorithm_checksum_sha256"),
            f"dynamic {split} algorithm checksum",
        )
        != policy.get("algorithm_checksum_sha256")
    ):
        raise FinalArtifactError(f"dynamic {split} plan provenance mismatch")

    source_motions = [
        _mapping(value, f"v27 {split} motion {index}")
        for index, value in enumerate(
            _sequence(source_manifest.get("motions"), f"v27 {split} motions")
        )
    ]
    source_by_path = {str(item["g1_file"]): item for item in source_motions}
    if len(source_by_path) != len(source_motions):
        raise FinalArtifactError(f"v27 {split} source paths are not unique")
    plan_records = [
        _mapping(value, f"dynamic {split} plan record {index}")
        for index, value in enumerate(
            _sequence(plan.get("records"), f"dynamic {split} plan records")
        )
    ]
    plan_by_source: dict[str, Mapping[str, Any]] = {}
    source_actions = {name: 0 for name in ("identity", "uniform_retime", "quarantine")}
    final_actions: dict[str, int] = {}
    derived_by_path: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(plan_records):
        source_relative = str(record.get("source_g1_file", ""))
        source = source_by_path.get(source_relative)
        if source is None or source_relative in plan_by_source:
            raise FinalArtifactError(
                f"dynamic {split} plan does not cover each v27 source exactly once"
            )
        plan_by_source[source_relative] = record
        source_action = record.get("source_action")
        audit = _mapping(record.get("audit"), f"dynamic {split} audit {index}")
        if source_action not in source_actions or audit.get("action") != source_action:
            raise FinalArtifactError(f"dynamic {split} source action mismatch at {index}")
        source_actions[str(source_action)] += 1
        final_action = str(record.get("final_action", ""))
        final_actions[final_action] = final_actions.get(final_action, 0) + 1
        if (
            _checksum(
                record.get("source_g1_checksum_sha256"),
                f"dynamic {split} source record checksum {index}",
            )
            != source.get("g1_checksum_sha256")
            or record.get("source_frame_count") != source.get("frame_count")
            or not math.isclose(
                _number(
                    record.get("source_duration_sec"),
                    f"dynamic {split} source record duration {index}",
                ),
                _number(
                    source.get("duration_sec"), f"v27 {split} source duration {index}"
                ),
                abs_tol=1.0e-9,
            )
        ):
            raise FinalArtifactError(f"dynamic {split} source record metadata mismatch")

        if record.get("status") == "derived_and_validated":
            if final_action not in {
                "identity_numeric_copy",
                "uniform_retime",
                "uniform_retime_with_post_fk_headroom",
            } or record.get("gate3_kinematic_complete") is not True:
                raise FinalArtifactError(f"dynamic {split} promoted plan record is invalid")
            derived_relative = str(record.get("derived_g1_file", ""))
            if derived_relative != source_relative or derived_relative in derived_by_path:
                raise FinalArtifactError(f"dynamic {split} derived path coverage mismatch")
            derivation = _mapping(
                record.get("derivation"), f"dynamic {split} derivation {index}"
            )
            derived_path = _pipeline_reference(
                derivation.get("path"), f"dynamic {split} derived archive {index}"
            )
            expected_derived_path = (WORK_ROOT / "dynamic" / "g1" / derived_relative).resolve()
            if derived_path != expected_derived_path:
                raise FinalArtifactError(f"dynamic {split} derived archive path mismatch")
            checksum = _checksum(
                derivation.get("checksum_sha256"),
                f"dynamic {split} derived checksum {index}",
            )
            size = _integer(
                derivation.get("bytes"), f"dynamic {split} derived size {index}", minimum=1
            )
            if (
                derived_path.stat().st_size != size
                or _sha256(derived_path) != checksum
                or derivation.get("retarget_version") != DYNAMIC_RETARGET_VERSION
                or derivation.get("source_action") != source_action
                or derivation.get("final_action") != final_action
                or derivation.get("output_frame_count") is None
                or derivation.get("output_duration_sec") is None
            ):
                raise FinalArtifactError(
                    f"dynamic {split} derived archive checksum/version mismatch"
                )
            derived_by_path[derived_relative] = record
            artifacts.add(
                derived_path,
                "dynamic_retargeted_motions",
                f"v28 dynamic {split} promoted motion",
            )
        elif record.get("status") in {
            "quarantined",
            "quarantined_after_derivation_audit",
        }:
            if not final_action.startswith("quarantine_"):
                raise FinalArtifactError(f"dynamic {split} quarantine action mismatch")
        else:
            raise FinalArtifactError(f"dynamic {split} has an unpublishable plan status")

    if set(plan_by_source) != set(source_by_path):
        raise FinalArtifactError(f"dynamic {split} plan source coverage is incomplete")
    derived_count = len(derived_by_path)
    source_count = len(source_motions)
    quarantined_count = source_count - derived_count
    recomputed_counts = {
        "source": source_count,
        **source_actions,
        "derived": derived_count,
        "final_quarantined": quarantined_count,
        "source_actions": source_actions,
        "final_actions": dict(sorted(final_actions.items())),
    }
    if not _same_json_value(plan.get("counts"), recomputed_counts):
        raise FinalArtifactError(f"dynamic {split} plan counts are inconsistent")
    if source_count != EXPECTED_SOURCE_SPLIT_COUNTS[split]:
        raise FinalArtifactError(f"dynamic {split} source population mismatch")
    if split in EXPECTED_CURRENT_DYNAMIC_COUNTS:
        if derived_count != EXPECTED_CURRENT_DYNAMIC_COUNTS[split]:
            raise FinalArtifactError(
                f"dynamic {split} must contain exactly "
                f"{EXPECTED_CURRENT_DYNAMIC_COUNTS[split]} promoted motions"
            )
    elif derived_count < 1:
        raise FinalArtifactError("dynamic test build promoted no eligible motion")

    promoted = [
        _mapping(value, f"dynamic {split} manifest motion {index}")
        for index, value in enumerate(
            _sequence(manifest.get("motions"), f"dynamic {split} manifest motions")
        )
    ]
    if [str(item.get("g1_file", "")) for item in promoted] != sorted(derived_by_path):
        raise FinalArtifactError(
            f"dynamic {split} manifest must be the sorted complete promoted set"
        )
    total_frames = 0
    total_duration = 0.0
    motion_set_payload: list[list[str]] = []
    for index, motion in enumerate(promoted):
        relative = str(motion.get("g1_file", ""))
        source = source_by_path[relative]
        plan_record = derived_by_path[relative]
        derivation = _mapping(
            plan_record.get("derivation"), f"dynamic {split} derivation {index}"
        )
        checksum = _checksum(
            motion.get("g1_checksum_sha256"),
            f"dynamic {split} manifest checksum {index}",
        )
        size = _integer(
            motion.get("g1_size_bytes"),
            f"dynamic {split} manifest size {index}",
            minimum=1,
        )
        frame_count = _integer(
            motion.get("frame_count"),
            f"dynamic {split} manifest frame count {index}",
            minimum=2,
        )
        duration = _number(
            motion.get("duration_sec"), f"dynamic {split} manifest duration {index}"
        )
        fps = _number(motion.get("fps"), f"dynamic {split} manifest fps {index}")
        if (
            motion.get("source_v27_g1_file") != relative
            or _checksum(
                motion.get("source_v27_g1_checksum_sha256"),
                f"dynamic {split} source checksum witness {index}",
            )
            != source.get("g1_checksum_sha256")
            or motion.get("source_file") != source.get("source_file")
            or motion.get("split") != split
            or motion.get("retarget_version") != DYNAMIC_RETARGET_VERSION
            or motion.get("gate3_kinematic_complete") is not True
            or _checksum(
                motion.get("dynamic_policy_checksum_sha256"),
                f"dynamic {split} policy witness {index}",
            )
            != policy_checksum
            or checksum != derivation.get("checksum_sha256")
            or size != derivation.get("bytes")
            or frame_count != derivation.get("output_frame_count")
            or not math.isclose(duration, float(derivation["output_duration_sec"]), abs_tol=1.0e-9)
            or not math.isclose(fps, 50.0, abs_tol=1.0e-12)
            or not math.isclose(duration, (frame_count - 1) / fps, abs_tol=1.0e-9)
            or motion.get("dynamic_source_action") != plan_record.get("source_action")
            or motion.get("dynamic_action") != plan_record.get("final_action")
            or not math.isclose(
                _number(motion.get("dynamic_scale_pre"), f"dynamic {split} scale {index}"),
                _number(derivation.get("source_scale_pre"), f"dynamic {split} source scale {index}"),
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                _number(motion.get("dynamic_effective_requested_scale"), f"dynamic {split} requested scale {index}"),
                _number(derivation.get("effective_requested_scale"), f"dynamic {split} derivation requested scale {index}"),
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                _number(motion.get("dynamic_realized_scale"), f"dynamic {split} realized scale {index}"),
                _number(derivation.get("realized_scale"), f"dynamic {split} derivation realized scale {index}"),
                abs_tol=1.0e-12,
            )
        ):
            raise FinalArtifactError(
                f"dynamic {split} promoted manifest record {index} is inconsistent"
            )
        total_frames += frame_count
        total_duration += duration
        motion_set_payload.append([split, relative, checksum])

    recomputed_motion_set = hashlib.sha256(
        json.dumps(motion_set_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        manifest.get("schema_version") != DYNAMIC_MANIFEST_SCHEMA
        or manifest.get("dataset") != "ACCAD"
        or manifest.get("kind") != f"split:{split}"
        or manifest.get("split") != split
        or manifest.get("gate") != "gate3_kinematic_complete+dynamic_prefilter_v1"
        or manifest.get("reference_root") != "work/dynamic/g1"
        or manifest.get("retarget_version") != DYNAMIC_RETARGET_VERSION
        or manifest.get("motion_set_checksum_sha256") != recomputed_motion_set
        or manifest.get("source_motion_set_checksum_sha256") != source_motion_set
        or manifest.get("source_manifest_checksum_sha256") != source_checksum
        or manifest.get("dynamic_policy_checksum_sha256") != policy_checksum
        or manifest.get("scale_plan_checksum_sha256") != plan_checksum
        or manifest.get("num_source_motions") != source_count
        or manifest.get("num_motions") != derived_count
        or manifest.get("num_source_policy_quarantined") != source_actions["quarantine"]
        or manifest.get("num_quarantined") != quarantined_count
        or manifest.get("num_frames") != total_frames
        or not math.isclose(
            _number(manifest.get("duration_sec"), f"dynamic {split} total duration"),
            total_duration,
            abs_tol=1.0e-8,
        )
    ):
        raise FinalArtifactError(f"dynamic {split} manifest publication mismatch")

    report_artifacts = _mapping(
        report.get("artifacts"), f"dynamic {split} report artifacts"
    )
    expected_report_artifacts = {
        "policy": (policy_path, policy_checksum),
        "plan": (plan_path, plan_checksum),
        "manifest": (manifest_path, manifest_checksum),
    }
    for name, (expected_path, expected_checksum) in expected_report_artifacts.items():
        record = _mapping(
            report_artifacts.get(name), f"dynamic {split} report {name}"
        )
        if (
            _pipeline_reference(record.get("path"), f"dynamic {split} report {name} path")
            != expected_path
            or _checksum(
                record.get("checksum_sha256"),
                f"dynamic {split} report {name} checksum",
            )
            != expected_checksum
        ):
            raise FinalArtifactError(f"dynamic {split} report {name} mismatch")
    if (
        report.get("schema_version") != DYNAMIC_REPORT_SCHEMA
        or report.get("status") != "completed"
        or report.get("split") != split
        or not _same_json_value(report.get("counts"), recomputed_counts)
        or report.get("validation_failures") != []
        or report.get("sealed_test_contract")
        != "test NPZ numerical contents are opened only by an explicit split=test invocation"
        or _pipeline_reference(
            report_artifacts.get("derived_reference_root"),
            f"dynamic {split} derived reference root",
            kind="directory",
        )
        != (WORK_ROOT / "dynamic" / "g1").resolve()
    ):
        raise FinalArtifactError(f"dynamic {split} report publication mismatch")

    artifacts.add(plan_path, "dynamic_retime", f"dynamic {split} scale plan")
    artifacts.add(manifest_path, "manifests", f"dynamic {split} promoted manifest")
    artifacts.add(report_path, "dynamic_retime", f"dynamic {split} build report")
    summary = {
        "status": "completed",
        "split": split,
        "source_motion_count": source_count,
        "promoted_motion_count": derived_count,
        "quarantined_motion_count": quarantined_count,
        "source_manifest_checksum_sha256": source_checksum,
        "policy_checksum_sha256": policy_checksum,
        "scale_plan_checksum_sha256": plan_checksum,
        "manifest_checksum_sha256": manifest_checksum,
        "all_promoted_archives_checksum_size_version_gate3_verified": True,
    }
    return manifest_path, manifest, summary


def _verify_manifest_selection(
    report: Mapping[str, Any],
    *,
    split: str,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    artifacts: _Artifacts,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    motion_input = _mapping(report.get("motion_input"), f"{split} motion_input")
    if motion_input.get("split") != split or motion_input.get("allow_missing") is not False:
        raise FinalArtifactError(f"{split} motion selection was partial or used the wrong split")
    expected_count = _integer(
        manifest.get("num_motions"), f"dynamic {split} manifest motion count", minimum=1
    )
    if motion_input.get("motion_count") != expected_count:
        raise FinalArtifactError(f"{split} motion selection must contain {expected_count} motions")
    # v2 runs started before the explicit coverage counters were added do not
    # contain these two convenience witnesses.  Missing fields are safe here:
    # the exact manifest checksum and every selected archive checksum are
    # independently recomputed below.  If a producer does emit them, however,
    # inconsistent values fail closed.
    if (
        "manifest_checksum_verified_motion_count" in motion_input
        and motion_input.get("manifest_checksum_verified_motion_count") != expected_count
    ) or (
        "manifest_promotion_verified" in motion_input
        and motion_input.get("manifest_promotion_verified") is not True
    ):
        raise FinalArtifactError(f"{split} motion selection coverage metadata is inconsistent")
    manifest_entries = _sequence(motion_input.get("manifests"), f"{split} input manifests")
    if len(manifest_entries) != 1:
        raise FinalArtifactError(f"{split} must use exactly one authoritative dynamic manifest")
    entry = _mapping(manifest_entries[0], f"{split} manifest provenance")
    declared_path = _inside(entry.get("path", ""), ACCAD_ROOT, f"{split} manifest path")
    if declared_path != manifest_path:
        raise FinalArtifactError(f"{split} report used a different manifest: {declared_path}")
    manifest_checksum = _sha256(manifest_path)
    if _checksum(entry.get("checksum_sha256"), f"{split} manifest checksum") != manifest_checksum:
        raise FinalArtifactError(f"{split} report manifest checksum mismatch")
    if (
        entry.get("split") != split
        or entry.get("resolved_count") != expected_count
        or (
            "archive_checksum_record_count" in entry
            and entry.get("archive_checksum_record_count") != expected_count
        )
    ):
        raise FinalArtifactError(f"{split} report manifest resolution is incomplete")

    expected: dict[Path, dict[str, Any]] = {}
    expected_keys: set[str] = set()
    reference_root = WORK_ROOT / "dynamic" / "g1"
    for index, value in enumerate(_sequence(manifest.get("motions"), f"{split} manifest motions")):
        motion = _mapping(value, f"{split} manifest motion {index}")
        path = _inside(reference_root / str(motion.get("g1_file", "")), ACCAD_ROOT, f"{split} G1 motion")
        checksum = _checksum(motion.get("g1_checksum_sha256"), f"{split} G1 checksum")
        if path in expected:
            raise FinalArtifactError(f"duplicate {split} motion path: {path}")
        motion_key = motion.get("source_file")
        if not isinstance(motion_key, str) or not motion_key:
            raise FinalArtifactError(f"{split} manifest motion {index} has no source_file key")
        if motion_key in expected_keys:
            raise FinalArtifactError(f"duplicate {split} motion key: {motion_key!r}")
        expected_keys.add(motion_key)
        frame_count = _integer(
            motion.get("frame_count"), f"{split} manifest frame_count {index}", minimum=2
        )
        fps = _number(motion.get("fps"), f"{split} manifest fps {index}")
        if not math.isclose(fps, 1.0 / EXPECTED_CONTROL_DT, abs_tol=1e-12):
            raise FinalArtifactError(f"{split} manifest motion {index} must be 50 Hz")
        duration_s = _number(
            motion.get("duration_sec"), f"{split} manifest duration {index}"
        )
        expected_duration_s = (frame_count - 1) / fps
        if not math.isclose(duration_s, expected_duration_s, abs_tol=1e-9):
            raise FinalArtifactError(
                f"{split} manifest motion {index} duration disagrees with frame_count/fps"
            )
        if (
            motion.get("split") != split
            or motion.get("retarget_version") != DYNAMIC_RETARGET_VERSION
            or motion.get("gate3_kinematic_complete") is not True
        ):
            raise FinalArtifactError(f"{split} manifest motion {index} is not Gate-3 complete")
        declared_size = _integer(
            motion.get("g1_size_bytes"), f"{split} manifest G1 size {index}", minimum=1
        )
        if path.stat().st_size != declared_size:
            raise FinalArtifactError(f"{split} manifest G1 size mismatch: {path}")
        expected[path] = {
            "path": path,
            "checksum_sha256": checksum,
            "motion_key": motion_key,
            "frame_count": frame_count,
            "duration_s": duration_s,
        }

    observed: dict[Path, str] = {}
    observed_order: list[Path] = []
    for index, value in enumerate(_sequence(motion_input.get("motions"), f"{split} selected motions")):
        motion = _mapping(value, f"{split} selected motion {index}")
        path = _inside(motion.get("path", ""), ACCAD_ROOT, f"{split} selected motion")
        checksum = _checksum(motion.get("checksum_sha256"), f"{split} selected checksum")
        if "bytes" in motion and _integer(motion["bytes"], f"{split} selected bytes") != path.stat().st_size:
            raise FinalArtifactError(f"{split} selected motion size mismatch: {path}")
        actual = _sha256(path)
        if actual != checksum:
            raise FinalArtifactError(f"{split} selected motion checksum mismatch: {path}")
        if path in observed:
            raise FinalArtifactError(f"duplicate {split} selected motion: {path}")
        observed[path] = actual
        observed_order.append(path)
    expected_checksums = {
        path: str(metadata["checksum_sha256"]) for path, metadata in expected.items()
    }
    if observed != expected_checksums:
        missing = sorted(str(path) for path in expected.keys() - observed.keys())
        extra = sorted(str(path) for path in observed.keys() - expected.keys())
        raise FinalArtifactError(f"{split} selected motion set differs from manifest; missing={missing}, extra={extra}")
    for path in sorted(observed, key=lambda item: item.as_posix()):
        artifacts.add(path, "dynamic_retargeted_motions", f"v28 dynamic {split} G1 motion")
    if observed_order != list(expected):
        raise FinalArtifactError(
            f"{split} selected motion order differs from the authoritative dynamic manifest"
        )
    digest_payload = [
        [_relative(path), checksum] for path, checksum in sorted(observed.items(), key=lambda item: str(item[0]))
    ]
    digest = hashlib.sha256(
        json.dumps(digest_payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    selection = {
        "split": split,
        "motion_count": expected_count,
        "manifest_path": _relative(manifest_path),
        "manifest_checksum_sha256": manifest_checksum,
        "verified_motion_selection_checksum_sha256": digest,
        "motion_keys": [str(expected[path]["motion_key"]) for path in observed_order],
    }
    authoritative_motions = [expected[path] for path in observed_order]
    return selection, authoritative_motions


def _verify_correspondence(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    correspondence = _mapping(report.get("correspondence"), f"{label} correspondence")
    expected_path = (PIPELINE_ROOT / "correspondence.yaml").resolve()
    path = _inside(correspondence.get("path", ""), ACCAD_ROOT, f"{label} correspondence path")
    if path != expected_path:
        raise FinalArtifactError(f"{label} used a non-authoritative correspondence file")
    checksum = _sha256(path)
    if _checksum(correspondence.get("checksum_sha256"), f"{label} correspondence checksum") != checksum:
        raise FinalArtifactError(f"{label} correspondence checksum mismatch")
    return {"path": _relative(path), "checksum_sha256": checksum}


def _verify_evaluation_isolation(value: Any, label: str) -> dict[str, Any]:
    isolation = _mapping(value, f"{label} evaluation isolation")
    if isolation.get("schema_version") != EVALUATION_ISOLATION_SCHEMA_VERSION:
        raise FinalArtifactError(f"{label} evaluation isolation schema mismatch")
    boolean_values = {
        "command_random_start": False,
        "policy_observation_corruption_enabled": False,
        "enhanced_determinism_enabled": True,
    }
    for key, expected in boolean_values.items():
        # JSON numbers compare equal to booleans in Python (0 == False and
        # 1 == True).  Require the actual JSON boolean type so provenance
        # evidence cannot pass through that aliasing behaviour.
        if isolation.get(key) is not expected:
            raise FinalArtifactError(f"{label} evaluation isolation {key} mismatch")
    active_event_terms = _mapping(
        isolation.get("active_event_terms"),
        f"{label} evaluation isolation active event terms",
    )
    if active_event_terms:
        raise FinalArtifactError(
            f"{label} evaluation isolation active_event_terms mismatch"
        )
    scalar_values = {
        "fixed_start_time_s": 0.0,
        "reward_integration_dt_s": EXPECTED_CONTROL_DT,
    }
    for key, expected in scalar_values.items():
        if not math.isclose(
            _number(isolation.get(key), f"{label} evaluation isolation {key}"),
            expected,
            abs_tol=1.0e-12,
        ):
            raise FinalArtifactError(f"{label} evaluation isolation {key} mismatch")
    for key in ("xy_offset_range_m", "target_yaw_range_rad"):
        values = _sequence(isolation.get(key), f"{label} evaluation isolation {key}")
        if len(values) != 2 or any(
            not math.isclose(
                _number(item, f"{label} evaluation isolation {key}"),
                0.0,
                abs_tol=1.0e-12,
            )
            for item in values
        ):
            raise FinalArtifactError(f"{label} evaluation isolation {key} must be zero")
    resolved_noise = _mapping(
        isolation.get("resolved_reset_state_noise"),
        f"{label} evaluation reset-state noise",
    )
    expected_noise_fields = {
        "joint_position_rad",
        "joint_velocity_radps",
        "root_linear_velocity_mps",
        "root_angular_velocity_radps",
    }
    if set(resolved_noise) != expected_noise_fields:
        raise FinalArtifactError(f"{label} evaluation reset-state noise fields mismatch")
    for field in sorted(expected_noise_fields):
        values = _sequence(
            resolved_noise.get(field),
            f"{label} evaluation reset-state noise {field}",
        )
        if len(values) != 2 or any(
            not math.isclose(
                _number(item, f"{label} evaluation reset-state noise {field}"),
                0.0,
                abs_tol=1.0e-12,
            )
            for item in values
        ):
            raise FinalArtifactError(f"{label} evaluation reset-state noise must be zero")
    reward_contract = _mapping(
        isolation.get("evaluation_reward_contract"),
        f"{label} evaluation reward contract",
    )
    if set(reward_contract) != {"alive", "failure"}:
        raise FinalArtifactError(f"{label} evaluation reward contract keys mismatch")
    alive_reward = _mapping(
        reward_contract.get("alive"), f"{label} evaluation alive reward"
    )
    failure_reward = _mapping(
        reward_contract.get("failure"), f"{label} evaluation failure reward"
    )
    if alive_reward.get("kind") != "per_second_rate" or not math.isclose(
        _number(alive_reward.get("weight"), f"{label} evaluation alive reward"),
        0.10,
        abs_tol=1.0e-12,
    ) or failure_reward.get("kind") != "one_shot_non_timeout_impulse" or not math.isclose(
        _number(failure_reward.get("impulse"), f"{label} evaluation failure reward"),
        -5.0,
        abs_tol=1.0e-12,
    ) or failure_reward.get("normalization") != "divide_by_live_step_dt" or (
        failure_reward.get("pure_timeouts_excluded") is not True
    ) or failure_reward.get("simultaneous_timeout_failure_penalized") is not True:
        raise FinalArtifactError(f"{label} evaluation reward contract mismatch")
    return dict(isolation)


def _expected_live_tensor_contract() -> dict[str, Any]:
    """Build the exact Isaac-manager witness implied by the static v5 contract."""

    groups: dict[str, Any] = {}
    for group_name, group_value in _mapping(
        EXPECTED_OBSERVATION_CONTRACT.get("observation_groups"),
        "expected observation groups",
    ).items():
        group = _mapping(group_value, f"expected {group_name} observation group")
        terms = [
            _mapping(term, f"expected {group_name} observation term")
            for term in _sequence(
                group.get("terms"), f"expected {group_name} observation terms"
            )
        ]
        dimensions = [int(term["dimension"]) for term in terms]
        dimension = int(group["dimension"])
        groups[str(group_name)] = {
            "term_names": [str(term["name"]) for term in terms],
            "term_dimensions": dimensions,
            "term_shapes": [[value] for value in dimensions],
            "group_shape": [dimension],
            "dimension": dimension,
            "concatenate_terms": True,
        }
    return {
        "schema_version": "accad_g1_live_tracking_tensor_contract_v1",
        "policy_contract_version": EXPECTED_TRACKING_POLICY_CONTRACT,
        "static_observation_contract_schema_version": EXPECTED_OBSERVATION_CONTRACT[
            "schema_version"
        ],
        "future_horizons_s": list(
            EXPECTED_OBSERVATION_CONTRACT["future_horizons_s"]
        ),
        "observation_groups": groups,
        "action": dict(EXPECTED_OBSERVATION_CONTRACT["action"]),
    }


EXPECTED_LIVE_TENSOR_CONTRACT: Mapping[str, Any] = _expected_live_tensor_contract()


def _same_json_value(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's ``True == 1`` type aliasing."""

    return json.dumps(
        left,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) == json.dumps(
        right,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _verify_tracking_tensor_contracts(
    contract: Mapping[str, Any], label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require canonical static semantics plus the resolved live tensor layout."""

    observation_contract = dict(
        _mapping(
            contract.get("observation_contract"),
            f"{label} observation contract",
        )
    )
    if not _same_json_value(observation_contract, EXPECTED_OBSERVATION_CONTRACT):
        raise FinalArtifactError(
            f"{label} observation contract differs from the canonical v5 contract"
        )
    live_tensor_contract = dict(
        _mapping(
            contract.get("live_tensor_contract"),
            f"{label} live tensor contract",
        )
    )
    if not _same_json_value(live_tensor_contract, EXPECTED_LIVE_TENSOR_CONTRACT):
        raise FinalArtifactError(
            f"{label} live tensor contract differs from the resolved v5 layout"
        )
    return observation_contract, live_tensor_contract


def _verify_warmup_observation_finiteness(
    value: Any,
    label: str,
    *,
    expected_num_envs: int,
) -> dict[str, Any]:
    """Verify that a warm-up reset observed both canonical finite groups."""

    witness = dict(_mapping(value, label))
    expected_tensors = [
        {
            "path": "observation.critic",
            "shape": [expected_num_envs, EXPECTED_RUNTIME["critic_observation_dim"]],
            "dtype": "torch.float32",
        },
        {
            "path": "observation.policy",
            "shape": [expected_num_envs, EXPECTED_RUNTIME["actor_observation_dim"]],
            "dtype": "torch.float32",
        },
    ]
    expected_elements = expected_num_envs * (
        EXPECTED_RUNTIME["actor_observation_dim"]
        + EXPECTED_RUNTIME["critic_observation_dim"]
    )
    if (
        _integer(witness.get("tensor_count"), f"{label} tensor_count", minimum=1)
        != len(expected_tensors)
        or _integer(
            witness.get("element_count"), f"{label} element_count", minimum=1
        )
        != expected_elements
        or _integer(witness.get("nonfinite_count"), f"{label} nonfinite_count")
        != 0
        or witness.get("all_finite") is not True
        or not _same_json_value(witness.get("tensors"), expected_tensors)
    ):
        raise FinalArtifactError(
            f"{label} does not prove finite canonical policy/critic observations"
        )
    return witness


def _verify_physics_cold_start_warmup(
    value: Any,
    label: str,
    *,
    expected_num_envs: int,
) -> dict[str, Any]:
    """Require the production 100-step, policy-independent cold-start witness."""

    warmup = dict(_mapping(value, f"{label} physics cold-start warmup"))
    if warmup.get("schema_version") != PHYSICS_WARMUP_SCHEMA_VERSION:
        raise FinalArtifactError(f"{label} physics warmup schema mismatch")
    if warmup.get("enabled") is not True or warmup.get("passed") is not True:
        raise FinalArtifactError(f"{label} physics warmup was not enabled and passed")
    requested = _integer(
        warmup.get("requested_control_steps"),
        f"{label} requested warmup control steps",
    )
    executed = _integer(
        warmup.get("executed_control_steps"),
        f"{label} executed warmup control steps",
    )
    if requested != EXPECTED_PHYSICS_WARMUP_STEPS or executed != requested:
        raise FinalArtifactError(
            f"{label} physics warmup must request and execute "
            f"{EXPECTED_PHYSICS_WARMUP_STEPS} control steps"
        )
    num_envs = _integer(
        warmup.get("num_environments"), f"{label} warmup num_environments", minimum=1
    )
    action_dim = _integer(
        warmup.get("action_dimension"), f"{label} warmup action_dimension", minimum=1
    )
    if (
        num_envs != expected_num_envs
        or action_dim != EXPECTED_RUNTIME["action_dim"]
        or not _same_json_value(
            warmup.get("zero_action_shape"),
            [expected_num_envs, EXPECTED_RUNTIME["action_dim"]],
        )
    ):
        raise FinalArtifactError(f"{label} physics warmup tensor dimensions mismatch")
    if warmup.get("policy_inference_used") is not False:
        raise FinalArtifactError(f"{label} physics warmup used policy inference")
    if warmup.get("purpose") != "prime_physics_and_contact_caches_only":
        raise FinalArtifactError(f"{label} physics warmup purpose mismatch")
    if not math.isclose(
        _number(
            warmup.get("final_reset_forced_motion_time_s"),
            f"{label} final reset motion time",
        ),
        0.0,
        abs_tol=1.0e-12,
    ) or warmup.get("final_reset_zeroed_reference_state_noise") is not True:
        raise FinalArtifactError(f"{label} physics warmup final reset contract mismatch")

    _verify_warmup_observation_finiteness(
        warmup.get("initial_reset_observations"),
        f"{label} warmup initial reset observations",
        expected_num_envs=expected_num_envs,
    )
    maximum_signals = expected_num_envs * EXPECTED_PHYSICS_WARMUP_STEPS
    for key in (
        "warmup_done_signals",
        "warmup_terminated_signals",
        "warmup_timeout_signals",
    ):
        if _integer(warmup.get(key), f"{label} {key}") > maximum_signals:
            raise FinalArtifactError(f"{label} {key} exceeds the warmup capacity")
    if (
        _integer(
            warmup.get("warmup_steps_with_any_done"),
            f"{label} warmup steps with any done",
        )
        > EXPECTED_PHYSICS_WARMUP_STEPS
    ):
        raise FinalArtifactError(f"{label} warmup done-step count is inconsistent")
    termination_counts = _mapping(
        warmup.get("warmup_termination_term_counts"),
        f"{label} warmup termination term counts",
    )
    if not termination_counts:
        raise FinalArtifactError(f"{label} warmup recorded no termination terms")
    for term_name, count in termination_counts.items():
        if not isinstance(term_name, str) or not term_name:
            raise FinalArtifactError(f"{label} warmup termination term name is invalid")
        if _integer(count, f"{label} warmup {term_name} count") > maximum_signals:
            raise FinalArtifactError(
                f"{label} warmup termination count exceeds the warmup capacity"
            )

    post_reset = _mapping(warmup.get("post_reset"), f"{label} warmup post_reset")
    for key in (
        "authoritative_motion_frame_zero",
        "episode_counters_zero",
        "action_state_zero",
        "passed",
    ):
        if post_reset.get(key) is not True:
            raise FinalArtifactError(f"{label} warmup post_reset {key} mismatch")
    tolerance = _number(
        post_reset.get("tolerance"), f"{label} warmup post_reset tolerance"
    )
    if not math.isclose(
        tolerance, EXPECTED_PHYSICS_WARMUP_TOLERANCE, abs_tol=1.0e-12
    ):
        raise FinalArtifactError(f"{label} warmup post_reset tolerance mismatch")
    maxima = _mapping(
        post_reset.get("maximum_absolute_values"),
        f"{label} warmup post_reset maximum values",
    )
    expected_maximum_fields = {
        "episode_length_buf",
        "motion_time",
        "raw_action",
        "normalized_residual",
        "previous_normalized_residual",
    }
    if set(maxima) != expected_maximum_fields or any(
        _number(maxima.get(key), f"{label} post_reset {key} maximum") > tolerance
        for key in expected_maximum_fields
    ):
        raise FinalArtifactError(f"{label} warmup post_reset zero-state invariants failed")
    reset_buffers = _mapping(
        post_reset.get("reset_buffers"), f"{label} warmup post_reset reset buffers"
    )
    expected_reset_buffers = {"reset_buf", "reset_terminated", "reset_time_outs"}
    if set(reset_buffers) != expected_reset_buffers:
        raise FinalArtifactError(f"{label} warmup post_reset reset buffers mismatch")
    for key in expected_reset_buffers:
        check = _mapping(reset_buffers.get(key), f"{label} post_reset {key}")
        if (
            _integer(check.get("nonzero_count"), f"{label} post_reset {key} count")
            != 0
            or check.get("all_clear") is not True
        ):
            raise FinalArtifactError(f"{label} warmup post_reset reset buffers are not clear")
    state_errors = _mapping(
        post_reset.get("reference_state_maximum_errors"),
        f"{label} warmup post_reset reference errors",
    )
    expected_state_error_fields = {
        "joint_position_rad",
        "root_position_m",
        "root_orientation_quaternion_l2",
    }
    if set(state_errors) != expected_state_error_fields or any(
        _number(state_errors.get(key), f"{label} post_reset {key} error") > tolerance
        for key in expected_state_error_fields
    ):
        raise FinalArtifactError(f"{label} warmup post_reset reference state mismatch")
    _verify_warmup_observation_finiteness(
        post_reset.get("observations"),
        f"{label} warmup post_reset observations",
        expected_num_envs=expected_num_envs,
    )
    return warmup


def _verify_runtime(
    runtime: Any,
    label: str,
    expected_count: int,
    *,
    suite: bool,
    authoritative_motions: Sequence[Mapping[str, Any]],
    expected_num_envs: int,
    recovery_training: bool = False,
) -> dict[str, Any]:
    contract = _mapping(runtime, f"{label} runtime contract")
    for key, expected in EXPECTED_RUNTIME.items():
        if contract.get(key) != expected:
            raise FinalArtifactError(f"{label} runtime {key} must be {expected}")
    observation_contract, live_tensor_contract = _verify_tracking_tensor_contracts(
        contract, label
    )
    physics_warmup = _verify_physics_cold_start_warmup(
        contract.get("physics_cold_start_warmup"),
        label,
        expected_num_envs=expected_num_envs,
    )
    if contract.get("motion_count") != expected_count:
        raise FinalArtifactError(f"{label} runtime motion_count mismatch")
    physics_key = "physics_dt_s" if suite else "physics_dt"
    control_key = "control_dt_s" if suite else "control_dt"
    if not math.isclose(_number(contract.get(physics_key), f"{label} physics dt"), EXPECTED_PHYSICS_DT, abs_tol=1e-12):
        raise FinalArtifactError(f"{label} physics dt mismatch")
    if not math.isclose(_number(contract.get(control_key), f"{label} control dt"), EXPECTED_CONTROL_DT, abs_tol=1e-12):
        raise FinalArtifactError(f"{label} control dt mismatch")
    if len(authoritative_motions) != expected_count:
        raise FinalArtifactError(f"{label} authoritative motion count mismatch")
    expected_keys = [str(motion["motion_key"]) for motion in authoritative_motions]
    contract_digest_key = (
        "library_contract_checksum_sha256" if suite else "contract_checksum_sha256"
    )
    contract_digest = _checksum(
        contract.get(contract_digest_key), f"{label} motion contract checksum"
    )
    evaluation_isolation: dict[str, Any] | None = None
    if suite:
        evaluation_isolation = _verify_evaluation_isolation(
            contract.get("evaluation_isolation"),
            label,
        )
        expected_motion_ids = list(range(expected_count))
        if contract.get("per_environment_motion_ids") != expected_motion_ids:
            raise FinalArtifactError(f"{label} per-environment motion mapping mismatch")
        runtime_motions = _sequence(contract.get("motions"), f"{label} runtime motions")
        if len(runtime_motions) != expected_count:
            raise FinalArtifactError(f"{label} runtime motion record count mismatch")
        for index, (value, expected_motion) in enumerate(
            zip(runtime_motions, authoritative_motions, strict=True)
        ):
            motion = _mapping(value, f"{label} runtime motion {index}")
            if motion.get("environment_id") != index or motion.get("motion_id") != index:
                raise FinalArtifactError(f"{label} runtime motion {index} ID mapping mismatch")
            if motion.get("motion_key") != expected_motion["motion_key"]:
                raise FinalArtifactError(f"{label} runtime motion {index} key mismatch")
            path = _inside(motion.get("path", ""), ACCAD_ROOT, f"{label} runtime motion {index}")
            if path != expected_motion["path"]:
                raise FinalArtifactError(f"{label} runtime motion {index} path mismatch")
            if (
                _checksum(
                    motion.get("checksum_sha256"),
                    f"{label} runtime motion {index} checksum",
                )
                != expected_motion["checksum_sha256"]
            ):
                raise FinalArtifactError(f"{label} runtime motion {index} checksum mismatch")
            if motion.get("frame_count") != expected_motion["frame_count"]:
                raise FinalArtifactError(f"{label} runtime motion {index} frame_count mismatch")
            if not math.isclose(
                _number(motion.get("duration_s"), f"{label} runtime motion {index} duration"),
                float(expected_motion["duration_s"]),
                abs_tol=RUNTIME_DURATION_ABS_TOL_S,
            ):
                raise FinalArtifactError(f"{label} runtime motion {index} duration mismatch")
    else:
        motion_keys = _sequence(contract.get("motion_keys"), f"{label} runtime motion keys")
        if motion_keys != expected_keys:
            raise FinalArtifactError(f"{label} runtime motion keys differ from Gate-3 manifest")
        if recovery_training:
            if contract.get("tracking_policy_contract_version") != EXPECTED_TRACKING_POLICY_CONTRACT:
                raise FinalArtifactError("training did not use the final recovery tracking contract")
            if contract.get("policy_wrapper_action_clip") is not None:
                raise FinalArtifactError("training wrapper hid raw actions before the action term")
            if not math.isclose(
                _number(contract.get("physical_action_clip"), "physical action clip"),
                1.0,
                abs_tol=1.0e-12,
            ):
                raise FinalArtifactError("training physical action clip mismatch")
            if not math.isclose(
                _number(
                    contract.get("frame_zero_start_probability"),
                    "frame-zero start probability",
                ),
                0.5,
                abs_tol=1.0e-12,
            ):
                raise FinalArtifactError("training frame-zero start probability must be 0.5")
            if contract.get("domain_randomization_events_enabled") is not False:
                raise FinalArtifactError("final recovery run must use nominal physics")
            if contract.get("random_episode_length_enabled") is not False:
                raise FinalArtifactError(
                    "final recovery run must disable runner episode-length randomization"
                )
            objective = _mapping(
                contract.get("training_objective"),
                "training recovery objective",
            )
            if objective.get("schema_version") != TRAINING_OBJECTIVE_SCHEMA_VERSION:
                raise FinalArtifactError("training recovery objective schema mismatch")
            if not math.isclose(
                _number(
                    objective.get("reset_state_noise_scale"),
                    "reset-state noise scale",
                ),
                EXPECTED_RESET_STATE_NOISE_SCALE,
                abs_tol=1.0e-12,
            ):
                raise FinalArtifactError("training reset-state noise scale mismatch")
            resolved_noise = _mapping(
                objective.get("resolved_reset_state_noise"),
                "resolved reset-state noise",
            )
            expected_noise_fields = (
                "joint_position_rad",
                "joint_velocity_radps",
                "root_linear_velocity_mps",
                "root_angular_velocity_radps",
            )
            if set(resolved_noise) != set(expected_noise_fields):
                raise FinalArtifactError("resolved reset-state noise fields mismatch")
            for field in expected_noise_fields:
                bounds = _sequence(resolved_noise.get(field), f"resolved {field} noise")
                if len(bounds) != 2 or any(
                    not math.isclose(
                        _number(value, f"resolved {field} noise"),
                        0.0,
                        abs_tol=1.0e-12,
                    )
                    for value in bounds
                ):
                    raise FinalArtifactError("final recovery reset-state noise must be zero")
            reward_weights = _mapping(
                objective.get("reward_weights"),
                "training recovery reward weights",
            )
            expected_reward_weight_keys = {
                "alive",
                "failure",
                "divergence_risk",
                "action_boundary_margin",
                "action_limiter_request_gap",
            }
            if set(reward_weights) != expected_reward_weight_keys or not math.isclose(
                _number(reward_weights.get("alive"), "alive reward weight"),
                EXPECTED_ALIVE_REWARD_WEIGHT,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                _number(reward_weights.get("failure"), "failure reward weight"),
                EXPECTED_FAILURE_REWARD_WEIGHT,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                _number(
                    reward_weights.get("divergence_risk"),
                    "divergence-risk reward weight",
                ),
                EXPECTED_DIVERGENCE_RISK_REWARD_WEIGHT,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                _number(
                    reward_weights.get("action_boundary_margin"),
                    "action-boundary reward weight",
                ),
                EXPECTED_ACTION_BOUNDARY_REWARD_WEIGHT,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                _number(
                    reward_weights.get("action_limiter_request_gap"),
                    "action-limiter-gap reward weight",
                ),
                EXPECTED_ACTION_LIMITER_GAP_REWARD_WEIGHT,
                abs_tol=1.0e-12,
            ):
                raise FinalArtifactError("training recovery reward weights mismatch")
            if not math.isclose(
                _number(
                    objective.get("reward_integration_dt_s"),
                    "training reward integration dt",
                ),
                EXPECTED_CONTROL_DT,
                abs_tol=1.0e-12,
            ):
                raise FinalArtifactError("training reward integration dt mismatch")
            reward_semantics = _mapping(
                objective.get("reward_semantics"),
                "training recovery reward semantics",
            )
            if set(reward_semantics) != expected_reward_weight_keys:
                raise FinalArtifactError("training recovery reward semantics keys mismatch")
            alive_semantics = _mapping(
                reward_semantics.get("alive"),
                "training alive reward semantics",
            )
            failure_semantics = _mapping(
                reward_semantics.get("failure"),
                "training failure reward semantics",
            )
            divergence_semantics = _mapping(
                reward_semantics.get("divergence_risk"),
                "training divergence-risk reward semantics",
            )
            boundary_semantics = _mapping(
                reward_semantics.get("action_boundary_margin"),
                "training action-boundary reward semantics",
            )
            limiter_semantics = _mapping(
                reward_semantics.get("action_limiter_request_gap"),
                "training action-limiter-gap reward semantics",
            )
            if (
                alive_semantics.get("kind") != "per_second_rate"
                or not math.isclose(
                    _number(alive_semantics.get("weight"), "training alive reward rate"),
                    EXPECTED_ALIVE_REWARD_WEIGHT,
                    abs_tol=1.0e-12,
                )
                or failure_semantics.get("kind")
                != "one_shot_non_timeout_impulse"
                or not math.isclose(
                    _number(
                        failure_semantics.get("impulse"),
                        "training failure reward impulse",
                    ),
                    EXPECTED_FAILURE_REWARD_WEIGHT,
                    abs_tol=1.0e-12,
                )
                or failure_semantics.get("normalization")
                != "divide_by_live_step_dt"
                or failure_semantics.get("pure_timeouts_excluded") is not True
                or failure_semantics.get("simultaneous_timeout_failure_penalized")
                is not True
                or divergence_semantics.get("kind")
                != "per_second_dense_squared_margin_barrier"
                or divergence_semantics.get("aggregation")
                != "maximum_across_exact_termination_components"
                or not math.isclose(
                    _number(
                        divergence_semantics.get("onset_fraction"),
                        "training divergence-risk onset fraction",
                    ),
                    EXPECTED_DIVERGENCE_RISK_ONSET_FRACTION,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    _number(
                        divergence_semantics.get("maximum_root_distance_m"),
                        "training divergence root threshold",
                    ),
                    EXPECTED_DIVERGENCE_MAX_ROOT_DISTANCE_M,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    _number(
                        divergence_semantics.get(
                            "maximum_mean_joint_error_rad"
                        ),
                        "training divergence joint threshold",
                    ),
                    EXPECTED_DIVERGENCE_MAX_MEAN_JOINT_ERROR_RAD,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    _number(
                        divergence_semantics.get(
                            "maximum_mean_body_distance_m"
                        ),
                        "training divergence body threshold",
                    ),
                    EXPECTED_DIVERGENCE_MAX_MEAN_BODY_DISTANCE_M,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    _number(
                        divergence_semantics.get("weight"),
                        "training divergence-risk reward weight",
                    ),
                    EXPECTED_DIVERGENCE_RISK_REWARD_WEIGHT,
                    abs_tol=1.0e-12,
                )
                or divergence_semantics.get(
                    "hard_termination_thresholds_unchanged"
                )
                is not True
                or boundary_semantics.get("kind")
                != "per_second_dense_squared_action_margin"
                or not math.isclose(
                    _number(
                        boundary_semantics.get("onset"),
                        "training action-boundary onset",
                    ),
                    EXPECTED_ACTION_BOUNDARY_ONSET,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    _number(
                        boundary_semantics.get("action_clip"),
                        "training action-boundary clip",
                    ),
                    1.0,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    _number(
                        boundary_semantics.get("weight"),
                        "training action-boundary reward weight",
                    ),
                    EXPECTED_ACTION_BOUNDARY_REWARD_WEIGHT,
                    abs_tol=1.0e-12,
                )
                or boundary_semantics.get("raw_action_preserved_for_acceptance")
                is not True
                or limiter_semantics.get("kind")
                != "per_second_clipped_request_minus_rate_limited_residual_l2"
                or not math.isclose(
                    _number(
                        limiter_semantics.get("action_clip"),
                        "training action-limiter clip",
                    ),
                    1.0,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    _number(
                        limiter_semantics.get("weight"),
                        "training action-limiter reward weight",
                    ),
                    EXPECTED_ACTION_LIMITER_GAP_REWARD_WEIGHT,
                    abs_tol=1.0e-12,
                )
            ):
                raise FinalArtifactError("training recovery reward semantics mismatch")
            ppo = _mapping(contract.get("ppo"), "training recovery PPO contract")
            if (
                not math.isclose(
                    _number(ppo.get("init_noise_std"), "initial action noise std"),
                    0.2,
                    abs_tol=1.0e-12,
                )
                or ppo.get("noise_std_type") != "log"
                or not math.isclose(
                    _number(ppo.get("entropy_coef"), "entropy coefficient"),
                    0.0,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    _number(ppo.get("learning_rate"), "PPO learning rate"),
                    EXPECTED_RECOVERY_PPO_LEARNING_RATE,
                    abs_tol=1.0e-12,
                )
                or ppo.get("schedule") != EXPECTED_RECOVERY_PPO_SCHEDULE
                or not math.isclose(
                    _number(ppo.get("desired_kl"), "PPO desired KL"),
                    EXPECTED_RECOVERY_PPO_DESIRED_KL,
                    abs_tol=1.0e-12,
                )
            ):
                raise FinalArtifactError("training recovery PPO contract mismatch")

    if suite or not recovery_training:
        if contract.get("policy_action_clip") is not None:
            raise FinalArtifactError(
                f"{label} policy wrapper must not clip actions before the action term"
            )
        if not math.isclose(
            _number(contract.get("physical_action_clip"), f"{label} physical action clip"),
            1.0,
            abs_tol=1.0e-12,
        ):
            raise FinalArtifactError(f"{label} physical action clip mismatch")

    identity = "\n".join(
        [
            f"contract {contract_digest}",
            "manifest -",
            "split -",
            *(
                f"{motion['checksum_sha256']} {motion['motion_key']}"
                for motion in authoritative_motions
            ),
        ]
    )
    expected_library_checksum = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    declared_library_checksum = _checksum(
        contract.get("library_checksum_sha256"), f"{label} library checksum"
    )
    if declared_library_checksum != expected_library_checksum:
        raise FinalArtifactError(
            f"{label} library checksum disagrees with the authoritative motion set"
        )
    mapping_payload = [
        [
            index,
            _relative(Path(motion["path"])),
            motion["checksum_sha256"],
            motion["motion_key"],
            motion["frame_count"],
        ]
        for index, motion in enumerate(authoritative_motions)
    ]
    mapping_checksum = hashlib.sha256(
        json.dumps(mapping_payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    verified_runtime = {
        **EXPECTED_RUNTIME,
        "observation_contract": observation_contract,
        "live_tensor_contract": live_tensor_contract,
        "physics_cold_start_warmup": physics_warmup,
        "motion_count": expected_count,
        "physics_dt_s": EXPECTED_PHYSICS_DT,
        "control_dt_s": EXPECTED_CONTROL_DT,
        "motion_contract_checksum_sha256": contract_digest,
        "library_checksum_sha256": declared_library_checksum,
        "verified_motion_mapping_checksum_sha256": mapping_checksum,
    }
    if evaluation_isolation is not None:
        verified_runtime["evaluation_isolation"] = evaluation_isolation
    return verified_runtime


def _independently_validate_exports(
    export_paths: Mapping[str, tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    """CPU-load and execute both final policy formats independently.

    This deliberately does not trust ``run_summary.json``'s structural
    validation record.  The ONNX reference implementation is used because
    onnxruntime is not a project dependency.
    """

    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - deployment environment guard
        raise FinalArtifactError(f"independent export validation dependency missing: {exc}") from exc

    input_array = np.zeros((1, EXPECTED_RUNTIME["actor_observation_dim"]), dtype=np.float32)
    jit_path = export_paths["torchscript"][0]
    try:
        module = torch.jit.load(str(jit_path), map_location="cpu")
        module.eval()
        with torch.inference_mode():
            jit_output = module(torch.from_numpy(input_array.copy()))
    except Exception as exc:
        raise FinalArtifactError(f"independent TorchScript load/inference failed: {exc}") from exc
    if not isinstance(jit_output, torch.Tensor):
        raise FinalArtifactError("independent TorchScript inference did not return a tensor")
    if list(jit_output.shape) != [1, EXPECTED_RUNTIME["action_dim"]]:
        raise FinalArtifactError(
            f"independent TorchScript output shape mismatch: {list(jit_output.shape)}"
        )
    if not bool(torch.isfinite(jit_output).all().item()):
        raise FinalArtifactError("independent TorchScript inference produced NaN or Inf")
    jit_array = jit_output.detach().cpu().numpy()

    try:
        import onnx
        from onnx.reference import ReferenceEvaluator
    except ImportError as exc:  # pragma: no cover - deployment environment guard
        raise FinalArtifactError(f"independent ONNX validation dependency missing: {exc}") from exc
    onnx_path = export_paths["onnx"][0]
    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise FinalArtifactError("ONNX policy must expose exactly one input and one output")
        input_info = model.graph.input[0]
        output_info = model.graph.output[0]
        input_shape = [
            int(dimension.dim_value)
            for dimension in input_info.type.tensor_type.shape.dim
        ]
        output_shape = [
            int(dimension.dim_value)
            for dimension in output_info.type.tensor_type.shape.dim
        ]
        expected_input = [1, EXPECTED_RUNTIME["actor_observation_dim"]]
        expected_output = [1, EXPECTED_RUNTIME["action_dim"]]
        if input_shape != expected_input or output_shape != expected_output:
            raise FinalArtifactError(
                "independent ONNX graph shape mismatch: "
                f"{input_shape} -> {output_shape}, expected {expected_input} -> {expected_output}"
            )
        outputs = ReferenceEvaluator(model).run(None, {input_info.name: input_array.copy()})
    except FinalArtifactError:
        raise
    except Exception as exc:
        raise FinalArtifactError(f"independent ONNX checker/reference inference failed: {exc}") from exc
    if len(outputs) != 1:
        raise FinalArtifactError(f"independent ONNX inference returned {len(outputs)} outputs")
    onnx_output = np.asarray(outputs[0])
    if list(onnx_output.shape) != [1, EXPECTED_RUNTIME["action_dim"]]:
        raise FinalArtifactError(
            f"independent ONNX inference output shape mismatch: {list(onnx_output.shape)}"
        )
    if not bool(np.isfinite(onnx_output).all()):
        raise FinalArtifactError("independent ONNX reference inference produced NaN or Inf")
    maximum_cross_format_difference = float(
        np.max(np.abs(jit_array.astype(np.float64) - onnx_output.astype(np.float64)))
    )
    if not bool(np.allclose(jit_array, onnx_output, rtol=1e-4, atol=1e-5)):
        raise FinalArtifactError(
            "TorchScript and ONNX zero-observation outputs disagree; "
            f"maximum absolute difference={maximum_cross_format_difference}"
        )

    return {
        "status": "passed",
        "device": "cpu",
        "input_contract": {
            "shape": [1, EXPECTED_RUNTIME["actor_observation_dim"]],
            "dtype": "float32",
            "fixture": "all-zero observation",
        },
        "output_contract": {
            "shape": [1, EXPECTED_RUNTIME["action_dim"]],
            "finite_required": True,
        },
        "cross_format_consistency": {
            "same_zero_observation": True,
            "rtol": 0.0001,
            "atol": 0.00001,
            "maximum_absolute_difference": maximum_cross_format_difference,
            "passed": True,
        },
        "torchscript": {
            "path": _relative(jit_path),
            "checksum_sha256": _sha256(jit_path),
            "loader": "torch.jit.load",
            "torch_version": str(torch.__version__),
            "output_shape": list(jit_array.shape),
            "output_dtype": str(jit_array.dtype),
            "finite_output": True,
        },
        "onnx": {
            "path": _relative(onnx_path),
            "checksum_sha256": _sha256(onnx_path),
            "checker_passed": True,
            "reference_evaluator_executed": True,
            "onnx_version": str(onnx.__version__),
            "input_name": input_info.name,
            "output_name": output_info.name,
            "output_shape": list(onnx_output.shape),
            "output_dtype": str(onnx_output.dtype),
            "finite_output": True,
        },
    }


def _verify_policy_initialization(
    report: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    artifacts: _Artifacts,
) -> dict[str, Any]:
    """Require model_599 policy-only initialization with untouched runner state."""

    witness = dict(
        _mapping(report.get("policy_initialization"), "training policy initialization")
    )
    runtime_witness = _mapping(
        runtime_contract.get("policy_initialization"),
        "training runtime policy initialization",
    )
    if not _same_json_value(witness, runtime_witness):
        raise FinalArtifactError(
            "training policy initialization differs from the runtime witness"
        )
    source = dict(_mapping(witness.get("source"), "policy initialization source"))
    initial_source = _mapping(
        report.get("policy_initialization_checkpoint"),
        "policy initialization checkpoint metadata",
    )
    if not _same_json_value(source, initial_source):
        raise FinalArtifactError(
            "policy initialization checkpoint metadata differs from the live witness"
        )
    source_path = _inside(
        source.get("path", ""), ACCAD_ROOT, "model_599 policy initialization checkpoint"
    )
    actual_checksum = _sha256(source_path)
    if (
        source_path.name != EXPECTED_POLICY_INITIALIZATION_FILENAME
        or source.get("iter") != EXPECTED_POLICY_INITIALIZATION_ITERATION
        or source.get("iteration") != EXPECTED_POLICY_INITIALIZATION_ITERATION
        or source.get("filename_iteration") != EXPECTED_POLICY_INITIALIZATION_ITERATION
        or _checksum(
            source.get("checksum_sha256"), "policy initialization source checksum"
        )
        != actual_checksum
        or _integer(
            source.get("bytes"), "policy initialization source size", minimum=1
        )
        != source_path.stat().st_size
        or _checksum(
            source.get("state_layout_checksum_sha256"),
            "policy initialization state-layout checksum",
        )
        == "0" * 64
        or _integer(
            source.get("state_key_count"),
            "policy initialization state-key count",
            minimum=1,
        )
        < 8
        or source.get("checkpoint_field_loaded") != "model_state_dict"
        or source.get("checkpoint_optimizer_state_present") is not True
    ):
        raise FinalArtifactError("model_599 policy initialization source contract mismatch")

    expected_normalizers = [
        "actor_obs_normalizer._mean",
        "actor_obs_normalizer._var",
        "actor_obs_normalizer._std",
        "actor_obs_normalizer.count",
        "critic_obs_normalizer._mean",
        "critic_obs_normalizer._var",
        "critic_obs_normalizer._std",
        "critic_obs_normalizer.count",
    ]
    load_contract = _mapping(
        witness.get("load_contract"), "policy initialization load contract"
    )
    if (
        witness.get("schema_version") != "accad_g1_policy_initialization_v1"
        or witness.get("mode") != "strict_model_state_dict_with_fresh_optimizer"
        or load_contract.get("torch_load_weights_only") is not True
        or load_contract.get("checkpoint_field_loaded") != "model_state_dict"
        or load_contract.get("strict") is not True
        or load_contract.get("optimizer_state_loaded") is not False
        or load_contract.get("normalizer_state_keys_loaded") != expected_normalizers
        or load_contract.get("missing_keys") != []
        or load_contract.get("unexpected_keys") != []
        or load_contract.get("all_loaded_tensors_exactly_equal") is not True
    ):
        raise FinalArtifactError("strict policy-only initialization load contract mismatch")

    fresh = _mapping(
        witness.get("fresh_runner_state"), "policy initialization fresh runner state"
    )
    before = dict(_mapping(fresh.get("before"), "runner state before initialization"))
    after = dict(_mapping(fresh.get("after"), "runner state after initialization"))
    if not _same_json_value(before, after):
        raise FinalArtifactError("runner or optimizer state changed during initialization")
    for phase, snapshot in (("before", before), ("after", after)):
        if (
            snapshot.get("current_learning_iteration") != 0
            or snapshot.get("tot_timesteps") != 0
            or snapshot.get("optimizer_state_entry_count") != 0
            or _integer(
                snapshot.get("optimizer_param_group_count"),
                f"optimizer param-group count {phase}",
                minimum=1,
            )
            != len(
                _sequence(
                    snapshot.get("optimizer_param_group_parameter_counts"),
                    f"optimizer parameter counts {phase}",
                )
            )
            or len(
                _sequence(
                    snapshot.get("optimizer_param_group_learning_rates"),
                    f"optimizer learning rates {phase}",
                )
            )
            != snapshot.get("optimizer_param_group_count")
        ):
            raise FinalArtifactError(
                f"policy initialization did not preserve fresh optimizer/counters {phase}"
            )
        if "tot_time" in snapshot and not math.isclose(
            _number(snapshot["tot_time"], f"runner total time {phase}"),
            0.0,
            abs_tol=1.0e-12,
        ):
            raise FinalArtifactError(f"runner total time was nonzero {phase} initialization")
    if (
        fresh.get("policy_object_preserved") is not True
        or fresh.get("optimizer_object_preserved") is not True
    ):
        raise FinalArtifactError("policy/optimizer object identity was not preserved")

    artifacts.add(
        source_path,
        "policy_initialization",
        "model_599 strict policy-only initialization source",
    )
    return {
        "schema_version": witness["schema_version"],
        "mode": witness["mode"],
        "source": {
            **source,
            "path": _relative(source_path),
            "checksum_sha256": actual_checksum,
        },
        "optimizer_state_loaded": False,
        "optimizer_state_entry_count_before": 0,
        "optimizer_state_entry_count_after": 0,
        "runner_current_learning_iteration_before": 0,
        "runner_current_learning_iteration_after": 0,
        "runner_tot_timesteps_before": 0,
        "runner_tot_timesteps_after": 0,
        "all_loaded_tensors_exactly_equal": True,
    }


def _verify_training(
    run_value: str | Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    artifacts: _Artifacts,
) -> dict[str, Any]:
    candidate = Path(run_value).expanduser().resolve()
    if candidate.is_dir():
        run_dir = _inside(candidate, ACCAD_ROOT, "training run", kind="directory")
        summary_path = _inside(run_dir / "run_summary.json", ACCAD_ROOT, "training summary")
    else:
        summary_path = _inside(candidate, ACCAD_ROOT, "training summary")
        if summary_path.name != "run_summary.json":
            raise FinalArtifactError("--training-run file must be named run_summary.json")
        run_dir = summary_path.parent
    required_runs_root = (WORK_ROOT / "training" / "runs").resolve()
    try:
        run_dir.relative_to(required_runs_root)
    except ValueError as exc:
        raise FinalArtifactError(f"training run must stay below {required_runs_root}") from exc

    report = _read_json(summary_path, "training summary")
    _require_schema_status(report, TRAINING_SCHEMA, "training summary")
    if (
        _integer(
            report.get("physics_warmup_steps_requested"),
            "training requested physics warmup steps",
        )
        != EXPECTED_PHYSICS_WARMUP_STEPS
    ):
        raise FinalArtifactError("training did not request the production physics warmup")
    if "resume_checkpoint" not in report or report["resume_checkpoint"] is not None:
        raise FinalArtifactError(
            "final recovery training must be a fresh run with policy-only initialization, "
            "not resume semantics"
        )
    if _inside(report.get("run_dir", ""), ACCAD_ROOT, "declared training run", kind="directory") != run_dir:
        raise FinalArtifactError("training summary run_dir does not match --training-run")
    selection, authoritative_motions = _verify_manifest_selection(
        report,
        split="train",
        manifest_path=manifest_path,
        manifest=manifest,
        artifacts=artifacts,
    )
    correspondence = _verify_correspondence(report, "training")
    runtime = _verify_runtime(
        report.get("runtime_contract"),
        "training",
        len(authoritative_motions),
        suite=False,
        authoritative_motions=authoritative_motions,
        expected_num_envs=EXPECTED_TRAINING_NUM_ENVS,
        recovery_training=True,
    )
    runtime_sidecar_path = _inside(
        run_dir / "runtime_contract.json",
        ACCAD_ROOT,
        "training runtime contract",
    )
    runtime_sidecar = _read_json(runtime_sidecar_path, "training runtime contract")
    if runtime_sidecar != report.get("runtime_contract"):
        raise FinalArtifactError(
            "training runtime sidecar differs from the embedded runtime contract"
        )

    raw_runtime_contract = _mapping(
        report.get("runtime_contract"), "training raw runtime contract"
    )
    if raw_runtime_contract.get("resume_checkpoint") is not None:
        raise FinalArtifactError("training runtime unexpectedly used resume semantics")
    policy_initialization = _verify_policy_initialization(
        report, raw_runtime_contract, artifacts
    )
    assignment = dict(
        _mapping(
            raw_runtime_contract.get("motion_assignment"),
            "training runtime motion assignment",
        )
    )
    if report.get("motion_assignment") != assignment:
        raise FinalArtifactError(
            "top-level training motion assignment differs from the runtime contract"
        )
    live_mapping_witness = assignment.pop("live_mapping_witness", None)
    expected_assignment = training_motion_assignment_contract(
        EXPECTED_TRAINING_MOTION_ASSIGNMENT,
        num_envs=EXPECTED_TRAINING_NUM_ENVS,
        motion_keys=[str(motion["motion_key"]) for motion in authoritative_motions],
        num_steps_per_env=EXPECTED_TRAINING_STEPS_PER_ENV,
        learning_updates=EXPECTED_TRAINING_UPDATES,
    )
    if assignment != expected_assignment:
        raise FinalArtifactError(
            "training motion assignment differs from the deterministic "
            "balanced-fixed production contract"
        )
    expected_live_mapping_witness = {
        "verified": True,
        "command_cfg_fixed_motion_ids_match": True,
        "live_motion_ids_match": True,
        "resolved_live_motion_id_count": EXPECTED_TRAINING_NUM_ENVS,
    }
    if live_mapping_witness != expected_live_mapping_witness:
        raise FinalArtifactError(
            "training motion assignment lacks the exact live-command mapping witness"
        )
    verified_motion_assignment = {
        **expected_assignment,
        "live_mapping_witness": expected_live_mapping_witness,
    }

    training = _mapping(report.get("training"), "training counters")
    num_envs = _integer(training.get("num_envs"), "training num_envs", minimum=1)
    num_steps_per_env = _integer(
        training.get("num_steps_per_env"), "training steps per env", minimum=1
    )
    requested = _integer(training.get("learning_updates_requested"), "requested updates", minimum=1)
    completed = _integer(training.get("learning_updates_completed"), "completed updates", minimum=1)
    runner_iteration = _integer(
        training.get("runner_current_learning_iteration"),
        "runner current learning iteration",
    )
    total_timesteps = _integer(
        training.get("total_timesteps"), "total timesteps", minimum=1
    )
    if (
        num_envs != EXPECTED_TRAINING_NUM_ENVS
        or num_steps_per_env != EXPECTED_TRAINING_STEPS_PER_ENV
        or requested != EXPECTED_TRAINING_UPDATES
        or completed != EXPECTED_TRAINING_UPDATES
        or runner_iteration != EXPECTED_FINAL_CHECKPOINT_ITERATION
        or total_timesteps != EXPECTED_TRAINING_TIMESTEPS
    ):
        raise FinalArtifactError(
            "training capacity contract mismatch; expected "
            f"{EXPECTED_TRAINING_NUM_ENVS} envs x {EXPECTED_TRAINING_STEPS_PER_ENV} steps "
            f"x {EXPECTED_TRAINING_UPDATES} updates = {EXPECTED_TRAINING_TIMESTEPS} transitions "
            f"and runner iteration {EXPECTED_FINAL_CHECKPOINT_ITERATION}"
        )

    ppo_execution = _mapping(training.get("ppo_execution"), "executed PPO state")
    executed_algorithm_lr = _number(
        ppo_execution.get("algorithm_learning_rate_final"),
        "executed final algorithm learning rate",
    )
    executed_optimizer_lrs = _sequence(
        ppo_execution.get("optimizer_param_group_learning_rates_final"),
        "executed final optimizer learning rates",
    )
    if (
        not math.isclose(
            executed_algorithm_lr,
            EXPECTED_RECOVERY_PPO_LEARNING_RATE,
            abs_tol=1.0e-12,
        )
        or not executed_optimizer_lrs
        or any(
            not math.isclose(
                _number(value, "executed optimizer learning rate"),
                EXPECTED_RECOVERY_PPO_LEARNING_RATE,
                abs_tol=1.0e-12,
            )
            for value in executed_optimizer_lrs
        )
    ):
        raise FinalArtifactError(
            "executed PPO learning rate differs from the recovery contract"
        )

    sampling = _mapping(report.get("motion_sampling"), "training motion sampling")
    if (
        sampling.get("motion_assignment_mode")
        != EXPECTED_TRAINING_MOTION_ASSIGNMENT
        or sampling.get("sampler")
        != "fixed_environment_motion_then_frame_zero_or_uniform_valid_start_time"
    ):
        raise FinalArtifactError(
            "training motion sampling does not use balanced-fixed assignment"
        )
    if sampling.get("zero_sampled_motion_count") != 0 or sampling.get("zero_sampled_motion_keys") != []:
        raise FinalArtifactError("one or more training motions were never sampled")
    if _integer(sampling.get("minimum_samples_per_motion"), "minimum samples per motion", minimum=1) < 1:
        raise FinalArtifactError("training motion coverage is incomplete")
    per_motion = _sequence(sampling.get("per_motion"), "per-motion sampling")
    dynamic_train_count = len(authoritative_motions)
    if len(per_motion) != dynamic_train_count:
        raise FinalArtifactError(
            f"per-motion sampling does not cover all {dynamic_train_count} dynamic train motions"
        )
    sample_counts: list[int] = []
    for index, (value, authoritative) in enumerate(
        zip(per_motion, authoritative_motions, strict=True)
    ):
        motion = _mapping(value, f"training per-motion sampling {index}")
        if motion.get("motion_id") != index:
            raise FinalArtifactError(f"training sampling motion_id {index} mismatch")
        if motion.get("motion_key") != authoritative["motion_key"]:
            raise FinalArtifactError(f"training sampling motion_key {index} mismatch")
        sample_counts.append(
            _integer(motion.get("samples"), f"training samples for motion {index}", minimum=1)
        )
    recomputed_total = sum(sample_counts)
    recomputed_minimum = min(sample_counts)
    recomputed_maximum = max(sample_counts)
    if (
        _integer(
            sampling.get("total_episode_samples"),
            "total training episode samples",
            minimum=dynamic_train_count,
        )
        != recomputed_total
        or sampling.get("minimum_samples_per_motion") != recomputed_minimum
        or sampling.get("maximum_samples_per_motion") != recomputed_maximum
    ):
        raise FinalArtifactError("training motion sampling aggregates are inconsistent")
    start_time = _mapping(sampling.get("start_time"), "training start-time sampling")
    if not math.isclose(
        _number(
            start_time.get("frame_zero_probability"),
            "training frame-zero sampling probability",
        ),
        0.5,
        abs_tol=1.0e-12,
    ):
        raise FinalArtifactError("training start-time mixture probability mismatch")
    frame_zero_samples = _integer(
        start_time.get("frame_zero_samples"), "training frame-zero samples", minimum=1
    )
    continuous_random_samples = _integer(
        start_time.get("continuous_random_samples"),
        "training continuous-random samples",
        minimum=1,
    )
    fixed_samples = _integer(
        start_time.get("fixed_samples"), "training fixed-start samples"
    )
    declared_start_total = _integer(
        start_time.get("total_samples"), "training start-time sample total", minimum=1
    )
    if (
        fixed_samples != EXPECTED_TRAINING_NUM_ENVS
        or frame_zero_samples + continuous_random_samples + fixed_samples
        != declared_start_total
        or declared_start_total != recomputed_total
    ):
        raise FinalArtifactError("training start-time sampling counters are inconsistent")

    checkpoints = _sequence(report.get("checkpoints"), "training checkpoints")
    if not checkpoints:
        raise FinalArtifactError("training summary has no checkpoint")
    checkpoint_records: list[tuple[int, Path, dict[str, Any]]] = []
    for index, record in enumerate(checkpoints):
        path, actual = _verify_declared_artifact(
            record, f"training checkpoint {index}", artifacts, "checkpoints", parent=run_dir
        )
        try:
            iteration = int(path.stem.rsplit("_", 1)[-1])
        except ValueError as exc:
            raise FinalArtifactError(f"invalid checkpoint filename: {path.name}") from exc
        checkpoint_records.append((iteration, path, actual))
    checkpoint_records.sort(key=lambda item: item[0])
    if len({item[0] for item in checkpoint_records}) != len(checkpoint_records):
        raise FinalArtifactError("duplicate checkpoint iteration")
    observed_checkpoint_iterations = tuple(item[0] for item in checkpoint_records)
    if observed_checkpoint_iterations != EXPECTED_CHECKPOINT_ITERATIONS:
        raise FinalArtifactError(
            "training checkpoint schedule mismatch; expected iterations "
            f"{EXPECTED_CHECKPOINT_ITERATIONS}, got {observed_checkpoint_iterations}"
        )
    final_iteration, final_checkpoint, final_record = checkpoint_records[-1]
    if (
        training.get("final_checkpoint_iteration_index") != final_iteration
        or final_iteration != EXPECTED_FINAL_CHECKPOINT_ITERATION
    ):
        raise FinalArtifactError("final checkpoint iteration does not match training summary")

    exports = _sequence(report.get("exports"), "training exports")
    export_paths: dict[str, tuple[Path, dict[str, Any]]] = {}
    for index, record in enumerate(exports):
        path, actual = _verify_declared_artifact(
            record, f"training export {index}", artifacts, "exports", parent=run_dir
        )
        format_name = "onnx" if path.suffix.lower() == ".onnx" else "torchscript" if path.suffix.lower() == ".pt" else ""
        if not format_name or format_name in export_paths:
            raise FinalArtifactError("training must contain one TorchScript and one ONNX export")
        export_paths[format_name] = (path, actual)
    if set(export_paths) != {"torchscript", "onnx"}:
        raise FinalArtifactError("training must contain both TorchScript and ONNX exports")
    validations: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(
        _sequence(report.get("export_validations"), "export validations")
    ):
        validation = _mapping(value, f"export validation {index}")
        format_name = str(validation.get("format"))
        if format_name in validations:
            raise FinalArtifactError(f"duplicate {format_name} export validation")
        validations[format_name] = validation
    if set(validations) != {"torchscript", "onnx"}:
        raise FinalArtifactError("training must contain exactly one validation per export format")
    torchscript = validations.get("torchscript")
    onnx = validations.get("onnx")
    if (
        torchscript is None
        or torchscript.get("finite_output") is not True
        or torchscript.get("input_shape")
        != [1, EXPECTED_RUNTIME["actor_observation_dim"]]
        or torchscript.get("output_shape") != [1, EXPECTED_RUNTIME["action_dim"]]
        or onnx is None
        or onnx.get("checker_passed") is not True
        or onnx.get("input_shape")
        != [1, EXPECTED_RUNTIME["actor_observation_dim"]]
        or onnx.get("output_shape") != [1, EXPECTED_RUNTIME["action_dim"]]
    ):
        raise FinalArtifactError(
            "training export validation did not pass the "
            f"{EXPECTED_RUNTIME['actor_observation_dim']}-to-"
            f"{EXPECTED_RUNTIME['action_dim']} contract"
        )
    for format_name, validation in (("torchscript", torchscript), ("onnx", onnx)):
        validation_path = _inside(validation.get("path", ""), ACCAD_ROOT, f"{format_name} validation path")
        if validation_path != export_paths[format_name][0]:
            raise FinalArtifactError(f"{format_name} validation names a different export")
    independent_export_validation = _independently_validate_exports(export_paths)
    # Detect a file replacement between the first checksum and executable load.
    for format_name, (path, _record) in export_paths.items():
        artifacts.add(path, "exports", f"independently executed {format_name} policy export")

    artifacts.add(summary_path, "reports", "final training summary")
    for relative, role in (
        ("runtime_contract.json", "training runtime contract"),
        ("params/env.yaml", "training environment config"),
        ("params/agent.yaml", "training PPO config"),
    ):
        artifacts.add(_inside(run_dir / relative, ACCAD_ROOT, role), "training_configuration", role)
    for event in sorted(run_dir.glob("events.out.tfevents.*")):
        artifacts.add(_inside(event, ACCAD_ROOT, "TensorBoard event"), "training_logs", "TensorBoard event")
    if not list(run_dir.glob("events.out.tfevents.*")):
        raise FinalArtifactError("training run has no TensorBoard event file")
    git_diff = run_dir / "git" / "IsaacLab.diff"
    if git_diff.is_file():
        artifacts.add(_inside(git_diff, ACCAD_ROOT, "training git diff"), "training_logs", "training git diff")

    return {
        "complete": True,
        "status": "completed",
        "run_directory": _relative(run_dir),
        "summary": _artifact(summary_path, role="final training summary"),
        "motion_selection": selection,
        "correspondence": correspondence,
        "runtime_contract": runtime,
        "policy_initialization": policy_initialization,
        "motion_assignment": verified_motion_assignment,
        "learning_updates_requested": requested,
        "learning_updates_completed": completed,
        "num_envs": num_envs,
        "num_steps_per_env": num_steps_per_env,
        "runner_current_learning_iteration": runner_iteration,
        "total_timesteps": total_timesteps,
        "motion_sampling": {
            "total_episode_samples": sampling.get("total_episode_samples"),
            "minimum_samples_per_motion": sampling.get("minimum_samples_per_motion"),
            "maximum_samples_per_motion": sampling.get("maximum_samples_per_motion"),
            "zero_sampled_motion_count": 0,
            "start_time": dict(start_time),
        },
        "final_checkpoint": final_record,
        "exports": {name: value[1] for name, value in sorted(export_paths.items())},
        "independent_export_validation": independent_export_validation,
    }


def _verify_predefined_acceptance(
    report: Mapping[str, Any], aggregate: Mapping[str, Any]
) -> dict[str, Any]:
    motion_input = _mapping(report.get("motion_input"), "suite motion_input")
    if motion_input.get("manifest_promotion_verified") is not True:
        raise FinalArtifactError("suite did not record checksum-verified Gate-3 promotion")
    source = dict(_mapping(report.get("acceptance"), "suite predefined acceptance"))
    expected = build_suite_acceptance(
        aggregate,
        manifest_promotion_verified=True,
    )
    if source != expected:
        raise FinalArtifactError(
            "suite acceptance block differs from an independent recomputation of the "
            "versioned predefined thresholds"
        )
    if report.get("passed") is not expected["passed"]:
        raise FinalArtifactError("suite passed flag disagrees with predefined acceptance")
    return {
        "policy": dict(DYNAMIC_ACCEPTANCE_POLICY),
        "thresholds": expected["thresholds"],
        "checks": expected["checks"],
        "verdict": expected["passed"],
        "independently_recomputed": True,
        "source_schema_version": source["schema_version"],
    }


def _verify_suite(
    report_value: str | Path,
    *,
    split: str,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    final_checkpoint: Mapping[str, Any],
    training_runtime: Mapping[str, Any],
    artifacts: _Artifacts,
) -> dict[str, Any]:
    report_path = _inside(report_value, ACCAD_ROOT, f"{split} suite report")
    report = _read_json(report_path, f"{split} suite report")
    _require_schema_status(
        report, SUITE_SCHEMA, f"{split} suite report", require_passed=True
    )
    if report_path.name != "suite_evaluation_summary.json":
        raise FinalArtifactError(f"{split} report must be named suite_evaluation_summary.json")
    selection, authoritative_motions = _verify_manifest_selection(
        report,
        split=split,
        manifest_path=manifest_path,
        manifest=manifest,
        artifacts=artifacts,
    )
    correspondence = _verify_correspondence(report, f"{split} suite")
    runtime = _verify_runtime(
        report.get("runtime_contract"),
        f"{split} suite",
        len(authoritative_motions),
        suite=True,
        authoritative_motions=authoritative_motions,
        expected_num_envs=len(authoritative_motions),
    )
    if (
        runtime["motion_contract_checksum_sha256"]
        != training_runtime["motion_contract_checksum_sha256"]
    ):
        raise FinalArtifactError(
            f"{split} suite motion contract differs from the training runtime"
        )
    for key, description in (
        ("observation_contract", "observation contract"),
        ("live_tensor_contract", "live tensor contract"),
    ):
        if not _same_json_value(runtime[key], training_runtime.get(key)):
            raise FinalArtifactError(
                f"{split} suite {description} differs from the training runtime"
            )

    checkpoint_path, checkpoint_actual = _verify_declared_artifact(
        report.get("checkpoint"), f"{split} checkpoint", artifacts, "checkpoints"
    )
    if (
        checkpoint_actual["checksum_sha256"] != final_checkpoint["checksum_sha256"]
        or _relative(checkpoint_path) != final_checkpoint["path"]
    ):
        raise FinalArtifactError(f"{split} suite did not use the final training checkpoint")

    contract = _mapping(report.get("evaluation_contract"), f"{split} evaluation contract")
    expected_contract = {
        "one_environment_per_selected_motion": True,
        "first_episode_only": True,
        "deterministic_policy_inference": True,
        "random_start": False,
        "state_and_observation_noise": False,
        "domain_randomization_events": False,
        "physx_enhanced_determinism": True,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) is not expected:
            raise FinalArtifactError(f"{split} evaluation contract {key} mismatch")
    if (
        _integer(
            contract.get("physics_warmup_steps_requested"),
            f"{split} requested physics warmup steps",
        )
        != EXPECTED_PHYSICS_WARMUP_STEPS
    ):
        raise FinalArtifactError(
            f"{split} evaluation did not request the production physics warmup"
        )
    if not isinstance(contract.get("success_definition"), str) or not contract["success_definition"]:
        raise FinalArtifactError(f"{split} evaluation has no success definition")

    motions = _sequence(report.get("motions"), f"{split} suite motions")
    expected_count = len(authoritative_motions)
    if len(motions) != expected_count:
        raise FinalArtifactError(f"{split} suite did not record every motion")
    aggregate = _mapping(report.get("aggregate"), f"{split} aggregate")
    if aggregate.get("motion_count") != expected_count:
        raise FinalArtifactError(f"{split} aggregate motion_count mismatch")
    if aggregate.get("finished_first_episodes") != expected_count:
        raise FinalArtifactError(f"{split} suite has unfinished first episodes")
    if aggregate.get("simulation_app_stopped_early") is not False:
        raise FinalArtifactError(f"{split} simulation stopped before suite completion")
    success_count = sum(
        _mapping(_mapping(motion, f"{split} motion").get("first_episode"), "first episode").get("success") is True
        for motion in motions
    )
    if aggregate.get("successful_motion_end_episodes") != success_count:
        raise FinalArtifactError(f"{split} aggregate success count disagrees with motion records")
    if success_count != expected_count:
        raise FinalArtifactError(
            f"{split} suite must cleanly succeed on every dynamic promoted motion"
        )
    expected_rate = success_count / expected_count
    if not math.isclose(_number(aggregate.get("success_rate"), f"{split} success rate"), expected_rate, abs_tol=1e-12):
        raise FinalArtifactError(f"{split} aggregate success rate is inconsistent")
    failed_motion_ids: list[Any] = []
    observed_termination_counts: dict[str, int] = {}
    for index, (motion, authoritative) in enumerate(
        zip(motions, authoritative_motions, strict=True)
    ):
        motion_record = _mapping(motion, f"{split} motion {index}")
        if (
            motion_record.get("environment_id") != index
            or motion_record.get("motion_id") != index
            or motion_record.get("motion_key") != authoritative["motion_key"]
        ):
            raise FinalArtifactError(f"{split} motion {index} identity mapping mismatch")
        motion_path = _inside(
            motion_record.get("path", ""), ACCAD_ROOT, f"{split} motion {index} path"
        )
        if motion_path != authoritative["path"]:
            raise FinalArtifactError(f"{split} motion {index} path mismatch")
        if (
            _checksum(
                motion_record.get("checksum_sha256"),
                f"{split} motion {index} checksum",
            )
            != authoritative["checksum_sha256"]
            or motion_record.get("frame_count") != authoritative["frame_count"]
        ):
            raise FinalArtifactError(f"{split} motion {index} archive metadata mismatch")
        if not math.isclose(
            _number(motion_record.get("duration_s"), f"{split} motion {index} duration"),
            float(authoritative["duration_s"]),
            abs_tol=RUNTIME_DURATION_ABS_TOL_S,
        ):
            raise FinalArtifactError(f"{split} motion {index} duration mismatch")
        first = _mapping(motion_record.get("first_episode"), "first episode")
        if first.get("finished") is not True:
            raise FinalArtifactError(f"{split} motion {index} first episode is unfinished")
        if first.get("success") is not True:
            failed_motion_ids.append(motion_record.get("motion_id"))
        flags = _mapping(first.get("termination_flags"), f"{split} termination flags {index}")
        for name, fired in flags.items():
            if not isinstance(fired, bool):
                raise FinalArtifactError(f"{split} termination flag {name!r} is not boolean")
            observed_termination_counts[name] = observed_termination_counts.get(name, 0) + int(fired)
    declared_termination_counts = _mapping(
        aggregate.get("termination_counts"), f"{split} termination counts"
    )
    for name in set(observed_termination_counts) | set(declared_termination_counts):
        declared = _integer(
            declared_termination_counts.get(name, 0),
            f"{split} termination count {name}",
        )
        if declared != observed_termination_counts.get(name, 0):
            raise FinalArtifactError(f"{split} termination count {name!r} is inconsistent")
    if aggregate.get("failed_or_incomplete_motion_ids") != failed_motion_ids:
        raise FinalArtifactError(f"{split} failed motion IDs disagree with motion records")
    acceptance = _verify_predefined_acceptance(report, aggregate)
    if acceptance["verdict"] is not True:
        raise FinalArtifactError(
            f"{split} suite did not pass the predefined dynamic acceptance policy"
        )

    artifacts.add(report_path, "reports", f"{split} deterministic suite report")
    evaluation_dir = report_path.parent
    required_evaluations_root = (WORK_ROOT / "training" / "evaluations").resolve()
    try:
        evaluation_dir.relative_to(required_evaluations_root)
    except ValueError as exc:
        raise FinalArtifactError(
            f"{split} evaluation must stay below {required_evaluations_root}"
        ) from exc
    declared_dir = _inside(report.get("evaluation_dir", ""), ACCAD_ROOT, f"{split} evaluation directory", kind="directory")
    if declared_dir != evaluation_dir:
        raise FinalArtifactError(f"{split} evaluation_dir does not match report location")
    runtime_sidecar_path = _inside(
        evaluation_dir / "runtime_contract.json",
        ACCAD_ROOT,
        f"{split} runtime contract",
    )
    runtime_sidecar = _read_json(runtime_sidecar_path, f"{split} runtime contract")
    if runtime_sidecar != report.get("runtime_contract"):
        raise FinalArtifactError(
            f"{split} runtime sidecar differs from the embedded runtime contract"
        )
    for relative, role in (
        ("runtime_contract.json", f"{split} runtime contract"),
        ("params/env.yaml", f"{split} environment config"),
        ("params/agent.yaml", f"{split} agent config"),
    ):
        artifacts.add(_inside(evaluation_dir / relative, ACCAD_ROOT, role), "evaluation_configuration", role)
    for index, value in enumerate(_sequence(report.get("exports"), f"{split} exports")):
        _verify_declared_artifact(value, f"{split} export {index}", artifacts, "exports", parent=evaluation_dir)
    for index, value in enumerate(_sequence(report.get("videos", []), f"{split} videos")):
        _verify_declared_artifact(value, f"{split} video {index}", artifacts, "videos", parent=evaluation_dir)

    measured = {
        "source_report_predefined_acceptance_passed": report.get("passed"),
        "all_motions_clean_motion_end_success": success_count == expected_count,
        "motion_count": expected_count,
        "finished_first_episodes": aggregate["finished_first_episodes"],
        "successful_motion_end_episodes": success_count,
        "success_rate": expected_rate,
        "failed_or_incomplete_motion_ids": aggregate.get("failed_or_incomplete_motion_ids"),
        "termination_counts": aggregate.get("termination_counts"),
        "return_mean": aggregate.get("return_mean"),
        "completion_ratio_mean": aggregate.get("completion_ratio_mean"),
        "macro_motion_metrics": aggregate.get("macro_motion_metrics"),
        "weighted_frame_or_element_metrics": aggregate.get("weighted_frame_or_element_metrics"),
        "aggregate_foot_contact": aggregate.get("aggregate_foot_contact"),
        "aggregate_stance_or_actual_contact_slide_speed_p95_mps": aggregate.get(
            "aggregate_stance_or_actual_contact_slide_speed_p95_mps"
        ),
    }
    return {
        "complete": True,
        "status": "completed",
        "split": split,
        "report": _artifact(report_path, role=f"{split} deterministic suite report"),
        "motion_selection": selection,
        "correspondence": correspondence,
        "runtime_contract": runtime,
        "checkpoint": {**checkpoint_actual, "path": _relative(checkpoint_path)},
        "measured": measured,
        "predefined_acceptance": acceptance,
    }


def _verify_representative_evaluation(
    report_value: str | Path,
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    final_checkpoint: Mapping[str, Any],
    training_runtime: Mapping[str, Any],
    artifacts: _Artifacts,
) -> dict[str, Any]:
    """Verify one representative selected from the built dynamic test set."""

    label = "representative held-out test"
    report_path = _inside(report_value, ACCAD_ROOT, f"{label} report")
    report = _read_json(report_path, f"{label} report")
    _require_schema_status(report, PLAY_SCHEMA, f"{label} report")
    if (
        _integer(
            report.get("physics_warmup_steps_requested"),
            "representative requested physics warmup steps",
        )
        != EXPECTED_PHYSICS_WARMUP_STEPS
    ):
        raise FinalArtifactError(
            "representative evaluation did not request the production physics warmup"
        )
    if report_path.name != "evaluation_summary.json":
        raise FinalArtifactError(
            "representative report must be named evaluation_summary.json"
        )
    selection, authoritative_motions = _verify_manifest_selection(
        report,
        split="test",
        manifest_path=manifest_path,
        manifest=manifest,
        artifacts=artifacts,
    )
    correspondence = _verify_correspondence(report, label)
    runtime_source = _mapping(report.get("runtime_contract"), f"{label} runtime")
    runtime = _verify_runtime(
        runtime_source,
        label,
        len(authoritative_motions),
        suite=False,
        authoritative_motions=authoritative_motions,
        expected_num_envs=1,
    )
    runtime["evaluation_isolation"] = _verify_evaluation_isolation(
        runtime_source.get("evaluation_isolation"),
        label,
    )
    if runtime_source.get("video_capture_started_after_physics_warmup") is not True:
        raise FinalArtifactError(
            "representative video capture did not start after physics warmup"
        )
    if (
        runtime["motion_contract_checksum_sha256"]
        != training_runtime["motion_contract_checksum_sha256"]
    ):
        raise FinalArtifactError(
            "representative evaluation motion contract differs from training"
        )
    for key, description in (
        ("observation_contract", "observation contract"),
        ("live_tensor_contract", "live tensor contract"),
    ):
        if not _same_json_value(runtime[key], training_runtime.get(key)):
            raise FinalArtifactError(
                f"representative evaluation {description} differs from training"
            )

    checkpoint_path, checkpoint_actual = _verify_declared_artifact(
        report.get("checkpoint"),
        "representative checkpoint",
        artifacts,
        "checkpoints",
    )
    if (
        checkpoint_actual["checksum_sha256"] != final_checkpoint["checksum_sha256"]
        or _relative(checkpoint_path) != final_checkpoint["path"]
    ):
        raise FinalArtifactError(
            "representative evaluation did not use the final training checkpoint"
        )

    motion_id = _integer(
        report.get("selected_motion_id"), "representative selected motion ID"
    )
    if motion_id >= len(authoritative_motions):
        raise FinalArtifactError(
            "representative motion ID is outside the dynamic test eligible set"
        )
    authoritative = authoritative_motions[motion_id]
    if runtime_source.get("selected_motion_id") != motion_id:
        raise FinalArtifactError("representative evaluation selected the wrong motion ID")
    if (
        "selected_motion_key" in report
        and report.get("selected_motion_key") != authoritative["motion_key"]
    ) or (
        "selected_motion_key" in runtime_source
        and runtime_source.get("selected_motion_key") != authoritative["motion_key"]
    ):
        raise FinalArtifactError(
            "representative selected key differs from the dynamic test manifest"
        )
    selected_path = _inside(
        report.get("selected_motion_path", ""),
        ACCAD_ROOT,
        "representative selected motion path",
    )
    if selected_path != authoritative["path"]:
        raise FinalArtifactError("representative evaluation selected the wrong motion path")

    evaluation = _mapping(report.get("evaluation"), "representative evaluation metrics")
    expected_frames = int(authoritative["frame_count"])
    if evaluation.get("expected_frame_count") != expected_frames:
        raise FinalArtifactError("representative expected frame count mismatch")
    steps_executed = _integer(
        evaluation.get("steps_executed"), "representative executed steps", minimum=1
    )
    if steps_executed != expected_frames:
        raise FinalArtifactError(
            "representative evaluation must record the fixed full-length step window"
        )
    first_lengths = _sequence(
        evaluation.get("first_episode_lengths_by_environment"),
        "representative first-episode lengths",
    )
    if len(first_lengths) != 1:
        raise FinalArtifactError(
            "representative evaluation must use exactly one environment"
        )
    first_length = _integer(
        evaluation.get("first_episode_length"),
        "representative first-episode length",
        minimum=1,
    )
    if _integer(first_lengths[0], "representative environment episode length", minimum=1) != first_length:
        raise FinalArtifactError("representative first-episode lengths disagree")
    if first_length > steps_executed:
        raise FinalArtifactError(
            "representative first episode is longer than the executed window"
        )
    completed_episodes = _integer(
        evaluation.get("completed_episodes"),
        "representative completed episode count",
        minimum=1,
    )
    expected_ratio = min(first_length / expected_frames, 1.0)
    if not math.isclose(
        _number(
            evaluation.get("first_episode_completion_ratio"),
            "representative completion ratio",
        ),
        expected_ratio,
        abs_tol=1.0e-12,
    ):
        raise FinalArtifactError("representative completion ratio is inconsistent")
    if evaluation.get("all_first_episodes_completed") is not True:
        raise FinalArtifactError("representative first episode did not finish")
    if not isinstance(evaluation.get("uninterrupted_full_clip_success"), bool):
        raise FinalArtifactError(
            "representative uninterrupted_full_clip_success must be boolean"
        )
    success_count = _integer(
        evaluation.get("first_episode_success_count"),
        "representative first-episode success count",
    )
    if success_count not in {0, 1}:
        raise FinalArtifactError("representative success count must be zero or one")
    uninterrupted = evaluation["uninterrupted_full_clip_success"]
    expected_uninterrupted = bool(
        evaluation["all_first_episodes_completed"] and success_count == 1
    )
    if uninterrupted is not expected_uninterrupted:
        raise FinalArtifactError("representative uninterrupted-success flag is inconsistent")

    evaluation_dir = report_path.parent
    required_root = (WORK_ROOT / "training" / "evaluations").resolve()
    try:
        evaluation_dir.relative_to(required_root)
    except ValueError as exc:
        raise FinalArtifactError(
            f"representative evaluation must stay below {required_root}"
        ) from exc
    declared_dir = _inside(
        report.get("evaluation_dir", ""),
        ACCAD_ROOT,
        "representative evaluation directory",
        kind="directory",
    )
    if declared_dir != evaluation_dir:
        raise FinalArtifactError("representative evaluation_dir does not match report")
    runtime_sidecar_path = _inside(
        evaluation_dir / "runtime_contract.json",
        ACCAD_ROOT,
        "representative runtime contract",
    )
    runtime_sidecar = _read_json(
        runtime_sidecar_path, "representative runtime contract"
    )
    if runtime_sidecar != dict(runtime_source):
        raise FinalArtifactError(
            "representative runtime sidecar differs from the embedded contract"
        )
    for relative, role in (
        ("runtime_contract.json", "representative runtime contract"),
        ("params/env.yaml", "representative environment config"),
        ("params/agent.yaml", "representative agent config"),
    ):
        artifacts.add(
            _inside(evaluation_dir / relative, ACCAD_ROOT, role),
            "evaluation_configuration",
            role,
        )
    for index, value in enumerate(
        _sequence(report.get("exports"), "representative exports")
    ):
        _verify_declared_artifact(
            value,
            f"representative export {index}",
            artifacts,
            "exports",
            parent=evaluation_dir,
        )
    videos = _sequence(report.get("videos"), "representative videos")
    if len(videos) != 1:
        raise FinalArtifactError("representative evaluation must contain exactly one video")
    video_source = _mapping(videos[0], "representative video")
    video_path, video_artifact = _verify_declared_artifact(
        video_source,
        "representative video",
        artifacts,
        "videos",
        parent=evaluation_dir,
    )
    if video_path.suffix.lower() != ".mp4":
        raise FinalArtifactError("representative video must be an MP4")
    expected_video_path = evaluation_dir / "videos" / "rl-video-step-0.mp4"
    if video_path != expected_video_path.resolve():
        raise FinalArtifactError(
            "representative video must be videos/rl-video-step-0.mp4"
        )
    discovered_videos = sorted((evaluation_dir / "videos").glob("*.mp4"))
    if [path.resolve() for path in discovered_videos] != [video_path]:
        raise FinalArtifactError(
            "representative videos directory must contain exactly the declared MP4"
        )
    capture_contract = _mapping(
        report.get("video_capture_contract"),
        "representative video capture contract",
    )
    if (
        capture_contract.get("mode")
        != "post_physics_pre_reset_terminal_frame_substitution"
        or capture_contract.get("single_environment") is not True
        or capture_contract.get("full_fixed_step_window") is not True
        or capture_contract.get("enabled") is not True
    ):
        raise FinalArtifactError(
            "representative video did not use the terminal-aware fixed-window contract"
        )
    captured_terminal_frames = _integer(
        capture_contract.get("terminal_frames_captured"),
        "representative terminal frames captured",
        minimum=1,
    )
    substituted_terminal_frames = _integer(
        capture_contract.get("terminal_frames_substituted"),
        "representative terminal frames substituted",
        minimum=1,
    )
    if (
        captured_terminal_frames != completed_episodes
        or substituted_terminal_frames != completed_episodes
    ):
        raise FinalArtifactError(
            "representative terminal-frame evidence disagrees with completed episodes"
        )
    try:
        independent_inspection = inspect_video(video_path)
    except VideoValidationError as exc:
        raise FinalArtifactError(f"representative video validation failed: {exc}") from exc
    recorded_inspection = dict(
        _mapping(video_source.get("inspection"), "representative video inspection")
    )
    if recorded_inspection != independent_inspection:
        raise FinalArtifactError(
            "representative video inspection differs from independent decoding"
        )
    decoded_frames = _integer(
        independent_inspection.get("decoded_frame_count"),
        "representative decoded video frames",
        minimum=1,
    )
    # RecordVideo normally emits one fewer frame than executed simulator steps.
    # Permit the initial-frame convention in either direction, but reject a
    # materially truncated visual record.
    if decoded_frames < max(1, steps_executed - 1) or decoded_frames > steps_executed + 1:
        raise FinalArtifactError(
            "representative video frame count does not cover the executed episode"
        )
    if not math.isclose(
        _number(independent_inspection.get("fps"), "representative video FPS"),
        1.0 / EXPECTED_CONTROL_DT,
        abs_tol=1.0e-6,
    ):
        raise FinalArtifactError("representative video is not 50 Hz")
    if independent_inspection.get("all_frames_black") is not False:
        raise FinalArtifactError("representative video is entirely black")
    if (
        independent_inspection.get("width_px") != 1280
        or independent_inspection.get("height_px") != 720
    ):
        raise FinalArtifactError("representative video must be 1280x720")
    if _number(
        independent_inspection.get("visually_nonblack_frame_fraction"),
        "representative visually nonblack frame fraction",
    ) < 0.98:
        raise FinalArtifactError("representative video is mostly black")
    if _number(
        independent_inspection.get("changed_frame_pair_fraction"),
        "representative changed-frame-pair fraction",
    ) < 0.05:
        raise FinalArtifactError(
            "representative video lacks sufficient temporal visual change"
        )
    if _number(
        independent_inspection.get("temporal_mean_absolute_pixel_change"),
        "representative temporal pixel change",
    ) < 0.05:
        raise FinalArtifactError(
            "representative video has negligible average temporal change"
        )

    artifacts.add(report_path, "reports", "representative held-out evaluation report")
    return {
        "complete": True,
        "status": "completed",
        "split": "test",
        "report": _artifact(
            report_path, role="representative held-out evaluation report"
        ),
        "motion_selection": selection,
        "correspondence": correspondence,
        "runtime_contract": runtime,
        "checkpoint": {**checkpoint_actual, "path": _relative(checkpoint_path)},
        "representative_motion": {
            "motion_id": motion_id,
            "motion_key": authoritative["motion_key"],
            "path": _relative(authoritative["path"]),
            "checksum_sha256": authoritative["checksum_sha256"],
            "frame_count": expected_frames,
            "first_episode_length": first_length,
            "first_episode_completion_ratio": expected_ratio,
            "uninterrupted_full_clip_success": uninterrupted,
            "completed_episodes_in_recording_window": completed_episodes,
        },
        "video": {
            **video_artifact,
            "path": _relative(video_path),
            "inspection": independent_inspection,
        },
        "video_capture_contract": dict(capture_contract),
    }


def _verify_pipeline_foundation(artifacts: _Artifacts) -> dict[str, Any]:
    paths = {
        "gate1": WORK_ROOT / "manifests" / "gate1_summary.json",
        "batch_items": WORK_ROOT / "reports" / "batch" / "batch_items.jsonl",
        "batch": WORK_ROOT / "reports" / "batch" / "batch_summary.json",
        "parallel": WORK_ROOT / "reports" / "batch" / "parallel_batch_summary.json",
        "gate3": WORK_ROOT / "manifests" / "g1_manifest_build.json",
        "kinematic": WORK_ROOT / "reports" / "B3_-_walk1_isaac_kinematic.json",
        "pd": WORK_ROOT / "reports" / "B3_-_walk1_isaac_pd_replay.json",
        "b3": WORK_ROOT / "g1" / "Female1Walking_c3d" / "B3_-_walk1_g1_50hz.npz",
        "kinematic_video": WORK_ROOT / "media" / "B3_-_walk1_isaac_kinematic.mp4",
        "pd_video": WORK_ROOT / "media" / "B3_-_walk1_isaac_pd_replay.mp4",
    }
    resolved = {name: _inside(path, ACCAD_ROOT, name) for name, path in paths.items()}
    gate1 = _read_json(resolved["gate1"], "Gate-1 summary")
    counts = _mapping(gate1.get("counts"), "Gate-1 counts")
    if gate1.get("schema_version") != "accad-gate1-v1" or gate1.get("passed") is not True or counts.get("eligible_files") != 249:
        raise FinalArtifactError("Gate 1 is not a passed 249-motion snapshot")
    batch = _read_json(resolved["batch"], "batch summary")
    batch_counts = _mapping(batch.get("counts"), "batch counts")
    settings = _mapping(batch.get("settings"), "batch settings")
    if not (
        batch.get("schema_version") == "accad_g1_batch_v1"
        and batch.get("all_selected_items_completed") is True
        and settings.get("max_items") is None
        and batch_counts.get("selected_total") == 249
        and batch_counts.get("completed") == 249
        and batch_counts.get("failed") == 0
    ):
        raise FinalArtifactError("canonical batch is not the complete 249-motion snapshot")
    jsonl = _mapping(_mapping(batch.get("artifacts"), "batch artifacts").get("items_jsonl"), "batch JSONL")
    if _inside(jsonl.get("path", ""), ACCAD_ROOT, "batch JSONL") != resolved["batch_items"]:
        raise FinalArtifactError("batch summary names a different batch_items.jsonl")
    if _checksum(jsonl.get("checksum_sha256"), "batch JSONL checksum") != _sha256(resolved["batch_items"]):
        raise FinalArtifactError("batch JSONL checksum mismatch")
    if jsonl.get("records") != 249 or _count_jsonl_objects(
        resolved["batch_items"], "batch JSONL"
    ) != 249:
        raise FinalArtifactError("batch JSONL must contain 249 records")
    parallel = _read_json(resolved["parallel"], "parallel batch summary")
    if not (
        parallel.get("schema_version") == "accad_g1_parallel_batch_v1"
        and parallel.get("status") == "completed"
        and parallel.get("success") is True
        and parallel.get("worker_problem_count") == 0
        and parallel.get("retarget_version") == RETARGET_VERSION
    ):
        raise FinalArtifactError("parallel batch summary is not complete and clean")
    gate3 = _read_json(resolved["gate3"], "Gate-3 manifest build")
    promoted = _mapping(gate3.get("counts"), "Gate-3 counts")
    if not (
        gate3.get("schema_version") == "accad_g1_gate3_manifest_build_v1"
        and gate3.get("status") == "completed"
        and gate3.get("retarget_version") == RETARGET_VERSION
        and promoted.get("promoted_total") == 249
        and promoted.get("source_completed_records") == 249
    ):
        raise FinalArtifactError("Gate-3 manifest promotion is incomplete")
    manifests = _mapping(gate3.get("manifests"), "Gate-3 promoted manifests")
    for name in ("train", "validation", "test", "combined"):
        declared = _mapping(manifests.get(name), f"Gate-3 {name} artifact")
        path = _inside(PIPELINE_ROOT / str(declared.get("path", "")), ACCAD_ROOT, f"Gate-3 {name} artifact")
        if _checksum(declared.get("checksum_sha256"), f"Gate-3 {name} checksum") != _sha256(path):
            raise FinalArtifactError(f"Gate-3 {name} manifest checksum mismatch")
        artifacts.add(path, "manifests", f"Gate-3 promoted {name} manifest")

    b3_checksum = _sha256(resolved["b3"])
    kinematic = _read_json(resolved["kinematic"], "Isaac kinematic report")
    kinematic_input = _mapping(kinematic.get("input"), "Isaac kinematic input")
    if not (
        kinematic.get("schema_version") == "accad_g1_isaac_validation_v3"
        and kinematic.get("mode") == "kinematic"
        and kinematic.get("execution_complete") is True
        and kinematic.get("passed") is True
        and kinematic_input.get("full_clip") is True
        and kinematic_input.get("validated_frames") == kinematic_input.get("total_frames")
        and kinematic_input.get("sha256") == b3_checksum
    ):
        raise FinalArtifactError("current B3 archive lacks a passing full Isaac kinematic replay")
    pd = _read_json(resolved["pd"], "Isaac PD report")
    pd_input = _mapping(pd.get("input"), "Isaac PD input")
    if not (
        pd.get("schema_version") == "accad_g1_isaac_validation_v3"
        and pd.get("mode") == "pd-replay"
        and pd.get("execution_complete") is True
        and pd_input.get("full_clip") is True
        and pd_input.get("validated_frames") == pd_input.get("total_frames")
        and pd_input.get("sha256") == b3_checksum
    ):
        raise FinalArtifactError("current B3 archive lacks a complete open-loop PD diagnostic")

    for report, report_input, video_name, label in (
        (kinematic, kinematic_input, "kinematic_video", "kinematic"),
        (pd, pd_input, "pd_video", "PD replay"),
    ):
        validator = _mapping(report.get("validator"), f"{label} validator")
        validator_path = _inside(
            validator.get("path", ""), ACCAD_ROOT, f"{label} validator path"
        )
        if (
            validator_path != (PIPELINE_ROOT / "run_isaac.py").resolve()
            or _checksum(validator.get("sha256"), f"{label} validator checksum")
            != _sha256(validator_path)
        ):
            raise FinalArtifactError(f"{label} report does not match the current validator")
        asset = _mapping(report.get("asset"), f"{label} asset provenance")
        for path_key, checksum_key, expected_path in (
            (
                "joint_contract",
                "joint_contract_sha256",
                PROJECT_ROOT / "configs" / "robot" / "g1_29dof.yaml",
            ),
            (
                "urdf",
                "urdf_sha256",
                PROJECT_ROOT
                / "third_party"
                / "unitree_ros"
                / "robots"
                / "g1_description"
                / "g1_29dof_rev_1_0.urdf",
            ),
        ):
            declared_path = _inside(
                asset.get(path_key, ""), PROJECT_ROOT, f"{label} {path_key}"
            )
            if (
                declared_path != expected_path.resolve()
                or _checksum(asset.get(checksum_key), f"{label} {checksum_key}")
                != _sha256(declared_path)
            ):
                raise FinalArtifactError(
                    f"{label} report does not match the current G1 {path_key}"
                )
        video = _mapping(report.get("video"), f"{label} video")
        if not (
            video.get("enabled") is True
            and video.get("all_black_frames") == 0
            and video.get("frames") == report_input.get("total_frames")
            and _inside(video.get("path", ""), ACCAD_ROOT, f"{label} video path")
            == resolved[video_name]
        ):
            raise FinalArtifactError(f"{label} video is missing, incomplete, or all black")

    for name in (
        "accad_inventory.json",
        "accad_quality.json",
        "accad_train.json",
        "accad_validation.json",
        "accad_test.json",
        "accad_excluded.json",
    ):
        artifacts.add(
            _inside(WORK_ROOT / "manifests" / name, ACCAD_ROOT, f"Gate-1 {name}"),
            "manifests",
            f"Gate-1 {name}",
        )

    for name, path in resolved.items():
        category = "videos" if name.endswith("video") else "manifests" if name in {"gate1", "batch_items", "batch", "parallel", "gate3"} else "reports" if name in {"kinematic", "pd"} else "references"
        artifacts.add(path, category, name.replace("_", " "))
    return {
        "gate1_complete": True,
        "gate3_kinematic_dataset_complete": True,
        "isaac_kinematic_complete": True,
        "open_loop_pd_execution_complete": True,
        "open_loop_pd_dynamics_passed": pd.get("dynamics_passed"),
        "b3_archive_checksum_sha256": b3_checksum,
        "retarget_version": RETARGET_VERSION,
        "motion_set_checksum_sha256": _checksum(
            gate3.get("motion_set_checksum_sha256"),
            "Gate-3 motion-set checksum",
        ),
    }


def _collect_code_and_config(artifacts: _Artifacts) -> None:
    for name in ENTRYPOINT_FILES:
        artifacts.add(_inside(PIPELINE_ROOT / name, ACCAD_ROOT, f"entry point {name}"), "code", f"entry point {name}")
    package_files = sorted((PIPELINE_ROOT / "accad_g1").glob("*.py"))
    if not package_files:
        raise FinalArtifactError("ACCAD package source files are missing")
    for path in package_files:
        artifacts.add(_inside(path, ACCAD_ROOT, "package source"), "code", "package source")
    for path in sorted((PIPELINE_ROOT / "tests").glob("test_*.py")):
        artifacts.add(_inside(path, ACCAD_ROOT, "CPU regression test"), "tests", "CPU regression test")
    for name in LOCAL_CONFIG_FILES:
        category = "documentation" if name == "README.md" else "configuration"
        path = _inside(PIPELINE_ROOT / name, ACCAD_ROOT, name)
        if name == "README.md":
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise FinalArtifactError(f"cannot read final README: {exc}") from exc
            present = [
                token for token in UNFINISHED_DOCUMENT_MARKERS if token in text
            ]
            if present:
                raise FinalArtifactError(
                    f"README still contains unfinished markers: {present}"
                )
        artifacts.add(path, category, name)
    model = _inside(PIPELINE_ROOT / "models" / "smplx" / "SMPLX_NEUTRAL.npz", ACCAD_ROOT, "SMPL-X neutral model")
    artifacts.add(model, "licensed_model", "SMPL-X neutral model")
    dependencies = (
        (PROJECT_ROOT / "configs" / "robot" / "g1_29dof.yaml", "G1 joint contract"),
        (
            PROJECT_ROOT
            / "third_party"
            / "unitree_ros"
            / "robots"
            / "g1_description"
            / "g1_29dof_rev_1_0.urdf",
            "G1 URDF",
        ),
    )
    for path, role in dependencies:
        artifacts.add(path, "external_read_only_dependencies", role, scope="project_read_only")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = _inside(path.parent, ACCAD_ROOT, "final report directory", kind="directory") / path.name
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def build_final_artifacts(
    *,
    training_run: str | Path,
    validation_report: str | Path,
    test_report: str | Path,
    representative_evaluation_report: str | Path,
    professor_report: str | Path,
) -> dict[str, Any]:
    """Validate all final inputs and publish two atomic completion witnesses."""

    professor = _inside(professor_report, ACCAD_ROOT, "professor report")
    if professor.suffix.lower() != ".md":
        raise FinalArtifactError("professor report must be a Markdown file")
    try:
        professor_text = professor.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise FinalArtifactError(f"cannot read professor report: {exc}") from exc
    if len(professor_text.strip()) < 100:
        raise FinalArtifactError("professor report is unexpectedly short")
    present = [
        token for token in UNFINISHED_DOCUMENT_MARKERS if token in professor_text
    ]
    if present:
        raise FinalArtifactError(f"professor report still contains unfinished markers: {present}")

    artifacts = _Artifacts()
    foundation = _verify_pipeline_foundation(artifacts)
    source_manifests = {
        split: _load_gate3_manifest(split, artifacts)
        for split in EXPECTED_SOURCE_SPLIT_COUNTS
    }
    dynamic_policy_path = _inside(
        WORK_ROOT / "dynamic" / "scale_policy.json",
        ACCAD_ROOT,
        "frozen dynamic scale policy",
    )
    dynamic_policy, dynamic_policy_checksum = _verify_dynamic_policy(
        dynamic_policy_path, source_manifests, artifacts
    )
    dynamic_stages: dict[str, dict[str, Any]] = {}
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for split in EXPECTED_SOURCE_SPLIT_COUNTS:
        dynamic_manifest_path, dynamic_manifest, dynamic_summary = _verify_dynamic_split(
            split,
            source_path=source_manifests[split][0],
            source_manifest=source_manifests[split][1],
            policy_path=dynamic_policy_path,
            policy=dynamic_policy,
            policy_checksum=dynamic_policy_checksum,
            artifacts=artifacts,
        )
        manifests[split] = (dynamic_manifest_path, dynamic_manifest)
        dynamic_stages[split] = dynamic_summary
    training = _verify_training(
        training_run, manifests["train"][0], manifests["train"][1], artifacts
    )
    validation = _verify_suite(
        validation_report,
        split="validation",
        manifest_path=manifests["validation"][0],
        manifest=manifests["validation"][1],
        final_checkpoint=training["final_checkpoint"],
        training_runtime=training["runtime_contract"],
        artifacts=artifacts,
    )
    test = _verify_suite(
        test_report,
        split="test",
        manifest_path=manifests["test"][0],
        manifest=manifests["test"][1],
        final_checkpoint=training["final_checkpoint"],
        training_runtime=training["runtime_contract"],
        artifacts=artifacts,
    )
    representative_evaluation = _verify_representative_evaluation(
        representative_evaluation_report,
        manifest_path=manifests["test"][0],
        manifest=manifests["test"][1],
        final_checkpoint=training["final_checkpoint"],
        training_runtime=training["runtime_contract"],
        artifacts=artifacts,
    )
    _collect_code_and_config(artifacts)
    artifacts.add(professor, "documentation", "professor-ready final report")
    professor_artifact = _artifact(professor, role="professor-ready final report")
    artifacts.verify_unchanged()
    generated_at = _utc_now()
    artifact_manifest_path = REPORT_ROOT / "final_artifact_manifest.json"
    completion_path = REPORT_ROOT / "final_completion.json"
    artifact_manifest = {
        "schema_version": ARTIFACT_SCHEMA,
        "status": "completed",
        "generated_at_utc": generated_at,
        "snapshot_semantics": (
            "Checksummed v27-to-v28 provenance and execution snapshot; both exhaustive "
            "dynamic promoted-motion suites passed the predefined acceptance policy."
        ),
        "pipeline_foundation": foundation,
        "dynamic_retime": dynamic_stages,
        "independent_policy_export_validation": training[
            "independent_export_validation"
        ],
        "artifact_root": ".",
        "artifact_unique_count": artifacts.unique_count,
        "artifacts": artifacts.grouped(),
    }
    _write_json_atomic(artifact_manifest_path, artifact_manifest)
    artifact_manifest_record = _artifact(
        artifact_manifest_path, role="final artifact manifest"
    )
    policy_verdict = bool(
        validation["predefined_acceptance"]["verdict"]
        and test["predefined_acceptance"]["verdict"]
    )
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "status": "completed",
        "generated_at_utc": generated_at,
        "completion_semantics": (
            "Dynamic promotion, model_599 policy-only initialized training, both exhaustive "
            "held-out suites, and a manifest-selected representative-video evaluation "
            "completed with verified contracts, artifacts, and passing acceptance criteria."
        ),
        "artifact_manifest": artifact_manifest_record,
        "pipeline_foundation": foundation,
        "dynamic_retime": dynamic_stages,
        "training": training,
        "validation": validation,
        "test": test,
        "representative_evaluation": representative_evaluation,
        "policy_acceptance": {
            "verdict": policy_verdict,
            "validation_verdict": validation["predefined_acceptance"]["verdict"],
            "test_verdict": test["predefined_acceptance"]["verdict"],
            "interpretation": (
                "Final publication requires both exhaustive dynamic suites to pass."
            ),
        },
        "professor_report": professor_artifact,
    }
    # Written last: this file is the commit witness.  run.py status verifies its
    # recorded artifact-manifest checksum, so an interrupted two-file publish
    # cannot be mistaken for a current completed snapshot.
    _write_json_atomic(completion_path, completion)
    return {
        "status": "completed",
        "artifact_manifest": {
            "path": str(artifact_manifest_path),
            "checksum_sha256": _sha256(artifact_manifest_path),
        },
        "completion": {
            "path": str(completion_path),
            "checksum_sha256": _sha256(completion_path),
        },
        "policy_acceptance_verdict": policy_verdict,
    }


__all__ = [
    "ACCEPTANCE_SCHEMA",
    "ARTIFACT_SCHEMA",
    "COMPLETION_SCHEMA",
    "DYNAMIC_ACCEPTANCE_POLICY",
    "FinalArtifactError",
    "build_final_artifacts",
]
