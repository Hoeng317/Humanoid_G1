"""Filesystem and input helpers for the standalone G1 tracking CLIs.

The module uses only the Python standard library at import time so command-line
scripts may safely import it *before* constructing Isaac Lab's ``AppLauncher``.
Motion-library imports are deliberately local to :func:`resolve_motion_inputs`
and therefore happen only after the simulator has started.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = PACKAGE_ROOT.parent
ACCAD_ROOT = PIPELINE_ROOT.parent
WORK_ROOT = PIPELINE_ROOT / "work"
TRAINING_ROOT = WORK_ROOT / "training"
RUNS_ROOT = TRAINING_ROOT / "runs"
EVALUATIONS_ROOT = TRAINING_ROOT / "evaluations"
# Final policy training consumes only Gate-3-promoted records with exact G1
# archive checksums.  ``accad_train.json`` is the upstream Gate-1 inventory and
# remains useful to the batch processor, but it is not an authoritative policy
# input once retarget validation has completed.
DEFAULT_MANIFEST = WORK_ROOT / "manifests" / "g1_train.json"
DEFAULT_REFERENCE_ROOT = WORK_ROOT / "g1"
DEFAULT_CORRESPONDENCE = PIPELINE_ROOT / "correspondence.yaml"
TRAINING_MOTION_ASSIGNMENT_SCHEMA_VERSION = "accad_g1_training_motion_assignment_v1"
TRAINING_MOTION_ASSIGNMENT_CHOICES = ("episode-uniform", "balanced-fixed")
DEFAULT_TRAINING_MOTION_ASSIGNMENT = "balanced-fixed"

_MODEL_CHECKPOINT_NAME = re.compile(r"model_(\d+)\.pt")
_NORMALIZER_STATE_KEYS = (
    "actor_obs_normalizer._mean",
    "actor_obs_normalizer._var",
    "actor_obs_normalizer._std",
    "actor_obs_normalizer.count",
    "critic_obs_normalizer._mean",
    "critic_obs_normalizer._var",
    "critic_obs_normalizer._std",
    "critic_obs_normalizer.count",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_below(path: str | Path, root: str | Path, *, must_exist: bool = False) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved_root = Path(root).expanduser().resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path must remain below {resolved_root}: {resolved}") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return slug[:80] or "run"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_artifact_dir(kind: str, name: str, *, timestamp: datetime | None = None) -> Path:
    """Atomically reserve a run/evaluation directory below ``work/training``."""

    if kind not in {"runs", "evaluations"}:
        raise ValueError("artifact kind must be 'runs' or 'evaluations'")
    root = RUNS_ROOT if kind == "runs" else EVALUATIONS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    now = timestamp or datetime.now()
    base = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{_slug(name)}"
    for index in range(1000):
        suffix = "" if index == 0 else f"_{index:03d}"
        candidate = root / f"{base}{suffix}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return path_below(candidate, TRAINING_ROOT)
    raise RuntimeError(f"cannot reserve a unique artifact directory for {base}")


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    destination = path_below(path, TRAINING_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


@dataclass(frozen=True)
class MotionInputSelection:
    paths: tuple[Path, ...]
    manifests: tuple[dict[str, Any], ...]
    checksum_verified_paths: tuple[Path, ...]
    split: str
    reference_root: Path
    allow_missing: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "reference_root": str(self.reference_root),
            "allow_missing": self.allow_missing,
            "motion_count": len(self.paths),
            "manifest_checksum_verified_motion_count": len(self.checksum_verified_paths),
            "manifest_promotion_verified": bool(
                self.manifests and len(self.checksum_verified_paths) == len(self.paths)
            ),
            "motions": [
                {
                    "path": str(path),
                    "checksum_sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in self.paths
            ],
            "manifests": list(self.manifests),
        }


def add_motion_input_args(parser: argparse.ArgumentParser, *, default_split: str = "train") -> None:
    parser.add_argument(
        "--motion",
        action="append",
        default=[],
        metavar="G1_NPZ",
        help="Gate-3 G1 NPZ below ACCAD; repeat for multiple clips.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        metavar="JSON",
        help="Gate-1/Gate-3 JSON manifest; repeat to merge manifests.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default=default_split,
        help=f"Manifest split (default: {default_split}).",
    )
    parser.add_argument(
        "--reference-root",
        default=str(DEFAULT_REFERENCE_ROOT),
        help="Root containing retargeted *_g1_50hz.npz files.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Use the generated subset of a manifest instead of requiring every clip.",
    )
    parser.add_argument(
        "--max-motions",
        type=positive_int,
        default=None,
        help="Deterministic lexicographic prefix for smoke/debug runs.",
    )
    parser.add_argument(
        "--correspondence",
        default=str(DEFAULT_CORRESPONDENCE),
        help="Correspondence YAML whose checksum must match every archive.",
    )
    parser.add_argument(
        "--skip-correspondence-check",
        action="store_true",
        help="Debug only: trust the correspondence checksum embedded in each archive.",
    )


def add_training_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    """Add the three mutually exclusive training checkpoint modes.

    ``--checkpoint`` and ``--resume`` are true RSL-RL resume modes: the runner
    restores its optimizer and iteration counter.  ``--initialize-policy-from``
    is intentionally different and copies only ``model_state_dict`` into a
    newly constructed runner.
    """

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimizer/counters from the latest local model_*.pt.",
    )
    group.add_argument(
        "--checkpoint",
        default=None,
        metavar="MODEL_PT",
        help="Resume optimizer/counters from an explicit RSL-RL checkpoint.",
    )
    group.add_argument(
        "--initialize-policy-from",
        default=None,
        metavar="MODEL_PT",
        help=(
            "Strictly initialize policy and actor/critic normalizers from an "
            "ACCAD-local model_*.pt while keeping fresh optimizer/counters."
        ),
    )


def resolve_motion_inputs(
    *,
    motions: Sequence[str | Path],
    manifests: Sequence[str | Path],
    split: str,
    reference_root: str | Path,
    allow_missing: bool,
    max_motions: int | None,
) -> MotionInputSelection:
    """Resolve direct and manifest inputs into one deterministic unique path list."""

    # Local import keeps torch out of pre-AppLauncher command-line parsing.
    from .motion_library import motion_paths_from_manifest

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")
    root = path_below(reference_root, ACCAD_ROOT, must_exist=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    manifest_values = list(manifests)
    if not motions and not manifest_values:
        manifest_values = [DEFAULT_MANIFEST]

    resolved: list[Path] = []
    expected_checksums: dict[Path, str] = {}
    manifest_metadata: list[dict[str, Any]] = []
    for value in motions:
        path = path_below(value, ACCAD_ROOT, must_exist=True)
        if not path.is_file() or path.suffix.lower() != ".npz":
            raise ValueError(f"direct motion must be an NPZ file: {path}")
        resolved.append(path)
    for value in manifest_values:
        manifest = path_below(value, ACCAD_ROOT, must_exist=True)
        selection = motion_paths_from_manifest(
            manifest,
            root,
            split=split,
            require_all=not allow_missing,
        )
        resolved.extend(selection.paths)
        for path, checksum in selection.expected_checksums.items():
            previous = expected_checksums.setdefault(path, checksum)
            if previous != checksum:
                raise ValueError(f"manifests disagree on checksum for {path}")
        manifest_metadata.append(
            {
                "path": str(selection.manifest_path),
                "checksum_sha256": selection.manifest_checksum_sha256,
                "split": selection.split,
                "resolved_count": len(selection.paths),
                "archive_checksum_record_count": len(selection.expected_checksums),
            }
        )
    paths = sorted(set(resolved), key=lambda path: path.as_posix())
    if max_motions is not None:
        if max_motions <= 0:
            raise ValueError("max_motions must be positive")
        paths = paths[:max_motions]
    if not paths:
        raise ValueError("motion selection is empty")
    for path in paths:
        expected = expected_checksums.get(path)
        if expected is not None:
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"manifest checksum mismatch for {path}: expected {expected}, got {actual}"
                )
    return MotionInputSelection(
        paths=tuple(paths),
        manifests=tuple(manifest_metadata),
        checksum_verified_paths=tuple(
            path for path in paths if path in expected_checksums
        ),
        split=split,
        reference_root=root,
        allow_missing=bool(allow_missing),
    )


def checkpoint_candidates(root: str | Path | None = None) -> tuple[Path, ...]:
    directory = path_below(RUNS_ROOT if root is None else root, TRAINING_ROOT)
    if not directory.exists():
        return ()
    candidates = [
        path
        for path in directory.rglob("model_*.pt")
        if path.is_file() and re.fullmatch(r"model_\d+\.pt", path.name)
    ]

    def key(path: Path) -> tuple[int, int, str]:
        iteration = int(path.stem.split("_")[-1])
        return (path.stat().st_mtime_ns, iteration, path.as_posix())

    return tuple(sorted(candidates, key=key))


def resolve_checkpoint(
    checkpoint: str | Path | None,
    *,
    latest_if_missing: bool,
) -> Path | None:
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".pt":
            raise FileNotFoundError(f"checkpoint is not an existing .pt file: {path}")
        return path
    if not latest_if_missing:
        return None
    candidates = checkpoint_candidates()
    if not candidates:
        raise FileNotFoundError(f"no model_*.pt checkpoint exists below {RUNS_ROOT}")
    return candidates[-1]


def checkpoint_training_contract_provenance(
    checkpoint: str | Path,
    *,
    expected_policy_contract_version: str,
    expected_training_objective_schema: str | None = None,
) -> dict[str, Any]:
    """Bind a checkpoint to its completed parent run and training contracts.

    Tensor shapes alone cannot distinguish checkpoints trained under different
    actuator semantics.  Official evaluation therefore requires the sibling
    ``runtime_contract.json`` and ``run_summary.json`` to agree on the exact
    policy contract.  Missing or incompatible metadata is returned as an
    explicit diagnostic result so callers may permit a labelled ablation while
    still failing closed for acceptance.
    """

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file() or _MODEL_CHECKPOINT_NAME.fullmatch(path.name) is None:
        raise ValueError(
            "checkpoint provenance requires an existing model_<iteration>.pt file"
        )
    if not isinstance(expected_policy_contract_version, str) or not (
        expected_policy_contract_version.strip()
    ):
        raise ValueError("expected_policy_contract_version must be a non-empty string")
    if expected_training_objective_schema is not None and (
        not isinstance(expected_training_objective_schema, str)
        or not expected_training_objective_schema.strip()
    ):
        raise ValueError(
            "expected_training_objective_schema must be None or a non-empty string"
        )

    runtime_path = path.parent / "runtime_contract.json"
    summary_path = path.parent / "run_summary.json"
    result: dict[str, Any] = {
        "schema_version": "accad_g1_checkpoint_training_contract_provenance_v2",
        "checkpoint_path": str(path),
        "checkpoint_checksum_sha256": sha256_file(path),
        "expected_policy_contract_version": expected_policy_contract_version,
        "expected_training_objective_schema": expected_training_objective_schema,
        "verified": False,
        "compatible": False,
        "runtime_contract": None,
        "run_summary": None,
        "reason": None,
    }
    missing = [
        candidate.name
        for candidate in (runtime_path, summary_path)
        if not candidate.is_file()
    ]
    if missing:
        result["reason"] = "missing_parent_run_metadata:" + ",".join(missing)
        return result

    def load_mapping(candidate: Path) -> Mapping[str, Any]:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint parent metadata: {candidate}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"checkpoint parent metadata must be an object: {candidate}")
        return payload

    runtime = load_mapping(runtime_path)
    summary = load_mapping(summary_path)
    summary_runtime = summary.get("runtime_contract")
    if not isinstance(summary_runtime, Mapping):
        result["reason"] = "run_summary_missing_runtime_contract"
        return result
    runtime_version = runtime.get("tracking_policy_contract_version")
    summary_version = summary_runtime.get("tracking_policy_contract_version")
    runtime_objective = runtime.get("training_objective")
    summary_objective = summary_runtime.get("training_objective")
    runtime_objective_schema = (
        runtime_objective.get("schema_version")
        if isinstance(runtime_objective, Mapping)
        else None
    )
    summary_objective_schema = (
        summary_objective.get("schema_version")
        if isinstance(summary_objective, Mapping)
        else None
    )
    result["runtime_contract"] = {
        "path": str(runtime_path),
        "checksum_sha256": sha256_file(runtime_path),
        "policy_contract_version": runtime_version,
        "training_objective_schema": runtime_objective_schema,
    }
    result["run_summary"] = {
        "path": str(summary_path),
        "checksum_sha256": sha256_file(summary_path),
        "status": summary.get("status"),
        "policy_contract_version": summary_version,
        "training_objective_schema": summary_objective_schema,
    }
    result["verified"] = True
    result["compatible"] = bool(
        summary.get("status") == "completed"
        and runtime_version == expected_policy_contract_version
        and summary_version == expected_policy_contract_version
        and (
            expected_training_objective_schema is None
            or (
                runtime_objective_schema == expected_training_objective_schema
                and summary_objective_schema == expected_training_objective_schema
            )
        )
    )
    if summary.get("status") != "completed":
        result["reason"] = "parent_training_run_not_completed"
    elif runtime_version != summary_version:
        result["reason"] = "parent_metadata_policy_contract_mismatch"
    elif runtime_version != expected_policy_contract_version:
        result["reason"] = "checkpoint_trained_under_different_policy_contract"
    elif expected_training_objective_schema is not None and (
        runtime_objective_schema != summary_objective_schema
    ):
        result["reason"] = "parent_metadata_training_objective_mismatch"
    elif expected_training_objective_schema is not None and (
        runtime_objective_schema != expected_training_objective_schema
    ):
        result["reason"] = "checkpoint_trained_under_different_training_objective"
    return result


@dataclass(frozen=True)
class PolicyInitializationSource:
    """Validated RSL-RL policy state used without resume semantics."""

    path: Path
    checksum_sha256: str
    size_bytes: int
    iteration: int
    filename_iteration: int
    state_layout_checksum_sha256: str
    state_key_count: int
    checkpoint_optimizer_state_present: bool
    model_state_dict: Mapping[str, Any] = field(repr=False, compare=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "checksum_sha256": self.checksum_sha256,
            "bytes": self.size_bytes,
            "iter": self.iteration,
            "iteration": self.iteration,
            "filename_iteration": self.filename_iteration,
            "state_layout_checksum_sha256": self.state_layout_checksum_sha256,
            "state_key_count": self.state_key_count,
            "checkpoint_field_loaded": "model_state_dict",
            "checkpoint_optimizer_state_present": self.checkpoint_optimizer_state_present,
        }


def resolve_policy_initialization_checkpoint(
    checkpoint: str | Path | None,
) -> Path | None:
    """Resolve an ACCAD-internal, conventionally named RSL-RL checkpoint."""

    if checkpoint is None:
        return None
    path = path_below(checkpoint, ACCAD_ROOT, must_exist=True)
    if not path.is_file() or _MODEL_CHECKPOINT_NAME.fullmatch(path.name) is None:
        raise ValueError(
            "policy initialization checkpoint must be an ACCAD-internal "
            f"model_<iteration>.pt file: {path}"
        )
    return path


def _state_layout_checksum(model_state_dict: Mapping[str, Any]) -> str:
    layout = [
        {
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in sorted(model_state_dict.items())
    ]
    encoded = json.dumps(layout, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_policy_initialization_source(
    checkpoint: str | Path | None,
) -> PolicyInitializationSource | None:
    """Load only the policy field from a validated ACCAD RSL-RL checkpoint.

    The checkpoint is deserialized with PyTorch's restricted weights-only
    loader.  Its optimizer payload is deliberately neither returned nor
    applied.  File identity is checked both before and after deserialization so
    the metadata describes the exact bytes that supplied the policy.
    """

    path = resolve_policy_initialization_checkpoint(checkpoint)
    if path is None:
        return None

    import torch

    before_size = path.stat().st_size
    before_checksum = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("policy initialization checkpoint must contain a mapping")
    if "model_state_dict" not in payload:
        raise ValueError("policy initialization checkpoint is missing model_state_dict")
    model_state_dict = payload["model_state_dict"]
    if not isinstance(model_state_dict, Mapping) or not model_state_dict:
        raise TypeError("checkpoint model_state_dict must be a non-empty mapping")
    if any(not isinstance(key, str) for key in model_state_dict):
        raise TypeError("checkpoint model_state_dict keys must all be strings")
    non_tensor_keys = [
        key for key, value in model_state_dict.items() if not isinstance(value, torch.Tensor)
    ]
    if non_tensor_keys:
        raise TypeError(
            "checkpoint model_state_dict values must all be tensors; invalid keys: "
            + ", ".join(sorted(non_tensor_keys))
        )
    missing_normalizers = sorted(set(_NORMALIZER_STATE_KEYS) - set(model_state_dict))
    if missing_normalizers:
        raise ValueError(
            "policy initialization checkpoint is missing actor/critic normalizer state: "
            + ", ".join(missing_normalizers)
        )

    iteration = payload.get("iter")
    if isinstance(iteration, bool) or not isinstance(iteration, Integral) or iteration < 0:
        raise ValueError("policy initialization checkpoint iter must be a non-negative integer")
    iteration = int(iteration)
    match = _MODEL_CHECKPOINT_NAME.fullmatch(path.name)
    assert match is not None
    filename_iteration = int(match.group(1))
    if filename_iteration != iteration:
        raise ValueError(
            "checkpoint filename/payload iteration mismatch: "
            f"filename={filename_iteration}, payload={iteration}"
        )

    after_size = path.stat().st_size
    after_checksum = sha256_file(path)
    if after_size != before_size or after_checksum != before_checksum:
        raise RuntimeError(f"policy initialization checkpoint changed while loading: {path}")
    return PolicyInitializationSource(
        path=path,
        checksum_sha256=after_checksum,
        size_bytes=after_size,
        iteration=iteration,
        filename_iteration=filename_iteration,
        state_layout_checksum_sha256=_state_layout_checksum(model_state_dict),
        state_key_count=len(model_state_dict),
        checkpoint_optimizer_state_present="optimizer_state_dict" in payload,
        model_state_dict=model_state_dict,
    )


def _fresh_runner_state(runner: Any, optimizer: Any) -> dict[str, Any]:
    iteration = getattr(runner, "current_learning_iteration", None)
    timesteps = getattr(runner, "tot_timesteps", None)
    if isinstance(iteration, bool) or not isinstance(iteration, Integral):
        raise TypeError("runner.current_learning_iteration must be an integer")
    if isinstance(timesteps, bool) or not isinstance(timesteps, Integral):
        raise TypeError("runner.tot_timesteps must be an integer")
    state = getattr(optimizer, "state", None)
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(state, Mapping):
        raise TypeError("runner optimizer state must be a mapping")
    if not isinstance(param_groups, list):
        raise TypeError("runner optimizer param_groups must be a list")
    snapshot: dict[str, Any] = {
        "current_learning_iteration": int(iteration),
        "tot_timesteps": int(timesteps),
        "optimizer_state_entry_count": len(state),
        "optimizer_param_group_count": len(param_groups),
        "optimizer_param_group_parameter_counts": [
            len(group.get("params", ())) for group in param_groups
        ],
        "optimizer_param_group_learning_rates": [
            float(group["lr"]) for group in param_groups
        ],
    }
    if hasattr(runner, "tot_time"):
        snapshot["tot_time"] = float(runner.tot_time)
    return snapshot


def _require_fresh_runner_state(snapshot: Mapping[str, Any], *, phase: str) -> None:
    if snapshot["current_learning_iteration"] != 0:
        raise ValueError(
            f"runner must have iteration 0 {phase} policy initialization; "
            f"got {snapshot['current_learning_iteration']}"
        )
    if snapshot["tot_timesteps"] != 0:
        raise ValueError(
            f"runner must have 0 timesteps {phase} policy initialization; "
            f"got {snapshot['tot_timesteps']}"
        )
    if snapshot["optimizer_state_entry_count"] != 0:
        raise ValueError(
            f"runner optimizer must be fresh {phase} policy initialization; "
            f"got {snapshot['optimizer_state_entry_count']} state entries"
        )


def initialize_runner_policy_from_source(
    runner: Any,
    source: PolicyInitializationSource,
) -> dict[str, Any]:
    """Strictly copy policy weights while proving runner state stays fresh."""

    import torch

    algorithm = getattr(runner, "alg", None)
    policy = getattr(algorithm, "policy", None)
    optimizer = getattr(algorithm, "optimizer", None)
    if policy is None or not callable(getattr(policy, "state_dict", None)):
        raise TypeError("runner.alg.policy must provide state_dict")
    if not callable(getattr(policy, "load_state_dict", None)):
        raise TypeError("runner.alg.policy must provide load_state_dict")
    if optimizer is None:
        raise TypeError("runner.alg.optimizer is required")

    policy_identity = id(policy)
    optimizer_identity = id(optimizer)
    before = _fresh_runner_state(runner, optimizer)
    _require_fresh_runner_state(before, phase="before")

    target_state = policy.state_dict()
    if not isinstance(target_state, Mapping):
        raise TypeError("runner policy state_dict must return a mapping")
    missing_target_normalizers = sorted(set(_NORMALIZER_STATE_KEYS) - set(target_state))
    if missing_target_normalizers:
        raise ValueError(
            "target policy is missing actor/critic normalizer state: "
            + ", ".join(missing_target_normalizers)
        )
    source_keys = set(source.model_state_dict)
    target_keys = set(target_state)
    if source_keys != target_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise ValueError(
            "policy state key mismatch for strict initialization; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key in sorted(target_keys):
        source_tensor = source.model_state_dict[key]
        target_tensor = target_state[key]
        if not isinstance(target_tensor, torch.Tensor):
            raise TypeError(f"target policy state is not a tensor: {key}")
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"policy tensor shape mismatch for {key}: "
                f"source={tuple(source_tensor.shape)}, target={tuple(target_tensor.shape)}"
            )
        if source_tensor.dtype != target_tensor.dtype:
            raise ValueError(
                f"policy tensor dtype mismatch for {key}: "
                f"source={source_tensor.dtype}, target={target_tensor.dtype}"
            )

    incompatible = policy.load_state_dict(source.model_state_dict, strict=True)
    missing_keys = list(getattr(incompatible, "missing_keys", ()))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", ()))
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "strict policy initialization unexpectedly returned incompatible keys: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    loaded_state = policy.state_dict()
    unequal_keys = []
    for key in sorted(source_keys):
        actual = loaded_state[key].detach()
        expected = source.model_state_dict[key].detach().to(device=actual.device)
        if not torch.equal(actual, expected):
            unequal_keys.append(key)
    if unequal_keys:
        raise RuntimeError(
            "strict policy initialization did not reproduce source tensors: "
            + ", ".join(unequal_keys)
        )

    if id(runner.alg.policy) != policy_identity:
        raise RuntimeError("policy object was replaced during initialization")
    if id(runner.alg.optimizer) != optimizer_identity:
        raise RuntimeError("optimizer object was replaced during initialization")
    after = _fresh_runner_state(runner, optimizer)
    _require_fresh_runner_state(after, phase="after")
    if before != after:
        raise RuntimeError(
            "runner or optimizer metadata changed during policy-only initialization: "
            f"before={before}, after={after}"
        )

    return {
        "schema_version": "accad_g1_policy_initialization_v1",
        "mode": "strict_model_state_dict_with_fresh_optimizer",
        "source": source.metadata(),
        "load_contract": {
            "torch_load_weights_only": True,
            "checkpoint_field_loaded": "model_state_dict",
            "strict": True,
            "optimizer_state_loaded": False,
            "normalizer_state_keys_loaded": list(_NORMALIZER_STATE_KEYS),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "all_loaded_tensors_exactly_equal": True,
        },
        "fresh_runner_state": {
            "before": before,
            "after": after,
            "policy_object_preserved": True,
            "optimizer_object_preserved": True,
        },
    }


def command_metadata(argv: Sequence[str]) -> dict[str, Any]:
    return {
        "argv": list(argv),
        "cwd": str(Path.cwd().resolve()),
        "started_at_utc": utc_now(),
        "pipeline_root": str(PIPELINE_ROOT),
        "training_root": str(TRAINING_ROOT),
    }


def validate_policy_exports(
    paths: Sequence[str | Path],
    *,
    actor_observation_dim: int,
    action_dim: int,
) -> list[dict[str, Any]]:
    """Load/check exported policies against the flat deployment contract.

    Imports are local so this module remains safe to import before AppLauncher.
    JIT is executed on a finite zero observation. ONNX is structurally checked;
    the installed exporter emits a static batch-one contract.
    """

    if actor_observation_dim <= 0 or action_dim <= 0:
        raise ValueError("policy dimensions must be positive")
    validations: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path.suffix == ".pt":
            import torch

            module = torch.jit.load(str(path), map_location="cpu")
            module.eval()
            with torch.inference_mode():
                output = module(torch.zeros((1, actor_observation_dim), dtype=torch.float32))
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"JIT export returned {type(output).__name__}, expected Tensor")
            if tuple(output.shape) != (1, action_dim):
                raise ValueError(
                    f"JIT export shape mismatch: expected (1, {action_dim}), got {tuple(output.shape)}"
                )
            if not bool(torch.isfinite(output).all()):
                raise ValueError("JIT export produced NaN or Inf")
            validations.append(
                {
                    "path": str(path),
                    "format": "torchscript",
                    "input_shape": [1, actor_observation_dim],
                    "output_shape": [1, action_dim],
                    "finite_output": True,
                }
            )
        elif path.suffix == ".onnx":
            import numpy as np
            import onnx
            from onnx.reference import ReferenceEvaluator

            model = onnx.load(str(path))
            onnx.checker.check_model(model)
            if len(model.graph.input) != 1 or len(model.graph.output) != 1:
                raise ValueError("ONNX policy must expose exactly one input and one output")
            input_shape = [
                int(dimension.dim_value)
                for dimension in model.graph.input[0].type.tensor_type.shape.dim
            ]
            output_shape = [
                int(dimension.dim_value)
                for dimension in model.graph.output[0].type.tensor_type.shape.dim
            ]
            if input_shape != [1, actor_observation_dim] or output_shape != [1, action_dim]:
                raise ValueError(
                    "ONNX export shape mismatch: "
                    f"expected [1, {actor_observation_dim}] -> [1, {action_dim}], "
                    f"got {input_shape} -> {output_shape}"
                )
            reference_outputs = ReferenceEvaluator(model).run(
                None,
                {
                    model.graph.input[0].name: np.zeros(
                        (1, actor_observation_dim), dtype=np.float32
                    )
                },
            )
            if len(reference_outputs) != 1:
                raise ValueError(
                    f"ONNX reference inference returned {len(reference_outputs)} outputs"
                )
            reference_output = np.asarray(reference_outputs[0])
            if reference_output.shape != (1, action_dim):
                raise ValueError(
                    "ONNX reference inference shape mismatch: "
                    f"expected {(1, action_dim)}, got {reference_output.shape}"
                )
            if not bool(np.isfinite(reference_output).all()):
                raise ValueError("ONNX reference inference produced NaN or Inf")
            validations.append(
                {
                    "path": str(path),
                    "format": "onnx",
                    "opset": max(entry.version for entry in model.opset_import),
                    "input_name": model.graph.input[0].name,
                    "input_shape": input_shape,
                    "output_name": model.graph.output[0].name,
                    "output_shape": output_shape,
                    "checker_passed": True,
                    "reference_inference_executed": True,
                    "finite_output": True,
                }
            )
        else:
            raise ValueError(f"unsupported policy export extension: {path}")
    return validations
