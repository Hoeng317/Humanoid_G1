"""Deterministic Gate 1 audit for the raw ACCAD AMASS archive.

The raw dataset is treated as immutable.  This module only reads ``*_stageii.npz``
files with NumPy's pickle loader disabled, then writes JSON manifests beneath the
caller-provided output directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "accad-gate1-v1"
_PIPELINE_DIRECTORY = "_g1_pipeline"
_REQUIRED_ARRAYS = ("trans", "root_orient", "pose_body", "betas")
_OPTIONAL_TIME_ARRAYS = {
    "pose_hand": 90,
    "pose_jaw": 3,
    "pose_eye": 6,
    "poses": 165,
}
_BODY_JOINT_NAMES = (
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
_SPLIT_NAMES = ("train", "validation", "test")
_PERSON_PREFIX = re.compile(r"^(Female\d+|Male\d+)")
_SUBJECT_DIRECTORY = re.compile(r"^s\d+$", re.IGNORECASE)
_DUPLICATE_ALIAS_SUFFIX = re.compile(r"(?:_a|_copy|_dup)$", re.IGNORECASE)
# The official ACCAD catalog lists this otherwise-unprefixed directory under
# "Male 2 Martial Arts Walks/Turns".  Treating it as a new actor would leak
# Male2 between train and evaluation splits.
_DIRECTORY_SUBJECT_OVERRIDES = {"MartialArtsWalksTurns_c3d": "Male2"}


def _native(value: Any) -> Any:
    """Convert NumPy scalar values to JSON-native values."""

    if isinstance(value, np.generic):
        return value.item()
    return value


def _finite_number(value: Any) -> float | None:
    """Return a finite scalar float, otherwise ``None``."""

    try:
        array = np.asarray(value)
        if array.size != 1:
            return None
        number = float(array.reshape(()))
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace ``path`` with a deterministic UTF-8 JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _scan_files(raw_root: Path, suffix: str) -> list[Path]:
    """Scan a raw-tree suffix while excluding every pipeline-owned subtree."""

    files: list[Path] = []
    for path in raw_root.rglob(f"*{suffix}"):
        if not path.is_file():
            continue
        relative = path.relative_to(raw_root)
        if _PIPELINE_DIRECTORY in relative.parts:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(raw_root).as_posix())


def _parse_identity(relative_path: Path) -> tuple[str, str, str, str]:
    """Return ``(subject, category, source_directory, motion_name)``."""

    if len(relative_path.parts) < 2:
        source_directory = "."
    else:
        source_directory = relative_path.parts[0]

    normalized_directory = source_directory.removesuffix("_c3d")
    prefix_match = _PERSON_PREFIX.match(normalized_directory)
    if source_directory in _DIRECTORY_SUBJECT_OVERRIDES:
        subject = _DIRECTORY_SUBJECT_OVERRIDES[source_directory]
        category = normalized_directory
    elif prefix_match:
        subject = prefix_match.group(1)
        category = normalized_directory[len(subject) :] or "Uncategorized"
    elif _SUBJECT_DIRECTORY.fullmatch(normalized_directory):
        subject = normalized_directory.lower()
        category = "Uncategorized"
    else:
        # Unknown unprefixed directories remain isolated subject groups rather
        # than being guessed into a named actor.
        subject = normalized_directory
        category = normalized_directory

    suffix = "_stageii"
    stem = relative_path.stem
    motion_name = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    return subject, category, source_directory, motion_name


def _rotvec_to_unit_quaternion(rotvec: np.ndarray) -> np.ndarray:
    """Convert ``[..., 3]`` axis-angle vectors to unit ``wxyz`` quaternions."""

    vectors = np.asarray(rotvec, dtype=np.float64)
    angles = np.linalg.norm(vectors, axis=-1, keepdims=True)
    half_angles = 0.5 * angles

    # sin(theta / 2) / theta has the series 1/2 - theta^2/48 + O(theta^4).
    small = angles < 1.0e-8
    scale = np.empty_like(angles)
    np.divide(np.sin(half_angles), angles, out=scale, where=~small)
    theta_squared = angles * angles
    scale[small] = (0.5 - theta_squared[small] / 48.0)

    quaternion = np.concatenate((np.cos(half_angles), vectors * scale), axis=-1)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return quaternion / np.maximum(norm, np.finfo(np.float64).eps)


def _rotvec_geodesic_steps(rotvec: np.ndarray) -> np.ndarray:
    """Compute shortest SO(3) distances between consecutive rotations.

    Axis-angle vectors are intentionally not subtracted.  The absolute
    quaternion inner product identifies the equivalent ``q`` and ``-q``
    representations before evaluating the geodesic angle.
    """

    rotations = np.asarray(rotvec, dtype=np.float64)
    if rotations.shape[0] < 2:
        return np.empty((0,) + rotations.shape[1:-1], dtype=np.float64)
    quaternions = _rotvec_to_unit_quaternion(rotations)
    dot = np.sum(quaternions[1:] * quaternions[:-1], axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _threshold(config: Mapping[str, Any], name: str, default: float | int) -> float:
    value = config.get(name, default)
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"quality.{name} must be numeric, got {value!r}") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"quality.{name} must be finite and non-negative")
    return result


def _threshold_with_alias(
    config: Mapping[str, Any],
    name: str,
    default: float | int,
    *aliases: str,
) -> float:
    if name in config:
        return _threshold(config, name, default)
    for alias in aliases:
        if alias in config:
            return _threshold({name: config[alias]}, name, default)
    return _threshold(config, name, default)


def _quality_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    quality = config.get("quality", {})
    if quality is None:
        quality = {}
    if not isinstance(quality, Mapping):
        raise ValueError("quality must be a mapping")

    settings: dict[str, Any] = {
        "min_frames": int(_threshold(quality, "min_frames", 2)),
        "min_duration_sec": _threshold(quality, "min_duration_sec", 0.1),
        "max_translation_step_m": _threshold_with_alias(
            quality, "max_translation_step_m", 0.25, "translation_step_warn_m"
        ),
        "max_translation_speed_mps": _threshold_with_alias(
            quality,
            "max_translation_speed_mps",
            15.0,
            "translation_speed_warn_mps",
        ),
        "max_rotation_step_rad": _threshold_with_alias(
            quality, "max_rotation_step_rad", 2.5, "rotation_step_warn_rad"
        ),
        "max_duplicate_fraction": _threshold_with_alias(
            quality, "max_duplicate_fraction", 0.20, "duplicate_ratio_warn"
        ),
        "duration_tolerance_frames": _threshold(
            quality, "duration_tolerance_frames", 1.01
        ),
    }
    if settings["max_duplicate_fraction"] > 1.0:
        raise ValueError("quality.max_duplicate_fraction must be in [0, 1]")

    configured_flags = quality.get(
        "auto_exclude_flags", config.get("exclude_quality_flags", [])
    )
    if configured_flags is None:
        configured_flags = []
    if isinstance(configured_flags, (str, bytes)) or not isinstance(
        configured_flags, Sequence
    ):
        raise ValueError("quality.auto_exclude_flags must be a list of flag names")
    settings["auto_exclude_flags"] = sorted({str(flag) for flag in configured_flags})
    settings["exclude_if_any_quality_flag"] = bool(
        quality.get(
            "exclude_if_any_quality_flag",
            quality.get("exclude_on_quality_warning", False),
        )
    )
    return settings


def _config_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    gate1 = config.get("gate1")
    if gate1 is None:
        return config
    if not isinstance(gate1, Mapping):
        raise ValueError("gate1 must be a mapping")
    return gate1


def _split_mapping(
    whole_config: Mapping[str, Any],
    gate_config: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Read the explicit subject split mapping with a few schema aliases."""

    candidate: Any = gate_config.get("split_subjects")
    if candidate is None:
        candidate = whole_config.get("split_subjects")
    if candidate is None:
        for owner in (gate_config, whole_config):
            splits = owner.get("splits") or owner.get("split")
            if isinstance(splits, Mapping):
                candidate = splits.get("subjects", splits)
                break
    if not isinstance(candidate, Mapping):
        raise ValueError(
            "config must define split_subjects as "
            "{train: [...], validation: [...], test: [...]}"
        )

    normalized: dict[str, list[str]] = {}
    for split_name in _SPLIT_NAMES:
        raw_subjects = candidate.get(split_name)
        if raw_subjects is None and split_name == "validation":
            raw_subjects = candidate.get("val")
        if raw_subjects is None:
            raise ValueError(f"split_subjects.{split_name} is required")
        if isinstance(raw_subjects, (str, bytes)) or not isinstance(
            raw_subjects, Sequence
        ):
            raise ValueError(f"split_subjects.{split_name} must be a list")
        normalized[split_name] = sorted({str(subject) for subject in raw_subjects})
        if len(normalized[split_name]) != len(raw_subjects):
            raise ValueError(f"split_subjects.{split_name} contains duplicates")
    return normalized


def _validate_subject_splits(
    split_subjects: Mapping[str, Sequence[str]], observed_subjects: Iterable[str]
) -> dict[str, Any]:
    observed = set(observed_subjects)
    owners: dict[str, str] = {}
    duplicate_owners: dict[str, list[str]] = {}
    for split_name in _SPLIT_NAMES:
        for subject in split_subjects[split_name]:
            if subject in owners:
                duplicate_owners.setdefault(subject, [owners[subject]]).append(split_name)
            else:
                owners[subject] = split_name

    assigned = set(owners)
    missing = sorted(observed - assigned)
    extra = sorted(assigned - observed)
    if duplicate_owners or missing or extra:
        parts: list[str] = []
        if duplicate_owners:
            parts.append(f"overlap={dict(sorted(duplicate_owners.items()))}")
        if missing:
            parts.append(f"unassigned={missing}")
        if extra:
            parts.append(f"unknown={extra}")
        raise ValueError("subject split is not disjoint/exhaustive: " + ", ".join(parts))

    return {
        "disjoint": True,
        "exhaustive": True,
        "observed_subjects": sorted(observed),
        "subject_to_split": dict(sorted(owners.items())),
    }


def _empty_record(relative_path: Path) -> dict[str, Any]:
    subject, category, source_directory, motion_name = _parse_identity(relative_path)
    return {
        "source_file": relative_path.as_posix(),
        "checksum_sha256": None,
        "subject": subject,
        "category": category,
        "source_directory": source_directory,
        "motion_name": motion_name,
        "gender": None,
        "surface_model_type": None,
        "num_betas": None,
        "fps": None,
        "num_frames": None,
        "duration_sec": None,
        "duration_metadata_sec": None,
        "keys": [],
        "array_shapes": {},
        "array_dtypes": {},
        "structural_valid": False,
        "valid": False,
        "exclude_reason": [],
        "quality_flags": [],
        "quality_metrics": {},
    }


def _safe_array(archive: Any, name: str) -> np.ndarray:
    """Load a numeric NPZ member without ever enabling object unpickling."""

    array = np.asarray(archive[name])
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a non-object numeric array")
    return array


def _audit_motion(path: Path, raw_root: Path, settings: Mapping[str, Any]) -> dict[str, Any]:
    relative = path.relative_to(raw_root)
    record = _empty_record(relative)
    structural_errors: list[str] = []
    quality_flags: list[str] = []

    try:
        record["checksum_sha256"] = _sha256(path)
    except OSError as error:
        structural_errors.append(f"checksum_error:{type(error).__name__}")

    arrays: dict[str, np.ndarray] = {}
    metadata_duration: float | None = None
    try:
        with np.load(path, allow_pickle=False) as archive:
            record["keys"] = sorted(str(key) for key in archive.files)
            missing = [name for name in _REQUIRED_ARRAYS if name not in archive.files]
            structural_errors.extend(f"missing_key:{name}" for name in missing)

            for name in (*_REQUIRED_ARRAYS, *_OPTIONAL_TIME_ARRAYS):
                if name not in archive.files:
                    continue
                try:
                    arrays[name] = _safe_array(archive, name)
                    record["array_shapes"][name] = list(arrays[name].shape)
                    record["array_dtypes"][name] = str(arrays[name].dtype)
                except (TypeError, ValueError) as error:
                    structural_errors.append(f"invalid_array:{name}:{type(error).__name__}")

            if "mocap_frame_rate" not in archive.files:
                structural_errors.append("missing_key:mocap_frame_rate")
            else:
                try:
                    fps = _finite_number(archive["mocap_frame_rate"])
                except (TypeError, ValueError):
                    fps = None
                if fps is None or fps <= 0.0:
                    structural_errors.append("invalid_fps")
                else:
                    record["fps"] = fps

            if "mocap_time_length" in archive.files:
                try:
                    metadata_duration = _finite_number(archive["mocap_time_length"])
                except (TypeError, ValueError):
                    metadata_duration = None
                record["duration_metadata_sec"] = metadata_duration
                if metadata_duration is None or metadata_duration <= 0.0:
                    quality_flags.append("invalid_duration_metadata")

            if "gender" in archive.files:
                try:
                    gender_array = np.asarray(archive["gender"])
                    if gender_array.size == 1 and not gender_array.dtype.hasobject:
                        record["gender"] = str(_native(gender_array.reshape(()).item()))
                    else:
                        quality_flags.append("invalid_gender_metadata")
                except (TypeError, ValueError):
                    quality_flags.append("invalid_gender_metadata")

            if "surface_model_type" in archive.files:
                try:
                    surface_array = np.asarray(archive["surface_model_type"])
                    if surface_array.size == 1 and not surface_array.dtype.hasobject:
                        record["surface_model_type"] = str(
                            _native(surface_array.reshape(()).item())
                        )
                    else:
                        quality_flags.append("invalid_surface_model_type_metadata")
                except (TypeError, ValueError):
                    quality_flags.append("invalid_surface_model_type_metadata")

            if "num_betas" in archive.files:
                try:
                    num_betas_value = _finite_number(archive["num_betas"])
                except (TypeError, ValueError):
                    num_betas_value = None
                if num_betas_value is None or not num_betas_value.is_integer():
                    quality_flags.append("invalid_num_betas_metadata")
                else:
                    record["num_betas"] = int(num_betas_value)
    except (OSError, ValueError, KeyError) as error:
        structural_errors.append(f"npz_read_error:{type(error).__name__}")

    trans = arrays.get("trans")
    root_orient = arrays.get("root_orient")
    pose_body = arrays.get("pose_body")
    betas = arrays.get("betas")

    frame_counts: dict[str, int] = {}
    expected_widths = {"trans": 3, "root_orient": 3, "pose_body": 63}
    for name, width in expected_widths.items():
        array = arrays.get(name)
        if array is None:
            continue
        if array.ndim != 2 or array.shape[1] != width:
            structural_errors.append(f"invalid_shape:{name}:expected_Tx{width}")
        else:
            frame_counts[name] = int(array.shape[0])

    if betas is not None and betas.shape != (16,):
        structural_errors.append("invalid_shape:betas:expected_16")

    for name, width in _OPTIONAL_TIME_ARRAYS.items():
        array = arrays.get(name)
        if array is None:
            continue
        if array.ndim != 2 or array.shape[1] != width:
            structural_errors.append(f"invalid_shape:{name}:expected_Tx{width}")
        else:
            frame_counts[name] = int(array.shape[0])

    if frame_counts:
        unique_frame_counts = set(frame_counts.values())
        if len(unique_frame_counts) != 1:
            structural_errors.append("frame_count_mismatch")
        else:
            num_frames = next(iter(unique_frame_counts))
            record["num_frames"] = num_frames
            if num_frames == 0:
                structural_errors.append("zero_length_motion")

    for name, array in arrays.items():
        if not bool(np.isfinite(array).all()):
            structural_errors.append(f"nonfinite:{name}")

    if record["gender"] is None:
        quality_flags.append("missing_gender_metadata")
    elif record["gender"].lower() != "neutral":
        quality_flags.append("gender_not_neutral")
    if record["surface_model_type"] is None:
        quality_flags.append("missing_surface_model_type_metadata")
    elif record["surface_model_type"].lower() != "smplx":
        quality_flags.append("surface_model_type_not_smplx")
    if record["num_betas"] is None:
        quality_flags.append("missing_num_betas_metadata")
    elif record["num_betas"] != 16:
        quality_flags.append("num_betas_mismatch")

    fps_value = record["fps"]
    num_frames_value = record["num_frames"]
    if fps_value is not None and num_frames_value is not None:
        duration = float(num_frames_value / fps_value)
        record["duration_sec"] = duration

        if num_frames_value < settings["min_frames"]:
            quality_flags.append("too_few_frames")
        if duration < settings["min_duration_sec"]:
            quality_flags.append("too_short")

        if metadata_duration is None:
            if "invalid_duration_metadata" not in quality_flags:
                quality_flags.append("missing_duration_metadata")
        elif metadata_duration > 0.0:
            # AMASS exporters use either T/fps or (T-1)/fps conventions.
            interval_duration = max(num_frames_value - 1, 0) / fps_value
            error = min(
                abs(metadata_duration - duration),
                abs(metadata_duration - interval_duration),
            )
            tolerance = settings["duration_tolerance_frames"] / fps_value
            record["quality_metrics"]["duration_error_sec"] = float(error)
            record["quality_metrics"]["duration_tolerance_sec"] = float(tolerance)
            if error > tolerance:
                quality_flags.append("duration_mismatch")

    arrays_usable = (
        trans is not None
        and root_orient is not None
        and pose_body is not None
        and trans.ndim == 2
        and root_orient.ndim == 2
        and pose_body.ndim == 2
        and trans.shape[1:] == (3,)
        and root_orient.shape[1:] == (3,)
        and pose_body.shape[1:] == (63,)
        and trans.shape[0] == root_orient.shape[0] == pose_body.shape[0]
        and bool(np.isfinite(trans).all())
        and bool(np.isfinite(root_orient).all())
        and bool(np.isfinite(pose_body).all())
    )
    if arrays_usable and trans.shape[0] >= 2 and fps_value is not None:
        translation_steps = np.linalg.norm(np.diff(trans.astype(np.float64), axis=0), axis=1)
        translation_transition = int(np.argmax(translation_steps))
        max_translation_step = float(np.max(translation_steps))
        max_translation_speed = float(max_translation_step * fps_value)

        root_steps = _rotvec_geodesic_steps(root_orient)
        body_steps = _rotvec_geodesic_steps(pose_body.reshape(pose_body.shape[0], 21, 3))
        root_transition = int(np.argmax(root_steps))
        body_transition, body_joint = (
            int(value) for value in np.unravel_index(np.argmax(body_steps), body_steps.shape)
        )
        max_root_rotation_step = float(np.max(root_steps))
        max_body_rotation_step = float(np.max(body_steps))
        max_rotation_step = max(max_root_rotation_step, max_body_rotation_step)

        duplicate_mask = (
            np.all(trans[1:] == trans[:-1], axis=1)
            & np.all(root_orient[1:] == root_orient[:-1], axis=1)
            & np.all(pose_body[1:] == pose_body[:-1], axis=1)
        )
        duplicate_indices = np.flatnonzero(duplicate_mask) + 1
        duplicate_count = int(duplicate_indices.size)
        duplicate_fraction = float(duplicate_count / (trans.shape[0] - 1))

        record["quality_metrics"].update(
            {
                "max_translation_step_m": max_translation_step,
                "max_translation_speed_mps": max_translation_speed,
                "max_translation_transition": [
                    translation_transition,
                    translation_transition + 1,
                ],
                "max_root_rotation_step_rad": max_root_rotation_step,
                "max_root_rotation_transition": [
                    root_transition,
                    root_transition + 1,
                ],
                "max_body_rotation_step_rad": max_body_rotation_step,
                "max_body_rotation_transition": [
                    body_transition,
                    body_transition + 1,
                ],
                "max_body_rotation_joint_index": body_joint,
                "max_body_rotation_joint_name": _BODY_JOINT_NAMES[body_joint],
                "max_rotation_step_rad": max_rotation_step,
                "duplicate_consecutive_frames": duplicate_count,
                "duplicate_transition_fraction": duplicate_fraction,
                "duplicate_frame_indices_sample": [
                    int(index) for index in duplicate_indices[:20]
                ],
                "duplicate_frame_indices_truncated": duplicate_count > 20,
            }
        )

        if max_translation_step > settings["max_translation_step_m"]:
            quality_flags.append("translation_step_jump")
        if max_translation_speed > settings["max_translation_speed_mps"]:
            quality_flags.append("translation_speed_jump")
        if max_rotation_step > settings["max_rotation_step_rad"]:
            quality_flags.append("rotation_geodesic_jump")
        if duplicate_fraction > settings["max_duplicate_fraction"]:
            quality_flags.append("excessive_duplicate_frames")

    record["structural_valid"] = not structural_errors
    record["exclude_reason"] = sorted(set(structural_errors))
    record["quality_flags"] = sorted(set(quality_flags))

    auto_flags = set(settings["auto_exclude_flags"])
    matched_auto_flags = sorted(auto_flags.intersection(record["quality_flags"]))
    if settings["exclude_if_any_quality_flag"]:
        matched_auto_flags = list(record["quality_flags"])
    if record["structural_valid"] and matched_auto_flags:
        record["exclude_reason"].extend(
            f"quality:{flag}" for flag in matched_auto_flags
        )
    record["exclude_reason"] = sorted(set(record["exclude_reason"]))
    record["valid"] = record["structural_valid"] and not matched_auto_flags
    return record


def _distribution(records: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(record[key]) for record in records)
    return dict(sorted(counts.items()))


def _mark_duplicate_files(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one deterministic canonical file for each identical SHA-256 group."""

    checksum_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        checksum = record.get("checksum_sha256")
        if checksum:
            checksum_groups.setdefault(str(checksum), []).append(record)

    report: list[dict[str, Any]] = []
    for checksum, members in sorted(checksum_groups.items()):
        if len(members) < 2:
            continue
        ordered = sorted(
            members,
            key=lambda record: (
                not bool(record["structural_valid"]),
                bool(_DUPLICATE_ALIAS_SUFFIX.search(str(record["motion_name"]))),
                str(record["source_file"]),
            ),
        )
        canonical = ordered[0]
        canonical_path = str(canonical["source_file"])
        duplicate_paths: list[str] = []
        for duplicate in ordered[1:]:
            duplicate_path = str(duplicate["source_file"])
            duplicate_paths.append(duplicate_path)
            duplicate["duplicate_file_of"] = canonical_path
            duplicate["quality_flags"] = sorted(
                set(duplicate["quality_flags"]) | {"duplicate_file_checksum"}
            )
            duplicate["exclude_reason"] = sorted(
                set(duplicate["exclude_reason"])
                | {f"duplicate_file_of:{canonical_path}"}
            )
            duplicate["valid"] = False
        canonical["duplicate_file_aliases"] = duplicate_paths
        report.append(
            {
                "checksum_sha256": checksum,
                "canonical_source_file": canonical_path,
                "duplicate_source_files": duplicate_paths,
                "group_size": len(ordered),
            }
        )
    return report


def _manifest_header(kind: str, raw_root: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "dataset": "ACCAD",
        "source_root_name": raw_root.name,
        "path_semantics": "source_file is POSIX and relative to source_root_name",
        "source_pattern": "*_stageii.npz",
        "pickle_loading_enabled": False,
    }


def audit_dataset(raw_root: Path, config: dict, output_root: Path) -> dict:
    """Audit ACCAD Stage-II motions and write the complete Gate 1 manifest set.

    Parameters
    ----------
    raw_root:
        ACCAD root containing raw category directories.  ``_g1_pipeline`` is
        always excluded from recursive scans.
    config:
        Either the Gate 1 mapping or a full configuration with a ``gate1``
        mapping.  An explicit subject-disjoint ``split_subjects`` mapping is
        required.  Quality flags are informational unless their names appear in
        ``quality.auto_exclude_flags`` (or ``exclude_if_any_quality_flag`` is
        true).
    output_root:
        Directory that receives the seven deterministic JSON reports.

    Returns
    -------
    dict
        The same summary written to ``gate1_summary.json``.
    """

    raw_root = Path(raw_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"ACCAD raw root is not a directory: {raw_root}")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    pipeline_root = (raw_root / _PIPELINE_DIRECTORY).resolve()
    try:
        output_root.relative_to(pipeline_root)
    except ValueError as error:
        raise ValueError(
            f"Gate 1 output must stay inside {pipeline_root}; got {output_root}"
        ) from error

    gate_config = _config_section(config)
    settings = _quality_settings(gate_config)
    split_subjects = _split_mapping(config, gate_config)

    stageii_files = _scan_files(raw_root, "_stageii.npz")
    stagei_files = _scan_files(raw_root, "_stagei.npz")
    if not stageii_files:
        raise FileNotFoundError(f"no *_stageii.npz files found under {raw_root}")

    records = [_audit_motion(path, raw_root, settings) for path in stageii_files]
    duplicate_checksum_groups = _mark_duplicate_files(records)
    split_validation = _validate_subject_splits(
        split_subjects, (record["subject"] for record in records)
    )
    subject_to_split = split_validation["subject_to_split"]

    split_records: dict[str, list[dict[str, Any]]] = {
        split_name: [] for split_name in _SPLIT_NAMES
    }
    excluded_records: list[dict[str, Any]] = []
    for record in records:
        if not record["valid"]:
            excluded_records.append(record)
            continue
        split_records[subject_to_split[record["subject"]]].append(record)

    inventory = _manifest_header("inventory", raw_root)
    inventory.update(
        {
            "counts": {
                "stagei_files": len(stagei_files),
                "stageii_files": len(records),
                "structural_valid": sum(
                    bool(record["structural_valid"]) for record in records
                ),
                "eligible": sum(bool(record["valid"]) for record in records),
                "excluded": len(excluded_records),
            },
            "subjects": _distribution(records, "subject"),
            "categories": _distribution(records, "category"),
            "directory_subject_overrides": _DIRECTORY_SUBJECT_OVERRIDES,
            "duplicate_checksum_groups": duplicate_checksum_groups,
            "motions": records,
        }
    )

    flag_counts = Counter(
        flag for record in records for flag in record["quality_flags"]
    )
    quality = _manifest_header("quality", raw_root)
    quality.update(
        {
            "thresholds": settings,
            "quality_flags_are_informational_by_default": True,
            "flag_counts": dict(sorted(flag_counts.items())),
            "motions_with_flags": sum(bool(record["quality_flags"]) for record in records),
            "motions": [
                {
                    "source_file": record["source_file"],
                    "subject": record["subject"],
                    "category": record["category"],
                    "structural_valid": record["structural_valid"],
                    "valid": record["valid"],
                    "quality_flags": record["quality_flags"],
                    "quality_metrics": record["quality_metrics"],
                }
                for record in records
            ],
        }
    )

    split_payloads: dict[str, dict[str, Any]] = {}
    for split_name in _SPLIT_NAMES:
        members = split_records[split_name]
        payload = _manifest_header(f"split:{split_name}", raw_root)
        payload.update(
            {
                "split": split_name,
                "subjects": list(split_subjects[split_name]),
                "num_motions": len(members),
                "num_frames": int(sum(record["num_frames"] for record in members)),
                "duration_sec": float(
                    sum(record["duration_sec"] for record in members)
                ),
                "categories": _distribution(members, "category"),
                "motions": members,
            }
        )
        split_payloads[split_name] = payload

    all_categories = sorted({str(record["category"]) for record in records})
    category_split_counts = {
        category: {
            split_name: sum(
                record["category"] == category
                for record in split_records[split_name]
            )
            for split_name in _SPLIT_NAMES
        }
        for category in all_categories
    }

    excluded = _manifest_header("excluded", raw_root)
    excluded.update(
        {
            "num_motions": len(excluded_records),
            "reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for record in excluded_records
                        for reason in record["exclude_reason"]
                    ).items()
                )
            ),
            "motions": excluded_records,
        }
    )

    expected_count = gate_config.get("expected_stageii_count")
    if expected_count is not None:
        try:
            expected_count = int(expected_count)
        except (TypeError, ValueError) as error:
            raise ValueError("expected_stageii_count must be an integer") from error
        if expected_count < 0:
            raise ValueError("expected_stageii_count must be non-negative")

    expected_stagei_count = gate_config.get("expected_stagei_count")
    if expected_stagei_count is not None:
        try:
            expected_stagei_count = int(expected_stagei_count)
        except (TypeError, ValueError) as error:
            raise ValueError("expected_stagei_count must be an integer") from error
        if expected_stagei_count < 0:
            raise ValueError("expected_stagei_count must be non-negative")

    all_checksums = all(record["checksum_sha256"] is not None for record in records)
    accounted_count = sum(len(items) for items in split_records.values()) + len(
        excluded_records
    )
    split_file_sets = {
        split_name: {record["source_file"] for record in members}
        for split_name, members in split_records.items()
    }
    file_disjoint = all(
        split_file_sets[left].isdisjoint(split_file_sets[right])
        for index, left in enumerate(_SPLIT_NAMES)
        for right in _SPLIT_NAMES[index + 1 :]
    )
    split_checksums = [
        str(record["checksum_sha256"])
        for members in split_records.values()
        for record in members
    ]

    checks = {
        "stageii_found": len(records) > 0,
        "expected_stagei_count_matches": (
            True
            if expected_stagei_count is None
            else len(stagei_files) == expected_stagei_count
        ),
        "expected_stageii_count_matches": (
            True if expected_count is None else len(records) == expected_count
        ),
        "all_stageii_have_sha256": all_checksums,
        "subject_split_disjoint": split_validation["disjoint"],
        "subject_split_exhaustive": split_validation["exhaustive"],
        "split_files_disjoint": file_disjoint,
        "split_checksums_unique": len(split_checksums) == len(set(split_checksums)),
        "all_stageii_accounted_for": accounted_count == len(records),
        "only_structurally_valid_files_in_splits": all(
            record["structural_valid"]
            for members in split_records.values()
            for record in members
        ),
    }
    summary = _manifest_header("gate1_summary", raw_root)
    summary.update(
        {
            "gate": 1,
            "passed": all(checks.values()),
            "checks": checks,
            "counts": {
                "stagei_files": len(stagei_files),
                "stageii_files": len(records),
                "eligible_files": len(records) - len(excluded_records),
                "excluded_files": len(excluded_records),
                "train_files": len(split_records["train"]),
                "validation_files": len(split_records["validation"]),
                "test_files": len(split_records["test"]),
            },
            "expected_stagei_count": expected_stagei_count,
            "expected_stageii_count": expected_count,
            "split_subjects": split_subjects,
            "subject_to_split": subject_to_split,
            "directory_subject_overrides": _DIRECTORY_SUBJECT_OVERRIDES,
            "category_split_counts": category_split_counts,
            "category_imbalance_note": (
                "Subject-disjoint ACCAD splits cannot preserve every category in every "
                "split; use these counts when interpreting validation/test metrics."
            ),
            "split_limitation_note": (
                "ACCAD has only three major named actors; the five sNNN subjects have "
                "one clip each, so a subject-disjoint validation split is necessarily "
                "small. Treat test(Male1) as the primary held-out-subject evaluation "
                "and consider major-actor cross-validation for final reporting."
            ),
            "duplicate_checksum_groups": duplicate_checksum_groups,
            "quality_flag_counts": dict(sorted(flag_counts.items())),
            "outputs": [
                "accad_inventory.json",
                "accad_quality.json",
                "accad_train.json",
                "accad_validation.json",
                "accad_test.json",
                "accad_excluded.json",
                "gate1_summary.json",
            ],
        }
    )

    _atomic_json_dump(output_root / "accad_inventory.json", inventory)
    _atomic_json_dump(output_root / "accad_quality.json", quality)
    _atomic_json_dump(output_root / "accad_train.json", split_payloads["train"])
    _atomic_json_dump(
        output_root / "accad_validation.json", split_payloads["validation"]
    )
    _atomic_json_dump(output_root / "accad_test.json", split_payloads["test"])
    _atomic_json_dump(output_root / "accad_excluded.json", excluded)
    _atomic_json_dump(output_root / "gate1_summary.json", summary)
    return summary


__all__ = ["audit_dataset"]
