"""Validated, vectorized motion-library access for G1 tracking policies.

This module deliberately has no Isaac Lab imports.  It loads canonical Gate-3
``accad_g1_reference_v1`` archives, verifies that every clip shares one exact
robot/action/body contract, packs them into contiguous PyTorch tensors, and
provides continuous-time batched queries suitable for an Isaac command term.

The public joint axis is always the interleaved ``policy_index`` order stored
in ``joint_names``.  It is never silently replaced with URDF/SDK order.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = "accad_g1_reference_v1"
_SHA256_LENGTH = 64

_REQUIRED_ARRAYS: dict[str, tuple[str, int]] = {
    "joint_pos": ("joint", 2),
    "joint_vel": ("joint", 2),
    "joint_acc": ("joint", 2),
    "root_pos_w": ("vec3", 2),
    "root_quat_wxyz": ("quat", 2),
    "root_lin_vel_w": ("vec3", 2),
    "root_ang_vel_w": ("vec3", 2),
    "body_pos_w": ("body_vec3", 3),
    "body_quat_wxyz": ("body_quat", 3),
    "body_lin_vel_w": ("body_vec3", 3),
    "body_ang_vel_w": ("body_vec3", 3),
    "foot_sole_position_w": ("feet", 3),
    "left_foot_contact": ("contact", 1),
    "right_foot_contact": ("contact", 1),
}

_REQUIRED_SCALARS = (
    "schema_version",
    "retarget_version",
    "fps",
    "frame_count",
    "coordinate_system",
    "quaternion_order",
    "position_unit",
    "angle_unit",
    "joint_order",
    "tracking_action_mode",
    "tracking_residual_scale",
    "tracking_action_clip",
    "tracking_normalized_delta_limit",
    "g1_contract_checksum_sha256",
    "g1_urdf_checksum_sha256",
    "correspondence_checksum_sha256",
)

_REQUIRED_NAMES = (
    "joint_names",
    "sdk_joint_names",
    "policy_to_sdk",
    "body_names",
    "tracking_body_names",
    "foot_sole_names",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scalar(arrays: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in arrays:
        raise ValueError(f"Gate-3 archive is missing scalar {name!r}")
    value = arrays[name]
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar; got shape {value.shape}")
    return value.item()


def _strings(array: np.ndarray, name: str) -> tuple[str, ...]:
    if array.ndim != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a one-dimensional string array")
    values = tuple(str(item) for item in array.tolist())
    if not values or any(not item for item in values):
        raise ValueError(f"{name} contains an empty name")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate names")
    return values


def _checksum_string(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != _SHA256_LENGTH:
        raise ValueError(f"{name} is not a 64-character SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ValueError(f"{name} is not hexadecimal") from exc
    return text.lower()


def _safe_npz(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() != ".npz" or not path.is_file():
        raise ValueError(f"motion archive must be an existing .npz file: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot safely load motion archive {path}: {exc}") from exc
    object_keys = [name for name, value in arrays.items() if value.dtype.hasobject]
    if object_keys:
        raise ValueError(f"object arrays are forbidden in Gate-3 archives: {object_keys}")
    return arrays


@dataclass(frozen=True)
class MotionContract:
    """Fields that must be identical across every packed motion."""

    schema_version: str
    retarget_version: str
    fps: float
    coordinate_system: str
    quaternion_order: str
    position_unit: str
    angle_unit: str
    joint_order: str
    joint_names: tuple[str, ...]
    sdk_joint_names: tuple[str, ...]
    policy_to_sdk: tuple[int, ...]
    body_names: tuple[str, ...]
    tracking_body_names: tuple[str, ...]
    foot_sole_names: tuple[str, ...]
    tracking_action_mode: str
    tracking_residual_scale: float
    tracking_action_clip: float
    tracking_normalized_delta_limit: float
    g1_contract_checksum_sha256: str
    g1_urdf_checksum_sha256: str
    correspondence_checksum_sha256: str

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)

    def digest(self) -> str:
        payload = {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }
        return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class _LoadedMotion:
    path: Path
    checksum_sha256: str
    key: str
    contract: MotionContract
    arrays: Mapping[str, np.ndarray]
    gate3_complete: bool

    @property
    def length(self) -> int:
        return int(self.arrays["joint_pos"].shape[0])

    @property
    def duration(self) -> float:
        return (self.length - 1) / self.contract.fps


@dataclass(frozen=True)
class ManifestSelection:
    """Deterministic result of resolving a JSON split/manifest."""

    manifest_path: Path
    manifest_checksum_sha256: str
    split: str | None
    paths: tuple[Path, ...]
    expected_checksums: Mapping[Path, str]


@dataclass(frozen=True)
class MotionQuery:
    """Continuous-time reference values.

    All tensor fields share the leading shape of the queried ``motion_id`` and
    ``time`` tensors.  ``query_future`` therefore returns ``[batch, horizon, …]``.
    """

    motion_id: torch.Tensor
    time: torch.Tensor
    phase: torch.Tensor
    frame_lower: torch.Tensor
    frame_upper: torch.Tensor
    blend: torch.Tensor
    valid: torch.Tensor
    terminal: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_acc: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_wxyz: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_wxyz: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    foot_sole_position_w: torch.Tensor
    foot_contact: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _contract_from_arrays(arrays: Mapping[str, np.ndarray]) -> MotionContract:
    for name in _REQUIRED_SCALARS:
        if name not in arrays:
            raise ValueError(f"Gate-3 archive is missing {name!r}")
    for name in _REQUIRED_NAMES:
        if name not in arrays:
            raise ValueError(f"Gate-3 archive is missing {name!r}")

    policy_to_sdk_array = np.asarray(arrays["policy_to_sdk"])
    if policy_to_sdk_array.ndim != 1 or policy_to_sdk_array.dtype.kind not in "iu":
        raise ValueError("policy_to_sdk must be a one-dimensional integer array")
    policy_to_sdk = tuple(int(item) for item in policy_to_sdk_array.tolist())
    if sorted(policy_to_sdk) != list(range(len(policy_to_sdk))):
        raise ValueError("policy_to_sdk must be a complete zero-based permutation")

    contract = MotionContract(
        schema_version=str(_scalar(arrays, "schema_version")),
        retarget_version=str(_scalar(arrays, "retarget_version")),
        fps=float(_scalar(arrays, "fps")),
        coordinate_system=str(_scalar(arrays, "coordinate_system")),
        quaternion_order=str(_scalar(arrays, "quaternion_order")),
        position_unit=str(_scalar(arrays, "position_unit")),
        angle_unit=str(_scalar(arrays, "angle_unit")),
        joint_order=str(_scalar(arrays, "joint_order")),
        joint_names=_strings(arrays["joint_names"], "joint_names"),
        sdk_joint_names=_strings(arrays["sdk_joint_names"], "sdk_joint_names"),
        policy_to_sdk=policy_to_sdk,
        body_names=_strings(arrays["body_names"], "body_names"),
        tracking_body_names=_strings(arrays["tracking_body_names"], "tracking_body_names"),
        foot_sole_names=_strings(arrays["foot_sole_names"], "foot_sole_names"),
        tracking_action_mode=str(_scalar(arrays, "tracking_action_mode")),
        tracking_residual_scale=float(_scalar(arrays, "tracking_residual_scale")),
        tracking_action_clip=float(_scalar(arrays, "tracking_action_clip")),
        tracking_normalized_delta_limit=float(
            _scalar(arrays, "tracking_normalized_delta_limit")
        ),
        g1_contract_checksum_sha256=_checksum_string(
            _scalar(arrays, "g1_contract_checksum_sha256"),
            "g1_contract_checksum_sha256",
        ),
        g1_urdf_checksum_sha256=_checksum_string(
            _scalar(arrays, "g1_urdf_checksum_sha256"),
            "g1_urdf_checksum_sha256",
        ),
        correspondence_checksum_sha256=_checksum_string(
            _scalar(arrays, "correspondence_checksum_sha256"),
            "correspondence_checksum_sha256",
        ),
    )
    if contract.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Gate-3 schema {contract.schema_version!r}; expected {SCHEMA_VERSION!r}"
        )
    if not np.isfinite(contract.fps) or contract.fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if contract.quaternion_order != "wxyz":
        raise ValueError("Gate-3 quaternion_order must be wxyz")
    if contract.coordinate_system != "right-handed_x-forward_y-left_z-up":
        raise ValueError("Gate-3 coordinate system must be right-handed X-forward/Y-left/Z-up")
    if contract.joint_order != "policy_index":
        raise ValueError("Gate-3 joint_order must be policy_index")
    if contract.position_unit != "meter" or contract.angle_unit != "radian":
        raise ValueError("Gate-3 units must be meter/radian")
    if len(contract.joint_names) != 29:
        raise ValueError(f"G1 contract requires 29 policy joints; got {len(contract.joint_names)}")
    if len(contract.sdk_joint_names) != 29 or len(contract.policy_to_sdk) != 29:
        raise ValueError("SDK name/map contract must contain 29 entries")
    expected_sdk = [""] * 29
    for name, sdk_index in zip(contract.joint_names, contract.policy_to_sdk, strict=True):
        expected_sdk[sdk_index] = name
    if tuple(expected_sdk) != contract.sdk_joint_names:
        raise ValueError("sdk_joint_names does not agree with policy_to_sdk")
    if len(contract.foot_sole_names) != 2:
        raise ValueError("foot_sole_names must contain left and right sole names")
    if not set(contract.tracking_body_names).issubset(contract.body_names):
        raise ValueError("tracking_body_names must be a subset of body_names")
    action_values = (
        contract.tracking_residual_scale,
        contract.tracking_action_clip,
        contract.tracking_normalized_delta_limit,
    )
    if not all(np.isfinite(value) and value > 0.0 for value in action_values):
        raise ValueError("tracking action scale/clip/delta must be finite and positive")
    return contract


def _validate_motion(path: Path, expected_checksum: str | None = None) -> _LoadedMotion:
    checksum = _sha256(path)
    if expected_checksum is not None and checksum != expected_checksum.lower():
        raise ValueError(
            f"motion checksum mismatch for {path}: expected {expected_checksum}, got {checksum}"
        )
    arrays = _safe_npz(path)
    contract = _contract_from_arrays(arrays)
    frame_count_value = arrays.get("frame_count")
    if frame_count_value is None or frame_count_value.ndim != 0 or frame_count_value.dtype.kind not in "iu":
        raise ValueError("frame_count must be an integer scalar")
    frame_count = int(frame_count_value.item())
    if frame_count < 2:
        raise ValueError("a motion must contain at least two frames")

    time = arrays.get("time")
    if time is None or time.shape != (frame_count,) or time.dtype.kind not in "f":
        raise ValueError(f"time must be a floating [{frame_count}] array")
    if not np.isfinite(time).all():
        raise ValueError("time contains NaN or Inf")
    expected_time = np.arange(frame_count, dtype=np.float64) / contract.fps
    if not np.allclose(time, expected_time, rtol=0.0, atol=2.0e-7):
        raise ValueError("time grid must start at zero and be uniformly sampled at fps")

    joint_count = contract.num_joints
    body_count = contract.num_bodies
    expected_shapes = {
        "joint": (frame_count, joint_count),
        "vec3": (frame_count, 3),
        "quat": (frame_count, 4),
        "body_vec3": (frame_count, body_count, 3),
        "body_quat": (frame_count, body_count, 4),
        "feet": (frame_count, 2, 3),
        "contact": (frame_count,),
    }
    for name, (shape_key, ndim) in _REQUIRED_ARRAYS.items():
        if name not in arrays:
            raise ValueError(f"Gate-3 archive is missing array {name!r}")
        value = arrays[name]
        if value.ndim != ndim or value.shape != expected_shapes[shape_key]:
            raise ValueError(
                f"{name} has shape {value.shape}; expected {expected_shapes[shape_key]}"
            )
        if shape_key == "contact":
            if value.dtype.kind not in "biuf":
                raise ValueError(f"{name} must be numeric or boolean")
        elif value.dtype.kind != "f":
            raise ValueError(f"{name} must be a floating-point array")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")

    for name in ("left_foot_contact", "right_foot_contact"):
        value = arrays[name]
        if np.any((value < 0) | (value > 1)):
            raise ValueError(f"{name} must contain boolean/probability values in [0, 1]")

    for name in ("root_quat_wxyz", "body_quat_wxyz"):
        norms = np.linalg.norm(arrays[name].astype(np.float64), axis=-1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-3):
            raise ValueError(f"{name} contains non-unit quaternions")

    if "joint_pos_sdk" not in arrays:
        raise ValueError("Gate-3 archive is missing joint_pos_sdk order witness")
    joint_pos_sdk = arrays["joint_pos_sdk"]
    if joint_pos_sdk.shape != (frame_count, joint_count):
        raise ValueError("joint_pos_sdk has the wrong shape")
    expected_sdk_values = np.empty_like(arrays["joint_pos"])
    expected_sdk_values[:, np.asarray(contract.policy_to_sdk)] = arrays["joint_pos"]
    if not np.allclose(joint_pos_sdk, expected_sdk_values, rtol=0.0, atol=1.0e-6):
        raise ValueError("joint_pos_sdk does not match policy_to_sdk and joint_pos")

    if "gate3_complete" in arrays:
        complete_value = arrays["gate3_complete"]
        if complete_value.ndim != 0 or complete_value.dtype.kind != "b":
            raise ValueError("gate3_complete must be a boolean scalar")
        gate3_complete = bool(complete_value.item())
    else:
        gate3_complete = False
    key = str(_scalar(arrays, "source_stageii_file")) if "source_stageii_file" in arrays else path.name
    if not key:
        raise ValueError("motion key is empty")
    return _LoadedMotion(
        path=path,
        checksum_sha256=checksum,
        key=key,
        contract=contract,
        arrays=arrays,
        gate3_complete=gate3_complete,
    )


def _manifest_records(document: Any, split: str | None) -> tuple[list[Any], str | None]:
    selected_split = split
    if isinstance(document, list):
        return list(document), selected_split
    if not isinstance(document, dict):
        raise ValueError("motion manifest must be a JSON list or mapping")

    if "splits" in document:
        splits = document["splits"]
        if not isinstance(splits, dict):
            raise ValueError("manifest.splits must be a mapping")
        if selected_split is None:
            raise ValueError("split is required for a manifest containing multiple splits")
        if selected_split not in splits:
            raise ValueError(f"split {selected_split!r} is absent from manifest.splits")
        records = splits[selected_split]
    elif selected_split is not None and selected_split in document:
        records = document[selected_split]
    elif "motions" in document:
        records = document["motions"]
        kind = str(document.get("kind", ""))
        if kind.startswith("split:"):
            manifest_split = kind.split(":", 1)[1]
            if selected_split is not None and selected_split != manifest_split:
                raise ValueError(
                    f"requested split {selected_split!r} does not match manifest {manifest_split!r}"
                )
            selected_split = manifest_split
    else:
        raise ValueError("manifest has no motions or selectable split")
    if not isinstance(records, list):
        raise ValueError("selected manifest records must be a list")
    return list(records), selected_split


def _record_path(record: Any, reference_root: Path) -> tuple[Path, str | None]:
    expected: str | None = None
    if isinstance(record, str):
        value = record
    elif isinstance(record, dict):
        direct_keys = (
            "g1_file",
            "reference_file",
            "retargeted_file",
            "output_file",
            "path",
        )
        value = next((record[key] for key in direct_keys if record.get(key)), None)
        if value is None:
            value = record.get("source_file")
        if not isinstance(value, str) or not value:
            raise ValueError("manifest record has no usable motion path")
        for key in (
            "g1_checksum_sha256",
            "reference_checksum_sha256",
            "archive_checksum_sha256",
            "file_checksum_sha256",
        ):
            if record.get(key) is not None:
                expected = _checksum_string(record[key], key)
                break
    else:
        raise ValueError("manifest records must be strings or mappings")

    raw_path = Path(value)
    if raw_path.name.endswith("_stageii.npz"):
        raw_path = raw_path.with_name(
            raw_path.name.removesuffix("_stageii.npz") + "_g1_50hz.npz"
        )
    resolved = raw_path.resolve() if raw_path.is_absolute() else (reference_root / raw_path).resolve()
    try:
        resolved.relative_to(reference_root)
    except ValueError as exc:
        raise ValueError(f"manifest path escapes reference root: {value}") from exc
    return resolved, expected


def motion_paths_from_manifest(
    manifest_path: str | Path,
    reference_root: str | Path,
    *,
    split: str | None = None,
    require_all: bool = True,
) -> ManifestSelection:
    """Resolve a split manifest to stable, lexicographically ordered Gate-3 files.

    Gate-1 records containing ``source_file=..._stageii.npz`` are mapped to
    ``<reference_root>/<same-directory>/<stem>_g1_50hz.npz``.  A purpose-built
    Gate-3 manifest may instead use ``g1_file``/``reference_file`` and may carry
    an archive-specific SHA-256 field.
    """

    manifest = Path(manifest_path).expanduser().resolve()
    root = Path(reference_root).expanduser().resolve()
    if not manifest.is_file():
        raise ValueError(f"manifest does not exist: {manifest}")
    if not root.is_dir():
        raise ValueError(f"reference root does not exist: {root}")
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON manifest {manifest}: {exc}") from exc
    records, selected_split = _manifest_records(document, split)
    resolved_records = [_record_path(record, root) for record in records]
    resolved_records.sort(key=lambda item: item[0].as_posix())
    paths = tuple(path for path, _ in resolved_records)
    if len(paths) != len(set(paths)):
        raise ValueError("manifest resolves the same Gate-3 file more than once")
    missing = [path for path in paths if not path.is_file()]
    if missing and require_all:
        sample = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} manifest Gate-3 files are missing; first entries: {sample}"
        )
    if missing:
        missing_set = set(missing)
        resolved_records = [item for item in resolved_records if item[0] not in missing_set]
        paths = tuple(path for path, _ in resolved_records)
    expected = {path: checksum for path, checksum in resolved_records if checksum is not None}
    return ManifestSelection(
        manifest_path=manifest,
        manifest_checksum_sha256=_sha256(manifest),
        split=selected_split,
        paths=paths,
        expected_checksums=expected,
    )


def _normalize_quaternion(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1.0e-12)


def _quat_slerp_shortest(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    q0 = _normalize_quaternion(q0)
    q1 = _normalize_quaternion(q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(0.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    alpha = blend
    while alpha.ndim < q0.ndim:
        alpha = alpha.unsqueeze(-1)
    safe = sin_theta.abs() > 1.0e-6
    denominator = torch.where(safe, sin_theta, torch.ones_like(sin_theta))
    spherical = (
        torch.sin((1.0 - alpha) * theta) / denominator * q0
        + torch.sin(alpha * theta) / denominator * q1
    )
    linear = (1.0 - alpha) * q0 + alpha * q1
    return _normalize_quaternion(torch.where(safe, spherical, linear))


class MotionLibrary:
    """Packed Gate-3 G1 references with vectorized continuous-time queries."""

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        expected_checksums: Mapping[str | Path, str] | None = None,
        manifest_checksum_sha256: str | None = None,
        split: str | None = None,
        require_gate3_complete: bool = False,
        reject_duplicate_content: bool = True,
    ):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("MotionLibrary dtype must be torch.float32 or torch.float64")
        if isinstance(paths, (str, Path)):
            path_values = [paths]
        else:
            path_values = list(paths)
        resolved = sorted(
            (Path(path).expanduser().resolve() for path in path_values),
            key=lambda path: path.as_posix(),
        )
        if not resolved:
            raise ValueError("MotionLibrary requires at least one Gate-3 archive")
        if len(resolved) != len(set(resolved)):
            raise ValueError("MotionLibrary paths contain duplicates")

        expected_by_path = {
            Path(path).expanduser().resolve(): _checksum_string(value, "expected checksum")
            for path, value in (expected_checksums or {}).items()
        }
        unknown_expected = set(expected_by_path) - set(resolved)
        if unknown_expected:
            raise ValueError(f"expected checksums include unloaded paths: {sorted(unknown_expected)}")
        motions = [
            _validate_motion(path, expected_by_path.get(path))
            for path in resolved
        ]
        first_contract = motions[0].contract
        for motion in motions[1:]:
            if motion.contract != first_contract:
                differing = [
                    field.name
                    for field in fields(MotionContract)
                    if getattr(motion.contract, field.name) != getattr(first_contract, field.name)
                ]
                raise ValueError(
                    f"motion contract mismatch in {motion.path}; differing fields: {differing}"
                )
        keys = [motion.key for motion in motions]
        if len(keys) != len(set(keys)):
            raise ValueError("motion source keys are not unique")
        checksums = [motion.checksum_sha256 for motion in motions]
        if reject_duplicate_content and len(checksums) != len(set(checksums)):
            raise ValueError("motion library contains checksum-identical archives")
        if require_gate3_complete and not all(motion.gate3_complete for motion in motions):
            incomplete = [motion.path for motion in motions if not motion.gate3_complete]
            raise ValueError(f"Gate-3 completion is required; incomplete files: {incomplete}")

        self.device = torch.device(device)
        self.dtype = dtype
        self.contract = first_contract
        self.split = split
        self.manifest_checksum_sha256 = manifest_checksum_sha256
        self.paths = tuple(motion.path for motion in motions)
        self.motion_keys = tuple(keys)
        self.checksums_sha256 = tuple(checksums)
        self.gate3_complete = tuple(motion.gate3_complete for motion in motions)

        lengths_cpu = torch.tensor([motion.length for motion in motions], dtype=torch.long)
        offsets_cpu = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(lengths_cpu, dim=0)[:-1]))
        self.lengths = lengths_cpu.to(self.device)
        self.offsets = offsets_cpu.to(self.device)
        self.durations = ((lengths_cpu - 1).to(torch.float64) / self.contract.fps).to(
            device=self.device, dtype=self.dtype
        )

        continuous_names = (
            "joint_pos",
            "joint_vel",
            "joint_acc",
            "root_pos_w",
            "root_quat_wxyz",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "body_pos_w",
            "body_quat_wxyz",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "foot_sole_position_w",
        )
        self._packed: dict[str, torch.Tensor] = {}
        for name in continuous_names:
            packed = np.concatenate(
                [np.asarray(motion.arrays[name], dtype=np.float64) for motion in motions],
                axis=0,
            )
            self._packed[name] = torch.as_tensor(packed, device=self.device, dtype=self.dtype)
        contacts = np.concatenate(
            [
                np.stack(
                    (
                        np.asarray(motion.arrays["left_foot_contact"], dtype=np.float32),
                        np.asarray(motion.arrays["right_foot_contact"], dtype=np.float32),
                    ),
                    axis=-1,
                )
                for motion in motions
            ],
            axis=0,
        )
        self._foot_contact = torch.as_tensor(contacts, device=self.device, dtype=self.dtype)

        identity = "\n".join(
            [
                f"contract {self.contract.digest()}",
                f"manifest {self.manifest_checksum_sha256 or '-'}",
                f"split {self.split or '-'}",
                *(f"{checksum} {key}" for checksum, key in zip(checksums, keys, strict=True)),
            ]
        )
        self.library_checksum_sha256 = _sha256_text(identity)

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        pattern: str = "*_g1_50hz.npz",
        **kwargs: Any,
    ) -> "MotionLibrary":
        directory = Path(root).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"motion directory does not exist: {directory}")
        return cls(sorted(directory.rglob(pattern)), **kwargs)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        reference_root: str | Path,
        *,
        split: str | None = None,
        require_all: bool = True,
        **kwargs: Any,
    ) -> "MotionLibrary":
        selection = motion_paths_from_manifest(
            manifest_path,
            reference_root,
            split=split,
            require_all=require_all,
        )
        return cls(
            selection.paths,
            expected_checksums=selection.expected_checksums,
            manifest_checksum_sha256=selection.manifest_checksum_sha256,
            split=selection.split,
            **kwargs,
        )

    @property
    def num_motions(self) -> int:
        return len(self.paths)

    @property
    def fps(self) -> float:
        return self.contract.fps

    @property
    def dt(self) -> float:
        return self.contract.dt

    def motion_id(self, key: str) -> int:
        try:
            return self.motion_keys.index(key)
        except ValueError as exc:
            raise KeyError(f"unknown motion key: {key}") from exc

    def valid_motion_mask(self, future_horizon_s: float = 0.0) -> torch.Tensor:
        horizon = float(future_horizon_s)
        if not np.isfinite(horizon) or horizon < 0.0:
            raise ValueError("future_horizon_s must be finite and non-negative")
        return self.durations >= horizon

    def valid_start_time_bounds(self, future_horizon_s: float = 0.0) -> torch.Tensor:
        """Return ``[0, duration-horizon]`` per motion; invalid highs are ``-1``."""

        horizon = float(future_horizon_s)
        mask = self.valid_motion_mask(horizon)
        high = self.durations - horizon
        high = torch.where(mask, high, torch.full_like(high, -1.0))
        return torch.stack((torch.zeros_like(high), high), dim=-1)

    def sample_start_times(
        self,
        motion_ids: torch.Tensor | Sequence[int],
        *,
        future_horizon_s: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        ids = torch.as_tensor(motion_ids, device=self.device, dtype=torch.long)
        self._validate_ids(ids)
        horizon = float(future_horizon_s)
        if not np.isfinite(horizon) or horizon < 0.0:
            raise ValueError("future_horizon_s must be finite and non-negative")
        maximum = self.durations[ids] - horizon
        if torch.any(maximum < 0.0):
            bad = ids[maximum < 0.0].detach().cpu().tolist()
            raise ValueError(f"motions are shorter than requested future horizon: {bad}")
        random = torch.rand(ids.shape, device=self.device, dtype=self.dtype, generator=generator)
        return random * maximum

    def sample(
        self,
        batch_size: int,
        *,
        future_horizon_s: float = 0.0,
        generator: torch.Generator | None = None,
        contact_mode: str = "nearest",
        contact_threshold: float = 0.5,
    ) -> MotionQuery:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        valid_ids = torch.where(self.valid_motion_mask(future_horizon_s))[0]
        if valid_ids.numel() == 0:
            raise ValueError("no motion is long enough for the requested future horizon")
        choices = torch.randint(
            valid_ids.numel(),
            (batch_size,),
            device=self.device,
            generator=generator,
        )
        ids = valid_ids[choices]
        times = self.sample_start_times(
            ids,
            future_horizon_s=future_horizon_s,
            generator=generator,
        )
        return self.query(
            ids,
            times,
            contact_mode=contact_mode,
            contact_threshold=contact_threshold,
        )

    def _validate_ids(self, ids: torch.Tensor) -> None:
        if ids.numel() == 0:
            return
        if torch.any(ids < 0) or torch.any(ids >= self.num_motions):
            raise IndexError(f"motion IDs must be in [0, {self.num_motions - 1}]")

    @staticmethod
    def _linear(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        alpha = blend
        while alpha.ndim < q0.ndim:
            alpha = alpha.unsqueeze(-1)
        return torch.lerp(q0, q1, alpha)

    def query(
        self,
        motion_ids: torch.Tensor | Sequence[int] | int,
        times: torch.Tensor | Sequence[float] | float,
        *,
        clamp: bool = True,
        contact_mode: str = "nearest",
        contact_threshold: float = 0.5,
    ) -> MotionQuery:
        """Query references at continuous seconds from each clip's first frame."""

        ids = torch.as_tensor(motion_ids, device=self.device, dtype=torch.long)
        query_time = torch.as_tensor(times, device=self.device, dtype=self.dtype)
        try:
            ids, query_time = torch.broadcast_tensors(ids, query_time)
        except RuntimeError as exc:
            raise ValueError("motion_ids and times are not broadcast-compatible") from exc
        self._validate_ids(ids)
        if not torch.isfinite(query_time).all():
            raise ValueError("query times contain NaN or Inf")
        durations = self.durations[ids]
        valid = (query_time >= 0.0) & (query_time <= durations)
        terminal = query_time >= durations
        if not clamp and not bool(valid.all()):
            raise ValueError("query time lies outside a motion's valid interval")
        clamped_time = torch.minimum(torch.clamp_min(query_time, 0.0), durations)

        frame = clamped_time * self.fps
        lower_local = torch.floor(frame).to(torch.long)
        last_local = self.lengths[ids] - 1
        lower_local = torch.minimum(lower_local, last_local)
        upper_local = torch.minimum(lower_local + 1, last_local)
        blend = (frame - lower_local.to(self.dtype)).clamp(0.0, 1.0)
        lower = self.offsets[ids] + lower_local
        upper = self.offsets[ids] + upper_local

        values: dict[str, torch.Tensor] = {}
        for name, packed in self._packed.items():
            low = packed[lower]
            high = packed[upper]
            if name in ("root_quat_wxyz", "body_quat_wxyz"):
                values[name] = _quat_slerp_shortest(low, high, blend)
            else:
                values[name] = self._linear(low, high, blend)

        if contact_mode == "nearest":
            contact_index = torch.where(blend < 0.5, lower, upper)
            contact = self._foot_contact[contact_index] >= contact_threshold
        elif contact_mode == "threshold":
            contact_probability = self._linear(
                self._foot_contact[lower], self._foot_contact[upper], blend
            )
            contact = contact_probability >= contact_threshold
        else:
            raise ValueError("contact_mode must be 'nearest' or 'threshold'")
        if not np.isfinite(contact_threshold) or not 0.0 <= contact_threshold <= 1.0:
            raise ValueError("contact_threshold must lie in [0, 1]")

        phase = torch.where(
            durations > 0.0,
            clamped_time / durations.clamp_min(torch.finfo(self.dtype).eps),
            torch.zeros_like(clamped_time),
        )
        return MotionQuery(
            motion_id=ids,
            time=clamped_time,
            phase=phase,
            frame_lower=lower_local,
            frame_upper=upper_local,
            blend=blend,
            valid=valid,
            terminal=terminal,
            joint_pos=values["joint_pos"],
            joint_vel=values["joint_vel"],
            joint_acc=values["joint_acc"],
            root_pos_w=values["root_pos_w"],
            root_quat_wxyz=values["root_quat_wxyz"],
            root_lin_vel_w=values["root_lin_vel_w"],
            root_ang_vel_w=values["root_ang_vel_w"],
            body_pos_w=values["body_pos_w"],
            body_quat_wxyz=values["body_quat_wxyz"],
            body_lin_vel_w=values["body_lin_vel_w"],
            body_ang_vel_w=values["body_ang_vel_w"],
            foot_sole_position_w=values["foot_sole_position_w"],
            foot_contact=contact,
        )

    def query_future(
        self,
        motion_ids: torch.Tensor | Sequence[int] | int,
        times: torch.Tensor | Sequence[float] | float,
        horizons_s: torch.Tensor | Sequence[float],
        *,
        clamp: bool = True,
        contact_mode: str = "nearest",
        contact_threshold: float = 0.5,
    ) -> MotionQuery:
        """Query current/future references with a final horizon dimension."""

        ids = torch.as_tensor(motion_ids, device=self.device, dtype=torch.long)
        base_time = torch.as_tensor(times, device=self.device, dtype=self.dtype)
        try:
            ids, base_time = torch.broadcast_tensors(ids, base_time)
        except RuntimeError as exc:
            raise ValueError("motion_ids and times are not broadcast-compatible") from exc
        horizons = torch.as_tensor(horizons_s, device=self.device, dtype=self.dtype)
        if horizons.ndim != 1 or horizons.numel() == 0:
            raise ValueError("horizons_s must be a non-empty one-dimensional sequence")
        if not torch.isfinite(horizons).all() or torch.any(horizons < 0.0):
            raise ValueError("future horizons must be finite and non-negative")
        expanded_ids = ids.unsqueeze(-1).expand(ids.shape + (horizons.numel(),))
        future_times = base_time.unsqueeze(-1) + horizons
        return self.query(
            expanded_ids,
            future_times,
            clamp=clamp,
            contact_mode=contact_mode,
            contact_threshold=contact_threshold,
        )


__all__ = [
    "ManifestSelection",
    "MotionContract",
    "MotionLibrary",
    "MotionQuery",
    "SCHEMA_VERSION",
    "motion_paths_from_manifest",
]
