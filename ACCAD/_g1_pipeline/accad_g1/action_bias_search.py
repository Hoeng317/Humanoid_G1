"""Pure helpers for deterministic actor-output bias search.

This module deliberately has no Isaac Lab imports.  Candidate construction,
ranking, and checkpoint patching can therefore be regression-tested on CPU.
The corresponding live simulator entry point is ``search_action_bias.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch


_ACTOR_BIAS_PATTERN = re.compile(
    r"^(?P<prefix>(?:.+\.)?actor)\.(?P<index>\d+)\.bias$"
)
BIAS_MODES = ("constant", "temporal_gaussian")


@dataclass(frozen=True)
class ActorOutputBiasSpec:
    """Resolved final actor linear layer inside an RSL-RL checkpoint."""

    bias_key: str
    weight_key: str
    actor_prefix: str
    module_index: int
    action_dim: int
    input_dim: int

    def metadata(self) -> dict[str, Any]:
        return {
            "bias_key": self.bias_key,
            "weight_key": self.weight_key,
            "actor_prefix": self.actor_prefix,
            "module_index": self.module_index,
            "action_dim": self.action_dim,
            "input_dim": self.input_dim,
        }


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def generate_action_bias_candidates(
    *,
    num_candidates: int,
    action_dim: int,
    std: float,
    seed: int,
    antithetic: bool = True,
) -> torch.Tensor:
    """Return a deterministic CPU float32 candidate table.

    Candidate zero is always the unmodified policy.  With antithetic sampling,
    subsequent rows are emitted as ``(+epsilon, -epsilon)`` pairs.  An even
    candidate count leaves one final seeded Gaussian row unpaired because the
    zero baseline consumes one row.
    """

    count = _positive_int(num_candidates, name="num_candidates")
    dimension = _positive_int(action_dim, name="action_dim")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    sigma = float(std)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("std must be finite and positive")
    if not isinstance(antithetic, bool):
        raise ValueError("antithetic must be boolean")

    candidates = torch.zeros((count, dimension), dtype=torch.float32, device="cpu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    candidate_id = 1
    while candidate_id < count:
        sample = torch.randn(dimension, generator=generator, dtype=torch.float32) * sigma
        candidates[candidate_id].copy_(sample)
        candidate_id += 1
        if antithetic and candidate_id < count:
            candidates[candidate_id].copy_(-sample)
            candidate_id += 1
    if not bool(torch.isfinite(candidates).all()):
        raise RuntimeError("generated action-bias table contains NaN or Inf")
    return candidates


def generate_temporal_action_bias_schedule(
    *,
    num_steps: int,
    num_candidates: int,
    action_dim: int,
    std: float,
    seed: int,
    antithetic: bool = True,
) -> torch.Tensor:
    """Return a reproducible ``[step, candidate, action]`` dither schedule.

    Candidate zero is identically zero at every step.  Each antithetic pair is
    formed from one complete independently sampled trajectory, so candidate
    ``2k`` is the exact negative of candidate ``2k-1`` over the full rollout.
    """

    steps = _positive_int(num_steps, name="num_steps")
    count = _positive_int(num_candidates, name="num_candidates")
    dimension = _positive_int(action_dim, name="action_dim")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    sigma = float(std)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("std must be finite and positive")
    if not isinstance(antithetic, bool):
        raise ValueError("antithetic must be boolean")

    schedule = torch.zeros(
        (steps, count, dimension), dtype=torch.float32, device="cpu"
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    candidate_id = 1
    while candidate_id < count:
        trajectory = (
            torch.randn(
                (steps, dimension), generator=generator, dtype=torch.float32
            )
            * sigma
        )
        schedule[:, candidate_id, :].copy_(trajectory)
        candidate_id += 1
        if antithetic and candidate_id < count:
            schedule[:, candidate_id, :].copy_(-trajectory)
            candidate_id += 1
    if not bool(torch.isfinite(schedule).all()):
        raise RuntimeError("generated temporal action-bias schedule contains NaN or Inf")
    return schedule


def validate_bias_mode(value: str) -> str:
    """Validate the public CLI/runtime mode without silently coercing values."""

    if value not in BIAS_MODES:
        raise ValueError(f"bias_mode must be one of {BIAS_MODES}, got {value!r}")
    return value


def checkpoint_patch_allowed(bias_mode: str) -> bool:
    """Only a constant output bias is equivalent to a final-layer bias edit."""

    return validate_bias_mode(bias_mode) == "constant"


def validate_target_yaw(value: float) -> float:
    """Return a finite fixed target yaw within the command's principal range."""

    if isinstance(value, bool):
        raise ValueError("target_yaw_rad must be numeric")
    yaw = float(value)
    if not math.isfinite(yaw) or not -math.pi <= yaw <= math.pi:
        raise ValueError("target_yaw_rad must be finite and within [-pi, pi]")
    return yaw


def bias_for_control_step(
    bias_tensor: torch.Tensor,
    *,
    bias_mode: str,
    step: int,
    num_candidates: int,
    action_dim: int,
) -> torch.Tensor:
    """Resolve the exact ``[candidate, action]`` tensor for one control step."""

    mode = validate_bias_mode(bias_mode)
    count = _positive_int(num_candidates, name="num_candidates")
    dimension = _positive_int(action_dim, name="action_dim")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if not isinstance(bias_tensor, torch.Tensor):
        raise TypeError("bias_tensor must be a torch.Tensor")
    if mode == "constant":
        expected = (count, dimension)
        if tuple(bias_tensor.shape) != expected:
            raise ValueError(
                f"constant bias tensor must have shape {expected}, got {tuple(bias_tensor.shape)}"
            )
        resolved = bias_tensor
    else:
        if bias_tensor.ndim != 3 or tuple(bias_tensor.shape[1:]) != (count, dimension):
            raise ValueError(
                "temporal bias tensor must have shape "
                f"[steps, {count}, {dimension}], got {tuple(bias_tensor.shape)}"
            )
        if step >= bias_tensor.shape[0]:
            raise IndexError(
                f"temporal bias schedule has {bias_tensor.shape[0]} steps, requested {step}"
            )
        resolved = bias_tensor[step]
    if not bool(torch.isfinite(resolved).all()):
        raise ValueError("resolved action bias contains NaN or Inf")
    return resolved


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and exact contiguous CPU bytes."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a torch.Tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def bias_statistics(value: torch.Tensor) -> dict[str, Any]:
    """Return JSON-safe magnitude statistics for one finite bias vector."""

    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() == 0:
        raise ValueError("bias must be a non-empty rank-one torch.Tensor")
    vector = value.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("bias contains NaN or Inf")
    return {
        "dimension": int(vector.numel()),
        "mean_absolute": float(vector.abs().mean().item()),
        "l2_norm": float(torch.linalg.vector_norm(vector).item()),
        "max_absolute": float(vector.abs().max().item()),
        "checksum_sha256": tensor_sha256(value),
    }


def temporal_bias_statistics(value: torch.Tensor) -> dict[str, Any]:
    """Return JSON-safe statistics for one ``[step, action]`` trajectory."""

    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
    ):
        raise ValueError("temporal bias must be a non-empty rank-two torch.Tensor")
    trajectory = value.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(trajectory).all()):
        raise ValueError("temporal bias contains NaN or Inf")
    per_step_l2 = torch.linalg.vector_norm(trajectory, dim=-1)
    return {
        "num_steps": int(trajectory.shape[0]),
        "dimension": int(trajectory.shape[1]),
        "mean_absolute": float(trajectory.abs().mean().item()),
        "root_mean_square": float(torch.sqrt(trajectory.square().mean()).item()),
        "l2_norm": float(torch.linalg.vector_norm(trajectory).item()),
        "mean_per_step_l2_norm": float(per_step_l2.mean().item()),
        "max_per_step_l2_norm": float(per_step_l2.max().item()),
        "max_absolute": float(trajectory.abs().max().item()),
        "checksum_sha256": tensor_sha256(value),
    }


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def candidate_ranking_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    """Build the documented lexicographic score for one candidate record.

    Clean motion completion dominates every diagnostic metric.  Incomplete
    candidates are first compared by clip completion, then return.  Tracking
    errors and perturbation magnitude break later ties; candidate id is the
    final stable tie-breaker, preferring the lower id (and hence zero at id 0).
    """

    try:
        candidate_id = int(record["candidate_id"])
        episode = record["first_episode"]
        metrics = record["metrics"]
        bias = record["bias"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("candidate record is missing its ranking contract") from exc
    if candidate_id < 0:
        raise ValueError("candidate_id must be non-negative")
    success = episode.get("success")
    if not isinstance(success, bool):
        raise ValueError("first_episode.success must be boolean")
    completion = _finite_number(
        episode.get("completion_ratio_length_over_archive_frames"),
        name="completion ratio",
    )
    episode_return = _finite_number(episode.get("return"), name="episode return")

    def negative_error(name: str) -> float:
        value = metrics.get(name)
        if value is None:
            return -math.inf
        error = _finite_number(value, name=name)
        if error < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return -error

    bias_l2 = _finite_number(bias.get("l2_norm"), name="bias.l2_norm")
    if bias_l2 < 0.0:
        raise ValueError("bias.l2_norm must be non-negative")
    return (
        float(success),
        completion,
        episode_return,
        negative_error("root_position_rmse_m"),
        negative_error("joint_position_rmse_rad"),
        negative_error("tracking_body_position_rmse_m"),
        -bias_l2,
        -float(candidate_id),
    )


def select_best_candidate(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select one best record while rejecting duplicate candidate ids."""

    if not records:
        raise ValueError("candidate records are empty")
    seen: set[int] = set()
    for record in records:
        try:
            candidate_id = int(record["candidate_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("candidate record has an invalid candidate_id") from exc
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
    return max(records, key=candidate_ranking_key)


def locate_actor_output_bias(
    model_state_dict: Mapping[str, Any], *, action_dim: int
) -> ActorOutputBiasSpec:
    """Resolve the unique final ``actor.<index>.bias`` and fail closed.

    The final actor module is identified independently from tensor shape: it is
    the highest numeric module index for the unique actor prefix.  Its paired
    weight and bias must then prove the requested output dimension.  This
    avoids silently patching a hidden layer merely because dimensions happen
    to coincide.
    """

    dimension = _positive_int(action_dim, name="action_dim")
    if not isinstance(model_state_dict, Mapping):
        raise TypeError("model_state_dict must be a mapping")
    matches: list[tuple[str, int, str]] = []
    for key in model_state_dict:
        if not isinstance(key, str):
            raise TypeError("model_state_dict keys must be strings")
        match = _ACTOR_BIAS_PATTERN.fullmatch(key)
        if match is not None:
            matches.append((match.group("prefix"), int(match.group("index")), key))
    if not matches:
        raise ValueError("checkpoint has no actor.<index>.bias entries")
    prefixes = {prefix for prefix, _, _ in matches}
    if len(prefixes) != 1:
        raise ValueError(f"checkpoint has ambiguous actor prefixes: {sorted(prefixes)}")
    actor_prefix = next(iter(prefixes))
    final_index = max(index for _, index, _ in matches)
    final_matches = [entry for entry in matches if entry[1] == final_index]
    if len(final_matches) != 1:
        raise ValueError("checkpoint final actor bias key is ambiguous")
    bias_key = final_matches[0][2]
    weight_key = f"{actor_prefix}.{final_index}.weight"
    if weight_key not in model_state_dict:
        raise ValueError(f"checkpoint is missing paired final actor weight: {weight_key}")
    bias = model_state_dict[bias_key]
    weight = model_state_dict[weight_key]
    if not isinstance(bias, torch.Tensor) or bias.ndim != 1:
        raise ValueError(f"{bias_key} must be a rank-one tensor")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError(f"{weight_key} must be a rank-two tensor")
    if not bias.dtype.is_floating_point or not weight.dtype.is_floating_point:
        raise ValueError("final actor weight and bias must be floating point")
    if tuple(bias.shape) != (dimension,) or int(weight.shape[0]) != dimension:
        raise ValueError(
            "final actor output dimension mismatch: "
            f"expected {dimension}, got bias {tuple(bias.shape)} and weight {tuple(weight.shape)}"
        )
    if not bool(torch.isfinite(bias).all()) or not bool(torch.isfinite(weight).all()):
        raise ValueError("final actor weight or bias contains NaN or Inf")
    return ActorOutputBiasSpec(
        bias_key=bias_key,
        weight_key=weight_key,
        actor_prefix=actor_prefix,
        module_index=final_index,
        action_dim=dimension,
        input_dim=int(weight.shape[1]),
    )


def _model_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint must contain a model_state_dict mapping")
    return state


def _resolved_below(path: str | Path, root: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved_root = Path(root).expanduser().resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"destination must remain below {resolved_root}: {resolved}") from exc
    return resolved


def write_actor_bias_checkpoint(
    *,
    source: str | Path,
    destination: str | Path,
    action_bias: torch.Tensor,
    action_dim: int,
    allowed_output_root: str | Path,
) -> dict[str, Any]:
    """Atomically write a checkpoint whose sole model change is actor bias.

    The source is loaded onto CPU with PyTorch's weights-only unpickler and is
    never opened for writing.  Destination overwrite is refused.  The newly
    written checkpoint is reloaded, every model tensor is compared, and the
    source checksum is re-read before success metadata is returned.
    """

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file() or source_path.suffix.lower() != ".pt":
        raise FileNotFoundError(f"source checkpoint is not an existing .pt file: {source_path}")
    destination_path = _resolved_below(destination, allowed_output_root)
    if destination_path.suffix.lower() != ".pt":
        raise ValueError("destination checkpoint must use the .pt extension")
    if destination_path == source_path:
        raise ValueError("destination must differ from source checkpoint")
    if destination_path.exists():
        raise FileExistsError(destination_path)
    vector = torch.as_tensor(action_bias).detach().to(device="cpu")
    dimension = _positive_int(action_dim, name="action_dim")
    if vector.ndim != 1 or tuple(vector.shape) != (dimension,):
        raise ValueError(f"action_bias must have shape ({dimension},)")
    if not vector.dtype.is_floating_point:
        vector = vector.to(torch.float32)
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("action_bias contains NaN or Inf")

    from .training_io import sha256_file

    source_checksum_before = sha256_file(source_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    state = _model_state_dict(checkpoint)
    spec = locate_actor_output_bias(state, action_dim=dimension)
    original_bias = state[spec.bias_key].detach().cpu().clone()
    converted_delta = vector.to(dtype=original_bias.dtype)
    patched_bias = original_bias + converted_delta
    if not bool(torch.isfinite(patched_bias).all()):
        raise ValueError("patched actor output bias contains NaN or Inf")
    state[spec.bias_key] = patched_bias

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, destination_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    verified_checkpoint = torch.load(destination_path, map_location="cpu", weights_only=True)
    verified_state = _model_state_dict(verified_checkpoint)
    if tuple(verified_state.keys()) != tuple(state.keys()):
        raise RuntimeError("patched checkpoint model_state_dict key order changed")
    expected_bias = original_bias + converted_delta
    for key, original_value in _model_state_dict(
        torch.load(source_path, map_location="cpu", weights_only=True)
    ).items():
        verified_value = verified_state[key]
        if key == spec.bias_key:
            if not isinstance(verified_value, torch.Tensor) or not torch.equal(
                verified_value, expected_bias
            ):
                raise RuntimeError("patched checkpoint actor bias verification failed")
        elif isinstance(original_value, torch.Tensor):
            if not isinstance(verified_value, torch.Tensor) or not torch.equal(
                verified_value, original_value
            ):
                raise RuntimeError(f"unexpected model tensor change while patching: {key}")
        elif verified_value != original_value:
            raise RuntimeError(f"unexpected model state change while patching: {key}")
    source_checksum_after = sha256_file(source_path)
    if source_checksum_after != source_checksum_before:
        raise RuntimeError("source checkpoint checksum changed during patching")

    return {
        "source": {
            "path": str(source_path),
            "checksum_sha256_before": source_checksum_before,
            "checksum_sha256_after": source_checksum_after,
            "preserved": True,
            "bytes": source_path.stat().st_size,
        },
        "output": {
            "path": str(destination_path),
            "checksum_sha256": sha256_file(destination_path),
            "bytes": destination_path.stat().st_size,
        },
        "actor_output_layer": spec.metadata(),
        "bias_delta": {
            **bias_statistics(converted_delta),
            "values": [float(value) for value in converted_delta.tolist()],
        },
        "source_actor_bias_checksum_sha256": tensor_sha256(original_bias),
        "patched_actor_bias_checksum_sha256": tensor_sha256(expected_bias),
        "verification": {
            "weights_only_cpu_reload": True,
            "model_state_dict_keys_unchanged": True,
            "non_target_model_values_unchanged": True,
            "target_equals_source_plus_delta": True,
        },
    }


def write_actor_output_scale_checkpoint(
    *,
    source: str | Path,
    destination: str | Path,
    output_scale: float,
    action_dim: int,
    allowed_output_root: str | Path,
) -> dict[str, Any]:
    """Atomically scale only the final actor affine layer of a checkpoint.

    Multiplying both the final weight and bias by the same positive scalar is
    exactly equivalent to multiplying the deterministic actor output.  This is
    useful for validation-only action calibration: it changes neither the
    observation normalizer nor any hidden/critic parameter.  The source is
    preserved and the destination is reloaded for a tensor-by-tensor audit.
    """

    if isinstance(output_scale, bool):
        raise ValueError("output_scale must be a finite positive number")
    factor = float(output_scale)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("output_scale must be a finite positive number")
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file() or source_path.suffix.lower() != ".pt":
        raise FileNotFoundError(
            f"source checkpoint is not an existing .pt file: {source_path}"
        )
    destination_path = _resolved_below(destination, allowed_output_root)
    if destination_path.suffix.lower() != ".pt":
        raise ValueError("destination checkpoint must use the .pt extension")
    if destination_path == source_path:
        raise ValueError("destination must differ from source checkpoint")
    if destination_path.exists():
        raise FileExistsError(destination_path)
    dimension = _positive_int(action_dim, name="action_dim")

    from .training_io import sha256_file

    source_checksum_before = sha256_file(source_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    state = _model_state_dict(checkpoint)
    spec = locate_actor_output_bias(state, action_dim=dimension)
    original_weight = state[spec.weight_key].detach().cpu().clone()
    original_bias = state[spec.bias_key].detach().cpu().clone()
    scaled_weight = original_weight * factor
    scaled_bias = original_bias * factor
    if not bool(torch.isfinite(scaled_weight).all()) or not bool(
        torch.isfinite(scaled_bias).all()
    ):
        raise ValueError("scaled actor output layer contains NaN or Inf")
    state[spec.weight_key] = scaled_weight
    state[spec.bias_key] = scaled_bias

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, destination_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    verified_checkpoint = torch.load(
        destination_path, map_location="cpu", weights_only=True
    )
    verified_state = _model_state_dict(verified_checkpoint)
    source_state = _model_state_dict(
        torch.load(source_path, map_location="cpu", weights_only=True)
    )
    if tuple(verified_state.keys()) != tuple(source_state.keys()):
        raise RuntimeError("scaled checkpoint model_state_dict key order changed")
    for key, original_value in source_state.items():
        verified_value = verified_state[key]
        if key == spec.weight_key:
            expected = original_weight * factor
        elif key == spec.bias_key:
            expected = original_bias * factor
        else:
            expected = original_value
        if isinstance(expected, torch.Tensor):
            if not isinstance(verified_value, torch.Tensor) or not torch.equal(
                verified_value, expected
            ):
                raise RuntimeError(
                    f"unexpected model tensor while scaling actor output: {key}"
                )
        elif verified_value != expected:
            raise RuntimeError(
                f"unexpected model state while scaling actor output: {key}"
            )
    source_checksum_after = sha256_file(source_path)
    if source_checksum_after != source_checksum_before:
        raise RuntimeError("source checkpoint checksum changed during scaling")

    return {
        "schema_version": "accad_g1_actor_output_scale_checkpoint_v1",
        "source": {
            "path": str(source_path),
            "checksum_sha256_before": source_checksum_before,
            "checksum_sha256_after": source_checksum_after,
            "preserved": True,
            "bytes": source_path.stat().st_size,
        },
        "output": {
            "path": str(destination_path),
            "checksum_sha256": sha256_file(destination_path),
            "bytes": destination_path.stat().st_size,
        },
        "output_scale": factor,
        "actor_output_layer": spec.metadata(),
        "source_weight_checksum_sha256": tensor_sha256(original_weight),
        "scaled_weight_checksum_sha256": tensor_sha256(scaled_weight),
        "source_bias_checksum_sha256": tensor_sha256(original_bias),
        "scaled_bias_checksum_sha256": tensor_sha256(scaled_bias),
        "verification": {
            "weights_only_cpu_reload": True,
            "model_state_dict_keys_unchanged": True,
            "only_final_actor_weight_and_bias_changed": True,
            "target_equals_source_times_scale": True,
        },
    }
