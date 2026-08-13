"""Rotation and time-resampling utilities for the ACCAD-to-G1 pipeline.

Conventions
-----------
* Quaternions are always ``[..., (w, x, y, z)]``.
* A quaternion represents the active rotation from local/body coordinates to
  world coordinates.
* Angular-velocity functions return one value per *interval*, hence an input
  with ``T`` frames produces ``T - 1`` velocity vectors.

Pose resampling is performed with quaternion SLERP.  Axis-angle vectors must
never be interpolated component by component because that does not follow the
shortest path on SO(3).
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation


FloatArray = NDArray[np.float64]


def _as_float_array(value: ArrayLike, *, last_dim: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != last_dim:
        raise ValueError(f"{name} must have shape [..., {last_dim}], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _validate_matrices(matrices: ArrayLike) -> FloatArray:
    array = np.asarray(matrices, dtype=np.float64)
    if array.ndim < 2 or array.shape[-2:] != (3, 3):
        raise ValueError(f"matrices must have shape [..., 3, 3], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrices contains NaN or infinity")
    return array


def _validate_times(times: ArrayLike, *, name: str, allow_single: bool = True) -> FloatArray:
    array = np.asarray(times, dtype=np.float64)
    if array.ndim != 1 or (array.size < 1 if allow_single else array.size < 2):
        minimum = 1 if allow_single else 2
        raise ValueError(f"{name} must be a one-dimensional array with at least {minimum} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    if array.size > 1 and np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def normalize_quaternion_wxyz(quaternions: ArrayLike, *, eps: float = 1.0e-12) -> FloatArray:
    """Return unit ``wxyz`` quaternions without changing their signs."""

    q = _as_float_array(quaternions, last_dim=4, name="quaternions")
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms < eps):
        raise ValueError("a zero-length quaternion cannot be normalized")
    return q / norms


# Short alias used by callers that already encode the convention in a variable name.
normalize_quaternion = normalize_quaternion_wxyz


def quaternion_sign_continuity(quaternions: ArrayLike, *, axis: int = 0) -> FloatArray:
    """Flip equivalent quaternion signs so adjacent samples have nonnegative dot products.

    Magnitudes are preserved; call :func:`normalize_quaternion_wxyz` separately
    when unit quaternions are required.
    """

    q = _as_float_array(quaternions, last_dim=4, name="quaternions").copy()
    if q.ndim < 2:
        raise ValueError("quaternions must contain a sequence dimension")
    sequence_axis = np.core.numeric.normalize_axis_index(axis, q.ndim - 1)
    q = np.moveaxis(q, sequence_axis, 0)
    for frame in range(1, q.shape[0]):
        flip = np.sum(q[frame - 1] * q[frame], axis=-1, keepdims=True) < 0.0
        q[frame] = np.where(flip, -q[frame], q[frame])
    return np.moveaxis(q, 0, sequence_axis)


def rotvec_to_matrix(rotvecs: ArrayLike) -> FloatArray:
    """Convert axis-angle rotation vectors in radians to rotation matrices."""

    rotvec = _as_float_array(rotvecs, last_dim=3, name="rotvecs")
    leading_shape = rotvec.shape[:-1]
    matrices = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
    return matrices.reshape(leading_shape + (3, 3))


def matrix_to_quaternion_wxyz(matrices: ArrayLike) -> FloatArray:
    """Convert rotation matrices to normalized ``wxyz`` quaternions."""

    matrix = _validate_matrices(matrices)
    leading_shape = matrix.shape[:-2]
    quaternion_xyzw = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_quat()
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]
    return quaternion_wxyz.reshape(leading_shape + (4,))


def rotvec_to_quaternion_wxyz(rotvecs: ArrayLike) -> FloatArray:
    """Convert axis-angle rotation vectors in radians to ``wxyz`` quaternions."""

    rotvec = _as_float_array(rotvecs, last_dim=3, name="rotvecs")
    leading_shape = rotvec.shape[:-1]
    quaternion_xyzw = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_quat()
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]
    return quaternion_wxyz.reshape(leading_shape + (4,))


def quaternion_wxyz_to_matrix(quaternions: ArrayLike) -> FloatArray:
    """Convert ``wxyz`` quaternions to rotation matrices."""

    q = normalize_quaternion_wxyz(quaternions)
    leading_shape = q.shape[:-1]
    quaternion_xyzw = q.reshape(-1, 4)[:, [1, 2, 3, 0]]
    matrices = Rotation.from_quat(quaternion_xyzw).as_matrix()
    return matrices.reshape(leading_shape + (3, 3))


def quaternion_multiply(left: ArrayLike, right: ArrayLike) -> FloatArray:
    """Hamilton product ``left * right`` for broadcastable ``wxyz`` arrays."""

    q_left = _as_float_array(left, last_dim=4, name="left")
    q_right = _as_float_array(right, last_dim=4, name="right")
    w1, x1, y1, z1 = np.moveaxis(q_left, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q_right, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def quaternion_conjugate(quaternions: ArrayLike) -> FloatArray:
    """Return quaternion conjugates in ``wxyz`` order."""

    q = _as_float_array(quaternions, last_dim=4, name="quaternions")
    result = q.copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_wxyz_to_rotvec(quaternions: ArrayLike, *, eps: float = 1.0e-12) -> FloatArray:
    """Convert quaternions to shortest-path rotation vectors in radians."""

    q = normalize_quaternion_wxyz(quaternions)
    # q and -q describe the same orientation.  w >= 0 selects angles in [0, pi].
    q = np.where(q[..., :1] < 0.0, -q, q)
    vector = q[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(q[..., :1], 0.0, 1.0))
    scale = np.divide(angle, vector_norm, out=np.full_like(angle, 2.0), where=vector_norm > eps)
    return vector * scale


def _interval_durations(sample_period_or_times: Union[float, ArrayLike], frames: int) -> FloatArray:
    if frames < 2:
        return np.empty((0,), dtype=np.float64)
    value = np.asarray(sample_period_or_times, dtype=np.float64)
    if value.ndim == 0:
        period = float(value)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("sample period must be finite and positive")
        return np.full((frames - 1,), period, dtype=np.float64)
    times = _validate_times(value, name="times")
    if times.size != frames:
        raise ValueError(f"times has {times.size} values but quaternions has {frames} frames")
    return np.diff(times)


def quaternion_angular_velocity_world(
    quaternions: ArrayLike, sample_period_or_times: Union[float, ArrayLike]
) -> FloatArray:
    """Return interval angular velocity expressed in world coordinates.

    For local-to-world orientations ``q``, each interval rotation is
    ``q[t+1] * conjugate(q[t])``.  The result has shape ``[T-1, ..., 3]``.
    """

    q = normalize_quaternion_wxyz(quaternions)
    if q.ndim < 2:
        raise ValueError("quaternions must have shape [T, ..., 4]")
    q = quaternion_sign_continuity(q, axis=0)
    dt = _interval_durations(sample_period_or_times, q.shape[0])
    relative = quaternion_multiply(q[1:], quaternion_conjugate(q[:-1]))
    rotation_step = quaternion_wxyz_to_rotvec(relative)
    reshape = (dt.size,) + (1,) * (rotation_step.ndim - 1)
    return rotation_step / dt.reshape(reshape)


def quaternion_angular_velocity_local(
    quaternions: ArrayLike, sample_period_or_times: Union[float, ArrayLike]
) -> FloatArray:
    """Return interval angular velocity expressed in the preceding local frame.

    For local-to-world orientations ``q``, each interval rotation is
    ``conjugate(q[t]) * q[t+1]``.  The result has shape ``[T-1, ..., 3]``.
    """

    q = normalize_quaternion_wxyz(quaternions)
    if q.ndim < 2:
        raise ValueError("quaternions must have shape [T, ..., 4]")
    q = quaternion_sign_continuity(q, axis=0)
    dt = _interval_durations(sample_period_or_times, q.shape[0])
    relative = quaternion_multiply(quaternion_conjugate(q[:-1]), q[1:])
    rotation_step = quaternion_wxyz_to_rotvec(relative)
    reshape = (dt.size,) + (1,) * (rotation_step.ndim - 1)
    return rotation_step / dt.reshape(reshape)


def so3_geodesic_steps(rotvecs: ArrayLike) -> FloatArray:
    """Return shortest SO(3) angle between consecutive rotvec frames.

    Input shape is ``[T, ..., 3]`` and output shape is ``[T-1, ...]`` in radians.
    """

    rotvec = _as_float_array(rotvecs, last_dim=3, name="rotvecs")
    if rotvec.ndim < 2:
        raise ValueError("rotvecs must have shape [T, ..., 3]")
    q = quaternion_sign_continuity(rotvec_to_quaternion_wxyz(rotvec), axis=0)
    relative = quaternion_multiply(quaternion_conjugate(q[:-1]), q[1:])
    return np.linalg.norm(quaternion_wxyz_to_rotvec(relative), axis=-1)


def source_time_grid(num_frames: int, fps: float, *, start_time: float = 0.0) -> FloatArray:
    """Build frame timestamps using the convention ``t[i] = start + i / fps``."""

    if isinstance(num_frames, bool) or int(num_frames) != num_frames or num_frames < 1:
        raise ValueError("num_frames must be a positive integer")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(start_time):
        raise ValueError("start_time must be finite")
    return float(start_time) + np.arange(int(num_frames), dtype=np.float64) / float(fps)


def target_time_grid(source_times: ArrayLike, target_fps: float) -> FloatArray:
    """Build a near-``target_fps`` grid that exactly preserves both endpoints.

    Rounding the number of intervals and then using ``linspace`` avoids silently
    shortening a clip when its duration is not an integer multiple of the target
    frame period.
    """

    times = _validate_times(source_times, name="source_times")
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("target_fps must be finite and positive")
    if times.size == 1:
        return times.copy()
    duration = float(times[-1] - times[0])
    intervals = max(1, int(np.floor(duration * float(target_fps) + 0.5)))
    return np.linspace(times[0], times[-1], intervals + 1, dtype=np.float64)


def _check_resampling_times(source_times: ArrayLike, target_times: ArrayLike, frames: int) -> tuple[FloatArray, FloatArray]:
    source = _validate_times(source_times, name="source_times")
    target = _validate_times(target_times, name="target_times")
    if source.size != frames:
        raise ValueError(f"source_times has {source.size} values but data has {frames} frames")
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(source[0]), abs(source[-1]))
    if target[0] < source[0] - tolerance or target[-1] > source[-1] + tolerance:
        raise ValueError("target_times must lie inside the source time range")
    return source, np.clip(target, source[0], source[-1])


def resample_quaternion_slerp(
    quaternions: ArrayLike, source_times: ArrayLike, target_times: ArrayLike
) -> FloatArray:
    """Resample ``[T, ..., 4]`` quaternions with shortest-path SLERP."""

    q = normalize_quaternion_wxyz(quaternions)
    if q.ndim < 2:
        raise ValueError("quaternions must have shape [T, ..., 4]")
    source, target = _check_resampling_times(source_times, target_times, q.shape[0])
    q = quaternion_sign_continuity(q, axis=0)

    if source.size == 1:
        if not np.allclose(target, source[0]):
            raise ValueError("a one-frame sequence can only be sampled at its timestamp")
        return np.broadcast_to(q[0], (target.size,) + q.shape[1:]).copy()

    upper = np.searchsorted(source, target, side="right")
    upper = np.clip(upper, 1, source.size - 1)
    lower = upper - 1
    fraction = (target - source[lower]) / (source[upper] - source[lower])
    fraction = fraction.reshape((target.size,) + (1,) * (q.ndim - 1))

    q0 = q[lower]
    q1 = q[upper]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    # Continuity makes dot nonnegative, but clipping absorbs floating-point drift.
    dot = np.clip(dot, 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    safe_denominator = np.where(sin_theta > 1.0e-8, sin_theta, 1.0)
    spherical = (
        np.sin((1.0 - fraction) * theta) / safe_denominator * q0
        + np.sin(fraction * theta) / safe_denominator * q1
    )
    linear = (1.0 - fraction) * q0 + fraction * q1
    result = np.where(dot > 0.9995, linear, spherical)
    return normalize_quaternion_wxyz(result)


def resample_translation_antialiased(
    translations: ArrayLike,
    source_times: ArrayLike,
    target_times: ArrayLike,
    *,
    filter_order: int = 4,
    cutoff_nyquist_fraction: float = 0.9,
) -> FloatArray:
    """Low-pass and resample translation trajectories along their first axis.

    A zero-phase Butterworth filter is applied only when downsampling.  Its
    cutoff is a fraction of the target Nyquist frequency, preventing energy
    above the representable band from aliasing.  Shape ``[T, ...]`` is allowed;
    for root translation the expected shape is ``[T, 3]``.  Original endpoints
    are restored when the target grid includes them exactly.
    """

    values = np.asarray(translations, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("translations must have shape [T, ...]")
    if not np.all(np.isfinite(values)):
        raise ValueError("translations contains NaN or infinity")
    source, target = _check_resampling_times(source_times, target_times, values.shape[0])
    if not isinstance(filter_order, (int, np.integer)) or filter_order < 1:
        raise ValueError("filter_order must be a positive integer")
    if not 0.0 < cutoff_nyquist_fraction < 1.0:
        raise ValueError("cutoff_nyquist_fraction must be between zero and one")

    if source.size == 1:
        if not np.allclose(target, source[0]):
            raise ValueError("a one-frame sequence can only be sampled at its timestamp")
        return np.broadcast_to(values[0], (target.size,) + values.shape[1:]).copy()

    source_dt = np.diff(source)
    uniform_tolerance = max(1.0e-9, 1.0e-4 * float(np.median(source_dt)))
    if np.max(np.abs(source_dt - np.median(source_dt))) > uniform_tolerance:
        raise ValueError("anti-alias filtering requires uniformly sampled source_times")
    source_rate = 1.0 / float(np.median(source_dt))
    if target.size > 1:
        target_rate = (target.size - 1) / float(target[-1] - target[0])
    else:
        target_rate = source_rate

    filtered = values
    normalized_cutoff = cutoff_nyquist_fraction * target_rate / source_rate
    if target_rate < source_rate and normalized_cutoff < 1.0:
        sos = butter(int(filter_order), normalized_cutoff, btype="lowpass", output="sos")
        # scipy's default padding can exceed short motion clips.  A bounded pad
        # still provides zero-phase filtering, while very short clips fall back
        # to interpolation because they do not contain enough temporal context.
        default_pad = 3 * (2 * sos.shape[0] + 1)
        pad_length = min(default_pad, source.size - 1)
        if pad_length >= 1:
            filtered = sosfiltfilt(sos, values, axis=0, padlen=pad_length)

    result = np.asarray(PchipInterpolator(source, filtered, axis=0)(target), dtype=np.float64)
    endpoint_tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(source[0]), abs(source[-1]))
    if abs(target[0] - source[0]) <= endpoint_tolerance:
        result[0] = values[0]
    if abs(target[-1] - source[-1]) <= endpoint_tolerance:
        result[-1] = values[-1]
    return result


def _self_check() -> None:
    source = source_time_grid(121, 120.0)
    target = target_time_grid(source, 50.0)
    q = rotvec_to_quaternion_wxyz(np.column_stack((np.zeros(121), np.zeros(121), source)))
    resampled = resample_quaternion_slerp(q, source, target)
    if resampled.shape != (51, 4) or not np.isclose(target[-1], source[-1]):
        raise RuntimeError("math3d self-check failed")


if __name__ == "__main__":
    _self_check()
    print("math3d self-check: OK")
