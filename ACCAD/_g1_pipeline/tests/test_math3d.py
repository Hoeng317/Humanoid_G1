from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from accad_g1.math3d import (  # noqa: E402
    matrix_to_quaternion_wxyz,
    quaternion_angular_velocity_local,
    quaternion_angular_velocity_world,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_sign_continuity,
    quaternion_wxyz_to_matrix,
    resample_quaternion_slerp,
    resample_translation_antialiased,
    rotvec_to_matrix,
    rotvec_to_quaternion_wxyz,
    so3_geodesic_steps,
    source_time_grid,
    target_time_grid,
)


def _quaternions_equivalent(actual: np.ndarray, expected: np.ndarray, atol: float = 1.0e-10) -> bool:
    actual = actual / np.linalg.norm(actual, axis=-1, keepdims=True)
    expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True)
    return bool(np.all(np.isclose(np.abs(np.sum(actual * expected, axis=-1)), 1.0, atol=atol)))


def test_identity_and_rotation_round_trips() -> None:
    identity_matrix = rotvec_to_matrix(np.zeros(3))
    np.testing.assert_allclose(identity_matrix, np.eye(3), atol=1.0e-12)
    identity_q = matrix_to_quaternion_wxyz(identity_matrix)
    assert _quaternions_equivalent(identity_q, np.array([1.0, 0.0, 0.0, 0.0]))

    rotvecs = np.array(
        [
            [0.2, -0.1, 0.3],
            [np.pi - 1.0e-7, 0.0, 0.0],
            [-0.4, 0.8, -0.2],
        ]
    )
    matrices = rotvec_to_matrix(rotvecs)
    quaternions = matrix_to_quaternion_wxyz(matrices)
    np.testing.assert_allclose(quaternion_wxyz_to_matrix(quaternions), matrices, atol=1.0e-10)
    assert _quaternions_equivalent(quaternions, rotvec_to_quaternion_wxyz(rotvecs))


def test_quaternion_product_conjugate_and_sign_continuity() -> None:
    q = rotvec_to_quaternion_wxyz(np.array([0.3, -0.4, 0.2]))
    identity = quaternion_multiply(q, quaternion_conjugate(q))
    np.testing.assert_allclose(identity, [1.0, 0.0, 0.0, 0.0], atol=1.0e-12)

    sequence = np.stack((q, -q, q, -q))
    continuous = quaternion_sign_continuity(sequence)
    assert np.all(np.sum(continuous[:-1] * continuous[1:], axis=-1) >= 0.0)
    np.testing.assert_allclose(continuous, np.broadcast_to(q, continuous.shape), atol=1.0e-12)


def test_world_and_local_angular_velocity_conventions() -> None:
    times = np.linspace(0.0, 0.5, 6)
    world_rate = np.array([1.2, 0.0, 0.0])
    initial = rotvec_to_quaternion_wxyz(np.array([0.0, 0.0, np.pi / 2.0]))
    delta_world = rotvec_to_quaternion_wxyz(times[:, None] * world_rate)
    orientations = quaternion_multiply(delta_world, initial)

    measured_world = quaternion_angular_velocity_world(orientations, times)
    np.testing.assert_allclose(measured_world, np.broadcast_to(world_rate, measured_world.shape), atol=1.0e-10)

    expected_local = quaternion_wxyz_to_matrix(initial).T @ world_rate
    measured_local = quaternion_angular_velocity_local(orientations, times)
    np.testing.assert_allclose(measured_local, np.broadcast_to(expected_local, measured_local.shape), atol=1.0e-10)


def test_so3_geodesic_steps_are_angles_not_axis_angle_differences() -> None:
    angles = np.array([0.0, np.pi / 4.0, np.pi / 2.0])
    rotvecs = np.column_stack((np.zeros_like(angles), np.zeros_like(angles), angles))
    np.testing.assert_allclose(so3_geodesic_steps(rotvecs), [np.pi / 4.0, np.pi / 4.0], atol=1.0e-12)


def test_target_grid_and_slerp_preserve_duration_endpoints_and_shape() -> None:
    source = source_time_grid(241, 120.0)
    target = target_time_grid(source, 50.0)
    assert source[-1] == 2.0
    assert target.shape == (101,)
    assert target[0] == source[0]
    assert target[-1] == source[-1]

    key_times = np.array([0.0, 1.0])
    key_rotvecs = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi]])
    key_quaternions = rotvec_to_quaternion_wxyz(key_rotvecs)
    sample_times = np.array([0.0, 0.5, 1.0])
    sampled = resample_quaternion_slerp(key_quaternions, key_times, sample_times)
    assert sampled.shape == (3, 4)
    np.testing.assert_allclose(sampled[[0, -1]], key_quaternions, atol=1.0e-12)
    expected_mid = rotvec_to_matrix(np.array([0.0, 0.0, np.pi / 2.0]))
    np.testing.assert_allclose(quaternion_wxyz_to_matrix(sampled[1]), expected_mid, atol=1.0e-10)


def test_antialiased_translation_resampling_shape_endpoints_and_finiteness() -> None:
    source = source_time_grid(241, 120.0)
    target = target_time_grid(source, 50.0)
    translations = np.column_stack(
        (
            source,
            np.sin(2.0 * np.pi * 2.0 * source) + 0.2 * np.sin(2.0 * np.pi * 40.0 * source),
            0.2 * source**2,
        )
    )
    result = resample_translation_antialiased(translations, source, target)
    assert result.shape == (target.size, 3)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result[0], translations[0], atol=1.0e-12)
    np.testing.assert_allclose(result[-1], translations[-1], atol=1.0e-12)

    # A 40 Hz component is above the 25 Hz Nyquist limit at 50 Hz.  Direct
    # interpolation would alias it into the output; the prefilter must strongly
    # suppress it (ignore restored endpoints when measuring the RMS).
    high_frequency_only = np.sin(2.0 * np.pi * 40.0 * source)[:, None]
    filtered_high = resample_translation_antialiased(high_frequency_only, source, target)
    direct_high = np.interp(target, source, high_frequency_only[:, 0])
    filtered_rms = np.sqrt(np.mean(filtered_high[2:-2, 0] ** 2))
    direct_rms = np.sqrt(np.mean(direct_high[2:-2] ** 2))
    assert filtered_rms < 0.05 * direct_rms
