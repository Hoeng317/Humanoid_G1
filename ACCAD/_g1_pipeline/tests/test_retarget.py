from __future__ import annotations

import numpy as np
import torch

from accad_g1.g1_kinematics import TorchG1Kinematics
from accad_g1.retarget import (
    FOOT_COLLISION_RADIUS_M,
    _contact_runs,
    _derive_stance_channels,
    _lock_contact_point_targets,
    _project_root_above_ground,
    _refine_output_stance_channels,
    _split_long_true_runs,
)


def test_contact_runs_include_boundaries() -> None:
    mask = np.asarray([True, True, False, True, False, False, True], dtype=bool)
    assert _contact_runs(mask) == [(0, 1), (3, 3), (6, 6)]


def test_long_stance_runs_get_explicit_reanchor_boundaries() -> None:
    mask = np.ones(12, dtype=bool)
    bounded = _split_long_true_runs(mask, maximum_frames=5)
    assert _contact_runs(bounded) == [(0, 4), (6, 10)]
    assert not bounded[5]
    assert not bounded[11]


def test_heel_and_toe_targets_lock_independently() -> None:
    targets = np.zeros((6, 4, 3), dtype=np.float64)
    targets[:, :, 0] = np.arange(6)[:, None]
    targets[:, :, 2] = 0.2
    contacts = np.zeros((6, 4), dtype=bool)
    contacts[1:4, 0] = True
    contacts[3:6, 1] = True

    locked, runs = _lock_contact_point_targets(targets, contacts)

    np.testing.assert_allclose(locked[1:4, 0, 0], 2.0)
    np.testing.assert_allclose(locked[3:6, 1, 0], 4.0)
    np.testing.assert_allclose(locked[contacts, 2], FOOT_COLLISION_RADIUS_M)
    np.testing.assert_allclose(locked[0, :, 0], targets[0, :, 0])
    assert {(run["point"], run["start_frame"], run["end_frame"]) for run in runs} == {
        ("left_heel", 1, 3),
        ("left_toe", 3, 5),
    }


def test_stance_is_stricter_than_contact_for_a_sliding_toe() -> None:
    targets = np.zeros((12, 4, 3), dtype=np.float64)
    contacts = np.zeros((12, 4), dtype=bool)
    contacts[:, 1] = True
    # Continuous 0.30 m/s toe drag at 50 Hz is contact, never rigid stance.
    targets[:, 1, 0] = np.arange(12) * (0.30 / 50.0)

    stance, runs = _derive_stance_channels(
        targets,
        contacts,
        fps=50.0,
        speed_threshold_mps=0.18,
        target_drift_limit_m=0.025,
        minimum_frames=3,
    )

    assert np.all(contacts[:, 1])
    assert not np.any(stance[:, 1])
    assert not np.any(stance[:, [0, 2, 3]])
    assert runs == []


def test_long_slow_contact_is_split_into_bounded_stance_runs() -> None:
    targets = np.zeros((20, 4, 3), dtype=np.float64)
    contacts = np.zeros((20, 4), dtype=bool)
    contacts[:, 0] = True
    targets[:, 0, 0] = np.arange(20) * 0.002  # 0.10 m/s at 50 Hz.

    stance, _ = _derive_stance_channels(
        targets,
        contacts,
        fps=50.0,
        speed_threshold_mps=0.18,
        target_drift_limit_m=0.010,
        minimum_frames=3,
    )

    assert np.all(~stance[:, 1:])
    runs = _contact_runs(stance[:, 0])
    assert len(runs) >= 2
    for start, end in runs:
        assert end - start + 1 >= 3
        drift = np.linalg.norm(targets[end, 0, :2] - targets[start, 0, :2])
        assert drift <= 0.010 + 1.0e-12


def test_non_upright_contact_is_not_promoted_to_stance() -> None:
    targets = np.zeros((8, 4, 3), dtype=np.float64)
    contacts = np.ones((8, 4), dtype=bool)
    eligible = np.asarray([False, False, False, False, True, True, True, True])

    stance, _ = _derive_stance_channels(
        targets,
        contacts,
        fps=50.0,
        speed_threshold_mps=0.18,
        target_drift_limit_m=0.025,
        minimum_frames=3,
        eligible_frames=eligible,
    )

    assert not np.any(stance[:4])
    assert np.all(stance[4:])


def test_output_stance_refinement_removes_retained_g1_slide() -> None:
    points = np.zeros((16, 4, 3), dtype=np.float64)
    preliminary = np.zeros((16, 4), dtype=bool)
    preliminary[:, 1] = True
    # The source-side classifier called the whole interval stance, but the
    # final G1 toe starts sliding at 0.50 m/s halfway through the clip.
    points[8:, 1, 0] = np.arange(8) * 0.01
    solver = {
        "stance_speed_threshold_mps": 0.18,
        "stance_target_drift_limit_m": 0.025,
        "stance_minimum_frames": 3,
        "stance_maximum_anchor_frames": 50,
    }

    refined, runs = _refine_output_stance_channels(
        points, preliminary, fps=50.0, solver=solver
    )

    assert np.all(refined <= preliminary)
    assert np.any(refined[:7, 1])
    assert not np.any(refined[8:, 1])
    assert all(run["source"].startswith("stationary_final_g1") for run in runs)


def test_ground_projection_is_feasible_even_after_optimizer_limit_saturates() -> None:
    solver = TorchG1Kinematics(device="cpu", dtype=torch.float64)
    q = solver.contract.default[None].astype(np.float32)
    quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    with torch.inference_mode():
        transforms = solver.forward(
            torch.as_tensor(q, dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.float64),
            torch.as_tensor(quat, dtype=torch.float64),
        )
        nominal_min_z = float(
            solver.foot_support_positions(transforms)[..., 2].min()
        )

    # Start with the optimizer exactly at its +0.30 m correction limit while
    # the collision spheres still need another 10.5 mm to clear the margin.
    current_z = -nominal_min_z + FOOT_COLLISION_RADIUS_M - 0.010
    root = np.asarray([[0.0, 0.0, current_z]], dtype=np.float32)
    base = root.copy()
    base[:, 2] -= 0.30
    config = {
        "solver": {
            "post_projection_ground_margin_m": 0.0005,
            "root_z_correction_limit_m": 0.30,
            "nonupright_root_z_correction_limit_m": 0.40,
        }
    }
    targets = {
        "stance_eligible": np.ones(1, dtype=bool),
        "root_position": base,
    }

    projected, metadata = _project_root_above_ground(
        q, root, quat, targets, config, solver
    )

    with torch.inference_mode():
        final_support = solver.foot_support_positions(
            solver.forward(
                torch.as_tensor(q, dtype=torch.float64),
                torch.as_tensor(projected, dtype=torch.float64),
                torch.as_tensor(quat, dtype=torch.float64),
            )
        )
    assert float(final_support[..., 2].min()) >= (
        FOOT_COLLISION_RADIUS_M + 0.0005 - 1.0e-7
    )
    assert metadata["maximum_unresolved_shift_m"] == 0.0
    assert 0.010 < metadata["maximum_limit_excess_m"] < 0.011


def test_default_support_geometry_matches_urdf_spheres() -> None:
    solver = TorchG1Kinematics(device="cpu", dtype=torch.float64)
    q = torch.as_tensor(solver.contract.default[None], dtype=torch.float64)
    transforms = solver.forward(
        q,
        torch.zeros((1, 3), dtype=torch.float64),
        torch.as_tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
    )
    support = solver.foot_support_positions(transforms).numpy()
    sole = solver.sole_positions(transforms).numpy()

    assert support.shape == (1, 2, 4, 3)
    np.testing.assert_allclose(support.mean(axis=2), sole, atol=1.0e-12)
