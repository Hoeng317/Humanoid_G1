from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from accad_g1.human import (  # noqa: E402
    HAND_POSE_SIZE,
    StageIIFormatError,
    _continuous_quaternion_wxyz,
    _global_rotations,
    _infer_ground_contacts,
    _resample_boolean_majority,
    _resample_axis_angle,
    _resample_translation,
    _rotation_matrices,
    _target_sample_times,
    check_human_prerequisites,
    load_stageii_numeric,
)


def _stageii_payload(frame_count: int = 4) -> dict[str, np.ndarray]:
    root = np.zeros((frame_count, 3), dtype=np.float64)
    body = np.zeros((frame_count, 63), dtype=np.float64)
    jaw = np.zeros((frame_count, 3), dtype=np.float64)
    eye = np.zeros((frame_count, 6), dtype=np.float64)
    hand = np.arange(frame_count * 90, dtype=np.float64).reshape(frame_count, 90)
    return {
        "gender": np.asarray("neutral"),
        "surface_model_type": np.asarray("smplx"),
        "mocap_frame_rate": np.asarray(120.0),
        "mocap_time_length": np.asarray(frame_count / 120.0),
        "trans": np.zeros((frame_count, 3), dtype=np.float64),
        "poses": np.concatenate((root, body, jaw, eye, hand), axis=1),
        "betas": np.zeros(16, dtype=np.float64),
        "num_betas": np.asarray(16, dtype=np.int64),
        "root_orient": root,
        "pose_body": body,
        "pose_hand": hand,
        "pose_jaw": jaw,
        "pose_eye": eye,
        # A real stage-II archive can have object marker metadata.  Successful
        # loading proves the strict loader did not index this untrusted field.
        "markers_latent_vids": np.asarray({"must_not_be_loaded": True}, dtype=object),
    }


def test_loader_uses_no_pickle_and_splits_hands_exactly(tmp_path: Path) -> None:
    path = tmp_path / "motion_stageii.npz"
    payload = _stageii_payload()
    np.savez(path, **payload)

    motion = load_stageii_numeric(path)

    assert motion["frame_count"] == 4
    assert motion["left_hand_pose"].shape == (4, HAND_POSE_SIZE)
    assert motion["right_hand_pose"].shape == (4, HAND_POSE_SIZE)
    np.testing.assert_array_equal(
        motion["left_hand_pose"], payload["pose_hand"][:, :45]
    )
    np.testing.assert_array_equal(
        motion["right_hand_pose"], payload["pose_hand"][:, 45:]
    )
    assert motion["full_pose"].shape == (4, 165)


def test_loader_rejects_wrong_hand_width(tmp_path: Path) -> None:
    path = tmp_path / "bad_hand_stageii.npz"
    payload = _stageii_payload()
    payload["pose_hand"] = np.zeros((4, 89), dtype=np.float64)
    np.savez(path, **payload)

    with pytest.raises(StageIIFormatError, match="pose_hand"):
        load_stageii_numeric(path)


def test_loader_rejects_component_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad_components_stageii.npz"
    payload = _stageii_payload()
    payload["poses"] = payload["poses"].copy()
    payload["poses"][0, 100] = 99.0
    np.savez(path, **payload)

    with pytest.raises(StageIIFormatError, match="inconsistent"):
        load_stageii_numeric(path)


def test_preflight_reports_missing_licensed_model_and_prepends_vendor(
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "pipeline"
    vendor = pipeline / "vendor"
    model_root = pipeline / "body_models"
    vendor.mkdir(parents=True)
    original_path = list(sys.path)
    try:
        report = check_human_prerequisites(pipeline, model_root, vendor)
        assert report["ok"] is False
        assert report["vendor_added_to_sys_path"] is True
        assert sys.path[0] == str(vendor.resolve())
        assert report["model"]["available"] is False
        assert report["model"]["path"].endswith(
            "body_models/smplx/SMPLX_NEUTRAL.npz"
        )
        assert any("SMPLX_NEUTRAL.npz" in item for item in report["next_actions"])
    finally:
        sys.path[:] = original_path


def test_preflight_inspects_present_but_invalid_model(tmp_path: Path) -> None:
    pipeline = tmp_path / "pipeline"
    vendor = pipeline / "vendor"
    model_root = pipeline / "body_models"
    model_dir = model_root / "smplx"
    model_dir.mkdir(parents=True)
    np.savez(
        model_dir / "SMPLX_NEUTRAL.npz",
        v_template=np.zeros((10, 3), dtype=np.float32),
    )

    report = check_human_prerequisites(pipeline, model_root, vendor)

    assert report["ok"] is False
    assert report["model"]["available"] is True
    assert report["model"]["valid"] is False
    assert "missing SMPL-X model keys" in report["model"]["error"]


def test_production_matrix_conversion_enforces_quaternion_sign_continuity() -> None:
    # Matrix -> quaternion has an arbitrary q/-q choice.  A smooth full turn
    # reliably crosses SciPy's representation branch and used to create a sign
    # jump in the exact production conversion path.
    local_axis_angle = np.zeros((1001, 1, 3), dtype=np.float64)
    local_axis_angle[:, 0, 2] = np.linspace(0.0, 2.0 * np.pi, 1001)
    quaternions = _continuous_quaternion_wxyz(
        _rotation_matrices(local_axis_angle)
    )
    adjacent_dot = np.sum(quaternions[1:] * quaternions[:-1], axis=-1)
    assert np.all(adjacent_dot >= 0.0)


def test_human_resampling_wrappers_and_parent_accumulation() -> None:
    frames = 241
    source_fps = 120.0
    target_times = _target_sample_times(frames, source_fps, 50.0)
    np.testing.assert_allclose(np.diff(target_times), 0.02)
    assert target_times.shape == (101,)

    source_times = np.arange(frames, dtype=np.float64) / source_fps
    translation = np.column_stack((source_times, source_times**2, source_times * 0.0))
    sampled_translation = _resample_translation(
        translation, source_fps, target_times
    )
    assert sampled_translation.shape == (101, 3)
    assert np.isfinite(sampled_translation).all()

    local_axis_angle = np.zeros((frames, 2, 3), dtype=np.float64)
    local_axis_angle[:, 0, 2] = np.linspace(0.0, np.pi, frames)
    sampled_axis_angle = _resample_axis_angle(
        local_axis_angle, source_fps, target_times
    )
    assert sampled_axis_angle.shape == (101, 2, 3)
    assert np.isfinite(sampled_axis_angle).all()

    local_matrices = _rotation_matrices(sampled_axis_angle[:3])
    global_matrices = _global_rotations(local_matrices, np.asarray([-1, 0]))
    np.testing.assert_allclose(global_matrices[:, 0], local_matrices[:, 0])
    np.testing.assert_allclose(
        global_matrices[:, 1], local_matrices[:, 0] @ local_matrices[:, 1]
    )


def test_name_verified_ground_and_contact_hysteresis() -> None:
    names = [
        "left_heel",
        "left_big_toe",
        "left_small_toe",
        "right_heel",
        "right_big_toe",
        "right_small_toe",
    ]
    frames = 60
    positions = np.zeros((frames, len(names), 3), dtype=np.float32)
    # Stable contact for the first half, then a clear swing above the floor.
    positions[:30, :, 2] = 0.01
    positions[30:, :, 0] = np.linspace(0.0, 0.4, 30)[:, None]
    positions[30:, :, 2] = np.linspace(0.09, 0.16, 30)[:, None]

    result = _infer_ground_contacts(positions, names, 50.0)

    assert abs(result["ground_height_w"] - 0.01) < 1.0e-5
    assert result["contact_keypoint_position_w"].shape == (frames, 4, 3)
    for key in (
        "left_heel_contact",
        "left_toe_contact",
        "right_heel_contact",
        "right_toe_contact",
    ):
        contact = result[key]
        assert contact.dtype == np.bool_
        assert np.all(contact[2:27])
        assert not np.any(contact[35:])


def test_adaptive_ground_tracks_slow_fit_drift() -> None:
    names = [
        "left_heel",
        "left_big_toe",
        "left_small_toe",
        "right_heel",
        "right_big_toe",
        "right_small_toe",
    ]
    frames = 400
    positions = np.zeros((frames, len(names), 3), dtype=np.float32)
    drift = np.linspace(0.01, 0.08, frames, dtype=np.float32)
    positions[:, :, 2] = drift[:, None]

    result = _infer_ground_contacts(positions, names, 50.0)

    baseline = result["ground_baseline_height_w"]
    assert baseline.shape == (frames,)
    assert 0.06 < float(baseline[-1] - baseline[0]) < 0.08
    assert np.mean(result["ground_baseline_confidence"]) > 0.95
    for key in (
        "left_heel_contact",
        "left_toe_contact",
        "right_heel_contact",
        "right_toe_contact",
    ):
        assert np.mean(result[key]) > 0.95


def test_adaptive_ground_does_not_follow_a_slow_jump_apex() -> None:
    names = [
        "left_heel",
        "left_big_toe",
        "left_small_toe",
        "right_heel",
        "right_big_toe",
        "right_small_toe",
    ]
    fps = 50.0
    frames = 250
    positions = np.zeros((frames, len(names), 3), dtype=np.float32)
    height = np.zeros(frames, dtype=np.float32)
    height[50:100] = np.linspace(0.0, 0.5, 50, dtype=np.float32)
    height[100:150] = 0.5
    height[150:200] = np.linspace(0.5, 0.0, 50, dtype=np.float32)
    positions[:, :, 2] = height[:, None]

    result = _infer_ground_contacts(positions, names, fps)

    # A rate-limited lower envelope stays near the physical floor even though
    # every foot is slow at the 0.5 m apex.
    assert float(np.max(result["ground_baseline_height_w"][90:160])) < 0.04
    for key in (
        "left_heel_contact",
        "left_toe_contact",
        "right_heel_contact",
        "right_toe_contact",
    ):
        assert not np.any(result[key][105:145])


def test_contact_resampling_uses_majority_and_nearest_tie() -> None:
    source = np.asarray([False, True, True, False, False, True], dtype=bool)
    target_times = np.asarray([0.0, 0.2, 0.4], dtype=np.float64)

    sampled = _resample_boolean_majority(source, 10.0, target_times, 5.0)

    np.testing.assert_array_equal(sampled, np.asarray([False, True, False]))
