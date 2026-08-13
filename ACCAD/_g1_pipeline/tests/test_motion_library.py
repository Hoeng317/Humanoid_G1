from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from accad_g1.motion_library import MotionLibrary, motion_paths_from_manifest


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_motion(
    path: Path,
    *,
    source: str,
    offset: float = 0.0,
    fps: float = 2.0,
    body_names: tuple[str, ...] = ("pelvis", "torso_link"),
    action_scale: float = 0.25,
    complete: bool = True,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = 3
    joints = 29
    policy_names = tuple(f"joint_{index:02d}" for index in range(joints))
    policy_to_sdk = np.arange(joints - 1, -1, -1, dtype=np.int64)
    sdk_names = [""] * joints
    for name, sdk_index in zip(policy_names, policy_to_sdk, strict=True):
        sdk_names[int(sdk_index)] = name

    frame_value = np.arange(frames, dtype=np.float32)[:, None]
    joint_pos = offset + frame_value + np.arange(joints, dtype=np.float32)[None, :] * 0.01
    joint_vel = np.broadcast_to(np.arange(joints, dtype=np.float32)[None, :] + 1.0, (frames, joints)).copy()
    joint_acc = np.zeros((frames, joints), dtype=np.float32)
    joint_pos_sdk = np.empty_like(joint_pos)
    joint_pos_sdk[:, policy_to_sdk] = joint_pos

    root_pos = np.concatenate((frame_value, frame_value * 2.0, frame_value * 0.0), axis=1)
    # Frame 1 is the antipodal representation of frame 0.  Correct shortest-
    # path SLERP must remain at identity between these two samples.
    root_quat = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    body_count = len(body_names)
    body_pos = np.zeros((frames, body_count, 3), dtype=np.float32)
    body_pos[..., 0] = frame_value
    body_pos[..., 1] = np.arange(body_count, dtype=np.float32)[None, :]
    body_quat = np.broadcast_to(root_quat[:, None, :], (frames, body_count, 4)).copy()
    body_lin_vel = np.ones((frames, body_count, 3), dtype=np.float32)
    body_ang_vel = np.zeros((frames, body_count, 3), dtype=np.float32)
    feet = np.zeros((frames, 2, 3), dtype=np.float32)
    feet[..., 0] = frame_value
    feet[:, 1, 1] = -0.2

    np.savez_compressed(
        path,
        schema_version=np.asarray("accad_g1_reference_v1"),
        retarget_version=np.asarray("test-retarget-v1"),
        fps=np.asarray(fps, dtype=np.float64),
        time=np.arange(frames, dtype=np.float64) / fps,
        frame_count=np.asarray(frames, dtype=np.int64),
        coordinate_system=np.asarray("right-handed_x-forward_y-left_z-up"),
        quaternion_order=np.asarray("wxyz"),
        position_unit=np.asarray("meter"),
        angle_unit=np.asarray("radian"),
        joint_order=np.asarray("policy_index"),
        joint_names=np.asarray(policy_names),
        sdk_joint_names=np.asarray(sdk_names),
        policy_to_sdk=policy_to_sdk,
        joint_pos=joint_pos,
        joint_pos_sdk=joint_pos_sdk,
        joint_vel=joint_vel,
        joint_acc=joint_acc,
        root_pos_w=root_pos.astype(np.float32),
        root_quat_wxyz=root_quat,
        root_lin_vel_w=np.ones((frames, 3), dtype=np.float32),
        root_ang_vel_w=np.zeros((frames, 3), dtype=np.float32),
        body_names=np.asarray(body_names),
        body_pos_w=body_pos,
        body_quat_wxyz=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
        tracking_body_names=np.asarray(body_names),
        foot_sole_names=np.asarray(("left_sole", "right_sole")),
        foot_sole_position_w=feet,
        left_foot_contact=np.asarray((False, True, True)),
        right_foot_contact=np.asarray((True, False, False)),
        source_stageii_file=np.asarray(source),
        g1_contract_checksum_sha256=np.asarray("a" * 64),
        g1_urdf_checksum_sha256=np.asarray("b" * 64),
        correspondence_checksum_sha256=np.asarray("c" * 64),
        tracking_action_mode=np.asarray("reference_residual"),
        tracking_residual_scale=np.asarray(action_scale, dtype=np.float32),
        tracking_action_clip=np.asarray(1.0, dtype=np.float32),
        tracking_normalized_delta_limit=np.asarray(0.15, dtype=np.float32),
        gate3_complete=np.asarray(complete),
    )
    return path


def test_vectorized_linear_and_shortest_quaternion_interpolation(tmp_path: Path):
    second = _write_motion(tmp_path / "z" / "second_g1_50hz.npz", source="z/second_stageii.npz", offset=10.0)
    first = _write_motion(tmp_path / "a" / "first_g1_50hz.npz", source="a/first_stageii.npz")
    library = MotionLibrary([second, first])

    assert library.num_motions == 2
    assert library.motion_keys == ("a/first_stageii.npz", "z/second_stageii.npz")
    assert library.contract.joint_order == "policy_index"
    assert library.lengths.tolist() == [3, 3]
    assert library.durations.tolist() == pytest.approx([1.0, 1.0])
    assert len(library.library_checksum_sha256) == 64

    result = library.query(torch.tensor([0, 1]), torch.tensor([0.25, 0.25]))
    assert result.joint_pos.shape == (2, 29)
    assert result.joint_pos[:, 0].tolist() == pytest.approx([0.5, 10.5])
    assert result.root_pos_w[:, 0].tolist() == pytest.approx([0.5, 0.5])
    assert torch.allclose(result.root_quat_wxyz.abs()[:, 0], torch.ones(2), atol=1.0e-6)
    assert torch.allclose(
        torch.linalg.vector_norm(result.body_quat_wxyz, dim=-1),
        torch.ones((2, 2)),
        atol=1.0e-6,
    )
    assert result.frame_lower.tolist() == [0, 0]
    assert result.frame_upper.tolist() == [1, 1]
    assert result.blend.tolist() == pytest.approx([0.5, 0.5])


def test_contact_modes_terminal_clamping_and_future_shape(tmp_path: Path):
    path = _write_motion(tmp_path / "motion_g1_50hz.npz", source="motion_stageii.npz")
    library = MotionLibrary(path)

    nearest = library.query(0, 0.25, contact_mode="nearest")
    assert nearest.foot_contact.tolist() == [True, False]
    threshold = library.query(0, 0.25, contact_mode="threshold", contact_threshold=0.75)
    assert threshold.foot_contact.tolist() == [False, False]

    outside = library.query(torch.tensor([0, 0]), torch.tensor([-0.1, 2.0]))
    assert outside.valid.tolist() == [False, False]
    assert outside.terminal.tolist() == [False, True]
    assert outside.time.tolist() == pytest.approx([0.0, 1.0])
    with pytest.raises(ValueError, match="outside"):
        library.query(0, 1.1, clamp=False)

    future = library.query_future(
        torch.tensor([0, 0]),
        torch.tensor([0.0, 0.5]),
        [0.0, 0.25, 0.5],
    )
    assert future.joint_pos.shape == (2, 3, 29)
    assert future.body_pos_w.shape == (2, 3, 2, 3)
    assert future.foot_contact.shape == (2, 3, 2)
    assert future.terminal.tolist() == [[False, False, False], [False, False, True]]


def test_deterministic_sampling_and_valid_start_bounds(tmp_path: Path):
    paths = [
        _write_motion(tmp_path / f"{index}_g1_50hz.npz", source=f"{index}_stageii.npz", offset=float(index))
        for index in range(3)
    ]
    library = MotionLibrary(paths)
    bounds = library.valid_start_time_bounds(0.4)
    np.testing.assert_allclose(bounds.cpu().numpy(), np.asarray([[0.0, 0.6]] * 3), atol=1.0e-6)

    generator_a = torch.Generator().manual_seed(1234)
    generator_b = torch.Generator().manual_seed(1234)
    sample_a = library.sample(64, future_horizon_s=0.4, generator=generator_a)
    sample_b = library.sample(64, future_horizon_s=0.4, generator=generator_b)
    assert torch.equal(sample_a.motion_id, sample_b.motion_id)
    assert torch.equal(sample_a.time, sample_b.time)
    assert bool(torch.all(sample_a.time <= 0.6 + 1.0e-6))


def test_contract_checksum_order_and_completion_are_enforced(tmp_path: Path):
    first = _write_motion(tmp_path / "first_g1_50hz.npz", source="first_stageii.npz")
    mismatch = _write_motion(
        tmp_path / "mismatch_g1_50hz.npz",
        source="mismatch_stageii.npz",
        action_scale=0.2,
    )
    with pytest.raises(ValueError, match="tracking_residual_scale"):
        MotionLibrary([first, mismatch])

    incomplete = _write_motion(
        tmp_path / "incomplete_g1_50hz.npz",
        source="incomplete_stageii.npz",
        complete=False,
    )
    with pytest.raises(ValueError, match="completion"):
        MotionLibrary(incomplete, require_gate3_complete=True)

    with pytest.raises(ValueError, match="checksum mismatch"):
        MotionLibrary(first, expected_checksums={first: "0" * 64})

    unsafe = tmp_path / "unsafe.npz"
    np.savez(unsafe, payload=np.asarray([{"unsafe": True}], dtype=object))
    with pytest.raises(ValueError, match="safely load"):
        MotionLibrary(unsafe)


def test_gate1_split_manifest_resolution_is_sorted_and_hashed(tmp_path: Path):
    reference_root = tmp_path / "g1"
    alpha = _write_motion(
        reference_root / "CategoryA" / "A_g1_50hz.npz",
        source="CategoryA/A_stageii.npz",
    )
    zeta = _write_motion(
        reference_root / "CategoryZ" / "Z_g1_50hz.npz",
        source="CategoryZ/Z_stageii.npz",
    )
    manifest = tmp_path / "accad_train.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "split:train",
                "motions": [
                    {"source_file": "CategoryZ/Z_stageii.npz"},
                    {"source_file": "CategoryA/A_stageii.npz"},
                ],
            }
        ),
        encoding="utf-8",
    )
    selection = motion_paths_from_manifest(manifest, reference_root, split="train")
    assert selection.paths == (alpha.resolve(), zeta.resolve())
    assert selection.split == "train"
    assert selection.manifest_checksum_sha256 == _digest(manifest)

    library = MotionLibrary.from_manifest(manifest, reference_root, split="train")
    assert library.motion_keys == (
        "CategoryA/A_stageii.npz",
        "CategoryZ/Z_stageii.npz",
    )
    assert library.manifest_checksum_sha256 == _digest(manifest)
    assert library.split == "train"


def test_gate3_manifest_checksum_and_path_confinement(tmp_path: Path):
    reference_root = tmp_path / "g1"
    motion = _write_motion(
        reference_root / "Category" / "clip_g1_50hz.npz",
        source="Category/clip_stageii.npz",
    )
    manifest = tmp_path / "gate3.json"
    manifest.write_text(
        json.dumps(
            {
                "motions": [
                    {
                        "g1_file": "Category/clip_g1_50hz.npz",
                        "archive_checksum_sha256": _digest(motion),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    library = MotionLibrary.from_manifest(manifest, reference_root)
    assert library.checksums_sha256 == (_digest(motion),)

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["motions"][0]["archive_checksum_sha256"] = "0" * 64
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        MotionLibrary.from_manifest(manifest, reference_root)

    manifest.write_text(json.dumps({"motions": ["../escape_g1_50hz.npz"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        motion_paths_from_manifest(manifest, reference_root)


def test_current_b3_gate3_archive_is_compatible_when_present():
    path = (
        Path(__file__).resolve().parents[1]
        / "work"
        / "g1"
        / "Female1Walking_c3d"
        / "B3_-_walk1_g1_50hz.npz"
    )
    if not path.is_file():
        pytest.skip("Gate-3 B3 artifact has not been generated")
    library = MotionLibrary(path)
    assert library.num_motions == 1
    assert library.fps == pytest.approx(50.0)
    assert library.lengths.tolist() == [381]
    query = library.query_future(0, 0.0, [0.0, 0.04, 0.08, 0.16])
    assert query.joint_pos.shape == (4, 29)
    assert bool(torch.isfinite(query.body_pos_w).all())
