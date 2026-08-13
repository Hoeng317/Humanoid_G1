from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
import yaml

from accad_g1.g1_validation import (
    DEFAULT_CORRESPONDENCE_PATH,
    DETAILED_CONTACT_KEYS,
    EXPECTED_BODY_NAMES,
    EXPECTED_FOOT_SUPPORT_NAMES,
    G1_CONTRACT_PATH,
    G1_URDF_PATH,
    validate_g1_archive,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _gradient(values: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(values, 1.0 / fps, axis=0, edge_order=2)


def _angular_velocity(quaternion_wxyz: np.ndarray, fps: float) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64).copy()
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    for frame in range(1, len(q)):
        flip = np.sum(q[frame - 1] * q[frame], axis=-1, keepdims=True) < 0.0
        q[frame] = np.where(flip, -q[frame], q[frame])
    shape = q.shape[:-1]
    rotation = Rotation.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
    rotation = rotation.reshape(shape + (3, 3))
    relative = rotation[1:] @ np.swapaxes(rotation[:-1], -1, -2)
    interval = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
    interval = interval.reshape(relative.shape[:-2] + (3,)) * fps
    output = np.empty(shape + (3,), dtype=np.float64)
    output[0] = interval[0]
    output[-1] = interval[-1]
    output[1:-1] = 0.5 * (interval[:-1] + interval[1:])
    return output


def _contract() -> dict[str, object]:
    payload = yaml.safe_load(G1_CONTRACT_PATH.read_text(encoding="utf-8"))
    joints = sorted(payload["joints"], key=lambda item: int(item["policy_index"]))
    policy_names = tuple(str(item["name"]) for item in joints)
    policy_to_sdk = np.asarray([int(item["sdk_index"]) for item in joints], dtype=np.int64)
    sdk_names = [""] * 29
    for name, index in zip(policy_names, policy_to_sdk, strict=True):
        sdk_names[int(index)] = name
    return {
        "policy_names": policy_names,
        "sdk_names": tuple(sdk_names),
        "policy_to_sdk": policy_to_sdk,
        "default": np.asarray([float(item["default"]) for item in joints]),
    }


def _support_geometry() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = ET.parse(G1_URDF_PATH).getroot()
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        link = next(item for item in root.findall("link") if item.attrib["name"] == name)
        centers = []
        radii = []
        for collision in link.findall("collision"):
            sphere = collision.find("./geometry/sphere")
            if sphere is None:
                continue
            origin = collision.find("origin")
            centers.append(
                [float(value) for value in origin.attrib.get("xyz", "0 0 0").split()]
            )
            radii.append(float(sphere.attrib["radius"]))
        result[name] = (np.asarray(centers), np.asarray(radii))
    return result


def _pin_fk(
    joint_position_sdk: np.ndarray,
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    body_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = pin.buildModelFromUrdf(str(G1_URDF_PATH), pin.JointModelFreeFlyer())
    data = model.createData()
    frame_ids = [model.getFrameId(name) for name in body_names]
    support = _support_geometry()
    frames = len(joint_position_sdk)
    body_position = np.empty((frames, 32, 3), dtype=np.float64)
    body_quaternion = np.empty((frames, 32, 4), dtype=np.float64)
    support_position = np.empty((frames, 2, 4, 3), dtype=np.float64)
    for frame in range(frames):
        q_pin = np.concatenate(
            (
                root_position[frame],
                root_quaternion[frame, [1, 2, 3, 0]],
                joint_position_sdk[frame],
            )
        )
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacements(model, data)
        for body, frame_id in enumerate(frame_ids):
            placement = data.oMf[frame_id]
            body_position[frame, body] = placement.translation
            xyzw = np.asarray(pin.Quaternion(placement.rotation).coeffs())
            body_quaternion[frame, body] = xyzw[[3, 0, 1, 2]]
        for side, link_name in enumerate(
            ("left_ankle_roll_link", "right_ankle_roll_link")
        ):
            placement = data.oMf[model.getFrameId(link_name)]
            local_centers, _ = support[link_name]
            support_position[frame, side] = (
                placement.translation[None]
                + (placement.rotation @ local_centers.T).T
            )
    for frame in range(1, frames):
        flip = np.sum(
            body_quaternion[frame - 1] * body_quaternion[frame],
            axis=-1,
            keepdims=True,
        ) < 0.0
        body_quaternion[frame] = np.where(
            flip, -body_quaternion[frame], body_quaternion[frame]
        )
    return body_position, body_quaternion, support_position


def _build_archive(
    directory: Path,
    *,
    joint_overrides: dict[str, float] | None = None,
    root_x_step_m: float = 0.0,
    root_z_offset_m: float = 0.0,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    fps = 50.0
    frames = 6
    contract = _contract()
    q_policy = np.broadcast_to(contract["default"], (frames, 29)).copy()
    for name, value in (joint_overrides or {}).items():
        q_policy[:, contract["policy_names"].index(name)] = value
    q_sdk = np.empty_like(q_policy)
    q_sdk[:, contract["policy_to_sdk"]] = q_policy

    correspondence = yaml.safe_load(
        DEFAULT_CORRESPONDENCE_PATH.read_text(encoding="utf-8")
    )
    body_names = tuple(str(name) for name in correspondence["isaac_body_names"])
    assert set(body_names) == EXPECTED_BODY_NAMES
    root_quaternion = np.zeros((frames, 4), dtype=np.float64)
    root_quaternion[:, 0] = 1.0
    root_position = np.zeros((frames, 3), dtype=np.float64)

    # Determine the exact flat-ground root height from every URDF support
    # sphere surface, rather than from the ankle frame or a copied constant.
    _, _, zero_support = _pin_fk(q_sdk, root_position, root_quaternion, body_names)
    support_geometry = _support_geometry()
    radii = np.stack(
        (
            support_geometry["left_ankle_roll_link"][1],
            support_geometry["right_ankle_roll_link"][1],
        )
    )
    minimum_surface_z = float(np.min(zero_support[..., 2] - radii[None]))
    root_position[:, 2] = -minimum_surface_z + root_z_offset_m
    root_position[:, 0] = np.arange(frames) * root_x_step_m

    body_position, body_quaternion, support_position = _pin_fk(
        q_sdk, root_position, root_quaternion, body_names
    )
    sole_position = support_position.mean(axis=2)
    contact_point_position = np.stack(
        (
            support_position[:, 0, :2].mean(axis=1),
            support_position[:, 0, 2:].mean(axis=1),
            support_position[:, 1, :2].mean(axis=1),
            support_position[:, 1, 2:].mean(axis=1),
        ),
        axis=1,
    )
    contact = np.ones(frames, dtype=bool)

    source = directory / "source_human.npz"
    np.savez_compressed(
        source,
        schema_version=np.asarray("accad_human_reconstruction_v1"),
        marker=np.arange(3, dtype=np.float32),
    )
    stageii = directory / "source_stageii.npz"
    np.savez_compressed(stageii, poses=np.zeros((frames, 165), dtype=np.float32))

    joint_velocity = _gradient(q_policy, fps)
    payload = {
        "schema_version": np.asarray("accad_g1_reference_v1"),
        "retarget_version": np.asarray("g1-sequence-retarget-v3-test"),
        "fps": np.asarray(fps),
        "time": np.arange(frames, dtype=np.float64) / fps,
        "frame_count": np.asarray(frames, dtype=np.int64),
        "coordinate_system": np.asarray("right-handed_x-forward_y-left_z-up"),
        "quaternion_order": np.asarray("wxyz"),
        "position_unit": np.asarray("meter"),
        "angle_unit": np.asarray("radian"),
        "joint_order": np.asarray("policy_index"),
        "joint_names": np.asarray(contract["policy_names"]),
        "sdk_joint_names": np.asarray(contract["sdk_names"]),
        "policy_to_sdk": contract["policy_to_sdk"],
        "joint_pos": q_policy.astype(np.float32),
        "joint_pos_sdk": q_sdk.astype(np.float32),
        "joint_vel": joint_velocity.astype(np.float32),
        "joint_acc": _gradient(joint_velocity, fps).astype(np.float32),
        "root_pos_w": root_position.astype(np.float32),
        "root_quat_wxyz": root_quaternion.astype(np.float32),
        "root_lin_vel_w": _gradient(root_position, fps).astype(np.float32),
        "root_ang_vel_w": _angular_velocity(root_quaternion, fps).astype(np.float32),
        "body_names": np.asarray(body_names),
        "body_pos_w": body_position.astype(np.float32),
        "body_quat_wxyz": body_quaternion.astype(np.float32),
        "body_lin_vel_w": _gradient(body_position, fps).astype(np.float32),
        "body_ang_vel_w": _angular_velocity(body_quaternion, fps).astype(np.float32),
        "tracking_body_names": np.asarray(correspondence["tracking_body_names"]),
        "foot_sole_names": np.asarray(("left_sole", "right_sole")),
        "foot_sole_position_w": sole_position.astype(np.float32),
        "foot_support_names": np.asarray(EXPECTED_FOOT_SUPPORT_NAMES),
        "foot_support_position_w": support_position.reshape(frames, 8, 3).astype(
            np.float32
        ),
        "foot_collision_radius_m": np.asarray(float(radii[0, 0]), dtype=np.float32),
        "left_foot_contact": contact,
        "right_foot_contact": contact,
        "contact_point_names": np.asarray(
            ("left_heel", "left_toe", "right_heel", "right_toe")
        ),
        "contact_point_position_w": contact_point_position.astype(np.float32),
        "left_heel_contact": contact,
        "left_toe_contact": contact,
        "right_heel_contact": contact,
        "right_toe_contact": contact,
        "morphology_scale": np.asarray(1.0, dtype=np.float32),
        "human_leg_height_m": np.asarray(0.8, dtype=np.float32),
        "g1_leg_height_m": np.asarray(-minimum_surface_z, dtype=np.float32),
        "source_file": np.asarray(str(source.resolve())),
        "source_checksum_sha256": np.asarray(_sha256(source)),
        "source_human_schema": np.asarray("accad_human_reconstruction_v1"),
        "source_stageii_file": np.asarray(str(stageii.resolve())),
        "source_stageii_checksum_sha256": np.asarray(_sha256(stageii)),
        "g1_contract_path": np.asarray(str(G1_CONTRACT_PATH.resolve())),
        "g1_contract_checksum_sha256": np.asarray(_sha256(G1_CONTRACT_PATH)),
        "g1_urdf_path": np.asarray(str(G1_URDF_PATH.resolve())),
        "g1_urdf_checksum_sha256": np.asarray(_sha256(G1_URDF_PATH)),
        "correspondence_path": np.asarray(str(DEFAULT_CORRESPONDENCE_PATH.resolve())),
        "correspondence_checksum_sha256": np.asarray(
            _sha256(DEFAULT_CORRESPONDENCE_PATH)
        ),
        "solver_name": np.asarray("synthetic_test_fixture"),
        "solver_iterations": np.asarray(1, dtype=np.int64),
        "tracking_action_mode": np.asarray("reference_residual"),
        "tracking_residual_scale": np.asarray(0.25, dtype=np.float32),
        "tracking_action_clip": np.asarray(1.0, dtype=np.float32),
        "tracking_normalized_delta_limit": np.asarray(0.15, dtype=np.float32),
        "contact_lock_run_count": np.asarray(4, dtype=np.int64),
        "gate3_complete": np.asarray(False),
        "gate3_status": np.asarray("synthetic basis; independent validation pending"),
    }
    archive = directory / "g1_reference.npz"
    np.savez_compressed(archive, **payload)
    return archive


def _rewrite(path: Path, destination: Path, **updates: np.ndarray) -> Path:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload.update(updates)
    np.savez_compressed(destination, **payload)
    return destination


def _check(result: dict[str, object], group: str, name: str) -> dict[str, object]:
    return next(item for item in result["checks"][group] if item["name"] == name)


def test_nominal_archive_passes_independent_kinematic_gate(tmp_path: Path) -> None:
    archive = _build_archive(tmp_path)
    result = validate_g1_archive(archive, expected_source_path=tmp_path / "source_human.npz")

    assert result["ok"] is True
    assert result["gate3_kinematic_complete"] is True
    assert result["gate3_complete"] is False
    assert result["isaac_replay"]["status"] == "pending"
    assert all(result["statuses"].values())
    assert result["metrics"]["pinocchio_fk"]["version"] == pin.__version__
    assert (
        result["metrics"]["joint_limits"]["acceleration"]["contract_limit_available"]
        is False
    )


def test_current_b3_archive_passes_all_offline_gate3_groups() -> None:
    archive = (
        DEFAULT_CORRESPONDENCE_PATH.parent
        / "work"
        / "g1"
        / "Female1Walking_c3d"
        / "B3_-_walk1_g1_50hz.npz"
    )
    assert archive.is_file(), "the current B3 Gate-3 basis must be generated before acceptance"

    result = validate_g1_archive(archive)

    assert result["gate3_kinematic_complete"] is True
    assert all(result["statuses"].values())
    assert result["required_failures"] == []


def test_policy_sdk_mismatch_is_detected(tmp_path: Path) -> None:
    archive = _build_archive(tmp_path)
    with np.load(archive, allow_pickle=False) as source:
        wrong_sdk = source["joint_pos_sdk"].copy()
    wrong_sdk[:, 0] += 0.1
    tampered = _rewrite(archive, tmp_path / "wrong_sdk.npz", joint_pos_sdk=wrong_sdk)

    result = validate_g1_archive(tampered)

    assert result["gate3_kinematic_complete"] is False
    assert _check(result, "contract", "policy_sdk_consistency")["passed"] is False


def test_pinocchio_fk_and_provenance_tampering_are_detected(tmp_path: Path) -> None:
    archive = _build_archive(tmp_path)
    with np.load(archive, allow_pickle=False) as source:
        wrong_body = source["body_pos_w"].copy()
    wrong_body[2, 7, 0] += 0.01
    tampered = _rewrite(
        archive,
        tmp_path / "wrong_fk.npz",
        body_pos_w=wrong_body,
        source_checksum_sha256=np.asarray("0" * 64),
    )

    result = validate_g1_archive(tampered)

    assert _check(
        result, "pinocchio_fk", "stored_body_position_matches_pinocchio"
    )["passed"] is False
    assert _check(result, "provenance", "source_human_sha256")["passed"] is False


def test_pinocchio_endpoint_roundoff_does_not_false_fail_body_velocity(
    tmp_path: Path,
) -> None:
    archive = _build_archive(tmp_path)
    with np.load(archive, allow_pickle=False) as source:
        body_position = source["body_pos_w"].copy()

    # Model the alternating micrometre-scale position differences observed
    # between float32 Torch FK and float64 Pinocchio FK.  At the last frame the
    # edge_order=2 one-sided derivative amplifies these differences above the
    # 2e-4 interior tolerance, despite the positions and serialized derivative
    # remaining mutually consistent.
    body_position[-3:, 28, 0] += np.asarray(
        (1.5e-6, -1.5e-6, 1.5e-6), dtype=np.float32
    )
    body_linear_velocity = _gradient(body_position, 50.0).astype(np.float32)
    endpoint_roundoff = _rewrite(
        archive,
        tmp_path / "endpoint_roundoff.npz",
        body_pos_w=body_position,
        body_lin_vel_w=body_linear_velocity,
    )

    result = validate_g1_archive(endpoint_roundoff)

    serialized = _check(
        result,
        "numeric",
        "recomputed_body_lin_vel_w_from_serialized_body_position",
    )
    interior = _check(
        result, "numeric", "recomputed_body_lin_vel_w_from_pinocchio"
    )
    endpoint = _check(
        result,
        "numeric",
        "recomputed_body_lin_vel_w_endpoint_from_pinocchio",
    )
    assert result["gate3_kinematic_complete"] is True
    assert serialized["passed"] is True
    assert interior["passed"] is True
    assert endpoint["passed"] is True
    assert endpoint["maximum_absolute_error"] > 2.0e-4
    assert endpoint["maximum_absolute_error"] <= 8.0e-4


def test_endpoint_velocity_corruption_is_caught_by_serialized_position_check(
    tmp_path: Path,
) -> None:
    archive = _build_archive(tmp_path)
    with np.load(archive, allow_pickle=False) as source:
        corrupt_velocity = source["body_lin_vel_w"].copy()
    # This perturbation is intentionally below the endpoint-only Pinocchio
    # tolerance.  The all-frame serialized-position check must still reject it.
    corrupt_velocity[-1, 0, 0] += np.float32(4.0e-4)
    corrupted = _rewrite(
        archive,
        tmp_path / "corrupt_endpoint_velocity.npz",
        body_lin_vel_w=corrupt_velocity,
    )

    result = validate_g1_archive(corrupted)

    serialized = _check(
        result,
        "numeric",
        "recomputed_body_lin_vel_w_from_serialized_body_position",
    )
    endpoint = _check(
        result,
        "numeric",
        "recomputed_body_lin_vel_w_endpoint_from_pinocchio",
    )
    assert serialized["passed"] is False
    assert serialized["maximum_absolute_error"] > 2.0e-4
    assert endpoint["passed"] is True
    assert result["gate3_kinematic_complete"] is False


def test_soft_limit_is_a_required_gate_but_acceleration_is_report_only(
    tmp_path: Path,
) -> None:
    # Knee q=0 is inside the URDF hard bound (-0.087267) but outside the
    # Isaac 0.9 soft bound (about +0.0611).
    archive = _build_archive(tmp_path, joint_overrides={"left_knee_joint": 0.0})
    result = validate_g1_archive(archive)

    assert _check(result, "limits", "hard_joint_position_limits")["passed"] is True
    assert _check(result, "limits", "soft_joint_position_limits")["passed"] is False
    acceleration = _check(result, "limits", "joint_acceleration_metrics")
    assert acceleration["passed"] is True
    assert acceleration["required"] is False
    assert result["gate3_kinematic_complete"] is False


def test_exact_support_spheres_detect_penetration_and_contact_slip(tmp_path: Path) -> None:
    penetrating = _build_archive(tmp_path / "penetrating", root_z_offset_m=-0.01)
    penetration_result = validate_g1_archive(penetrating)
    penetration_check = _check(
        penetration_result, "contact_geometry", "ground_penetration"
    )
    assert penetration_check["passed"] is False
    assert penetration_check["maximum_penetration_m"] > 0.009

    slipping = _build_archive(tmp_path / "slipping", root_x_step_m=0.02)
    slip_result = validate_g1_archive(slipping)
    assert _check(
        slip_result, "contact_geometry", "consecutive_contact_point_step"
    )["passed"] is False
    assert _check(
        slip_result, "contact_geometry", "stance_run_anchor_drift"
    )["passed"] is False


def test_object_array_is_never_unpickled(tmp_path: Path) -> None:
    archive = _build_archive(tmp_path)
    with np.load(archive, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    payload["unsafe_python_object"] = np.asarray([{"payload": "forbidden"}], dtype=object)
    unsafe = tmp_path / "unsafe.npz"
    np.savez_compressed(unsafe, **payload)

    result = validate_g1_archive(unsafe)

    assert result["gate3_kinematic_complete"] is False
    safe_check = _check(result, "schema", "safe_no_pickle_load")
    assert safe_check["passed"] is False
    assert "unsafe_python_object" in safe_check["unreadable"]


def test_legacy_foot_contact_archive_uses_sole_fallback(tmp_path: Path) -> None:
    archive = _build_archive(tmp_path)
    with np.load(archive, allow_pickle=False) as source:
        payload = {
            name: source[name]
            for name in source.files
            if name not in DETAILED_CONTACT_KEYS
        }
    payload["retarget_version"] = np.asarray("g1-sequence-retarget-v1")
    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(legacy, **payload)

    result = validate_g1_archive(legacy)

    assert result["gate3_kinematic_complete"] is True
    assert "sole" in result["metrics"]["foot_contact_geometry"]["left"][
        "contact_channels"
    ]
    assert any("sole-center fallback" in warning for warning in result["warnings"])
