from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import accad_g1.contact_feasibility as contact_module
from accad_g1.contact_feasibility import (
    CONTACT_GATE_VERSION,
    MANIFEST_SCHEMA_VERSION,
    ContactFeasibilityError,
    audit_contact_pair,
    build_contact_feasibility_split,
    centroidal_zmp_from_dynamics,
    five_frame_derivatives,
    load_urdf_asset,
    quaternion_wxyz_to_matrix,
    resolve_inertial_model,
    signed_distance_to_support,
)
from test_g1_validation import _build_archive


PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _extend_archive(source: Path, destination: Path, *, frames: int = 9, **scalars) -> Path:
    with np.load(source, allow_pickle=False) as archive:
        old_count = int(archive["frame_count"].item())
        payload: dict[str, np.ndarray] = {}
        for name in archive.files:
            value = archive[name]
            if value.ndim >= 1 and value.shape[0] == old_count:
                indices = np.minimum(np.arange(frames), old_count - 1)
                payload[name] = value[indices].copy()
            else:
                payload[name] = value.copy()
    payload["frame_count"] = np.asarray(frames, dtype=np.int64)
    payload["time"] = np.arange(frames, dtype=np.float64) / 50.0
    payload.update({name: np.asarray(value) for name, value in scalars.items()})
    np.savez_compressed(destination, **payload)
    return destination


def _write_split_fixture(
    directory: Path,
    *,
    split: str,
) -> tuple[Path, Path, Path, Path]:
    source_root = directory / f"{split}_v27"
    candidate_root = directory / f"{split}_v28"
    source_root.mkdir(parents=True)
    candidate_root.mkdir(parents=True)
    basis = _build_archive(directory / f"{split}_basis")
    relative = Path("synthetic/static_g1_50hz.npz")
    source_path = source_root / relative
    source_path.parent.mkdir(parents=True)
    _extend_archive(basis, source_path)
    v27_path = directory / f"g1_{split}_v27.json"
    source_record = {
        "g1_file": relative.as_posix(),
        "g1_checksum_sha256": _sha256(source_path),
        "g1_size_bytes": source_path.stat().st_size,
        "frame_count": 9,
        "duration_sec": 8.0 / 50.0,
        "fps": 50.0,
        "human_file": "unused_human_fixture.npz",
    }
    v27_document = {
        "schema_version": "accad_g1_gate3_manifest_v1",
        "dataset": "ACCAD",
        "kind": f"split:{split}",
        "split": split,
        "reference_root": source_root.relative_to(PIPELINE_ROOT).as_posix(),
        "motion_set_checksum_sha256": "1" * 64,
        "num_motions": 1,
        "motions": [source_record],
    }
    v27_path.write_text(json.dumps(v27_document), encoding="utf-8")

    candidate_path = candidate_root / relative
    candidate_path.parent.mkdir(parents=True)
    _extend_archive(
        source_path,
        candidate_path,
        retarget_version="g1-sequence-retarget-v28-synthetic",
        dynamic_source_g1_checksum_sha256=source_record["g1_checksum_sha256"],
        dynamic_source_manifest_checksum_sha256=_sha256(v27_path),
        dynamic_split=split,
        dynamic_source_action="identity",
        dynamic_final_action="identity_numeric_copy",
        dynamic_realized_scale=np.float64(1.0),
    )
    dynamic_policy_checksum = _sha256(contact_module.DEFAULT_DYNAMIC_POLICY_PATH)
    candidate_record = {
        **source_record,
        "source_v27_g1_checksum_sha256": source_record["g1_checksum_sha256"],
        "g1_checksum_sha256": _sha256(candidate_path),
        "g1_size_bytes": candidate_path.stat().st_size,
        "retarget_version": "g1-sequence-retarget-v28-synthetic",
    }
    v28_path = directory / f"g1_{split}_v28.json"
    v28_document = {
        "schema_version": "accad_g1_dynamic_manifest_v1",
        "dataset": "ACCAD",
        "kind": f"split:{split}",
        "split": split,
        "gate": "synthetic",
        "reference_root": candidate_root.relative_to(PIPELINE_ROOT).as_posix(),
        "retarget_version": "g1-sequence-retarget-v28-synthetic",
        "motion_set_checksum_sha256": "2" * 64,
        "source_manifest_checksum_sha256": _sha256(v27_path),
        "dynamic_policy_checksum_sha256": dynamic_policy_checksum,
        "num_motions": 1,
        "motions": [candidate_record],
    }
    v28_path.write_text(json.dumps(v28_document), encoding="utf-8")
    return v27_path, v28_path, source_path, candidate_path


def test_exact_urdf_inertials_include_fixed_children_without_scipy() -> None:
    asset = load_urdf_asset()
    assert asset.checksum_sha256 == "c0ae739c640c3e2c00d1bdd8810b5d6e59601487bd1a3995859f9543269ee5c8"
    assert asset.total_mass_kg == pytest.approx(33.34114202, abs=1.0e-12)
    assert len(asset.raw_inertials) == 35
    archive_path = (
        PIPELINE_ROOT
        / "work/dynamic/g1/Female1General_c3d/A1_-_Stand_g1_50hz.npz"
    )
    if not archive_path.is_file():
        pytest.skip("production G1 body contract is not installed")
    with np.load(archive_path, allow_pickle=False) as archive:
        body_names = tuple(str(value) for value in archive["body_names"].tolist())
    model = resolve_inertial_model(body_names, asset)
    anchors = dict(zip(model.source_links, model.anchor_links, strict=True))
    assert anchors["head_link"] == "torso_link"
    assert anchors["logo_link"] == "torso_link"
    assert anchors["pelvis_contour_link"] == "pelvis"
    assert model.total_mass_kg == pytest.approx(asset.total_mass_kg, abs=1.0e-12)


def test_quaternion_rotation_and_five_frame_quadratic_derivatives_are_exact() -> None:
    half = np.sqrt(0.5)
    rotation = quaternion_wxyz_to_matrix(np.asarray([half, 0.0, 0.0, half]))
    assert rotation @ np.asarray([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])

    fps = 50.0
    time = np.arange(9, dtype=np.float64) / fps
    values = np.stack((3.0 * time**2 + 2.0 * time + 1.0, -time**2), axis=1)
    velocity, acceleration = five_frame_derivatives(values, fps)
    assert velocity[2:-2, 0] == pytest.approx(6.0 * time[2:-2] + 2.0)
    assert velocity[2:-2, 1] == pytest.approx(-2.0 * time[2:-2])
    assert acceleration[2:-2] == pytest.approx(
        np.broadcast_to([6.0, -2.0], (5, 2))
    )
    assert np.isnan(velocity[:2]).all() and np.isnan(velocity[-2:]).all()


def test_support_signed_distance_handles_polygon_segment_and_point() -> None:
    square = np.asarray(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
    assert signed_distance_to_support(np.asarray([0.0, 0.0]), square) == pytest.approx(1.0)
    assert signed_distance_to_support(np.asarray([1.25, 0.0]), square) == pytest.approx(-0.25)
    segment = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    assert signed_distance_to_support(np.asarray([0.5, 0.2]), segment) == pytest.approx(-0.2)
    assert signed_distance_to_support(np.asarray([2.0, 0.0]), segment) == pytest.approx(-1.0)
    assert signed_distance_to_support(np.asarray([0.3, 0.4]), np.asarray(((0.0, 0.0),))) == pytest.approx(-0.5)


def test_uniform_slowdown_can_move_centroidal_zmp_out_of_single_support() -> None:
    com = np.asarray((0.10, 0.0, 0.70))
    acceleration = np.asarray((1.40, 0.0, 0.0))
    momentum_rate = np.zeros(3)
    source, source_normal = centroidal_zmp_from_dynamics(
        com,
        acceleration,
        momentum_rate,
        support_plane_z=0.0,
        total_mass_kg=33.0,
    )
    scale = 2.0
    slowed, slowed_normal = centroidal_zmp_from_dynamics(
        com,
        acceleration / scale**2,
        momentum_rate / scale**2,
        support_plane_z=0.0,
        total_mass_kg=33.0,
    )
    support = np.asarray(((-0.05, -0.03), (0.05, -0.03), (0.05, 0.03), (-0.05, 0.03)))
    assert source_normal == pytest.approx(slowed_normal)
    assert signed_distance_to_support(source, support) > 0.0
    assert signed_distance_to_support(slowed, support) < -0.017


def test_audit_only_is_default_and_manifest_publication_is_opt_in(
    pipeline_tmp_path: Path,
) -> None:
    v27, v28, _, _ = _write_split_fixture(pipeline_tmp_path, split="train")
    output = pipeline_tmp_path / "contact"
    report = build_contact_feasibility_split(
        split="train",
        v27_manifest_path=v27,
        v28_manifest_path=v28,
        output_root=output,
    )
    assert report["status"] == "completed"
    assert report["mode"] == "audit_only"
    assert report["publication"]["status"] == "not_requested"
    assert not (output / "g1_train.json").exists()

    published = build_contact_feasibility_split(
        split="train",
        v27_manifest_path=v27,
        v28_manifest_path=v28,
        output_root=output,
        force=True,
        publish_manifest=True,
    )
    assert published["publication"]["status"] == "published"
    manifest = json.loads((output / "g1_train.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["num_motions"] == 1
    assert manifest["motions"][0]["contact_v29_gate_version"] == CONTACT_GATE_VERSION


def test_validation_requires_unchanged_train_frozen_policy(
    pipeline_tmp_path: Path,
) -> None:
    train_v27, train_v28, _, _ = _write_split_fixture(pipeline_tmp_path, split="train")
    validation_v27, validation_v28, _, _ = _write_split_fixture(
        pipeline_tmp_path, split="validation"
    )
    output = pipeline_tmp_path / "contact_policy"
    build_contact_feasibility_split(
        split="train",
        v27_manifest_path=train_v27,
        v28_manifest_path=train_v28,
        output_root=output,
    )
    policy_path = output / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["algorithm"]["gravity_mps2"] = 9.80
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ContactFeasibilityError, match="altered"):
        build_contact_feasibility_split(
            split="validation",
            v27_manifest_path=validation_v27,
            v28_manifest_path=validation_v28,
            output_root=output,
        )


def test_train_and_validation_invocations_never_open_test_paths(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_v27, train_v28, _, _ = _write_split_fixture(pipeline_tmp_path, split="train")
    validation_v27, validation_v28, _, _ = _write_split_fixture(
        pipeline_tmp_path, split="validation"
    )
    forbidden_test = pipeline_tmp_path / "sealed_test_must_not_open.json"
    forbidden_test.write_text("this is deliberately invalid JSON", encoding="utf-8")
    monkeypatch.setitem(contact_module.DEFAULT_V27_MANIFESTS, "test", forbidden_test)
    monkeypatch.setitem(contact_module.DEFAULT_V28_MANIFESTS, "test", forbidden_test)
    opened_npz: list[Path] = []
    original_safe_npz = contact_module._safe_npz

    def recording_safe_npz(path: Path):
        assert "test" not in path.name.lower()
        opened_npz.append(path)
        return original_safe_npz(path)

    monkeypatch.setattr(contact_module, "_safe_npz", recording_safe_npz)
    output = pipeline_tmp_path / "sealed"
    train_report = build_contact_feasibility_split(
        split="train",
        v27_manifest_path=train_v27,
        v28_manifest_path=train_v28,
        output_root=output,
    )
    validation_report = build_contact_feasibility_split(
        split="validation",
        v27_manifest_path=validation_v27,
        v28_manifest_path=validation_v28,
        output_root=output,
    )
    assert train_report["test_opened_by_this_invocation"] is False
    assert validation_report["test_opened_by_this_invocation"] is False
    assert len(opened_npz) == 4


def test_current_qk_validation_reference_is_quarantined_without_name_exception() -> None:
    source_path = PIPELINE_ROOT / "work/g1/s007/QkWalk1_g1_50hz.npz"
    candidate_path = PIPELINE_ROOT / "work/dynamic/g1/s007/QkWalk1_g1_50hz.npz"
    if not source_path.is_file() or not candidate_path.is_file():
        pytest.skip("production validation references are not installed")
    with np.load(source_path, allow_pickle=False) as source, np.load(
        candidate_path, allow_pickle=False
    ) as candidate:
        result = audit_contact_pair(
            {name: source[name] for name in source.files},
            {name: candidate[name] for name in candidate.files},
        )
    assert result["passed"] is False
    assert result["frame_contract"]["realized_uniform_scale"] == pytest.approx(
        2.204724409448819
    )
    assert result["violations"]["centroidal_zmp_absolute"][
        "longest_consecutive_frames"
    ] >= 3
    assert result["quarantine_reasons"]
