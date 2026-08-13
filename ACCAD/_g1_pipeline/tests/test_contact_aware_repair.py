from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import accad_g1.contact_aware_repair as repair_module
import build_contact_aware_repair as repair_cli
from accad_g1.contact_aware_repair import (
    ContactAwareRepairError,
    LEG_JOINT_NAMES,
    REPAIR_SCHEMA_VERSION,
    REPAIR_VERSION,
    WAIST_BALANCE_JOINT_NAMES,
    build_contact_aware_repair_split,
    audit_contact_anchor_postconditions,
    differentiable_centroidal_signals,
    derive_contact_repaired_archive,
    derive_contact_repaired_payload,
    inertial_com_torch,
    plan_repair_targets,
)
from accad_g1.contact_feasibility import (
    audit_contact_pair,
    compute_clip_signals,
    resolve_inertial_model,
)
from accad_g1.g1_kinematics import TorchG1Kinematics
from accad_g1.motion_library import MotionLibrary
from accad_g1.training_io import resolve_motion_inputs
from test_g1_validation import _build_archive


PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extend_static_archive(source: Path, destination: Path, frames: int = 9) -> Path:
    with np.load(source, allow_pickle=False) as archive:
        old_count = int(archive["frame_count"].item())
        payload: dict[str, np.ndarray] = {}
        indices = np.minimum(np.arange(frames), old_count - 1)
        for name in archive.files:
            value = archive[name]
            payload[name] = (
                value[indices].copy()
                if value.ndim >= 1 and value.shape[0] == old_count
                else value.copy()
            )
    payload["frame_count"] = np.asarray(frames, dtype=np.int64)
    payload["time"] = np.arange(frames, dtype=np.float64) / 50.0
    np.savez_compressed(destination, **payload)
    return destination


def _static_pair(directory: Path) -> tuple[Path, Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    basis = _build_archive(directory / "basis")
    source_path = _extend_static_archive(basis, directory / "source_v27.npz")
    with np.load(source_path, allow_pickle=False) as archive:
        candidate = {name: archive[name].copy() for name in archive.files}
    candidate.update(
        {
            "retarget_version": np.asarray("g1-sequence-retarget-v28-synthetic"),
            "dynamic_source_g1_checksum_sha256": np.asarray(_sha256(source_path)),
            "dynamic_source_manifest_checksum_sha256": np.asarray("a" * 64),
            "dynamic_split": np.asarray("train"),
            "dynamic_source_action": np.asarray("identity"),
            "dynamic_final_action": np.asarray("identity_numeric_copy"),
            "dynamic_realized_scale": np.asarray(1.0, dtype=np.float64),
        }
    )
    candidate_path = directory / "candidate_v28.npz"
    np.savez_compressed(candidate_path, **candidate)
    with np.load(source_path, allow_pickle=False) as source_archive, np.load(
        candidate_path, allow_pickle=False
    ) as candidate_archive:
        source = {name: source_archive[name] for name in source_archive.files}
        loaded_candidate = {
            name: candidate_archive[name] for name in candidate_archive.files
        }
    return source_path, candidate_path, source, loaded_candidate


def test_exact_torch_inertial_com_matches_v29_static_signal(
    pipeline_tmp_path: Path,
) -> None:
    _, _, _, candidate = _static_pair(pipeline_tmp_path)
    signals = compute_clip_signals(candidate)
    body_names = tuple(str(name) for name in candidate["body_names"].tolist())
    model = resolve_inertial_model(body_names)
    solver = TorchG1Kinematics(device="cpu", dtype=torch.float64)
    transforms = solver.forward(
        torch.as_tensor(candidate["joint_pos"], dtype=torch.float64),
        torch.as_tensor(candidate["root_pos_w"], dtype=torch.float64),
        torch.as_tensor(candidate["root_quat_wxyz"], dtype=torch.float64),
    )
    differentiable = inertial_com_torch(transforms, model).detach().numpy()
    assert differentiable == pytest.approx(signals.com_position, abs=2.0e-7)


def test_differentiable_centroidal_path_matches_exact_v29_serialized_path(
    pipeline_tmp_path: Path,
) -> None:
    _, _, _, candidate = _static_pair(pipeline_tmp_path)
    q = np.asarray(candidate["joint_pos"], dtype=np.float64).copy()
    root = np.asarray(candidate["root_pos_w"], dtype=np.float64).copy()
    phase = np.linspace(0.0, np.pi, len(q))
    q[:, 0] += 0.025 * np.sin(phase)
    q[:, 1] -= 0.020 * np.sin(phase)
    root[:, 0] += 0.004 * np.sin(phase)
    payload = repair_module._recompute_payload(candidate, q, root)
    exact = compute_clip_signals(payload)
    body_names = tuple(str(name) for name in payload["body_names"].tolist())
    model = resolve_inertial_model(body_names)
    solver = TorchG1Kinematics(device="cpu", dtype=torch.float64)
    transforms = solver.forward(
        torch.as_tensor(payload["joint_pos"], dtype=torch.float64),
        torch.as_tensor(payload["root_pos_w"], dtype=torch.float64),
        torch.as_tensor(payload["root_quat_wxyz"], dtype=torch.float64),
    )
    differentiable = differentiable_centroidal_signals(
        transforms, model, fps=50.0
    )
    assert differentiable["com_position"].detach().numpy() == pytest.approx(
        exact.com_position, abs=5.0e-7
    )
    assert differentiable["com_velocity"].detach().numpy() == pytest.approx(
        exact.com_velocity, abs=2.0e-5
    )
    assert differentiable["com_acceleration"].detach().numpy()[2:-2] == pytest.approx(
        exact.com_acceleration[2:-2], abs=2.0e-3
    )
    assert differentiable["centroidal_momentum_rate"].detach().numpy()[2:-2] == pytest.approx(
        exact.centroidal_momentum_rate[2:-2], abs=2.0e-3
    )


def test_static_exact_pass_has_zero_seed_and_identity_preserves_contract(
    pipeline_tmp_path: Path,
) -> None:
    source_path, candidate_path, source, candidate = _static_pair(pipeline_tmp_path)
    targets = plan_repair_targets(source, candidate)
    assert targets.maximum_raw_correction_m == pytest.approx(0.0, abs=1.0e-12)
    assert not targets.zmp_mask.any()
    assert not targets.capture_mask.any()

    payload, derivation = derive_contact_repaired_payload(
        source,
        candidate,
        split="train",
        source_v27_checksum_sha256=_sha256(source_path),
        source_v28_checksum_sha256=_sha256(candidate_path),
        source_v27_manifest_checksum_sha256="b" * 64,
        source_v28_manifest_checksum_sha256="c" * 64,
        policy_checksum_sha256="d" * 64,
    )
    assert derivation["status"] == "identity_exact_v29_pass"
    assert payload["retarget_version"].item() == REPAIR_VERSION
    assert payload["contact_repair_schema_version"].item() == REPAIR_SCHEMA_VERSION
    assert np.array_equal(payload["time"], candidate["time"])
    assert np.array_equal(payload["root_quat_wxyz"], candidate["root_quat_wxyz"])
    contract_names = tuple(str(name) for name in candidate["joint_names"].tolist())
    variable_names = set(LEG_JOINT_NAMES) | set(WAIST_BALANCE_JOINT_NAMES)
    frozen = [index for index, name in enumerate(contract_names) if name not in variable_names]
    assert np.array_equal(payload["joint_pos"][:, frozen], candidate["joint_pos"][:, frozen])
    assert np.array_equal(
        payload["contact_repair_source_frame_coordinate"],
        np.arange(9, dtype=np.float64),
    )
    assert audit_contact_pair(source, payload)["passed"] is True


def test_published_v30_archive_is_consumed_by_training_and_evaluation_input_path(
    pipeline_tmp_path: Path,
) -> None:
    source_path, candidate_path, _, candidate = _static_pair(pipeline_tmp_path)
    reference_root = pipeline_tmp_path / "v30" / "g1"
    output = reference_root / "synthetic/static_g1_50hz.npz"
    result = derive_contact_repaired_archive(
        source_path,
        candidate_path,
        output,
        split="train",
        source_v27_manifest_checksum_sha256="a" * 64,
        source_v28_manifest_checksum_sha256="f" * 64,
        policy_checksum_sha256="1" * 64,
        expected_human_path=Path(candidate["source_file"].item()),
    )
    assert result["gate3_kinematic_complete"] is True
    assert result["serialized_metadata"] == {
        "frame_count": 9,
        "fps": 50.0,
        "duration_sec": 0.16,
        "split": "train",
    }
    manifest = pipeline_tmp_path / "v30" / "g1_train.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "accad_g1_contact_aware_repair_manifest_v1",
                "kind": "split:train",
                "split": "train",
                "reference_root": reference_root.relative_to(PIPELINE_ROOT).as_posix(),
                "motions": [
                    {
                        "g1_file": "synthetic/static_g1_50hz.npz",
                        "g1_checksum_sha256": result["checksum_sha256"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    selection = resolve_motion_inputs(
        motions=[],
        manifests=[manifest],
        split="train",
        reference_root=reference_root,
        allow_missing=False,
        max_motions=None,
    )
    assert selection.paths == (output.resolve(),)
    assert selection.checksum_verified_paths == (output.resolve(),)
    library = MotionLibrary.from_manifest(
        manifest,
        reference_root,
        split="train",
        device="cpu",
    )
    query = library.query(0, 0.04)
    assert library.num_motions == 1
    assert library.contract.retarget_version == REPAIR_VERSION
    assert query.joint_pos.shape == (29,)
    # train_tracking.py and evaluate_tracking_suite.py share
    # resolve_motion_inputs, while the live command consumes MotionLibrary.


def test_test_split_is_not_opened_without_explicit_validation_witness(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = False

    def forbidden(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("sealed test manifests must not be opened")

    monkeypatch.setattr(repair_module, "_load_split_manifests", forbidden)
    with pytest.raises(ContactAwareRepairError, match="train-frozen"):
        build_contact_aware_repair_split(
            split="test", output_root=pipeline_tmp_path / "sealed"
        )
    assert opened is False
    assert not (pipeline_tmp_path / "sealed" / "test_consumption_receipt.json").exists()


def test_low_level_test_derivation_cannot_bypass_one_shot_builder(
    pipeline_tmp_path: Path,
) -> None:
    source_path, candidate_path, source, candidate = _static_pair(pipeline_tmp_path)
    with pytest.raises(ContactAwareRepairError, match="low-level test repair is sealed"):
        derive_contact_repaired_payload(
            source,
            candidate,
            split="test",
            source_v27_checksum_sha256=_sha256(source_path),
            source_v28_checksum_sha256=_sha256(candidate_path),
            source_v27_manifest_checksum_sha256="a" * 64,
            source_v28_manifest_checksum_sha256="b" * 64,
            policy_checksum_sha256="c" * 64,
        )
    with pytest.raises(ContactAwareRepairError, match="archive access is sealed"):
        derive_contact_repaired_archive(
            pipeline_tmp_path / "must_not_open_source_test.npz",
            pipeline_tmp_path / "must_not_open_candidate_test.npz",
            pipeline_tmp_path / "must_not_write_test.npz",
            split="test",
            source_v27_manifest_checksum_sha256="a" * 64,
            source_v28_manifest_checksum_sha256="b" * 64,
            policy_checksum_sha256="c" * 64,
        )


def test_world_contact_anchor_gate_closes_uniform_root_lift_loophole(
    pipeline_tmp_path: Path,
) -> None:
    _, _, source, candidate = _static_pair(pipeline_tmp_path)
    lifted_root = np.asarray(candidate["root_pos_w"], dtype=np.float64).copy()
    lifted_root[:, 2] += 0.10
    lifted = repair_module._recompute_payload(
        candidate, candidate["joint_pos"], lifted_root
    )
    # Relative centroidal/support geometry is translation invariant, which is
    # why the independent absolute-world postcondition is necessary.
    assert audit_contact_pair(source, lifted)["passed"] is True
    result = audit_contact_anchor_postconditions(candidate, lifted)
    assert result["passed"] is False
    assert "active_contact_support_above_ground" in result["violations"]
    assert "active_contact_support_anchor_displacement" in result["violations"]

    source_with_stance = {name: value.copy() for name, value in candidate.items()}
    for name in (
        "left_heel_stance",
        "left_toe_stance",
        "right_heel_stance",
        "right_toe_stance",
    ):
        source_with_stance[name] = np.ones(9, dtype=bool)
    repaired_without_stance = {name: value.copy() for name, value in candidate.items()}
    with pytest.raises(ContactAwareRepairError, match="stance-field presence"):
        audit_contact_anchor_postconditions(
            source_with_stance, repaired_without_stance
        )


def test_serialized_root_xy_trust_region_is_radial_not_per_axis(
    pipeline_tmp_path: Path,
) -> None:
    _, _, source, candidate = _static_pair(pipeline_tmp_path)
    tampered = {name: value.copy() for name, value in candidate.items()}
    tampered["root_pos_w"][:, 0] += 0.30
    tampered["root_pos_w"][:, 1] += 0.30
    with pytest.raises(ContactAwareRepairError, match="radial XY"):
        repair_module._assert_immutable_contract(source, candidate, tampered)


def test_seed_trust_region_quarantines_instead_of_silent_clipping(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, source, candidate = _static_pair(pipeline_tmp_path)
    signals = compute_clip_signals(candidate)
    displaced = signals.centroidal_zmp.copy()
    displaced[signals.eligible_support, 0] += 2.0
    synthetic = repair_module.ClipSignals(
        **{
            **signals.__dict__,
            "centroidal_zmp": displaced,
            "zmp_margin_optimistic": np.where(
                signals.eligible_support, -2.0, signals.zmp_margin_optimistic
            ),
        }
    )
    calls = iter((compute_clip_signals(source), synthetic))
    monkeypatch.setattr(repair_module, "compute_clip_signals", lambda arrays: next(calls))
    with pytest.raises(repair_module.ContactAwareRepairQuarantine) as caught:
        plan_repair_targets(source, candidate)
    assert caught.value.evidence["action"] == "quarantine_without_clipping"
    assert caught.value.evidence["maximum_raw_correction_m"] > 0.65


def test_opposing_zmp_and_capture_targets_cannot_cancel_seed_trust_gate(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, source, candidate = _static_pair(pipeline_tmp_path)
    single_source = {name: value.copy() for name, value in source.items()}
    single_candidate = {name: value.copy() for name, value in candidate.items()}
    for arrays in (single_source, single_candidate):
        arrays["right_heel_contact"][:] = False
        arrays["right_toe_contact"][:] = False
        arrays["right_foot_contact"][:] = False
    source_signals = compute_clip_signals(single_source)
    candidate_signals = compute_clip_signals(single_candidate)
    zmp = candidate_signals.centroidal_zmp.copy()
    capture = candidate_signals.capture_point.copy()
    active = candidate_signals.eligible_single_support
    zmp[active, 0] += 2.0
    capture[active, 0] -= 2.0
    synthetic = repair_module.ClipSignals(
        **{
            **candidate_signals.__dict__,
            "centroidal_zmp": zmp,
            "capture_point": capture,
            "zmp_margin_optimistic": np.where(
                active, -2.0, candidate_signals.zmp_margin_optimistic
            ),
            "capture_margin_optimistic": np.where(
                active, -2.0, candidate_signals.capture_margin_optimistic
            ),
        }
    )
    calls = iter((source_signals, synthetic))
    monkeypatch.setattr(repair_module, "compute_clip_signals", lambda arrays: next(calls))
    with pytest.raises(repair_module.ContactAwareRepairQuarantine) as caught:
        plan_repair_targets(single_source, single_candidate)
    assert caught.value.evidence["maximum_individual_target_correction_m"] > 0.65


def test_test_receipt_creation_is_exclusive(pipeline_tmp_path: Path) -> None:
    receipt = pipeline_tmp_path / "receipt.json"
    repair_module._exclusive_json(receipt, {"status": "first"})
    with pytest.raises(ContactAwareRepairError, match="already been consumed"):
        repair_module._exclusive_json(receipt, {"status": "second"})
    assert json.loads(receipt.read_text(encoding="utf-8")) == {"status": "first"}


def test_validation_manifest_overrides_are_rejected_before_any_manifest_open(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = False

    def forbidden(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("non-train override must fail before manifest access")

    monkeypatch.setattr(repair_module, "_load_split_manifests", forbidden)
    with pytest.raises(ContactAwareRepairError, match="overrides are forbidden"):
        build_contact_aware_repair_split(
            split="validation",
            v27_manifest_path=pipeline_tmp_path / "cherry_pick_v27.json",
            v28_manifest_path=pipeline_tmp_path / "cherry_pick_v28.json",
            output_root=pipeline_tmp_path / "holdout",
        )
    assert opened is False


def test_canonical_validation_registry_rejects_manifest_metadata_forgery() -> None:
    path = repair_module.DEFAULT_V28_MANIFESTS["validation"].resolve()
    document = repair_module._load_json(path, "canonical validation manifest")
    repair_module._verify_canonical_manifest_commitment(
        stage="v28", split="validation", path=path, document=document
    )
    forged = {**document, "num_frames": int(document["num_frames"]) - 1}
    with pytest.raises(ContactAwareRepairError, match="num_frames commitment mismatch"):
        repair_module._verify_canonical_manifest_commitment(
            stage="v28", split="validation", path=path, document=forged
        )


def test_v28_record_metadata_must_match_serialized_archive_scalars(
    pipeline_tmp_path: Path,
) -> None:
    _, _, _, candidate = _static_pair(pipeline_tmp_path)
    record = {
        "frame_count": 9,
        "fps": 50.0,
        "duration_sec": 0.16,
        "split": "train",
    }
    actual = repair_module._validate_v28_record_metadata(
        record, candidate, split="train"
    )
    assert actual == {
        "frame_count": 9,
        "fps": 50.0,
        "duration_sec": 0.16,
        "split": "train",
    }
    attacks = (
        {**record, "frame_count": 8},
        {**record, "fps": 60.0},
        {**record, "duration_sec": 0.18},
        {**record, "split": "validation"},
    )
    for forged in attacks:
        with pytest.raises(ContactAwareRepairError, match="serialized archive metadata"):
            repair_module._validate_v28_record_metadata(
                forged, candidate, split="train"
            )


def _write_json(path: Path, document: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return _sha256(path)


def _write_current_policy(output_root: Path) -> tuple[Path, str]:
    v27_path = repair_module.DEFAULT_V27_MANIFESTS["train"].resolve()
    v28_path = repair_module.DEFAULT_V28_MANIFESTS["train"].resolve()
    v27, v28 = repair_module._load_split_manifests("train", v27_path, v28_path)
    policy = repair_module._policy_document(v27_path, v28_path, v27, v28)
    policy_path = output_root / "policy.json"
    return policy_path, _write_json(policy_path, policy)


def test_forged_validation_witness_without_declared_plan_cannot_unlock_test(
    pipeline_tmp_path: Path,
) -> None:
    output_root = pipeline_tmp_path / "forged_witness"
    policy_path, policy_checksum = _write_current_policy(output_root)
    plan_path = output_root / "g1_validation_plan.json"
    manifest_path = output_root / "g1_validation.json"
    report = {
        "schema_version": repair_module.REPORT_SCHEMA_VERSION,
        "status": "completed",
        "split": "validation",
        "counts": {"source": 1, "published": 1, "quarantined": 0},
        "acceptance": {
            "ready_for_test": True,
            "policy_checksum_sha256": policy_checksum,
        },
        "artifacts": {
            "policy": {
                "path": policy_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": policy_checksum,
            },
            "plan": {
                "path": plan_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": "0" * 64,
            },
            "manifest": {
                "path": manifest_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": "1" * 64,
            },
            "reference_root": (output_root / "g1").relative_to(
                PIPELINE_ROOT
            ).as_posix(),
        },
    }
    report_path = output_root / "g1_validation_report.json"
    report_checksum = _write_json(report_path, report)
    with pytest.raises(ContactAwareRepairError, match="repair plan is missing"):
        repair_module._authorize_test_once(
            output_root=output_root,
            policy_checksum=policy_checksum,
            witness_path=report_path,
            witness_checksum_sha256=report_checksum,
        )
    assert not (output_root / "test_consumption_receipt.json").exists()


def test_stale_validation_archive_cannot_unlock_test(
    pipeline_tmp_path: Path,
) -> None:
    output_root = pipeline_tmp_path / "stale_archive"
    policy_path, policy_checksum = _write_current_policy(output_root)
    reference_root = output_root / "g1"
    relative = "synthetic/validation_g1_50hz.npz"
    archive_path = reference_root / relative
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(b"committed-validation-archive")
    archive_checksum = _sha256(archive_path)
    archive_size = archive_path.stat().st_size
    metadata = {
        "frame_count": 9,
        "fps": 50.0,
        "duration_sec": 0.16,
        "split": "validation",
    }
    counts = {"source": 1, "published": 1, "quarantined": 0}
    plan_path = output_root / "g1_validation_plan.json"
    plan = {
        "schema_version": repair_module.PLAN_SCHEMA_VERSION,
        "status": "completed",
        "split": "validation",
        "inputs": {
            "v27_manifest_path": repair_module.CANONICAL_MANIFEST_REGISTRY["v27"][
                "validation"
            ]["path"],
            "v27_manifest_checksum_sha256": repair_module.CANONICAL_MANIFEST_REGISTRY[
                "v27"
            ]["validation"]["checksum_sha256"],
            "v28_manifest_path": repair_module.CANONICAL_MANIFEST_REGISTRY["v28"][
                "validation"
            ]["path"],
            "v28_manifest_checksum_sha256": repair_module.CANONICAL_MANIFEST_REGISTRY[
                "v28"
            ]["validation"]["checksum_sha256"],
        },
        "frozen_policy": {
            "path": policy_path.relative_to(PIPELINE_ROOT).as_posix(),
            "checksum_sha256": policy_checksum,
            "algorithm_checksum_sha256": repair_module._canonical_checksum(
                repair_module.REPAIR_ALGORITHM
            ),
        },
        "counts": counts,
        "records": [
            {
                "g1_file": relative,
                "status": "published",
                "derivation": {
                    "checksum_sha256": archive_checksum,
                    "bytes": archive_size,
                    "serialized_metadata": metadata,
                },
            }
        ],
    }
    plan_checksum = _write_json(plan_path, plan)
    motion = {
        "g1_file": relative,
        "g1_checksum_sha256": archive_checksum,
        "g1_size_bytes": archive_size,
        **metadata,
        "human_file": "work/human/unused.npz",
        "human_checksum_sha256": "2" * 64,
        "contact_repair_policy_checksum_sha256": policy_checksum,
        "contact_repair_exact_v29_gate_complete": True,
        "gate3_kinematic_complete": True,
    }
    motion_set_checksum = hashlib.sha256(
        json.dumps(
            [["validation", relative, archive_checksum]], separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    v28_commitment = repair_module.CANONICAL_MANIFEST_REGISTRY["v28"][
        "validation"
    ]
    manifest_path = output_root / "g1_validation.json"
    manifest = {
        "schema_version": repair_module.MANIFEST_SCHEMA_VERSION,
        "kind": "split:validation",
        "split": "validation",
        "reference_root": reference_root.relative_to(PIPELINE_ROOT).as_posix(),
        "motion_set_checksum_sha256": motion_set_checksum,
        "source_v28_motion_set_checksum_sha256": v28_commitment[
            "motion_set_checksum_sha256"
        ],
        "source_v28_manifest_checksum_sha256": v28_commitment[
            "checksum_sha256"
        ],
        "contact_repair_policy_checksum_sha256": policy_checksum,
        "contact_repair_plan_checksum_sha256": plan_checksum,
        "num_source_motions": 1,
        "num_motions": 1,
        "num_quarantined": 0,
        "num_frames": 9,
        "duration_sec": 0.16,
        "motions": [motion],
    }
    manifest_checksum = _write_json(manifest_path, manifest)
    report_path = output_root / "g1_validation_report.json"
    report = {
        "schema_version": repair_module.REPORT_SCHEMA_VERSION,
        "status": "completed",
        "split": "validation",
        "counts": counts,
        "acceptance": {
            "ready_for_test": True,
            "policy_checksum_sha256": policy_checksum,
        },
        "artifacts": {
            "policy": {
                "path": policy_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": policy_checksum,
            },
            "plan": {
                "path": plan_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": plan_checksum,
            },
            "manifest": {
                "path": manifest_path.relative_to(PIPELINE_ROOT).as_posix(),
                "checksum_sha256": manifest_checksum,
            },
            "reference_root": reference_root.relative_to(PIPELINE_ROOT).as_posix(),
        },
    }
    report_checksum = _write_json(report_path, report)
    archive_path.write_bytes(b"stale----validation-archive")
    with pytest.raises(ContactAwareRepairError, match="missing or stale"):
        repair_module._authorize_test_once(
            output_root=output_root,
            policy_checksum=policy_checksum,
            witness_path=report_path,
            witness_checksum_sha256=report_checksum,
        )
    assert not (output_root / "test_consumption_receipt.json").exists()


@pytest.mark.parametrize(
    ("split", "counts"),
    (
        ("validation", {"published": 0, "quarantined": 2}),
        ("validation", {"published": 1, "quarantined": 1}),
        ("test", {"published": 1, "quarantined": 1}),
    ),
)
def test_holdout_quarantine_is_failed_acceptance_and_production_cli_is_disabled(
    split: str,
    counts: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = repair_module._report_status(split, counts)
    assert status == "failed_acceptance"
    monkeypatch.setattr(
        repair_cli,
        "parse_args",
        lambda: SimpleNamespace(
            split=split,
            v27_manifest=None,
            v28_manifest=None,
            output_root=Path("unused"),
            force=False,
            validation_pass_witness=None,
            validation_pass_witness_sha256=None,
        ),
    )
    assert repair_cli.main() == 2
