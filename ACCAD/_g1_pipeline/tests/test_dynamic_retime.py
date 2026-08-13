from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import accad_g1.dynamic_retime as dynamic_retime_module
from accad_g1.dynamic_retime import (
    ARCHIVE_DERIVATION_SCHEMA_VERSION,
    DYNAMIC_RETARGET_VERSION,
    DynamicRetimeError,
    DynamicRetimeQuarantine,
    _retime_grid,
    audit_dynamic_feasibility,
    derive_retimed_archive,
    derive_retimed_payload,
)
from accad_g1.g1_kinematics import load_g1_contract
from accad_g1.g1_validation import validate_g1_archive
from test_g1_validation import _build_archive


PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def _audit_arrays(*, frames: int = 8, flight: slice | None = None) -> dict[str, np.ndarray]:
    contract = load_g1_contract()
    joint_position = np.broadcast_to(contract.default, (frames, 29)).copy()
    root_position = np.zeros((frames, 3), dtype=np.float64)
    root_position[:, 2] = 0.8
    root_quaternion = np.zeros((frames, 4), dtype=np.float64)
    root_quaternion[:, 0] = 1.0
    contact = np.ones(frames, dtype=bool)
    if flight is not None:
        contact[flight] = False
    return {
        "fps": np.asarray(50.0),
        "joint_pos": joint_position,
        "root_pos_w": root_position,
        "root_quat_wxyz": root_quaternion,
        "left_foot_contact": contact,
        "right_foot_contact": contact,
    }


def test_frozen_formula_uses_segment_rates_and_quantized_scale() -> None:
    arrays = _audit_arrays()
    # 0.05 m per 20 ms is 2.5 m/s, hence root factor 1.25 exactly.
    arrays["root_pos_w"][:, 0] = np.arange(8) * 0.05
    audit = audit_dynamic_feasibility(arrays)
    assert audit["factors"]["root_linear_speed"] == pytest.approx(1.25)
    assert audit["scale_pre"] == 1.25
    assert audit["playback_speed_equivalent"] == pytest.approx(0.8)
    assert audit["action"] == "uniform_retime"


def test_flight_policy_is_fail_closed_at_three_frames() -> None:
    two = _audit_arrays(flight=slice(2, 4))
    three = _audit_arrays(flight=slice(2, 5))
    two["root_pos_w"][:, 0] = np.arange(8) * 0.05
    three["root_pos_w"][:, 0] = np.arange(8) * 0.05
    audit_two = audit_dynamic_feasibility(two)
    audit_three = audit_dynamic_feasibility(three)
    assert audit_two["flight"]["longest_consecutive_frames"] == 2
    assert audit_two["action"] == "uniform_retime"
    assert audit_three["flight"]["longest_consecutive_frames"] == 3
    assert audit_three["action"] == "quarantine"


def test_retime_grid_preserves_50hz_endpoint_with_realized_headroom() -> None:
    indices, realized = _retime_grid(8, 1.15, 50.0)
    assert len(indices) == 10
    assert indices[0] == 0.0
    assert indices[-1] == 7.0
    assert realized == pytest.approx(9.0 / 7.0)
    assert realized >= 1.15
    assert np.all(np.diff(indices) > 0.0)


def test_uniform_retime_recomputes_fk_and_passes_independent_gate3(
    pipeline_tmp_path: Path,
) -> None:
    source = _build_archive(pipeline_tmp_path / "source")
    output = pipeline_tmp_path / "dynamic" / "g1_reference.npz"
    result = derive_retimed_archive(
        source,
        output,
        requested_scale=2.0,
        source_manifest_checksum_sha256="a" * 64,
        split="train",
    )
    assert result["source_frame_count"] == 6
    assert result["output_frame_count"] == 11
    with np.load(output, allow_pickle=False) as archive:
        assert archive["retarget_version"].item() == DYNAMIC_RETARGET_VERSION
        assert (
            archive["dynamic_derivation_schema_version"].item()
            == ARCHIVE_DERIVATION_SCHEMA_VERSION
        )
        assert archive["dynamic_requested_scale"].item() == 2.0
        assert archive["fps"].item() == 50.0
        assert np.array_equal(archive["time"], np.arange(11) / 50.0)
        expected_source = Path(archive["source_file"].item()).resolve()
    validation = validate_g1_archive(output, expected_source_path=expected_source)
    assert validation["gate3_kinematic_complete"] is True


def test_identity_derivative_preserves_every_frame_aligned_array(
    pipeline_tmp_path: Path,
) -> None:
    source = _build_archive(pipeline_tmp_path / "source_identity")
    output = pipeline_tmp_path / "dynamic" / "g1_identity.npz"
    result = derive_retimed_archive(
        source,
        output,
        requested_scale=1.0,
        source_manifest_checksum_sha256="d" * 64,
        split="train",
    )
    assert result["final_action"] == "identity_numeric_copy"
    assert result["headroom_attempts"][0]["post_scale_pre"] == 1.0
    with np.load(source, allow_pickle=False) as before, np.load(
        output, allow_pickle=False
    ) as after:
        frame_count = int(before["frame_count"].item())
        for name in before.files:
            value = before[name]
            if value.ndim >= 1 and value.shape[0] == frame_count:
                assert np.array_equal(after[name], value), name
        assert after["dynamic_derivation_mode"].item() == "identity_numeric_copy"


def test_post_fk_headroom_retry_has_strict_monotonic_witness(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _build_archive(pipeline_tmp_path / "source_retry")
    output = pipeline_tmp_path / "dynamic" / "g1_retry.npz"
    calls: list[float] = []

    def fake_payload(arrays, *, requested_scale: float, **kwargs):
        calls.append(float(requested_scale))
        indices, realized = _retime_grid(6, requested_scale, 50.0)
        failed = len(calls) == 1
        post_audit = {
            "scale_pre": 1.1 if failed else 1.0,
            "raw_scale": 1.07 if failed else 1.0,
            "factors": {"synthetic": 1.07 if failed else 1.0},
            "flight": {"longest_consecutive_frames": 0},
        }
        return {"payload": np.arange(len(indices))}, {
            "requested_scale": requested_scale,
            "realized_scale": realized,
            "initial_scale_pre": kwargs["initial_scale_pre"],
            "source_frame_count": 6,
            "output_frame_count": len(indices),
            "source_duration_sec": 0.1,
            "output_duration_sec": (len(indices) - 1) / 50.0,
            "ground_projection": {},
            "derivation_mode": "synthetic",
            "post_audit": post_audit,
        }

    monkeypatch.setattr(dynamic_retime_module, "derive_retimed_payload", fake_payload)
    result = derive_retimed_archive(
        source,
        output,
        requested_scale=2.0,
        source_manifest_checksum_sha256="e" * 64,
        split="train",
    )
    assert len(calls) == 2
    assert result["final_action"] == "uniform_retime_with_post_fk_headroom"
    assert result["monotonic_witness"] == {
        "requested_scale_strictly_increasing": True,
        "output_frame_count_strictly_increasing": True,
    }


def test_post_fk_extra_stretch_with_three_frame_flight_is_quarantined(
    pipeline_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _build_archive(pipeline_tmp_path / "source_flight")
    output = pipeline_tmp_path / "dynamic" / "must_not_publish.npz"

    def fake_payload(arrays, *, requested_scale: float, **kwargs):
        return {"payload": np.arange(11)}, {
            "requested_scale": requested_scale,
            "realized_scale": 2.0,
            "initial_scale_pre": kwargs["initial_scale_pre"],
            "source_frame_count": 6,
            "output_frame_count": 11,
            "source_duration_sec": 0.1,
            "output_duration_sec": 0.2,
            "ground_projection": {},
            "derivation_mode": "synthetic",
            "post_audit": {
                "scale_pre": 1.1,
                "raw_scale": 1.02,
                "factors": {"synthetic": 1.02},
                "flight": {"longest_consecutive_frames": 3},
            },
        }

    monkeypatch.setattr(dynamic_retime_module, "derive_retimed_payload", fake_payload)
    with pytest.raises(DynamicRetimeQuarantine) as caught:
        derive_retimed_archive(
            source,
            output,
            requested_scale=2.0,
            source_manifest_checksum_sha256="f" * 64,
            split="train",
        )
    assert caught.value.evidence["final_action"] == "quarantine_post_fk_headroom_flight"
    assert not output.exists()


def test_unknown_frame_aligned_array_is_rejected(
    pipeline_tmp_path: Path,
) -> None:
    source = _build_archive(pipeline_tmp_path / "source")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["uncontracted_signal"] = np.zeros((6, 2), dtype=np.float32)
    with pytest.raises(DynamicRetimeError, match="unrecognized frame-aligned"):
        derive_retimed_payload(
            arrays,
            requested_scale=1.0,
            source_archive_checksum_sha256="b" * 64,
            source_manifest_checksum_sha256="c" * 64,
            split="train",
        )


def test_production_train_scale_plan_regression_matches_frozen_audit() -> None:
    manifest_path = PIPELINE_ROOT / "work" / "manifests" / "g1_train.json"
    if not manifest_path.is_file():
        pytest.skip("production train manifest is not installed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_root = PIPELINE_ROOT / manifest["reference_root"]
    actions: list[str] = []
    scales: list[float] = []
    selected: dict[str, dict[str, object]] = {}
    for record in manifest["motions"]:
        path = reference_root / record["g1_file"]
        with np.load(path, allow_pickle=False) as archive:
            audit = audit_dynamic_feasibility(
                {name: archive[name] for name in archive.files}
            )
        actions.append(audit["action"])
        scales.append(audit["scale_pre"])
        if any(name in record["g1_file"] for name in ("A1_-_Stand", "B3_-_walk1", "C3_-_Run")):
            selected[record["g1_file"]] = audit

    assert {name: actions.count(name) for name in set(actions)} == {
        "identity": 16,
        "uniform_retime": 93,
        "quarantine": 106,
    }
    assert np.percentile(scales, (50, 95, 100)).tolist() == pytest.approx(
        [1.75, 3.1, 11.0]
    )
    stand = next(value for name, value in selected.items() if "A1_-_Stand" in name)
    walk = next(value for name, value in selected.items() if "B3_-_walk1" in name)
    run = next(value for name, value in selected.items() if "C3_-_Run" in name)
    assert (stand["scale_pre"], stand["flight"]["longest_consecutive_frames"]) == (1.0, 0)
    assert (walk["scale_pre"], walk["flight"]["longest_consecutive_frames"]) == (1.35, 0)
    assert (run["scale_pre"], run["flight"]["longest_consecutive_frames"]) == (2.6, 13)
    assert run["action"] == "quarantine"
