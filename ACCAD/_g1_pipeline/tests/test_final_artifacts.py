from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from accad_g1 import final_artifacts
from accad_g1.evaluation_metrics import build_suite_acceptance
from accad_g1.tracking_task import (
    RECOVERY_ALIVE_REWARD_WEIGHT,
    RECOVERY_FAILURE_REWARD_WEIGHT,
    RECOVERY_PPO_DESIRED_KL,
    RECOVERY_PPO_LEARNING_RATE,
    RECOVERY_PPO_SCHEDULE,
    RECOVERY_RESET_STATE_NOISE_SCALE,
    TRACKING_POLICY_CONTRACT_VERSION,
    tracking_observation_contract,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_checksum(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def test_finalizer_recovery_constants_match_the_live_task_contract() -> None:
    assert final_artifacts.EXPECTED_TRACKING_POLICY_CONTRACT == TRACKING_POLICY_CONTRACT_VERSION
    assert final_artifacts.EXPECTED_RECOVERY_PPO_LEARNING_RATE == RECOVERY_PPO_LEARNING_RATE
    assert final_artifacts.EXPECTED_RECOVERY_PPO_SCHEDULE == RECOVERY_PPO_SCHEDULE
    assert final_artifacts.EXPECTED_RECOVERY_PPO_DESIRED_KL == RECOVERY_PPO_DESIRED_KL
    assert final_artifacts.EXPECTED_RESET_STATE_NOISE_SCALE == RECOVERY_RESET_STATE_NOISE_SCALE
    assert final_artifacts.EXPECTED_ALIVE_REWARD_WEIGHT == RECOVERY_ALIVE_REWARD_WEIGHT
    assert final_artifacts.EXPECTED_FAILURE_REWARD_WEIGHT == RECOVERY_FAILURE_REWARD_WEIGHT
    assert final_artifacts.EXPECTED_TRAINING_MOTION_ASSIGNMENT == "balanced-fixed"
    assert final_artifacts.EXPECTED_SOURCE_SPLIT_COUNTS == {
        "train": 215,
        "validation": 5,
        "test": 29,
    }
    assert final_artifacts.EXPECTED_CURRENT_DYNAMIC_COUNTS == {
        "train": 103,
        "validation": 2,
    }
    assert not hasattr(final_artifacts, "REPRESENTATIVE_TEST_MOTION_ID")
    assert not hasattr(final_artifacts, "REPRESENTATIVE_TEST_MOTION_KEY")
    assert final_artifacts.EXPECTED_RUNTIME == {
        "action_dim": 29,
        "actor_observation_dim": 293,
        "critic_observation_dim": 505,
    }
    assert dict(final_artifacts.EXPECTED_OBSERVATION_CONTRACT) == (
        tracking_observation_contract()
    )


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _declared(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "checksum_sha256": _sha256(path),
    }


def _library_checksum(
    contract_digest: str,
    motion_paths: list[Path],
    motion_keys: list[str],
) -> str:
    identity = "\n".join(
        [
            f"contract {contract_digest}",
            "manifest -",
            "split -",
            *(
                f"{_sha256(path)} {key}"
                for path, key in zip(motion_paths, motion_keys, strict=True)
            ),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _tracking_tensor_contract_fields() -> dict[str, Any]:
    """Return independent JSON values so each fixture surface can be mutated."""

    return json.loads(
        json.dumps(
            {
                "observation_contract": final_artifacts.EXPECTED_OBSERVATION_CONTRACT,
                "live_tensor_contract": final_artifacts.EXPECTED_LIVE_TENSOR_CONTRACT,
            },
            allow_nan=False,
        )
    )


def _warmup_observation_witness(num_envs: int) -> dict[str, Any]:
    tensors = [
        {
            "path": "observation.critic",
            "shape": [num_envs, 505],
            "dtype": "torch.float32",
        },
        {
            "path": "observation.policy",
            "shape": [num_envs, 293],
            "dtype": "torch.float32",
        },
    ]
    return {
        "tensor_count": 2,
        "element_count": num_envs * (505 + 293),
        "nonfinite_count": 0,
        "all_finite": True,
        "tensors": tensors,
    }


def _physics_warmup_witness(num_envs: int) -> dict[str, Any]:
    observation = _warmup_observation_witness(num_envs)
    return {
        "schema_version": final_artifacts.PHYSICS_WARMUP_SCHEMA_VERSION,
        "enabled": True,
        "requested_control_steps": 100,
        "executed_control_steps": 100,
        "num_environments": num_envs,
        "action_dimension": 29,
        "zero_action_shape": [num_envs, 29],
        "policy_inference_used": False,
        "purpose": "prime_physics_and_contact_caches_only",
        "initial_reset_observations": observation,
        "warmup_done_signals": 0,
        "warmup_terminated_signals": 0,
        "warmup_timeout_signals": 0,
        "warmup_steps_with_any_done": 0,
        "warmup_termination_term_counts": {
            "motion_end": 0,
            "fall": 0,
            "tracking_divergence": 0,
            "joint_limits": 0,
            "nonfinite": 0,
        },
        "final_reset_forced_motion_time_s": 0.0,
        "final_reset_zeroed_reference_state_noise": True,
        "post_reset": {
            "authoritative_motion_frame_zero": True,
            "episode_counters_zero": True,
            "action_state_zero": True,
            "maximum_absolute_values": {
                "episode_length_buf": 0.0,
                "motion_time": 0.0,
                "raw_action": 0.0,
                "normalized_residual": 0.0,
                "previous_normalized_residual": 0.0,
            },
            "reset_buffers": {
                "reset_buf": {"nonzero_count": 0, "all_clear": True},
                "reset_terminated": {"nonzero_count": 0, "all_clear": True},
                "reset_time_outs": {"nonzero_count": 0, "all_clear": True},
            },
            "reference_state_maximum_errors": {
                "joint_position_rad": 0.0,
                "root_position_m": 0.0,
                "root_orientation_quaternion_l2": 0.0,
            },
            "observations": json.loads(json.dumps(observation)),
            "tolerance": 1.0e-5,
            "passed": True,
        },
        "passed": True,
    }


def _make_policy_exports(torchscript_path: Path, onnx_path: Path) -> None:
    """Create executable 293-to-29 policies without touching CUDA."""

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    import torch

    torchscript_path.parent.mkdir(parents=True, exist_ok=True)
    linear = torch.nn.Linear(293, 29, bias=True, device="cpu")
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
    linear.eval()
    traced = torch.jit.trace(linear, torch.zeros((1, 293), dtype=torch.float32))
    torch.jit.save(traced, str(torchscript_path))

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    observation = helper.make_tensor_value_info(
        "observation", onnx.TensorProto.FLOAT, [1, 293]
    )
    action = helper.make_tensor_value_info("action", onnx.TensorProto.FLOAT, [1, 29])
    weights = numpy_helper.from_array(
        np.zeros((293, 29), dtype=np.float32), name="weights"
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["observation", "weights"], ["action"])],
        "tiny_zero_policy",
        [observation],
        [action],
        [weights],
    )
    model = helper.make_model(
        graph,
        producer_name="test_final_artifacts",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))


def _motion_input(
    split: str,
    manifest_path: Path,
    motion_paths: list[Path],
) -> dict[str, Any]:
    return {
        "split": split,
        "allow_missing": False,
        "motion_count": len(motion_paths),
        "manifest_checksum_verified_motion_count": len(motion_paths),
        "manifest_promotion_verified": True,
        "manifests": [
            {
                "path": str(manifest_path.resolve()),
                "checksum_sha256": _sha256(manifest_path),
                "split": split,
                "resolved_count": len(motion_paths),
                "archive_checksum_record_count": len(motion_paths),
            }
        ],
        "motions": [_declared(path) for path in motion_paths],
    }


def _suite_aggregate(count: int, success_count: int) -> dict[str, Any]:
    failed_ids = list(range(success_count, count))
    return {
        "motion_count": count,
        "finished_first_episodes": count,
        "successful_motion_end_episodes": success_count,
        "success_rate": success_count / count,
        "simulation_app_stopped_early": False,
        "failed_or_incomplete_motion_ids": failed_ids,
        "termination_counts": {
            "motion_end": success_count,
            "fall": count - success_count,
            "tracking_divergence": 0,
            "joint_limits": 0,
            "nonfinite": 0,
        },
        "return_mean": 1.0,
        "completion_ratio_mean": 1.0,
        "macro_motion_metrics": {
            "mean_absolute_joint_error_rad": 0.01,
            "mean_root_position_error_m": 0.01,
            "mean_tracking_body_position_error_m": 0.01,
        },
        "weighted_frame_or_element_metrics": {
            "raw_action_clip_fraction": 0.0,
        },
        "aggregate_foot_contact": {"overall": {"f1": 1.0}},
        "aggregate_stance_or_actual_contact_slide_speed_p95_mps": 0.01,
    }


@pytest.fixture
def complete_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    project = tmp_path / "humanoid_G1"
    accad = project / "ACCAD"
    pipeline = accad / "_g1_pipeline"
    package = pipeline / "accad_g1"
    work = pipeline / "work"
    reports = work / "reports"
    manifests_root = work / "manifests"
    reports.mkdir(parents=True)
    package.mkdir(parents=True)

    monkeypatch.setattr(final_artifacts, "PROJECT_ROOT", project)
    monkeypatch.setattr(final_artifacts, "ACCAD_ROOT", accad)
    monkeypatch.setattr(final_artifacts, "PIPELINE_ROOT", pipeline)
    monkeypatch.setattr(final_artifacts, "PACKAGE_ROOT", package)
    monkeypatch.setattr(final_artifacts, "WORK_ROOT", work)
    monkeypatch.setattr(final_artifacts, "REPORT_ROOT", reports)
    representative_video_inspection = {
        "decoder": "opencv",
        "decoded_frame_count": 51,
        "container_declared_frame_count": 51,
        "fps": 50.0,
        "width_px": 1280,
        "height_px": 720,
        "duration_s": 1.02,
        "all_black_frame_count": 1,
        "all_black_frame_fraction": 1.0 / 51.0,
        "near_black_frame_count": 1,
        "visually_nonblack_frame_fraction": 50.0 / 51.0,
        "all_frames_black": False,
        "frame_pair_count": 50,
        "changed_frame_pair_count": 45,
        "changed_frame_pair_fraction": 0.9,
        "temporal_mean_absolute_pixel_change": 2.0,
        "decoded_pixel_min": 0,
        "decoded_pixel_max": 240,
    }
    monkeypatch.setattr(
        final_artifacts,
        "inspect_video",
        lambda _path: dict(representative_video_inspection),
    )
    monkeypatch.setattr(
        final_artifacts,
        "ENTRYPOINT_FILES",
        ("run_isaac.py", "build_final_artifacts.py"),
    )
    monkeypatch.setattr(
        final_artifacts,
        "LOCAL_CONFIG_FILES",
        ("config.yaml", "correspondence.yaml", "requirements-motion.txt", "README.md"),
    )

    for relative in (
        "run_isaac.py",
        "build_final_artifacts.py",
        "accad_g1/__init__.py",
        "accad_g1/final_artifacts.py",
        "tests/test_smoke.py",
        "config.yaml",
        "correspondence.yaml",
        "requirements-motion.txt",
        "README.md",
    ):
        _write_bytes(pipeline / relative, f"fixture:{relative}\n".encode())
    model_path = _write_bytes(
        pipeline / "models" / "smplx" / "SMPLX_NEUTRAL.npz", b"licensed-smplx"
    )
    assert model_path.is_file()
    joint_contract = _write_bytes(
        project / "configs" / "robot" / "g1_29dof.yaml", b"joints: 29\n"
    )
    urdf = _write_bytes(
        project
        / "third_party"
        / "unitree_ros"
        / "robots"
        / "g1_description"
        / "g1_29dof_rev_1_0.urdf",
        b"<robot name='g1'/>\n",
    )
    correspondence = pipeline / "correspondence.yaml"

    split_counts = {"train": 215, "validation": 5, "test": 29}
    split_manifests: dict[str, Path] = {}
    split_motions: dict[str, list[Path]] = {}
    split_records: dict[str, list[dict[str, Any]]] = {}
    combined_records: list[dict[str, Any]] = []
    for split, count in split_counts.items():
        paths: list[Path] = []
        records: list[dict[str, Any]] = []
        for index in range(count):
            relative = Path(split) / f"motion_{index:03d}_g1_50hz.npz"
            motion = _write_bytes(
                work / "g1" / relative,
                f"{split}:{index}:retargeted-g1-motion".encode(),
            )
            paths.append(motion)
            record = {
                "motion_id": f"motion_{index:03d}",
                "source_file": f"{split}/motion_{index:03d}_stageii.npz",
                "g1_file": relative.as_posix(),
                "g1_checksum_sha256": _sha256(motion),
                "g1_size_bytes": motion.stat().st_size,
                "split": split,
                "retarget_version": final_artifacts.RETARGET_VERSION,
                "fps": 50.0,
                "frame_count": 51 + index,
                "duration_sec": (50 + index) / 50.0,
                "gate3_kinematic_complete": True,
            }
            records.append(record)
            combined_records.append({"split": split, **record})
        manifest = _write_json(
            manifests_root / f"g1_{split}.json",
            {
                "schema_version": final_artifacts.GATE3_MANIFEST_SCHEMA,
                "dataset": "ACCAD",
                "kind": f"split:{split}",
                "split": split,
                "gate": "gate3_kinematic_complete",
                "reference_root": "work/g1",
                "motion_set_checksum_sha256": "a" * 64,
                "num_motions": count,
                "retarget_version": final_artifacts.RETARGET_VERSION,
                "motions": records,
            },
        )
        split_manifests[split] = manifest
        split_motions[split] = paths
        split_records[split] = records
    combined_manifest = _write_json(
        manifests_root / "g1_all.json",
        {
            "schema_version": final_artifacts.GATE3_MANIFEST_SCHEMA,
            "split": "combined",
            "num_motions": 249,
            "retarget_version": final_artifacts.RETARGET_VERSION,
            "motions": combined_records,
        },
    )

    dynamic_root = work / "dynamic"
    dynamic_reference_root = dynamic_root / "g1"
    policy_path = _write_json(
        dynamic_root / "scale_policy.json",
        {
            "schema_version": final_artifacts.DYNAMIC_POLICY_SCHEMA,
            "status": "frozen_from_train",
            "created_at_utc": "2026-08-09T00:00:00+00:00",
            "algorithm": final_artifacts.DYNAMIC_POLICY_ALGORITHM,
            "algorithm_checksum_sha256": _canonical_checksum(
                final_artifacts.DYNAMIC_POLICY_ALGORITHM
            ),
            "train_source_manifest": {
                "path": "work/manifests/g1_train.json",
                "checksum_sha256": _sha256(split_manifests["train"]),
                "motion_set_checksum_sha256": "a" * 64,
                "num_motions": 215,
            },
            "split_policy": {
                "train": "allowed_and_freezes_policy",
                "validation": "allowed_only_with_existing_frozen_train_policy",
                "test": "sealed_until_explicit_split_test_invocation",
            },
        },
    )
    policy_checksum = _sha256(policy_path)
    dynamic_counts = {"train": 103, "validation": 2, "test": 3}
    dynamic_manifests: dict[str, Path] = {}
    dynamic_motions: dict[str, list[Path]] = {}
    dynamic_records: dict[str, list[dict[str, Any]]] = {}
    for split, source_count in split_counts.items():
        promoted_count = dynamic_counts[split]
        plan_records: list[dict[str, Any]] = []
        promoted: list[dict[str, Any]] = []
        promoted_paths: list[Path] = []
        for index, source in enumerate(split_records[split]):
            source_base = {
                "source_g1_file": source["g1_file"],
                "source_g1_checksum_sha256": source["g1_checksum_sha256"],
                "source_frame_count": source["frame_count"],
                "source_duration_sec": source["duration_sec"],
            }
            if index >= promoted_count:
                plan_records.append(
                    {
                        **source_base,
                        "audit": {"action": "quarantine"},
                        "source_action": "quarantine",
                        "status": "quarantined",
                        "final_action": "quarantine_source_flight_policy",
                        "quarantine_reason": "fixture frozen dynamic flight policy",
                    }
                )
                continue
            derived_path = _write_bytes(
                dynamic_reference_root / source["g1_file"],
                f"dynamic:{split}:{index}:v28".encode(),
            )
            promoted_paths.append(derived_path)
            derivation = {
                "path": str(derived_path.resolve()),
                "bytes": derived_path.stat().st_size,
                "checksum_sha256": _sha256(derived_path),
                "retarget_version": final_artifacts.DYNAMIC_RETARGET_VERSION,
                "source_action": "identity",
                "final_action": "identity_numeric_copy",
                "source_scale_pre": 1.0,
                "effective_requested_scale": 1.0,
                "realized_scale": 1.0,
                "output_frame_count": source["frame_count"],
                "output_duration_sec": source["duration_sec"],
            }
            plan_records.append(
                {
                    **source_base,
                    "audit": {"action": "identity"},
                    "source_action": "identity",
                    "status": "derived_and_validated",
                    "final_action": "identity_numeric_copy",
                    "derived_g1_file": source["g1_file"],
                    "derivation": derivation,
                    "gate3_kinematic_complete": True,
                }
            )
            promoted.append(
                {
                    **source,
                    "source_v27_g1_file": source["g1_file"],
                    "source_v27_g1_checksum_sha256": source[
                        "g1_checksum_sha256"
                    ],
                    "g1_checksum_sha256": _sha256(derived_path),
                    "g1_size_bytes": derived_path.stat().st_size,
                    "retarget_version": final_artifacts.DYNAMIC_RETARGET_VERSION,
                    "dynamic_source_action": "identity",
                    "dynamic_action": "identity_numeric_copy",
                    "dynamic_scale_pre": 1.0,
                    "dynamic_effective_requested_scale": 1.0,
                    "dynamic_realized_scale": 1.0,
                    "dynamic_policy_checksum_sha256": policy_checksum,
                    "gate3_kinematic_complete": True,
                }
            )
        plan_counts = {
            "source": source_count,
            "identity": promoted_count,
            "uniform_retime": 0,
            "quarantine": source_count - promoted_count,
            "derived": promoted_count,
            "final_quarantined": source_count - promoted_count,
            "source_actions": {
                "identity": promoted_count,
                "uniform_retime": 0,
                "quarantine": source_count - promoted_count,
            },
            "final_actions": {
                "identity_numeric_copy": promoted_count,
                "quarantine_source_flight_policy": source_count - promoted_count,
            },
        }
        plan_path = _write_json(
            dynamic_root / f"g1_{split}_scale_plan.json",
            {
                "schema_version": final_artifacts.DYNAMIC_PLAN_SCHEMA,
                "status": "completed",
                "split": split,
                "created_at_utc": "2026-08-09T00:00:00+00:00",
                "source_manifest": {
                    "path": f"work/manifests/g1_{split}.json",
                    "checksum_sha256": _sha256(split_manifests[split]),
                    "motion_set_checksum_sha256": "a" * 64,
                },
                "frozen_policy": {
                    "path": "work/dynamic/scale_policy.json",
                    "checksum_sha256": policy_checksum,
                    "algorithm_checksum_sha256": _canonical_checksum(
                        final_artifacts.DYNAMIC_POLICY_ALGORITHM
                    ),
                },
                "counts": plan_counts,
                "records": plan_records,
            },
        )
        promoted.sort(key=lambda record: record["g1_file"])
        promoted_paths.sort(key=lambda path: path.relative_to(dynamic_reference_root).as_posix())
        motion_set_checksum = hashlib.sha256(
            json.dumps(
                [
                    [split, record["g1_file"], record["g1_checksum_sha256"]]
                    for record in promoted
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        dynamic_manifest = _write_json(
            dynamic_root / f"g1_{split}.json",
            {
                "schema_version": final_artifacts.DYNAMIC_MANIFEST_SCHEMA,
                "dataset": "ACCAD",
                "kind": f"split:{split}",
                "split": split,
                "gate": "gate3_kinematic_complete+dynamic_prefilter_v1",
                "reference_root": "work/dynamic/g1",
                "retarget_version": final_artifacts.DYNAMIC_RETARGET_VERSION,
                "motion_set_checksum_sha256": motion_set_checksum,
                "source_motion_set_checksum_sha256": "a" * 64,
                "source_manifest_checksum_sha256": _sha256(
                    split_manifests[split]
                ),
                "dynamic_policy_checksum_sha256": policy_checksum,
                "scale_plan_checksum_sha256": _sha256(plan_path),
                "num_source_motions": source_count,
                "num_motions": promoted_count,
                "num_source_policy_quarantined": source_count - promoted_count,
                "num_quarantined": source_count - promoted_count,
                "num_frames": sum(record["frame_count"] for record in promoted),
                "duration_sec": sum(record["duration_sec"] for record in promoted),
                "motions": promoted,
            },
        )
        _write_json(
            dynamic_root / f"g1_{split}_report.json",
            {
                "schema_version": final_artifacts.DYNAMIC_REPORT_SCHEMA,
                "status": "completed",
                "split": split,
                "started_at_utc": "2026-08-09T00:00:00+00:00",
                "finished_at_utc": "2026-08-09T00:01:00+00:00",
                "counts": plan_counts,
                "validation_failures": [],
                "sealed_test_contract": (
                    "test NPZ numerical contents are opened only by an explicit "
                    "split=test invocation"
                ),
                "artifacts": {
                    "policy": {
                        "path": "work/dynamic/scale_policy.json",
                        "checksum_sha256": policy_checksum,
                    },
                    "plan": {
                        "path": f"work/dynamic/g1_{split}_scale_plan.json",
                        "checksum_sha256": _sha256(plan_path),
                    },
                    "manifest": {
                        "path": f"work/dynamic/g1_{split}.json",
                        "checksum_sha256": _sha256(dynamic_manifest),
                    },
                    "derived_reference_root": "work/dynamic/g1",
                },
            },
        )
        dynamic_manifests[split] = dynamic_manifest
        dynamic_motions[split] = promoted_paths
        dynamic_records[split] = promoted

    _write_json(
        manifests_root / "gate1_summary.json",
        {
            "schema_version": "accad-gate1-v1",
            "passed": True,
            "counts": {"eligible_files": 249},
        },
    )
    for name in (
        "accad_inventory.json",
        "accad_quality.json",
        "accad_train.json",
        "accad_validation.json",
        "accad_test.json",
        "accad_excluded.json",
    ):
        _write_json(manifests_root / name, {"fixture": name})

    batch_items = reports / "batch" / "batch_items.jsonl"
    batch_items.parent.mkdir(parents=True, exist_ok=True)
    batch_items.write_text(
        "".join(json.dumps({"index": index}) + "\n" for index in range(249)),
        encoding="utf-8",
    )
    _write_json(
        reports / "batch" / "batch_summary.json",
        {
            "schema_version": "accad_g1_batch_v1",
            "all_selected_items_completed": True,
            "settings": {"max_items": None},
            "counts": {"selected_total": 249, "completed": 249, "failed": 0},
            "artifacts": {
                "items_jsonl": {
                    "path": str(batch_items.resolve()),
                    "checksum_sha256": _sha256(batch_items),
                    "records": 249,
                }
            },
        },
    )
    _write_json(
        reports / "batch" / "parallel_batch_summary.json",
        {
            "schema_version": "accad_g1_parallel_batch_v1",
            "status": "completed",
            "success": True,
            "worker_problem_count": 0,
            "retarget_version": final_artifacts.RETARGET_VERSION,
        },
    )
    gate3_paths = {
        **split_manifests,
        "combined": combined_manifest,
    }
    _write_json(
        manifests_root / "g1_manifest_build.json",
        {
            "schema_version": "accad_g1_gate3_manifest_build_v1",
            "status": "completed",
            "retarget_version": final_artifacts.RETARGET_VERSION,
            "motion_set_checksum_sha256": "d" * 64,
            "counts": {"promoted_total": 249, "source_completed_records": 249},
            "manifests": {
                name: {
                    "path": path.relative_to(pipeline).as_posix(),
                    "checksum_sha256": _sha256(path),
                }
                for name, path in gate3_paths.items()
            },
        },
    )

    b3 = _write_bytes(
        work / "g1" / "Female1Walking_c3d" / "B3_-_walk1_g1_50hz.npz",
        b"canonical-b3-retarget",
    )
    validator = pipeline / "run_isaac.py"
    validation_asset = {
        "joint_contract": str(joint_contract.resolve()),
        "joint_contract_sha256": _sha256(joint_contract),
        "urdf": str(urdf.resolve()),
        "urdf_sha256": _sha256(urdf),
    }
    for mode, report_name, video_name, passed in (
        (
            "kinematic",
            "B3_-_walk1_isaac_kinematic.json",
            "B3_-_walk1_isaac_kinematic.mp4",
            True,
        ),
        (
            "pd-replay",
            "B3_-_walk1_isaac_pd_replay.json",
            "B3_-_walk1_isaac_pd_replay.mp4",
            False,
        ),
    ):
        video = _write_bytes(work / "media" / video_name, f"video:{mode}".encode())
        _write_json(
            reports / report_name,
            {
                "schema_version": "accad_g1_isaac_validation_v3",
                "mode": mode,
                "execution_complete": True,
                "passed": passed,
                "dynamics_passed": passed,
                "input": {
                    "full_clip": True,
                    "validated_frames": 20,
                    "total_frames": 20,
                    "sha256": _sha256(b3),
                },
                "validator": {
                    "path": str(validator.resolve()),
                    "sha256": _sha256(validator),
                },
                "asset": validation_asset,
                "video": {
                    "enabled": True,
                    "path": str(video.resolve()),
                    "frames": 20,
                    "all_black_frames": 0,
                },
            },
        )

    run_dir = work / "training" / "runs" / "fixture_final"
    for relative in (
        "runtime_contract.json",
        "params/env.yaml",
        "params/agent.yaml",
        "events.out.tfevents.fixture",
    ):
        _write_bytes(run_dir / relative, f"fixture:{relative}\n".encode())
    checkpoints = [
        _write_bytes(run_dir / f"model_{iteration}.pt", f"checkpoint:{iteration}".encode())
        for iteration in final_artifacts.EXPECTED_CHECKPOINT_ITERATIONS
    ]
    checkpoint = checkpoints[-1]
    initialization_checkpoint = _write_bytes(
        work
        / "training"
        / "runs"
        / "fixture_v5_source"
        / "model_599.pt",
        b"fixture-model-599-policy-and-optimizer-state",
    )
    initialization_source = {
        "path": str(initialization_checkpoint.resolve()),
        "checksum_sha256": _sha256(initialization_checkpoint),
        "bytes": initialization_checkpoint.stat().st_size,
        "iter": 599,
        "iteration": 599,
        "filename_iteration": 599,
        "state_layout_checksum_sha256": "c" * 64,
        "state_key_count": 42,
        "checkpoint_field_loaded": "model_state_dict",
        "checkpoint_optimizer_state_present": True,
    }
    fresh_runner_snapshot = {
        "current_learning_iteration": 0,
        "tot_timesteps": 0,
        "optimizer_state_entry_count": 0,
        "optimizer_param_group_count": 1,
        "optimizer_param_group_parameter_counts": [42],
        "optimizer_param_group_learning_rates": [
            final_artifacts.EXPECTED_RECOVERY_PPO_LEARNING_RATE
        ],
        "tot_time": 0.0,
    }
    policy_initialization = {
        "schema_version": "accad_g1_policy_initialization_v1",
        "mode": "strict_model_state_dict_with_fresh_optimizer",
        "source": initialization_source,
        "load_contract": {
            "torch_load_weights_only": True,
            "checkpoint_field_loaded": "model_state_dict",
            "strict": True,
            "optimizer_state_loaded": False,
            "normalizer_state_keys_loaded": [
                "actor_obs_normalizer._mean",
                "actor_obs_normalizer._var",
                "actor_obs_normalizer._std",
                "actor_obs_normalizer.count",
                "critic_obs_normalizer._mean",
                "critic_obs_normalizer._var",
                "critic_obs_normalizer._std",
                "critic_obs_normalizer.count",
            ],
            "missing_keys": [],
            "unexpected_keys": [],
            "all_loaded_tensors_exactly_equal": True,
        },
        "fresh_runner_state": {
            "before": fresh_runner_snapshot,
            "after": json.loads(json.dumps(fresh_runner_snapshot)),
            "policy_object_preserved": True,
            "optimizer_object_preserved": True,
        },
    }
    torchscript = run_dir / "exported" / "policy.pt"
    onnx = run_dir / "exported" / "policy.onnx"
    _make_policy_exports(torchscript, onnx)
    common_correspondence = {
        "path": str(correspondence.resolve()),
        "checksum_sha256": _sha256(correspondence),
    }
    motion_contract_checksum = "b" * 64
    training_motion_keys = [
        str(record["source_file"]) for record in dynamic_records["train"]
    ]
    training_motion_assignment = final_artifacts.training_motion_assignment_contract(
        final_artifacts.EXPECTED_TRAINING_MOTION_ASSIGNMENT,
        num_envs=final_artifacts.EXPECTED_TRAINING_NUM_ENVS,
        motion_keys=training_motion_keys,
        num_steps_per_env=final_artifacts.EXPECTED_TRAINING_STEPS_PER_ENV,
        learning_updates=final_artifacts.EXPECTED_TRAINING_UPDATES,
    )
    training_motion_assignment["live_mapping_witness"] = {
        "verified": True,
        "command_cfg_fixed_motion_ids_match": True,
        "live_motion_ids_match": True,
        "resolved_live_motion_id_count": final_artifacts.EXPECTED_TRAINING_NUM_ENVS,
    }
    training_sample_counts = [
        int(record["environment_count"]) + 1
        for record in training_motion_assignment["per_motion"]
    ]
    training_summary = _write_json(
        run_dir / "run_summary.json",
        {
            "schema_version": final_artifacts.TRAINING_SCHEMA,
            "status": "completed",
            "run_dir": str(run_dir.resolve()),
            "resume_checkpoint": None,
            "policy_initialization_checkpoint": initialization_source,
            "policy_initialization": policy_initialization,
            "physics_warmup_steps_requested": 100,
            "motion_assignment": json.loads(
                json.dumps(training_motion_assignment)
            ),
            "motion_input": _motion_input(
                "train", dynamic_manifests["train"], dynamic_motions["train"]
            ),
            "correspondence": common_correspondence,
            "runtime_contract": {
                **final_artifacts.EXPECTED_RUNTIME,
                **_tracking_tensor_contract_fields(),
                "physics_cold_start_warmup": _physics_warmup_witness(2048),
                "tracking_policy_contract_version": final_artifacts.EXPECTED_TRACKING_POLICY_CONTRACT,
                "resume_checkpoint": None,
                "policy_initialization": policy_initialization,
                "motion_count": dynamic_counts["train"],
                "physics_dt": 0.005,
                "control_dt": 0.02,
                "motion_keys": training_motion_keys,
                "motion_assignment": json.loads(
                    json.dumps(training_motion_assignment)
                ),
                "contract_checksum_sha256": motion_contract_checksum,
                "library_checksum_sha256": _library_checksum(
                    motion_contract_checksum,
                    dynamic_motions["train"],
                    training_motion_keys,
                ),
                "physical_action_clip": 1.0,
                "policy_wrapper_action_clip": None,
                "frame_zero_start_probability": 0.5,
                "domain_randomization_events_enabled": False,
                "random_episode_length_enabled": False,
                "training_objective": {
                    "schema_version": final_artifacts.TRAINING_OBJECTIVE_SCHEMA_VERSION,
                    "reset_state_noise_scale": final_artifacts.EXPECTED_RESET_STATE_NOISE_SCALE,
                    "resolved_reset_state_noise": {
                        "joint_position_rad": [0.0, 0.0],
                        "joint_velocity_radps": [0.0, 0.0],
                        "root_linear_velocity_mps": [0.0, 0.0],
                        "root_angular_velocity_radps": [0.0, 0.0],
                    },
                    "reward_weights": {
                        "alive": final_artifacts.EXPECTED_ALIVE_REWARD_WEIGHT,
                        "failure": final_artifacts.EXPECTED_FAILURE_REWARD_WEIGHT,
                        "divergence_risk": final_artifacts.EXPECTED_DIVERGENCE_RISK_REWARD_WEIGHT,
                        "action_boundary_margin": final_artifacts.EXPECTED_ACTION_BOUNDARY_REWARD_WEIGHT,
                        "action_limiter_request_gap": final_artifacts.EXPECTED_ACTION_LIMITER_GAP_REWARD_WEIGHT,
                    },
                    "reward_semantics": {
                        "alive": {
                            "kind": "per_second_rate",
                            "weight": final_artifacts.EXPECTED_ALIVE_REWARD_WEIGHT,
                        },
                        "failure": {
                            "kind": "one_shot_non_timeout_impulse",
                            "impulse": final_artifacts.EXPECTED_FAILURE_REWARD_WEIGHT,
                            "normalization": "divide_by_live_step_dt",
                            "pure_timeouts_excluded": True,
                            "simultaneous_timeout_failure_penalized": True,
                        },
                        "divergence_risk": {
                            "kind": "per_second_dense_squared_margin_barrier",
                            "aggregation": "maximum_across_exact_termination_components",
                            "onset_fraction": final_artifacts.EXPECTED_DIVERGENCE_RISK_ONSET_FRACTION,
                            "maximum_root_distance_m": final_artifacts.EXPECTED_DIVERGENCE_MAX_ROOT_DISTANCE_M,
                            "maximum_mean_joint_error_rad": final_artifacts.EXPECTED_DIVERGENCE_MAX_MEAN_JOINT_ERROR_RAD,
                            "maximum_mean_body_distance_m": final_artifacts.EXPECTED_DIVERGENCE_MAX_MEAN_BODY_DISTANCE_M,
                            "weight": final_artifacts.EXPECTED_DIVERGENCE_RISK_REWARD_WEIGHT,
                            "hard_termination_thresholds_unchanged": True,
                        },
                        "action_boundary_margin": {
                            "kind": "per_second_dense_squared_action_margin",
                            "onset": final_artifacts.EXPECTED_ACTION_BOUNDARY_ONSET,
                            "action_clip": 1.0,
                            "weight": final_artifacts.EXPECTED_ACTION_BOUNDARY_REWARD_WEIGHT,
                            "raw_action_preserved_for_acceptance": True,
                        },
                        "action_limiter_request_gap": {
                            "kind": "per_second_clipped_request_minus_rate_limited_residual_l2",
                            "action_clip": 1.0,
                            "weight": final_artifacts.EXPECTED_ACTION_LIMITER_GAP_REWARD_WEIGHT,
                        },
                    },
                    "reward_integration_dt_s": 0.02,
                },
                "ppo": {
                    "init_noise_std": 0.2,
                    "noise_std_type": "log",
                    "entropy_coef": 0.0,
                    "learning_rate": final_artifacts.EXPECTED_RECOVERY_PPO_LEARNING_RATE,
                    "schedule": final_artifacts.EXPECTED_RECOVERY_PPO_SCHEDULE,
                    "desired_kl": final_artifacts.EXPECTED_RECOVERY_PPO_DESIRED_KL,
                },
            },
            "training": {
                "num_envs": final_artifacts.EXPECTED_TRAINING_NUM_ENVS,
                "num_steps_per_env": final_artifacts.EXPECTED_TRAINING_STEPS_PER_ENV,
                "learning_updates_requested": final_artifacts.EXPECTED_TRAINING_UPDATES,
                "learning_updates_completed": final_artifacts.EXPECTED_TRAINING_UPDATES,
                "runner_current_learning_iteration": final_artifacts.EXPECTED_FINAL_CHECKPOINT_ITERATION,
                "total_timesteps": final_artifacts.EXPECTED_TRAINING_TIMESTEPS,
                "final_checkpoint_iteration_index": final_artifacts.EXPECTED_FINAL_CHECKPOINT_ITERATION,
                "ppo_execution": {
                    "algorithm_learning_rate_final": final_artifacts.EXPECTED_RECOVERY_PPO_LEARNING_RATE,
                    "optimizer_param_group_learning_rates_final": [
                        final_artifacts.EXPECTED_RECOVERY_PPO_LEARNING_RATE
                    ],
                },
            },
            "motion_sampling": {
                "sampler": "fixed_environment_motion_then_frame_zero_or_uniform_valid_start_time",
                "motion_assignment_mode": "balanced-fixed",
                "total_episode_samples": sum(training_sample_counts),
                "minimum_samples_per_motion": min(training_sample_counts),
                "maximum_samples_per_motion": max(training_sample_counts),
                "zero_sampled_motion_count": 0,
                "zero_sampled_motion_keys": [],
                "start_time": {
                    "frame_zero_probability": 0.5,
                    "frame_zero_samples": 52,
                    "continuous_random_samples": 51,
                    "fixed_samples": final_artifacts.EXPECTED_TRAINING_NUM_ENVS,
                    "total_samples": sum(training_sample_counts),
                },
                "per_motion": [
                    {
                        "motion_id": index,
                        "motion_key": training_motion_keys[index],
                        "samples": training_sample_counts[index],
                    }
                    for index in range(dynamic_counts["train"])
                ],
            },
            "checkpoints": [_declared(path) for path in checkpoints],
            "exports": [_declared(torchscript), _declared(onnx)],
            "export_validations": [
                {
                    "format": "torchscript",
                    "path": str(torchscript.resolve()),
                    "input_shape": [1, 293],
                    "output_shape": [1, 29],
                    "finite_output": True,
                },
                {
                    "format": "onnx",
                    "path": str(onnx.resolve()),
                    "input_shape": [1, 293],
                    "output_shape": [1, 29],
                    "checker_passed": True,
                },
            ],
        },
    )
    training_summary_payload = json.loads(training_summary.read_text(encoding="utf-8"))
    _write_json(
        run_dir / "runtime_contract.json",
        training_summary_payload["runtime_contract"],
    )

    suite_reports: dict[str, Path] = {}
    for split in ("validation", "test"):
        evaluation_dir = work / "training" / "evaluations" / f"fixture_{split}"
        for relative in ("runtime_contract.json", "params/env.yaml", "params/agent.yaml"):
            _write_bytes(evaluation_dir / relative, f"fixture:{split}:{relative}\n".encode())
        count = dynamic_counts[split]
        success_count = count
        aggregate = _suite_aggregate(count, success_count)
        motion_keys = [str(record["source_file"]) for record in dynamic_records[split]]
        runtime_motions = [
            {
                "environment_id": index,
                "motion_id": index,
                "motion_key": motion_keys[index],
                "path": str(dynamic_motions[split][index].resolve()),
                "checksum_sha256": _sha256(dynamic_motions[split][index]),
                "frame_count": dynamic_records[split][index]["frame_count"],
                "duration_s": dynamic_records[split][index]["duration_sec"],
            }
            for index in range(count)
        ]
        motions = []
        for index in range(count):
            success = index < success_count
            motions.append(
                {
                    **runtime_motions[index],
                    "first_episode": {
                        "finished": True,
                        "success": success,
                        "termination_flags": {
                            "motion_end": success,
                            "fall": not success,
                            "tracking_divergence": False,
                            "joint_limits": False,
                            "nonfinite": False,
                        },
                    },
                }
            )
        acceptance = build_suite_acceptance(
            aggregate, manifest_promotion_verified=True
        )
        suite_reports[split] = _write_json(
            evaluation_dir / "suite_evaluation_summary.json",
            {
                "schema_version": final_artifacts.SUITE_SCHEMA,
                "status": "completed",
                "passed": acceptance["passed"],
                "evaluation_dir": str(evaluation_dir.resolve()),
                "motion_input": _motion_input(
                    split, dynamic_manifests[split], dynamic_motions[split]
                ),
                "correspondence": common_correspondence,
                "runtime_contract": {
                    **final_artifacts.EXPECTED_RUNTIME,
                    **_tracking_tensor_contract_fields(),
                    "physics_cold_start_warmup": _physics_warmup_witness(count),
                    "motion_count": count,
                    "physics_dt_s": 0.005,
                    "control_dt_s": 0.02,
                    "per_environment_motion_ids": list(range(count)),
                    "motions": runtime_motions,
                    "library_contract_checksum_sha256": motion_contract_checksum,
                    "library_checksum_sha256": _library_checksum(
                        motion_contract_checksum,
                        dynamic_motions[split],
                        motion_keys,
                    ),
                    "policy_action_clip": None,
                    "physical_action_clip": 1.0,
                    "evaluation_isolation": {
                        "schema_version": final_artifacts.EVALUATION_ISOLATION_SCHEMA_VERSION,
                        "command_random_start": False,
                        "fixed_start_time_s": 0.0,
                        "xy_offset_range_m": [0.0, 0.0],
                        "target_yaw_range_rad": [0.0, 0.0],
                        "resolved_reset_state_noise": {
                            "joint_position_rad": [0.0, 0.0],
                            "joint_velocity_radps": [0.0, 0.0],
                            "root_linear_velocity_mps": [0.0, 0.0],
                            "root_angular_velocity_radps": [0.0, 0.0],
                        },
                        "policy_observation_corruption_enabled": False,
                        "active_event_terms": {},
                        "enhanced_determinism_enabled": True,
                        "evaluation_reward_contract": {
                            "alive": {"kind": "per_second_rate", "weight": 0.10},
                            "failure": {
                                "kind": "one_shot_non_timeout_impulse",
                                "impulse": -5.0,
                                "normalization": "divide_by_live_step_dt",
                                "pure_timeouts_excluded": True,
                                "simultaneous_timeout_failure_penalized": True,
                            },
                        },
                        "reward_integration_dt_s": 0.02,
                    },
                },
                "checkpoint": _declared(checkpoint),
                "evaluation_contract": {
                    "one_environment_per_selected_motion": True,
                    "first_episode_only": True,
                    "deterministic_policy_inference": True,
                    "random_start": False,
                    "state_and_observation_noise": False,
                    "domain_randomization_events": False,
                    "physx_enhanced_determinism": True,
                    "physics_warmup_steps_requested": 100,
                    "success_definition": "clean motion_end on the finished first episode",
                },
                "motions": motions,
                "aggregate": aggregate,
                "acceptance": acceptance,
                "exports": [],
                "videos": [],
            },
        )
        suite_payload = json.loads(suite_reports[split].read_text(encoding="utf-8"))
        _write_json(
            evaluation_dir / "runtime_contract.json",
            suite_payload["runtime_contract"],
        )

    representative_dir = (
        work / "training" / "evaluations" / "fixture_representative_test"
    )
    for relative in ("runtime_contract.json", "params/env.yaml", "params/agent.yaml"):
        _write_bytes(
            representative_dir / relative,
            f"fixture:representative:{relative}\n".encode(),
        )
    representative_video = _write_bytes(
        representative_dir / "videos" / "rl-video-step-0.mp4",
        b"fixture-representative-mp4",
    )
    representative_index = 1
    representative_record = dynamic_records["test"][representative_index]
    representative_path = dynamic_motions["test"][representative_index]
    test_motion_keys = [
        str(record["source_file"]) for record in dynamic_records["test"]
    ]
    representative_runtime_contract = {
        **final_artifacts.EXPECTED_RUNTIME,
        **_tracking_tensor_contract_fields(),
        "physics_cold_start_warmup": _physics_warmup_witness(1),
        "motion_count": dynamic_counts["test"],
        "physics_dt": 0.005,
        "control_dt": 0.02,
        "motion_keys": test_motion_keys,
        "selected_motion_id": representative_index,
        "selected_motion_key": test_motion_keys[representative_index],
        "contract_checksum_sha256": motion_contract_checksum,
        "library_checksum_sha256": _library_checksum(
            motion_contract_checksum,
            dynamic_motions["test"],
            test_motion_keys,
        ),
        "policy_action_clip": None,
        "physical_action_clip": 1.0,
        "video_capture_started_after_physics_warmup": True,
        "evaluation_isolation": {
            "schema_version": final_artifacts.EVALUATION_ISOLATION_SCHEMA_VERSION,
            "command_random_start": False,
            "fixed_start_time_s": 0.0,
            "xy_offset_range_m": [0.0, 0.0],
            "target_yaw_range_rad": [0.0, 0.0],
            "resolved_reset_state_noise": {
                "joint_position_rad": [0.0, 0.0],
                "joint_velocity_radps": [0.0, 0.0],
                "root_linear_velocity_mps": [0.0, 0.0],
                "root_angular_velocity_radps": [0.0, 0.0],
            },
            "policy_observation_corruption_enabled": False,
            "active_event_terms": {},
            "enhanced_determinism_enabled": True,
            "evaluation_reward_contract": {
                "alive": {"kind": "per_second_rate", "weight": 0.10},
                "failure": {
                    "kind": "one_shot_non_timeout_impulse",
                    "impulse": -5.0,
                    "normalization": "divide_by_live_step_dt",
                    "pure_timeouts_excluded": True,
                    "simultaneous_timeout_failure_penalized": True,
                },
            },
            "reward_integration_dt_s": 0.02,
        },
    }
    _write_json(
        representative_dir / "runtime_contract.json",
        representative_runtime_contract,
    )
    representative_report = _write_json(
        representative_dir / "evaluation_summary.json",
        {
            "schema_version": final_artifacts.PLAY_SCHEMA,
            "status": "completed",
            "evaluation_dir": str(representative_dir.resolve()),
            "motion_input": _motion_input(
                "test", dynamic_manifests["test"], dynamic_motions["test"]
            ),
            "correspondence": common_correspondence,
            "checkpoint": _declared(checkpoint),
            "selected_motion_id": representative_index,
            "selected_motion_key": test_motion_keys[representative_index],
            "selected_motion_path": str(representative_path.resolve()),
            "physics_warmup_steps_requested": 100,
            "runtime_contract": representative_runtime_contract,
            "evaluation": {
                "steps_executed": representative_record["frame_count"],
                "completed_episodes": 1,
                "expected_frame_count": representative_record["frame_count"],
                "first_episode_length": representative_record["frame_count"],
                "first_episode_lengths_by_environment": [
                    representative_record["frame_count"]
                ],
                "first_episode_success_count": 1,
                "all_first_episodes_completed": True,
                "first_episode_completion_ratio": 1.0,
                "uninterrupted_full_clip_success": True,
            },
            "exports": [],
            "videos": [
                {
                    **_declared(representative_video),
                    "inspection": representative_video_inspection,
                }
            ],
            "video_capture_contract": {
                "mode": "post_physics_pre_reset_terminal_frame_substitution",
                "single_environment": True,
                "full_fixed_step_window": True,
                "enabled": True,
                "terminal_frames_captured": 1,
                "terminal_frames_substituted": 1,
            },
        },
    )

    professor_report = _write_bytes(
        reports / "PROFESSOR_FINAL_REPORT_KO.md",
        (
            "# ACCAD-to-G1 최종 보고서\n\n"
            "본 문서는 격리된 통합 테스트용 최종 보고서입니다. "
            "학습 실행, 검증 분할, 테스트 분할의 실행 완료 여부와 정책 수용 기준을 "
            "서로 구분해 기록하며 모든 근거 산출물은 SHA-256으로 고정합니다.\n"
        ).encode("utf-8"),
    )
    return {
        "project": project,
        "accad": accad,
        "pipeline": pipeline,
        "work": work,
        "reports": reports,
        "run_dir": run_dir,
        "training_summary": training_summary,
        "checkpoint": checkpoint,
        "validation_report": suite_reports["validation"],
        "test_report": suite_reports["test"],
        "representative_report": representative_report,
        "professor_report": professor_report,
        "torchscript": torchscript,
        "onnx": onnx,
    }


def test_complete_dynamic_snapshot_requires_exhaustive_acceptance(
    complete_snapshot: dict[str, Any],
) -> None:
    snapshot = complete_snapshot
    result = final_artifacts.build_final_artifacts(
        training_run=snapshot["run_dir"],
        validation_report=snapshot["validation_report"],
        test_report=snapshot["test_report"],
        representative_evaluation_report=snapshot["representative_report"],
        professor_report=snapshot["professor_report"],
    )

    manifest_path = snapshot["reports"] / "final_artifact_manifest.json"
    completion_path = snapshot["reports"] / "final_completion.json"
    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert result["policy_acceptance_verdict"] is True
    assert result["artifact_manifest"]["checksum_sha256"] == _sha256(manifest_path)
    assert result["completion"]["checksum_sha256"] == _sha256(completion_path)
    assert artifact_manifest["schema_version"] == final_artifacts.ARTIFACT_SCHEMA
    assert artifact_manifest["status"] == "completed"
    assert completion["schema_version"] == final_artifacts.COMPLETION_SCHEMA
    assert completion["status"] == "completed"
    assert completion["artifact_manifest"]["checksum_sha256"] == _sha256(manifest_path)
    assert completion["training"]["complete"] is True
    assert completion["dynamic_retime"]["train"]["source_motion_count"] == 215
    assert completion["dynamic_retime"]["train"]["promoted_motion_count"] == 103
    assert completion["dynamic_retime"]["validation"]["promoted_motion_count"] == 2
    assert completion["dynamic_retime"]["test"]["promoted_motion_count"] == 3
    assert completion["training"]["policy_initialization"]["source"][
        "iteration"
    ] == 599
    assert completion["training"]["policy_initialization"][
        "optimizer_state_loaded"
    ] is False
    assert completion["training"]["motion_assignment"]["mode"] == "balanced-fixed"
    assert completion["training"]["motion_assignment"][
        "all_motions_have_nonzero_environments"
    ] is True
    assert completion["training"]["motion_assignment"][
        "maximum_environment_count_difference"
    ] == 1
    assert completion["training"]["motion_assignment"][
        "theoretical_total_training_transitions"
    ] == final_artifacts.EXPECTED_TRAINING_TIMESTEPS
    assert completion["training"]["motion_sampling"]["start_time"][
        "fixed_samples"
    ] == final_artifacts.EXPECTED_TRAINING_NUM_ENVS
    assert completion["validation"]["complete"] is True
    assert completion["test"]["complete"] is True
    assert completion["validation"]["status"] == "completed"
    assert completion["test"]["status"] == "completed"
    assert completion["representative_evaluation"]["complete"] is True
    assert completion["representative_evaluation"]["status"] == "completed"
    assert completion["training"]["runtime_contract"][
        "physics_cold_start_warmup"
    ]["executed_control_steps"] == 100
    assert completion["validation"]["runtime_contract"][
        "physics_cold_start_warmup"
    ]["passed"] is True
    assert completion["test"]["runtime_contract"]["physics_cold_start_warmup"][
        "passed"
    ] is True
    assert completion["representative_evaluation"]["runtime_contract"][
        "physics_cold_start_warmup"
    ]["post_reset"]["authoritative_motion_frame_zero"] is True
    assert (
        completion["representative_evaluation"]["representative_motion"][
            "motion_id"
        ]
        == 1
    )
    representative_source = json.loads(
        complete_snapshot["representative_report"].read_text(encoding="utf-8")
    )
    assert (
        completion["representative_evaluation"]["video"]["inspection"]
        == representative_source["videos"][0]["inspection"]
    )
    assert completion["validation"]["predefined_acceptance"]["verdict"] is True
    assert completion["test"]["predefined_acceptance"]["verdict"] is True
    assert completion["policy_acceptance"] == {
        "verdict": True,
        "validation_verdict": True,
        "test_verdict": True,
        "interpretation": (
            "Final publication requires both exhaustive dynamic suites to pass."
        ),
    }

    independent = artifact_manifest["independent_policy_export_validation"]
    assert independent == completion["training"]["independent_export_validation"]
    assert independent["status"] == "passed"
    assert independent["device"] == "cpu"
    assert independent["input_contract"]["shape"] == [1, 293]
    assert independent["output_contract"]["shape"] == [1, 29]
    assert independent["torchscript"]["checksum_sha256"] == _sha256(
        snapshot["torchscript"]
    )
    assert independent["onnx"]["checksum_sha256"] == _sha256(snapshot["onnx"])
    assert independent["torchscript"]["finite_output"] is True
    assert independent["onnx"]["reference_evaluator_executed"] is True


def test_dynamic_plan_must_cover_source_as_promoted_plus_quarantine(
    complete_snapshot: dict[str, Any],
) -> None:
    plan_path = complete_snapshot["work"] / "dynamic" / "g1_train_scale_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["counts"]["final_quarantined"] -= 1
    _write_json(plan_path, plan)

    with pytest.raises(
        final_artifacts.FinalArtifactError,
        match="dynamic train plan counts are inconsistent",
    ):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["training_summary"],
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot[
                "representative_report"
            ],
            professor_report=complete_snapshot["professor_report"],
        )


def test_dynamic_promoted_archive_checksum_is_recomputed(
    complete_snapshot: dict[str, Any],
) -> None:
    manifest_path = complete_snapshot["work"] / "dynamic" / "g1_validation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    motion_path = (
        complete_snapshot["work"]
        / "dynamic"
        / "g1"
        / manifest["motions"][0]["g1_file"]
    )
    motion_path.write_bytes(motion_path.read_bytes() + b"tampered")

    with pytest.raises(
        final_artifacts.FinalArtifactError,
        match="dynamic validation derived archive checksum/version mismatch",
    ):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["training_summary"],
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot[
                "representative_report"
            ],
            professor_report=complete_snapshot["professor_report"],
        )


@pytest.mark.parametrize(
    "mutation",
    ("optimizer_loaded", "optimizer_state_nonzero", "wrong_source_iteration"),
)
def test_model599_policy_only_initialization_witness_fails_closed(
    complete_snapshot: dict[str, Any], mutation: str
) -> None:
    summary_path = complete_snapshot["training_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    witnesses = [
        summary["policy_initialization"],
        summary["runtime_contract"]["policy_initialization"],
    ]
    if mutation == "optimizer_loaded":
        for witness in witnesses:
            witness["load_contract"]["optimizer_state_loaded"] = True
    elif mutation == "optimizer_state_nonzero":
        for witness in witnesses:
            witness["fresh_runner_state"]["before"][
                "optimizer_state_entry_count"
            ] = 1
            witness["fresh_runner_state"]["after"][
                "optimizer_state_entry_count"
            ] = 1
    else:
        for witness in witnesses:
            witness["source"]["iteration"] = 598
        summary["policy_initialization_checkpoint"]["iteration"] = 598
    _write_json(summary_path, summary)
    _write_json(
        complete_snapshot["run_dir"] / "runtime_contract.json",
        summary["runtime_contract"],
    )

    with pytest.raises(final_artifacts.FinalArtifactError, match="initialization"):
        final_artifacts.build_final_artifacts(
            training_run=summary_path,
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot[
                "representative_report"
            ],
            professor_report=complete_snapshot["professor_report"],
        )


def test_final_suite_rejects_one_failed_dynamic_promoted_motion(
    complete_snapshot: dict[str, Any],
) -> None:
    report_path = complete_snapshot["validation_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failed = report["motions"][-1]["first_episode"]
    failed["success"] = False
    failed["termination_flags"]["motion_end"] = False
    failed["termination_flags"]["fall"] = True
    report["aggregate"] = _suite_aggregate(2, 1)
    report["acceptance"] = build_suite_acceptance(
        report["aggregate"], manifest_promotion_verified=True
    )
    report["passed"] = report["acceptance"]["passed"]
    _write_json(report_path, report)

    with pytest.raises(
        final_artifacts.FinalArtifactError, match="validation suite report did not pass"
    ):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["training_summary"],
            validation_report=report_path,
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot[
                "representative_report"
            ],
            professor_report=complete_snapshot["professor_report"],
        )


def test_representative_key_is_bound_to_dynamic_test_manifest(
    complete_snapshot: dict[str, Any],
) -> None:
    report_path = complete_snapshot["representative_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["selected_motion_key"] = "forged/not-in-dynamic-manifest.npz"
    _write_json(report_path, report)

    with pytest.raises(
        final_artifacts.FinalArtifactError,
        match="selected key differs from the dynamic test manifest",
    ):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["training_summary"],
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=report_path,
            professor_report=complete_snapshot["professor_report"],
        )


def test_explicit_professor_report_outside_accad_is_rejected_without_outputs(
    complete_snapshot: dict[str, Any], tmp_path: Path
) -> None:
    outside = tmp_path / "outside-professor-report.md"
    outside.write_text("outside " * 30, encoding="utf-8")

    with pytest.raises(final_artifacts.FinalArtifactError, match="must remain below"):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["run_dir"],
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot["representative_report"],
            professor_report=outside,
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


def test_suite_checkpoint_mismatch_is_rejected_without_outputs(
    complete_snapshot: dict[str, Any],
) -> None:
    test_report_path = complete_snapshot["test_report"]
    test_report = json.loads(test_report_path.read_text(encoding="utf-8"))
    alternate = _write_bytes(test_report_path.parent / "model_2.pt", b"wrong-checkpoint")
    test_report["checkpoint"] = _declared(alternate)
    _write_json(test_report_path, test_report)

    with pytest.raises(
        final_artifacts.FinalArtifactError,
        match="did not use the final training checkpoint",
    ):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["training_summary"],
            validation_report=complete_snapshot["validation_report"],
            test_report=test_report_path,
            representative_evaluation_report=complete_snapshot["representative_report"],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


def test_short_training_cannot_be_finalized(
    complete_snapshot: dict[str, Any],
) -> None:
    summary_path = complete_snapshot["training_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["training"].update(
        {
            "learning_updates_requested": 5,
            "learning_updates_completed": 5,
            "runner_current_learning_iteration": 4,
            "total_timesteps": 2048 * 32 * 5,
            "final_checkpoint_iteration_index": 4,
        }
    )
    _write_json(summary_path, summary)

    with pytest.raises(
        final_artifacts.FinalArtifactError,
        match="training capacity contract mismatch",
    ):
        final_artifacts.build_final_artifacts(
            training_run=summary_path,
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot["representative_report"],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


@pytest.mark.parametrize("resume_value", ["__missing__", "model_0.pt"])
def test_non_fresh_training_cannot_be_finalized(
    complete_snapshot: dict[str, Any], resume_value: str
) -> None:
    summary_path = complete_snapshot["training_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if resume_value == "__missing__":
        summary.pop("resume_checkpoint")
    else:
        summary["resume_checkpoint"] = resume_value
    _write_json(summary_path, summary)

    with pytest.raises(
        final_artifacts.FinalArtifactError,
        match="final recovery training must be a fresh run",
    ):
        final_artifacts.build_final_artifacts(
            training_run=summary_path,
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot["representative_report"],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_contract_version", "final recovery tracking contract"),
        ("adaptive_schedule", "training recovery PPO contract mismatch"),
        ("wrong_configured_lr", "training recovery PPO contract mismatch"),
        ("wrong_desired_kl", "training recovery PPO contract mismatch"),
        (
            "random_episode_length",
            "must disable runner episode-length randomization",
        ),
        ("wrong_objective_schema", "training recovery objective schema mismatch"),
        ("wrong_reset_noise_scale", "training reset-state noise scale mismatch"),
        ("nonzero_resolved_noise", "reset-state noise must be zero"),
        ("wrong_alive_weight", "training recovery reward weights mismatch"),
        ("wrong_failure_weight", "training recovery reward weights mismatch"),
        (
            "wrong_executed_algorithm_lr",
            "executed PPO learning rate differs from the recovery contract",
        ),
        (
            "empty_executed_optimizer_lrs",
            "executed PPO learning rate differs from the recovery contract",
        ),
        (
            "wrong_executed_optimizer_lr",
            "executed PPO learning rate differs from the recovery contract",
        ),
    ],
)
def test_fixed_lr_recovery_contract_cannot_be_forged(
    complete_snapshot: dict[str, Any], mutation: str, message: str
) -> None:
    summary_path = complete_snapshot["training_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if mutation == "wrong_contract_version":
        summary["runtime_contract"]["tracking_policy_contract_version"] = "forged"
    elif mutation == "adaptive_schedule":
        summary["runtime_contract"]["ppo"]["schedule"] = "adaptive"
    elif mutation == "wrong_configured_lr":
        summary["runtime_contract"]["ppo"]["learning_rate"] = 5.0e-4
    elif mutation == "wrong_desired_kl":
        summary["runtime_contract"]["ppo"]["desired_kl"] = 0.02
    elif mutation == "random_episode_length":
        summary["runtime_contract"]["random_episode_length_enabled"] = True
    elif mutation == "wrong_objective_schema":
        summary["runtime_contract"]["training_objective"]["schema_version"] = "forged"
    elif mutation == "wrong_reset_noise_scale":
        summary["runtime_contract"]["training_objective"][
            "reset_state_noise_scale"
        ] = 0.1
    elif mutation == "nonzero_resolved_noise":
        summary["runtime_contract"]["training_objective"][
            "resolved_reset_state_noise"
        ]["joint_position_rad"] = [-0.01, 0.01]
    elif mutation == "wrong_alive_weight":
        summary["runtime_contract"]["training_objective"]["reward_weights"][
            "alive"
        ] = 0.1
    elif mutation == "wrong_failure_weight":
        summary["runtime_contract"]["training_objective"]["reward_weights"][
            "failure"
        ] = -5.0
    elif mutation == "wrong_executed_algorithm_lr":
        summary["training"]["ppo_execution"]["algorithm_learning_rate_final"] = 1.0e-5
    elif mutation == "empty_executed_optimizer_lrs":
        summary["training"]["ppo_execution"][
            "optimizer_param_group_learning_rates_final"
        ] = []
    else:
        summary["training"]["ppo_execution"][
            "optimizer_param_group_learning_rates_final"
        ] = [1.0e-5]
    _write_json(summary_path, summary)

    with pytest.raises(final_artifacts.FinalArtifactError, match=message):
        final_artifacts.build_final_artifacts(
            training_run=summary_path,
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot["representative_report"],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


def test_training_runtime_sidecar_must_equal_embedded_contract(
    complete_snapshot: dict[str, Any],
) -> None:
    sidecar_path = complete_snapshot["run_dir"] / "runtime_contract.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["ppo"]["learning_rate"] = 5.0e-4
    _write_json(sidecar_path, sidecar)

    with pytest.raises(
        final_artifacts.FinalArtifactError,
        match="training runtime sidecar differs from the embedded runtime contract",
    ):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["training_summary"],
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot["representative_report"],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


@pytest.mark.parametrize("split", ["validation", "test"])
def test_suite_runtime_sidecar_must_equal_embedded_contract(
    complete_snapshot: dict[str, Any], split: str
) -> None:
    report_path = complete_snapshot[f"{split}_report"]
    sidecar_path = report_path.parent / "runtime_contract.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["physics_dt_s"] = 0.01
    _write_json(sidecar_path, sidecar)

    with pytest.raises(
        final_artifacts.FinalArtifactError,
        match=f"{split} runtime sidecar differs from the embedded runtime contract",
    ):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["training_summary"],
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot["representative_report"],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


@pytest.mark.parametrize(
    ("report_key", "mutation", "message"),
    [
        (
            "training_summary",
            "wrong_term_order",
            "training observation contract differs from the canonical v5 contract",
        ),
        (
            "validation_report",
            "wrong_frame",
            "validation suite observation contract differs from the canonical v5 contract",
        ),
        (
            "representative_report",
            "wrong_scale",
            "representative held-out test observation contract differs from the canonical v5 contract",
        ),
        (
            "test_report",
            "wrong_live_dimension",
            "test suite live tensor contract differs from the resolved v5 layout",
        ),
    ],
)
def test_v5_observation_and_live_tensor_contract_mutations_fail_closed(
    complete_snapshot: dict[str, Any],
    report_key: str,
    mutation: str,
    message: str,
) -> None:
    """Static semantics and live Isaac tensor axes are mandatory on every path."""

    report_path = complete_snapshot[report_key]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runtime = report["runtime_contract"]
    if mutation == "wrong_term_order":
        terms = runtime["observation_contract"]["observation_groups"]["policy"][
            "terms"
        ]
        terms[0], terms[1] = terms[1], terms[0]
    elif mutation == "wrong_frame":
        runtime["observation_contract"]["observation_groups"]["policy"]["terms"][
            1
        ]["frame"] = "world"
    elif mutation == "wrong_scale":
        runtime["observation_contract"]["observation_groups"]["policy"]["terms"][
            6
        ]["scale"] = 1.0
    else:
        runtime["live_tensor_contract"]["observation_groups"]["policy"][
            "term_dimensions"
        ][0] += 1
    _write_json(report_path, report)
    _write_json(report_path.parent / "runtime_contract.json", runtime)

    with pytest.raises(final_artifacts.FinalArtifactError, match=message):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["training_summary"],
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot[
                "representative_report"
            ],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


@pytest.mark.parametrize(
    ("report_key", "mutation"),
    [
        ("training_summary", "schema"),
        ("validation_report", "disabled"),
        ("test_report", "requested_steps"),
        ("representative_report", "executed_steps"),
        ("training_summary", "action_dimension"),
        ("validation_report", "policy_inference"),
        ("test_report", "post_reset_passed"),
        ("representative_report", "frame_zero"),
        ("training_summary", "episode_counters"),
        ("validation_report", "action_state"),
        ("test_report", "reset_buffer"),
        ("representative_report", "reference_state"),
        ("training_summary", "finite_observation"),
        ("representative_report", "video_before_warmup"),
    ],
)
def test_production_physics_warmup_witness_mutations_fail_closed(
    complete_snapshot: dict[str, Any], report_key: str, mutation: str
) -> None:
    report_path = complete_snapshot[report_key]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runtime = report["runtime_contract"]
    warmup = runtime["physics_cold_start_warmup"]
    post_reset = warmup["post_reset"]
    if mutation == "schema":
        warmup["schema_version"] = "forged"
    elif mutation == "disabled":
        warmup["enabled"] = False
    elif mutation == "requested_steps":
        warmup["requested_control_steps"] = 99
    elif mutation == "executed_steps":
        warmup["executed_control_steps"] = 99
    elif mutation == "action_dimension":
        warmup["action_dimension"] = 28
    elif mutation == "policy_inference":
        warmup["policy_inference_used"] = True
    elif mutation == "post_reset_passed":
        post_reset["passed"] = False
    elif mutation == "frame_zero":
        post_reset["authoritative_motion_frame_zero"] = False
    elif mutation == "episode_counters":
        post_reset["episode_counters_zero"] = False
    elif mutation == "action_state":
        post_reset["action_state_zero"] = False
    elif mutation == "reset_buffer":
        post_reset["reset_buffers"]["reset_buf"] = {
            "nonzero_count": 1,
            "all_clear": False,
        }
    elif mutation == "reference_state":
        post_reset["reference_state_maximum_errors"]["root_position_m"] = 0.1
    elif mutation == "finite_observation":
        post_reset["observations"]["nonfinite_count"] = 1
        post_reset["observations"]["all_finite"] = False
    else:
        runtime["video_capture_started_after_physics_warmup"] = False
    _write_json(report_path, report)
    _write_json(report_path.parent / "runtime_contract.json", runtime)

    with pytest.raises(final_artifacts.FinalArtifactError, match="warmup"):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["training_summary"],
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot[
                "representative_report"
            ],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


@pytest.mark.parametrize(
    ("report_key", "field_parent"),
    [
        ("training_summary", "report"),
        ("validation_report", "evaluation_contract"),
        ("representative_report", "report"),
    ],
)
def test_production_physics_warmup_request_metadata_is_required(
    complete_snapshot: dict[str, Any], report_key: str, field_parent: str
) -> None:
    report_path = complete_snapshot[report_key]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    parent = report if field_parent == "report" else report[field_parent]
    parent["physics_warmup_steps_requested"] = 0
    _write_json(report_path, report)

    with pytest.raises(final_artifacts.FinalArtifactError, match="physics warmup"):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["training_summary"],
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot[
                "representative_report"
            ],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "random_start",
        "observation_corruption",
        "active_event",
        "enhanced_determinism",
        "reset_noise",
        "evaluation_reward",
        "boolean_type_alias",
    ],
)
def test_suite_resolved_evaluation_isolation_fails_closed(
    complete_snapshot: dict[str, Any], mutation: str
) -> None:
    report_path = complete_snapshot["validation_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    isolation = report["runtime_contract"]["evaluation_isolation"]
    if mutation == "random_start":
        isolation["command_random_start"] = True
    elif mutation == "observation_corruption":
        isolation["policy_observation_corruption_enabled"] = True
    elif mutation == "active_event":
        isolation["active_event_terms"] = {"interval": ["push"]}
    elif mutation == "enhanced_determinism":
        isolation["enhanced_determinism_enabled"] = False
    elif mutation == "reset_noise":
        isolation["resolved_reset_state_noise"]["joint_position_rad"] = [
            -0.01,
            0.01,
        ]
    elif mutation == "evaluation_reward":
        isolation["evaluation_reward_contract"]["failure"]["impulse"] = -10.0
    else:
        isolation["command_random_start"] = 0
    _write_json(report_path, report)
    _write_json(report_path.parent / "runtime_contract.json", report["runtime_contract"])

    with pytest.raises(
        final_artifacts.FinalArtifactError,
        match="validation suite evaluation",
    ):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["training_summary"],
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot["representative_report"],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


@pytest.mark.parametrize(
    ("report_key", "field", "value", "label"),
    [
        ("validation_report", "policy_action_clip", 1.0, "validation suite"),
        ("validation_report", "physical_action_clip", 0.5, "validation suite"),
        ("representative_report", "policy_action_clip", 1.0, "representative held-out test"),
        ("representative_report", "physical_action_clip", 0.5, "representative held-out test"),
    ],
)
def test_evaluation_action_path_fails_closed(
    complete_snapshot: dict[str, Any],
    report_key: str,
    field: str,
    value: float,
    label: str,
) -> None:
    report_path = complete_snapshot[report_key]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["runtime_contract"][field] = value
    _write_json(report_path, report)
    _write_json(report_path.parent / "runtime_contract.json", report["runtime_contract"])

    with pytest.raises(final_artifacts.FinalArtifactError, match=label):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["training_summary"],
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot["representative_report"],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


def test_inconsistent_training_motion_sampling_cannot_be_finalized(
    complete_snapshot: dict[str, Any],
) -> None:
    summary_path = complete_snapshot["training_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["motion_sampling"]["per_motion"][1]["motion_id"] = 0
    _write_json(summary_path, summary)

    with pytest.raises(
        final_artifacts.FinalArtifactError,
        match="sampling motion_id 1 mismatch",
    ):
        final_artifacts.build_final_artifacts(
            training_run=summary_path,
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot["representative_report"],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            "mapping_digest",
            "deterministic balanced-fixed production contract",
        ),
        (
            "environment_count",
            "deterministic balanced-fixed production contract",
        ),
        (
            "all_nonzero_witness",
            "deterministic balanced-fixed production contract",
        ),
        (
            "live_mapping_witness",
            "live-command mapping witness",
        ),
        (
            "top_level_disagreement",
            "top-level training motion assignment differs",
        ),
        (
            "sampling_mode",
            "does not use balanced-fixed assignment",
        ),
    ],
)
def test_balanced_fixed_training_assignment_cannot_be_forged(
    complete_snapshot: dict[str, Any], mutation: str, message: str
) -> None:
    summary_path = complete_snapshot["training_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime_assignment = summary["runtime_contract"]["motion_assignment"]
    top_level_assignment = summary["motion_assignment"]
    if mutation == "mapping_digest":
        runtime_assignment["mapping_digest_sha256"] = "0" * 64
        top_level_assignment["mapping_digest_sha256"] = "0" * 64
    elif mutation == "environment_count":
        runtime_assignment["per_motion"][0]["environment_count"] = 0
        top_level_assignment["per_motion"][0]["environment_count"] = 0
    elif mutation == "all_nonzero_witness":
        runtime_assignment["all_motions_nonzero_witness"]["verified"] = False
        top_level_assignment["all_motions_nonzero_witness"]["verified"] = False
    elif mutation == "live_mapping_witness":
        runtime_assignment["live_mapping_witness"]["live_motion_ids_match"] = False
        top_level_assignment["live_mapping_witness"]["live_motion_ids_match"] = False
    elif mutation == "top_level_disagreement":
        top_level_assignment["mapping_digest_sha256"] = "0" * 64
    else:
        summary["motion_sampling"]["motion_assignment_mode"] = "episode-uniform"
    _write_json(summary_path, summary)
    _write_json(
        summary_path.parent / "runtime_contract.json",
        summary["runtime_contract"],
    )

    with pytest.raises(final_artifacts.FinalArtifactError, match=message):
        final_artifacts.build_final_artifacts(
            training_run=summary_path,
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot[
                "representative_report"
            ],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


def test_suite_motion_identity_mapping_cannot_be_forged(
    complete_snapshot: dict[str, Any],
) -> None:
    report_path = complete_snapshot["validation_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["runtime_contract"]["motions"][1]["motion_id"] = 0
    _write_json(report_path, report)

    with pytest.raises(
        final_artifacts.FinalArtifactError,
        match="runtime motion 1 ID mapping mismatch",
    ):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["run_dir"],
            validation_report=report_path,
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=complete_snapshot["representative_report"],
            professor_report=complete_snapshot["professor_report"],
        )

    assert not (complete_snapshot["reports"] / "final_artifact_manifest.json").exists()
    assert not (complete_snapshot["reports"] / "final_completion.json").exists()


def test_suite_float32_duration_rounding_is_accepted(
    complete_snapshot: dict[str, Any],
) -> None:
    """Isaac's float32 duration witness may differ by several 1e-7 seconds."""

    report_path = complete_snapshot["validation_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["runtime_contract"]["motions"][0]["duration_s"] += 8.0e-7
    report["motions"][0]["duration_s"] += 8.0e-7
    _write_json(report_path, report)
    _write_json(report_path.parent / "runtime_contract.json", report["runtime_contract"])

    result = final_artifacts.build_final_artifacts(
        training_run=complete_snapshot["run_dir"],
        validation_report=report_path,
        test_report=complete_snapshot["test_report"],
        representative_evaluation_report=complete_snapshot["representative_report"],
        professor_report=complete_snapshot["professor_report"],
    )

    assert result["status"] == "completed"


def test_representative_video_inspection_mismatch_is_rejected(
    complete_snapshot: dict[str, Any],
) -> None:
    report_path = complete_snapshot["representative_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["videos"][0]["inspection"]["decoded_frame_count"] = 1
    _write_json(report_path, report)

    with pytest.raises(
        final_artifacts.FinalArtifactError,
        match="inspection differs from independent decoding",
    ):
        final_artifacts.build_final_artifacts(
            training_run=complete_snapshot["run_dir"],
            validation_report=complete_snapshot["validation_report"],
            test_report=complete_snapshot["test_report"],
            representative_evaluation_report=report_path,
            professor_report=complete_snapshot["professor_report"],
        )
