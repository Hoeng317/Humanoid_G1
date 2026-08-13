#!/usr/bin/env python3
"""Deterministically evaluate one first episode for every selected G1 motion.

The complete suite runs in one Isaac Lab process with one environment assigned
to each loaded archive.  Automatic environment resets are unavoidable inside
``ManagerBasedRLEnv``, so all metrics and termination witnesses are frozen at
an environment's first done signal and later episodes are ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
PROJECT_SOURCE = PROJECT_ROOT / "source"
for path in (PIPELINE_ROOT, PROJECT_SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from accad_g1.training_io import (  # noqa: E402
    ACCAD_ROOT,
    WORK_ROOT,
    add_motion_input_args,
    checkpoint_training_contract_provenance,
    command_metadata,
    new_artifact_dir,
    nonnegative_int,
    path_below,
    resolve_checkpoint,
    resolve_motion_inputs,
    sha256_file,
    utc_now,
    validate_policy_exports,
    write_json_atomic,
)
from accad_g1.physics_warmup import (  # noqa: E402
    DEFAULT_PHYSICS_WARMUP_STEPS,
    run_physics_cold_start_warmup,
)

from isaaclab.app import AppLauncher  # noqa: E402


PARSER = argparse.ArgumentParser(
    description=(
        "Evaluate every selected held-out G1 motion in one deterministic Isaac Lab process "
        "and record each environment's first episode only."
    )
)
add_motion_input_args(PARSER, default_split="validation")
PARSER.add_argument("--checkpoint", default=None, help="Default: latest local model_*.pt.")
PARSER.add_argument(
    "--action-source",
    choices=("checkpoint", "zero"),
    default="checkpoint",
    help=(
        "Deterministic action source. 'checkpoint' preserves the official suite; "
        "'zero' is an acceptance-ineligible reference-PD feasibility diagnostic."
    ),
)
PARSER.add_argument(
    "--allow-legacy-dynamics-ablation",
    action="store_true",
    help=(
        "Allow a checkpoint whose parent run used a different or unverifiable "
        "actuator/training-objective contract. The result is forced acceptance-ineligible "
        "and export is disabled."
    ),
)
PARSER.add_argument(
    "--max-steps",
    type=nonnegative_int,
    default=0,
    help="Global control-step cap; 0 uses the longest archive frame count plus one.",
)
PARSER.add_argument("--seed", type=nonnegative_int, default=42)
PARSER.add_argument(
    "--physics-warmup-steps",
    type=nonnegative_int,
    default=DEFAULT_PHYSICS_WARMUP_STEPS,
    help=(
        "Policy-independent zero-residual control steps used to prime physics/contact "
        "caches before an authoritative frame-0 reset (default: 100; pass 0 only "
        "for a cold-start diagnostic)."
    ),
)
PARSER.add_argument(
    "--export",
    choices=("none", "jit", "onnx", "both"),
    default="none",
    help="Optionally export the loaded actor into this evaluation directory.",
)
PARSER.add_argument("--evaluation-name", default="heldout_suite")
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import importlib.metadata  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from accad_g1.evaluation_metrics import (  # noqa: E402
    build_suite_acceptance,
    contact_precision_recall_f1,
)
from accad_g1.tracking_task import (  # noqa: E402
    G1_CONTRACT,
    G1_POLICY_JOINT_NAMES,
    ReferenceAwareFallTermination,
    TRACKING_POLICY_CONTRACT_VERSION,
    TRAINING_OBJECTIVE_SCHEMA_VERSION,
    TrackingDivergenceTermination,
    clean_motion_end_success,
    deterministic_suite_actions,
    make_tracking_env_cfg,
    make_tracking_runner_cfg,
    quat_error_angle_wxyz,
    reference_residual_target,
    reference_velocity_target,
    resolved_evaluation_isolation,
    resolved_live_tracking_tensor_contract,
    tracking_observation_contract,
)
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _export_policy(runner: OnPolicyRunner, export_dir: Path, mode: str) -> list[Path]:
    if mode == "none":
        return []
    policy = runner.alg.policy
    normalizer = getattr(policy, "actor_obs_normalizer", None)
    export_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if mode in {"jit", "both"}:
        export_policy_as_jit(
            policy,
            normalizer=normalizer,
            path=str(export_dir),
            filename="policy.pt",
        )
        outputs.append(export_dir / "policy.pt")
    if mode in {"onnx", "both"}:
        export_policy_as_onnx(
            policy,
            normalizer=normalizer,
            path=str(export_dir),
            filename="policy.onnx",
        )
        outputs.append(export_dir / "policy.onnx")
    return outputs


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _safe_mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _safe_min(values: list[float]) -> float | None:
    return None if not values else float(min(values))


def _safe_max(values: list[float]) -> float | None:
    return None if not values else float(max(values))


def _safe_p95(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95.0))


def _json_number(value: torch.Tensor, env_id: int) -> float | None:
    numeric = float(value[env_id].item())
    return numeric if math.isfinite(numeric) else None


def _json_bool(value: torch.Tensor, env_id: int) -> bool:
    return bool(value[env_id].item())


def _update_episode_maximum(
    values: torch.Tensor,
    *,
    active: torch.Tensor,
    control_step: int,
    maxima: torch.Tensor,
    maximum_steps: torch.Tensor,
) -> None:
    """Update a finite first-episode maximum and its first attaining step."""

    eligible = active & torch.isfinite(values)
    improved = eligible & (values.to(maxima.dtype) > maxima)
    maxima[:] = torch.where(improved, values.to(maxima.dtype), maxima)
    maximum_steps[:] = torch.where(
        improved,
        torch.full_like(maximum_steps, int(control_step)),
        maximum_steps,
    )


def _named_true_joints(mask: torch.Tensor, env_id: int) -> list[str]:
    indices = torch.where(mask[env_id])[0].detach().cpu().tolist()
    return [G1_POLICY_JOINT_NAMES[int(index)] for index in indices]


def _ratio_by_joint(values: torch.Tensor, env_id: int) -> dict[str, float | None]:
    return {
        name: _json_number(values[:, joint_id], env_id)
        for joint_id, name in enumerate(G1_POLICY_JOINT_NAMES)
    }


def _terminal_transition_witness(
    *,
    env_id: int,
    control_step: int,
    termination_flags: dict[str, bool],
    fall_witness: dict[str, Any],
    divergence_witness: dict[str, Any],
    fall_thresholds: dict[str, Any],
    divergence_thresholds: dict[str, Any],
    actions: torch.Tensor,
    raw_action_clip: float,
    raw_clip_mask: torch.Tensor,
    target_clamped: torch.Tensor,
    rate_limited: torch.Tensor,
    velocity_target_clamped: torch.Tensor,
) -> dict[str, Any]:
    """Serialize the exact first terminal transition into JSON-safe values."""

    action_count = int(actions.shape[-1])
    height = _json_bool(fall_witness["height_deficit_exceeded"], env_id)
    orientation = _json_bool(fall_witness["orientation_error_exceeded"], env_id)
    pelvis = _json_bool(fall_witness["unexpected_pelvis_contact"], env_id)
    root = _json_bool(divergence_witness["root_distance_exceeded"], env_id)
    joint = _json_bool(divergence_witness["mean_joint_error_exceeded"], env_id)
    body = _json_bool(divergence_witness["mean_body_distance_exceeded"], env_id)
    result: dict[str, Any] = {
        "schema_version": "accad_g1_terminal_transition_witness_v1",
        "control_step_index_one_based": int(control_step),
        "transition_index_zero_based": int(control_step - 1),
        "reference_time_s": _json_number(fall_witness["reference_time_s"], env_id),
        "state_timing": (
            "post_physics_final_decimation_substep_before_Isaac_automatic_reset; "
            "reference held during this control transition"
        ),
        "termination_flags": dict(termination_flags),
        "fall": {
            "fired": _json_bool(fall_witness["fall"], env_id),
            "root_height_deficit_m": _json_number(
                fall_witness["root_height_deficit_m"], env_id
            ),
            "maximum_root_height_deficit_m": float(
                fall_thresholds["maximum_root_height_deficit"]
            ),
            "height_deficit_exceeded": height,
            "root_orientation_error_rad": _json_number(
                fall_witness["root_orientation_error_rad"], env_id
            ),
            "maximum_orientation_error_rad": float(
                fall_thresholds["maximum_orientation_error"]
            ),
            "orientation_error_exceeded": orientation,
            "reference_pelvis_height_m": _json_number(
                fall_witness["reference_pelvis_height_m"], env_id
            ),
            "reference_pelvis_tilt_rad": _json_number(
                fall_witness["reference_pelvis_tilt_rad"], env_id
            ),
            "pelvis_contact_force_n": _json_number(
                fall_witness["pelvis_contact_force_n"], env_id
            ),
            "pelvis_contact_threshold_n": float(
                fall_thresholds["pelvis_contact_threshold"]
            ),
            "pelvis_contact_reference_height_m": float(
                fall_thresholds["pelvis_contact_reference_height"]
            ),
            "maximum_pelvis_contact_reference_tilt_rad": float(
                fall_thresholds["maximum_pelvis_contact_reference_tilt"]
            ),
            "reference_height_eligible": _json_bool(
                fall_witness["reference_height_eligible"], env_id
            ),
            "reference_tilt_eligible": _json_bool(
                fall_witness["reference_tilt_eligible"], env_id
            ),
            "pelvis_contact_applicable": _json_bool(
                fall_witness["pelvis_contact_applicable"], env_id
            ),
            "pelvis_force_exceeded": _json_bool(
                fall_witness["pelvis_force_exceeded"], env_id
            ),
            "unexpected_pelvis_contact": pelvis,
            "fired_subcauses": [
                name
                for name, fired in (
                    ("height_deficit", height),
                    ("orientation_error", orientation),
                    ("unexpected_pelvis_contact", pelvis),
                )
                if fired
            ],
        },
        "tracking_divergence": {
            "fired": _json_bool(divergence_witness["diverged"], env_id),
            "root_distance_m": _json_number(
                divergence_witness["root_distance_m"], env_id
            ),
            "maximum_root_distance_m": float(
                divergence_thresholds["maximum_root_distance"]
            ),
            "root_distance_exceeded": root,
            "mean_absolute_joint_error_rad": _json_number(
                divergence_witness["mean_absolute_joint_error_rad"], env_id
            ),
            "maximum_mean_joint_error_rad": float(
                divergence_thresholds["maximum_mean_joint_error"]
            ),
            "mean_joint_error_exceeded": joint,
            "mean_tracking_body_distance_m": _json_number(
                divergence_witness["mean_tracking_body_distance_m"], env_id
            ),
            "maximum_mean_tracking_body_distance_m": float(
                divergence_thresholds["maximum_mean_body_distance"]
            ),
            "mean_body_distance_exceeded": body,
            "fired_subcauses": [
                name
                for name, fired in (
                    ("root_distance", root),
                    ("mean_joint_error", joint),
                    ("mean_tracking_body_distance", body),
                )
                if fired
            ],
        },
        "action_safety": {
            "raw_action_clip": float(raw_action_clip),
            "maximum_absolute_raw_action": _json_number(
                torch.abs(actions).max(dim=-1).values, env_id
            ),
            "raw_action_clip_count": int(raw_clip_mask[env_id].sum().item()),
            "raw_action_clip_fraction": float(
                raw_clip_mask[env_id].to(torch.float64).mean().item()
            ),
            "raw_action_clipped_joints": _named_true_joints(raw_clip_mask, env_id),
            "normalized_rate_limit_count": int(rate_limited[env_id].sum().item()),
            "normalized_rate_limit_fraction": float(
                rate_limited[env_id].to(torch.float64).mean().item()
            ),
            "normalized_rate_limited_joints": _named_true_joints(rate_limited, env_id),
            "target_clamp_count": int(target_clamped[env_id].sum().item()),
            "target_clamp_fraction": float(
                target_clamped[env_id].to(torch.float64).mean().item()
            ),
            "target_clamped_joints": _named_true_joints(target_clamped, env_id),
            "reference_velocity_target_clamp_count": int(
                velocity_target_clamped[env_id].sum().item()
            ),
            "reference_velocity_target_clamp_fraction": float(
                velocity_target_clamped[env_id].to(torch.float64).mean().item()
            ),
            "reference_velocity_target_clamped_joints": _named_true_joints(
                velocity_target_clamped, env_id
            ),
            "action_dimension": action_count,
        },
    }
    effort_available = bool(divergence_witness.get("effort_available", False))
    effort: dict[str, Any] = {
        "available": effort_available,
        "sample_semantics": "final physics substep of the terminal control transition",
        "effort_limit_source": "configs/robot/g1_29dof.yaml policy_index order",
    }
    if effort_available:
        computed_ratio = divergence_witness["computed_abs_torque_to_effort_ratio"]
        applied_ratio = divergence_witness["applied_abs_torque_to_effort_ratio"]
        computed_exceeded = divergence_witness["computed_effort_limit_exceeded"]
        applied_exceeded = divergence_witness["applied_effort_limit_exceeded"]
        effort.update(
            {
                "maximum_abs_computed_torque_to_effort_ratio": _json_number(
                    computed_ratio.max(dim=-1).values, env_id
                ),
                "maximum_abs_applied_torque_to_effort_ratio": _json_number(
                    applied_ratio.max(dim=-1).values, env_id
                ),
                "computed_effort_limit_exceeded_count": int(
                    computed_exceeded[env_id].sum().item()
                ),
                "applied_effort_limit_exceeded_count": int(
                    applied_exceeded[env_id].sum().item()
                ),
                "computed_effort_limit_exceeded_joints": _named_true_joints(
                    computed_exceeded, env_id
                ),
                "applied_effort_limit_exceeded_joints": _named_true_joints(
                    applied_exceeded, env_id
                ),
                "computed_abs_torque_to_effort_ratio_by_joint": _ratio_by_joint(
                    computed_ratio, env_id
                ),
                "applied_abs_torque_to_effort_ratio_by_joint": _ratio_by_joint(
                    applied_ratio, env_id
                ),
            }
        )
    else:
        effort["unavailable_reason"] = divergence_witness.get(
            "effort_unavailable_reason"
        )
    result["joint_effort"] = effort
    return result


def _episode_extrema_record(
    maxima: dict[str, torch.Tensor],
    maximum_steps: dict[str, torch.Tensor],
    env_id: int,
) -> dict[str, Any]:
    return {
        name: {
            "value": _json_number(values, env_id),
            "control_step_index_one_based": (
                int(maximum_steps[name][env_id].item())
                if int(maximum_steps[name][env_id].item()) > 0
                else None
            ),
        }
        for name, values in maxima.items()
    }


def _motion_metric_records(
    *,
    runtime_motions: list[dict[str, Any]],
    step_dt: float,
    episode_returns: torch.Tensor,
    episode_lengths: torch.Tensor,
    finished: torch.Tensor,
    success: list[bool],
    termination_flags: list[dict[str, bool]],
    finish_steps: list[int | None],
    joint_abs_sum: torch.Tensor,
    joint_sq_sum: torch.Tensor,
    joint_samples: torch.Tensor,
    root_distance_sum: torch.Tensor,
    root_distance_sq_sum: torch.Tensor,
    root_orientation_sum: torch.Tensor,
    root_orientation_sq_sum: torch.Tensor,
    root_samples: torch.Tensor,
    body_distance_sum: torch.Tensor,
    body_distance_sq_sum: torch.Tensor,
    body_samples: torch.Tensor,
    contact_match_sum: torch.Tensor,
    contact_samples: torch.Tensor,
    contact_tp: torch.Tensor,
    contact_fp: torch.Tensor,
    contact_fn: torch.Tensor,
    action_abs_sum: torch.Tensor,
    action_abs_max: torch.Tensor,
    action_clip_sum: torch.Tensor,
    action_target_clamp_sum: torch.Tensor,
    action_rate_limit_sum: torch.Tensor,
    velocity_target_clamp_sum: torch.Tensor,
    action_samples: torch.Tensor,
    slide_samples: list[list[list[float]]],
    timeout_term_names: set[str],
    max_steps: int,
    app_stopped: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for env_id, runtime in enumerate(runtime_motions):
        length = int(episode_lengths[env_id].item())
        complete = bool(finished[env_id].item())
        flags = termination_flags[env_id]
        fired = [name for name, value in flags.items() if value]
        if success[env_id]:
            outcome = "motion_end_success"
        elif complete:
            outcome = "non_timeout_termination" if any(
                value
                for name, value in flags.items()
                if value and name not in timeout_term_names
            ) else "ended_without_clean_motion_end"
        elif app_stopped:
            outcome = "simulation_stopped_before_done"
        else:
            outcome = "max_steps_reached_before_done"

        joint_count = int(joint_samples[env_id].item())
        root_count = int(root_samples[env_id].item())
        body_count = int(body_samples[env_id].item())
        contact_count = int(contact_samples[env_id].item())
        action_count = int(action_samples[env_id].item())
        per_foot: dict[str, Any] = {}
        for foot_id, foot_name in enumerate(("left", "right")):
            foot_scores = contact_precision_recall_f1(
                int(contact_tp[env_id, foot_id].item()),
                int(contact_fp[env_id, foot_id].item()),
                int(contact_fn[env_id, foot_id].item()),
            )
            foot_scores["stance_or_actual_contact_slide_speed_p95_mps"] = _safe_p95(
                slide_samples[env_id][foot_id]
            )
            per_foot[foot_name] = foot_scores
        all_slide = slide_samples[env_id][0] + slide_samples[env_id][1]
        frame_count = int(runtime["frame_count"])
        records.append(
            {
                **runtime,
                "first_episode": {
                    "finished": complete,
                    "success": bool(success[env_id]),
                    "outcome": outcome,
                    "return": float(episode_returns[env_id].item()),
                    "length_steps": length,
                    "simulated_duration_s": float(length * step_dt),
                    "completion_ratio_length_over_archive_frames": float(length / frame_count),
                    "global_finish_step": finish_steps[env_id],
                    "termination_flags": flags,
                    "fired_termination_terms": fired,
                    "max_steps_contract": max_steps,
                },
                "metrics": {
                    "sample_semantics": (
                        "pre-step commanded/physical state for each first-episode transition; "
                        "includes the terminal-reference transition and excludes Isaac's internal "
                        "post-done reset state"
                    ),
                    "sampled_state_steps": root_count,
                    "mean_absolute_joint_error_rad": _safe_ratio(
                        float(joint_abs_sum[env_id].item()), joint_count
                    ),
                    "joint_position_rmse_rad": None
                    if joint_count == 0
                    else math.sqrt(float(joint_sq_sum[env_id].item()) / joint_count),
                    "mean_root_position_error_m": _safe_ratio(
                        float(root_distance_sum[env_id].item()), root_count
                    ),
                    "root_position_rmse_m": None
                    if root_count == 0
                    else math.sqrt(float(root_distance_sq_sum[env_id].item()) / root_count),
                    "mean_root_orientation_error_rad": _safe_ratio(
                        float(root_orientation_sum[env_id].item()), root_count
                    ),
                    "root_orientation_rmse_rad": None
                    if root_count == 0
                    else math.sqrt(float(root_orientation_sq_sum[env_id].item()) / root_count),
                    "mean_tracking_body_position_error_m": _safe_ratio(
                        float(body_distance_sum[env_id].item()), body_count
                    ),
                    "tracking_body_position_rmse_m": None
                    if body_count == 0
                    else math.sqrt(float(body_distance_sq_sum[env_id].item()) / body_count),
                    "foot_contact_match_fraction": _safe_ratio(
                        int(contact_match_sum[env_id].item()), contact_count
                    ),
                    "foot_contact": per_foot,
                    "stance_or_actual_contact_slide_speed_p95_mps": _safe_p95(all_slide),
                    "mean_absolute_raw_action": _safe_ratio(
                        float(action_abs_sum[env_id].item()), action_count
                    ),
                    "maximum_absolute_raw_action": float(action_abs_max[env_id].item()),
                    "raw_action_clip_fraction": _safe_ratio(
                        int(action_clip_sum[env_id].item()), action_count
                    ),
                    "predicted_target_clamp_fraction": _safe_ratio(
                        int(action_target_clamp_sum[env_id].item()), action_count
                    ),
                    "normalized_action_rate_limit_fraction": _safe_ratio(
                        int(action_rate_limit_sum[env_id].item()), action_count
                    ),
                    "reference_velocity_target_clamp_fraction": _safe_ratio(
                        int(velocity_target_clamp_sum[env_id].item()), action_count
                    ),
                },
            }
        )
    return records


def main() -> None:
    if not ARGS.motion and not ARGS.manifest:
        # Unlike the generic training input helper, suite evaluation defaults
        # to the final Gate-3 manifest for the requested held-out split.
        ARGS.manifest = [str(WORK_ROOT / "manifests" / f"g1_{ARGS.split}.json")]
    selection = resolve_motion_inputs(
        motions=ARGS.motion,
        manifests=ARGS.manifest,
        split=ARGS.split,
        reference_root=ARGS.reference_root,
        allow_missing=ARGS.allow_missing,
        max_motions=ARGS.max_motions,
    )
    correspondence = None
    if not ARGS.skip_correspondence_check:
        correspondence = path_below(ARGS.correspondence, ACCAD_ROOT, must_exist=True)
        if not correspondence.is_file():
            raise FileNotFoundError(correspondence)
    if ARGS.action_source == "zero":
        if ARGS.allow_legacy_dynamics_ablation:
            raise ValueError(
                "--allow-legacy-dynamics-ablation requires --action-source checkpoint"
            )
        if ARGS.checkpoint is not None:
            raise ValueError("--checkpoint cannot be combined with --action-source zero")
        if ARGS.export != "none":
            raise ValueError("policy export is unavailable for --action-source zero")
        checkpoint = None
        checkpoint_provenance = None
        legacy_dynamics_ablation = False
        official_acceptance_eligible = False
    else:
        checkpoint = resolve_checkpoint(ARGS.checkpoint, latest_if_missing=True)
        assert checkpoint is not None
        checkpoint_provenance = checkpoint_training_contract_provenance(
            checkpoint,
            expected_policy_contract_version=TRACKING_POLICY_CONTRACT_VERSION,
            expected_training_objective_schema=TRAINING_OBJECTIVE_SCHEMA_VERSION,
        )
        legacy_dynamics_ablation = not bool(checkpoint_provenance["compatible"])
        if legacy_dynamics_ablation and not ARGS.allow_legacy_dynamics_ablation:
            raise RuntimeError(
                "checkpoint was not trained under the active actuator/objective contracts; "
                "use --allow-legacy-dynamics-ablation only for an acceptance-ineligible "
                f"diagnostic ({checkpoint_provenance['reason']})"
            )
        if legacy_dynamics_ablation and ARGS.export != "none":
            raise ValueError("legacy dynamics ablations cannot export a policy")
        official_acceptance_eligible = not legacy_dynamics_ablation

    evaluation_dir = new_artifact_dir("evaluations", ARGS.evaluation_name)
    params_dir = evaluation_dir / "params"
    params_dir.mkdir()
    initial_report = command_metadata(sys.argv)
    initial_report.update(
        {
            "schema_version": (
                "accad_g1_tracking_suite_evaluation_v2"
                if official_acceptance_eligible
                else "accad_g1_tracking_suite_diagnostic_v1"
            ),
            "status": "initializing",
            "evaluation_dir": str(evaluation_dir),
            "evaluation_contract": {
                "one_environment_per_selected_motion": True,
                "first_episode_only": True,
                "action_source": ARGS.action_source,
                "deterministic_policy_inference": ARGS.action_source == "checkpoint",
                "deterministic_zero_action_generation": ARGS.action_source == "zero",
                "official_acceptance_eligible": official_acceptance_eligible,
                "legacy_dynamics_ablation": legacy_dynamics_ablation,
                "random_start": False,
                "state_and_observation_noise": False,
                "domain_randomization_events": False,
                "physx_enhanced_determinism": True,
                "physics_warmup_steps_requested": int(ARGS.physics_warmup_steps),
                "gate3_promotion_contract": (
                    "archive gate3_complete is intentionally generation-time false; finalization "
                    "requires every selected archive checksum to be covered by a promoted G1 manifest"
                ),
                "success_definition": (
                    "motion_end fired and no non-timeout termination term fired on the same step"
                ),
            },
            "checkpoint": None
            if checkpoint is None
            else {
                "path": str(checkpoint),
                "checksum_sha256": sha256_file(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "training_contract_provenance": checkpoint_provenance,
            },
            "motion_input": selection.metadata(),
            "correspondence": None
            if correspondence is None
            else {
                "path": str(correspondence),
                "checksum_sha256": sha256_file(correspondence),
            },
        }
    )
    write_json_atomic(evaluation_dir / "suite_evaluation_summary.json", initial_report)

    env_cfg = make_tracking_env_cfg(
        selection.paths,
        num_envs=len(selection.paths),
        eval_mode=True,
        correspondence_path=correspondence,
        fixed_motion_ids=tuple(range(len(selection.paths))),
        random_start=False,
        # ``gate3_complete`` is intentionally false inside generated NPZs.
        # Gate-3 promotion is external: resolve_motion_inputs verifies every
        # archive checksum from the promoted g1_<split>.json manifest, and the
        # acceptance block fails closed when that coverage is absent. Direct
        # --motion remains useful for schema-validated diagnostics only.
        require_gate3_complete=False,
        seed=ARGS.seed,
    )
    env_cfg.sim.physx.enable_enhanced_determinism = True
    agent_cfg = make_tracking_runner_cfg()
    agent_cfg.seed = ARGS.seed
    if getattr(ARGS, "device", None) is not None:
        env_cfg.sim.device = ARGS.device
        agent_cfg.device = ARGS.device
    env_cfg.log_dir = str(evaluation_dir)
    dump_yaml(str(params_dir / "env.yaml"), env_cfg)
    dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)

    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ARGS.seed)

    base_env: ManagerBasedRLEnv | None = None
    wrapped: RslRlVecEnvWrapper | None = None
    wall_start = time.perf_counter()
    try:
        base_env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        physics_warmup = run_physics_cold_start_warmup(
            base_env,
            int(ARGS.physics_warmup_steps),
            expected_action_dim=env_cfg.policy_action_dim,
        )
        observation_contract = tracking_observation_contract()
        live_tensor_contract = resolved_live_tracking_tensor_contract(base_env)
        wrapped = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        runner: OnPolicyRunner | None = None
        policy: Any | None = None
        policy_module: Any | None = None
        exports: list[Path] = []
        export_validations: list[dict[str, Any]] = []
        if ARGS.action_source == "checkpoint":
            assert checkpoint is not None
            runner = OnPolicyRunner(
                wrapped,
                agent_cfg.to_dict(),
                log_dir=None,
                device=agent_cfg.device,
            )
            print(f"[INFO] Loading checkpoint: {checkpoint}")
            runner.load(str(checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=wrapped.device)
            policy_module = runner.alg.policy
            exports = _export_policy(runner, evaluation_dir / "exported", ARGS.export)
            export_validations = validate_policy_exports(
                exports,
                actor_observation_dim=env_cfg.actor_observation_dim,
                action_dim=env_cfg.policy_action_dim,
            )
        else:
            print("[INFO] Zero-residual reference-PD diagnostic; no checkpoint loaded")

        command = base_env.command_manager.get_term("motion")
        action_term = base_env.action_manager.get_term("joint_residual")
        assigned_motion_ids = [int(value) for value in command.motion_ids.detach().cpu().tolist()]
        expected_motion_ids = list(range(len(selection.paths)))
        if assigned_motion_ids != expected_motion_ids:
            raise RuntimeError(
                "per-environment fixed motion mapping changed at runtime: "
                f"expected {expected_motion_ids}, got {assigned_motion_ids}"
            )
        runtime_motions = [
            {
                "environment_id": motion_id,
                "motion_id": motion_id,
                "motion_key": command.library.motion_keys[motion_id],
                "path": str(command.library.paths[motion_id]),
                "checksum_sha256": command.library.checksums_sha256[motion_id],
                "frame_count": int(command.library.lengths[motion_id].item()),
                "duration_s": float(command.library.durations[motion_id].item()),
            }
            for motion_id in expected_motion_ids
        ]
        runtime_contract = {
            "library_checksum_sha256": command.library.library_checksum_sha256,
            "library_contract_checksum_sha256": command.library.contract.digest(),
            "motion_count": command.library.num_motions,
            "per_environment_motion_ids": assigned_motion_ids,
            "motions": runtime_motions,
            "actor_observation_dim": env_cfg.actor_observation_dim,
            "critic_observation_dim": env_cfg.critic_observation_dim,
            "action_dim": env_cfg.policy_action_dim,
            "observation_contract": observation_contract,
            "live_tensor_contract": live_tensor_contract,
            "physics_dt_s": env_cfg.sim.dt,
            "control_dt_s": env_cfg.sim.dt * env_cfg.decimation,
            "action_source": ARGS.action_source,
            "official_acceptance_eligible": official_acceptance_eligible,
            "legacy_dynamics_ablation": legacy_dynamics_ablation,
            "checkpoint_training_contract_provenance": checkpoint_provenance,
            "policy_action_clip": agent_cfg.clip_actions,
            "physical_action_clip": action_term.cfg.action_clip,
            "joint_effort_diagnostics": {
                "joint_names": list(G1_POLICY_JOINT_NAMES),
                "effort_limits_nm": [float(value) for value in G1_CONTRACT.effort],
                "sample_semantics": (
                    "final physics substep of each control transition; terminal sample "
                    "captured before Isaac automatic reset"
                ),
            },
            "evaluation_isolation": resolved_evaluation_isolation(base_env),
            "physics_cold_start_warmup": physics_warmup,
            "device": str(wrapped.device),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "isaaclab": _package_version("isaaclab"),
                "isaaclab_rl": _package_version("isaaclab-rl"),
                "rsl_rl": _package_version("rsl-rl"),
            },
        }
        write_json_atomic(evaluation_dir / "runtime_contract.json", runtime_contract)

        num_envs = base_env.num_envs
        device = wrapped.device
        observations = wrapped.get_observations()
        active = torch.ones(num_envs, device=device, dtype=torch.bool)
        finished = torch.zeros(num_envs, device=device, dtype=torch.bool)
        episode_returns = torch.zeros(num_envs, device=device, dtype=torch.float64)
        episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
        success = [False] * num_envs
        termination_flags: list[dict[str, bool]] = [dict() for _ in range(num_envs)]
        finish_steps: list[int | None] = [None] * num_envs

        float_accumulators = {
            name: torch.zeros(num_envs, device=device, dtype=torch.float64)
            for name in (
                "joint_abs_sum",
                "joint_sq_sum",
                "root_distance_sum",
                "root_distance_sq_sum",
                "root_orientation_sum",
                "root_orientation_sq_sum",
                "body_distance_sum",
                "body_distance_sq_sum",
                "action_abs_sum",
                "action_abs_max",
            )
        }
        count_accumulators = {
            name: torch.zeros(num_envs, device=device, dtype=torch.long)
            for name in (
                "joint_samples",
                "root_samples",
                "body_samples",
                "contact_match_sum",
                "contact_samples",
                "action_clip_sum",
                "action_target_clamp_sum",
                "action_rate_limit_sum",
                "velocity_target_clamp_sum",
                "action_samples",
            )
        }
        contact_tp = torch.zeros(num_envs, 2, device=device, dtype=torch.long)
        contact_fp = torch.zeros_like(contact_tp)
        contact_fn = torch.zeros_like(contact_tp)
        slide_samples: list[list[list[float]]] = [[[], []] for _ in range(num_envs)]

        term_names = tuple(base_env.termination_manager.active_terms)
        timeout_term_names = {
            name
            for name in term_names
            if bool(base_env.termination_manager.get_term_cfg(name).time_out)
        }
        if "motion_end" not in timeout_term_names:
            raise RuntimeError("motion_end must be an explicit timeout-style termination term")
        fall_term_cfg = base_env.termination_manager.get_term_cfg("fall")
        divergence_term_cfg = base_env.termination_manager.get_term_cfg(
            "tracking_divergence"
        )
        fall_term = fall_term_cfg.func
        divergence_term = divergence_term_cfg.func
        if not isinstance(fall_term, ReferenceAwareFallTermination):
            raise RuntimeError("fall termination does not expose the exact witness contract")
        if not isinstance(divergence_term, TrackingDivergenceTermination):
            raise RuntimeError(
                "tracking_divergence termination does not expose the exact witness contract"
            )
        if fall_term_cfg.params.get("capture_witness") is not True:
            raise RuntimeError("evaluation fall witness capture is disabled")
        if divergence_term_cfg.params.get("capture_witness") is not True:
            raise RuntimeError("evaluation divergence witness capture is disabled")
        fall_thresholds = {
            name: value
            for name, value in fall_term_cfg.params.items()
            if name != "capture_witness"
        }
        divergence_thresholds = {
            name: value
            for name, value in divergence_term_cfg.params.items()
            if name != "capture_witness"
        }
        diagnostic_names = (
            "maximum_root_height_deficit_m",
            "maximum_root_orientation_error_rad",
            "maximum_pelvis_contact_force_n",
            "maximum_root_distance_m",
            "maximum_mean_absolute_joint_error_rad",
            "maximum_mean_tracking_body_distance_m",
            "maximum_absolute_raw_action",
            "maximum_raw_action_clip_fraction",
            "maximum_normalized_action_rate_limit_fraction",
            "maximum_target_clamp_fraction",
            "maximum_reference_velocity_target_clamp_fraction",
            "maximum_abs_computed_torque_to_effort_ratio",
            "maximum_abs_applied_torque_to_effort_ratio",
        )
        diagnostic_maxima = {
            name: torch.full(
                (num_envs,),
                -torch.inf,
                device=device,
                dtype=torch.float64,
            )
            for name in diagnostic_names
        }
        diagnostic_maximum_steps = {
            name: torch.zeros(num_envs, device=device, dtype=torch.long)
            for name in diagnostic_names
        }
        terminal_witnesses: list[dict[str, Any] | None] = [None] * num_envs
        max_steps = (
            int(ARGS.max_steps)
            if ARGS.max_steps > 0
            else int(command.library.lengths.max().item()) + 1
        )
        # The recovery runner deliberately leaves the RSL wrapper unclipped so
        # the action term and metrics see the true policy output.  Dynamic
        # safety and the acceptance threshold are owned by the physical action
        # term's immutable Gate-3 +/-1 contract.
        raw_clip = float(action_term.cfg.action_clip)
        steps_executed = 0
        print(
            f"[INFO] Deterministic suite: {num_envs} motions, one env each, "
            f"max_steps={max_steps}"
        )

        while (
            SIMULATION_APP.is_running()
            and steps_executed < max_steps
            and bool(active.any())
        ):
            active_before = active.clone()
            active_float = active_before.to(torch.float64)
            robot = command.robot

            # State metrics are sampled before step(). Isaac resets done envs
            # internally, so this ordering prevents reset poses contaminating
            # the terminal transition of the first episode.
            actual_q = robot.data.joint_pos[:, command.policy_joint_ids]
            joint_error = actual_q - command.joint_pos
            float_accumulators["joint_abs_sum"] += (
                torch.abs(joint_error).sum(dim=-1).to(torch.float64) * active_float
            )
            float_accumulators["joint_sq_sum"] += (
                torch.square(joint_error).sum(dim=-1).to(torch.float64) * active_float
            )
            count_accumulators["joint_samples"] += (
                active_before.to(torch.long) * joint_error.shape[-1]
            )

            root_distance = torch.linalg.vector_norm(
                robot.data.root_link_pos_w - command.root_pos_w, dim=-1
            )
            root_orientation = quat_error_angle_wxyz(
                command.root_quat_wxyz, robot.data.root_link_quat_w
            )
            float_accumulators["root_distance_sum"] += root_distance.to(torch.float64) * active_float
            float_accumulators["root_distance_sq_sum"] += torch.square(root_distance).to(
                torch.float64
            ) * active_float
            float_accumulators["root_orientation_sum"] += root_orientation.to(
                torch.float64
            ) * active_float
            float_accumulators["root_orientation_sq_sum"] += torch.square(
                root_orientation
            ).to(torch.float64) * active_float
            count_accumulators["root_samples"] += active_before.to(torch.long)

            actual_body = robot.data.body_pos_w[:, command.tracking_body_ids]
            body_distance = torch.linalg.vector_norm(
                actual_body - command.tracking_body_pos_w, dim=-1
            )
            float_accumulators["body_distance_sum"] += (
                body_distance.sum(dim=-1).to(torch.float64) * active_float
            )
            float_accumulators["body_distance_sq_sum"] += (
                torch.square(body_distance).sum(dim=-1).to(torch.float64) * active_float
            )
            count_accumulators["body_samples"] += (
                active_before.to(torch.long) * body_distance.shape[-1]
            )

            reference_contact = command.foot_contact
            actual_contact = command.actual_foot_contact
            active_feet = active_before.unsqueeze(-1)
            matches = (reference_contact == actual_contact) & active_feet
            count_accumulators["contact_match_sum"] += matches.sum(dim=-1).to(torch.long)
            count_accumulators["contact_samples"] += active_before.to(torch.long) * 2
            contact_tp += (reference_contact & actual_contact & active_feet).to(torch.long)
            contact_fp += ((~reference_contact) & actual_contact & active_feet).to(torch.long)
            contact_fn += (reference_contact & (~actual_contact) & active_feet).to(torch.long)

            slide_speed = torch.linalg.vector_norm(
                robot.data.body_lin_vel_w[:, command.foot_body_ids, :2], dim=-1
            )
            slide_mask = (reference_contact | actual_contact) & active_feet
            slide_env, slide_foot = torch.where(slide_mask)
            if slide_env.numel() > 0:
                selected_slide = slide_speed[slide_env, slide_foot]
                for env_id, foot_id, value in zip(
                    slide_env.detach().cpu().tolist(),
                    slide_foot.detach().cpu().tolist(),
                    selected_slide.detach().cpu().tolist(),
                    strict=True,
                ):
                    slide_samples[int(env_id)][int(foot_id)].append(float(value))

            with torch.inference_mode():
                actions = deterministic_suite_actions(
                    ARGS.action_source,
                    observations,
                    env_cfg.policy_action_dim,
                    checkpoint_policy=policy,
                )
                if actions.shape != (num_envs, env_cfg.policy_action_dim):
                    raise RuntimeError(
                        f"{ARGS.action_source} action shape changed: expected "
                        f"{(num_envs, env_cfg.policy_action_dim)}, got {tuple(actions.shape)}"
                    )
                if not bool(torch.isfinite(actions[active_before]).all()):
                    raise RuntimeError(
                        f"{ARGS.action_source} produced NaN or Inf for an unfinished first episode"
                    )
                # Completed environments keep simulating because Isaac's vector
                # environment auto-resets them.  Their later episodes are
                # outside this evaluator's contract and must not contaminate
                # action metrics or cause a policy-output failure.
                actions = torch.where(
                    active_before.unsqueeze(-1), actions, torch.zeros_like(actions)
                )
                absolute_actions = torch.abs(actions)
                raw_clip_mask = absolute_actions > raw_clip
                float_accumulators["action_abs_sum"] += (
                    absolute_actions.sum(dim=-1).to(torch.float64) * active_float
                )
                float_accumulators["action_abs_max"] = torch.where(
                    active_before,
                    torch.maximum(
                        float_accumulators["action_abs_max"],
                        absolute_actions.max(dim=-1).values.to(torch.float64),
                    ),
                    float_accumulators["action_abs_max"],
                )
                count_accumulators["action_clip_sum"] += (
                    raw_clip_mask & active_feet.expand_as(actions)
                ).sum(dim=-1).to(torch.long)
                effective_actions = actions.clamp(-raw_clip, raw_clip)
                hard = robot.data.joint_pos_limits[:, action_term.joint_ids]
                soft = robot.data.soft_joint_pos_limits[:, action_term.joint_ids]
                predicted_target = reference_residual_target(
                    effective_actions,
                    command.joint_pos,
                    action_term.normalized_residual,
                    residual_scale=action_term.cfg.residual_scale,
                    action_clip=action_term.cfg.action_clip,
                    normalized_delta_limit=action_term.cfg.normalized_delta_limit,
                    hard_limits=hard,
                    soft_limits=soft,
                    target_limit_mode=action_term.cfg.target_limit_mode,
                )
                predicted_velocity_target = reference_velocity_target(
                    command.joint_vel,
                    scale=action_term.cfg.reference_velocity_feedforward_scale,
                    velocity_limits=robot.data.soft_joint_vel_limits[
                        :, action_term.joint_ids
                    ],
                )
                count_accumulators["action_target_clamp_sum"] += (
                    predicted_target.target_clamped & active_feet.expand_as(actions)
                ).sum(dim=-1).to(torch.long)
                count_accumulators["action_rate_limit_sum"] += (
                    predicted_target.rate_limited & active_feet.expand_as(actions)
                ).sum(dim=-1).to(torch.long)
                count_accumulators["velocity_target_clamp_sum"] += (
                    predicted_velocity_target.target_clamped
                    & active_feet.expand_as(actions)
                ).sum(dim=-1).to(torch.long)
                count_accumulators["action_samples"] += (
                    active_before.to(torch.long) * actions.shape[-1]
                )

                observations, rewards, dones, _ = wrapped.step(actions)
                if policy_module is not None:
                    policy_module.reset(dones)

            # These stateful termination terms execute after the final physics
            # substep and before Isaac resets done environments.  Reading their
            # detached snapshots here avoids both a one-transition lag and reset
            # pose contamination.
            fall_witness = fall_term.last_witness
            divergence_witness = divergence_term.last_witness
            if fall_witness is None or divergence_witness is None:
                raise RuntimeError("termination witness was not captured during env.step()")
            if fall_witness["capture_sequence"] != divergence_witness["capture_sequence"]:
                raise RuntimeError("fall and divergence witness sequences differ")
            step_index = steps_executed + 1
            diagnostic_step_values = {
                "maximum_root_height_deficit_m": fall_witness[
                    "root_height_deficit_m"
                ],
                "maximum_root_orientation_error_rad": fall_witness[
                    "root_orientation_error_rad"
                ],
                "maximum_pelvis_contact_force_n": fall_witness[
                    "pelvis_contact_force_n"
                ],
                "maximum_root_distance_m": divergence_witness["root_distance_m"],
                "maximum_mean_absolute_joint_error_rad": divergence_witness[
                    "mean_absolute_joint_error_rad"
                ],
                "maximum_mean_tracking_body_distance_m": divergence_witness[
                    "mean_tracking_body_distance_m"
                ],
                "maximum_absolute_raw_action": absolute_actions.max(dim=-1).values,
                "maximum_raw_action_clip_fraction": raw_clip_mask.to(torch.float64).mean(
                    dim=-1
                ),
                "maximum_normalized_action_rate_limit_fraction": predicted_target.rate_limited.to(
                    torch.float64
                ).mean(dim=-1),
                "maximum_target_clamp_fraction": predicted_target.target_clamped.to(
                    torch.float64
                ).mean(dim=-1),
                "maximum_reference_velocity_target_clamp_fraction": (
                    predicted_velocity_target.target_clamped.to(torch.float64).mean(dim=-1)
                ),
            }
            if bool(divergence_witness.get("effort_available", False)):
                diagnostic_step_values.update(
                    {
                        "maximum_abs_computed_torque_to_effort_ratio": divergence_witness[
                            "computed_abs_torque_to_effort_ratio"
                        ].max(dim=-1).values,
                        "maximum_abs_applied_torque_to_effort_ratio": divergence_witness[
                            "applied_abs_torque_to_effort_ratio"
                        ].max(dim=-1).values,
                    }
                )
            for name, values in diagnostic_step_values.items():
                _update_episode_maximum(
                    values,
                    active=active_before,
                    control_step=step_index,
                    maxima=diagnostic_maxima[name],
                    maximum_steps=diagnostic_maximum_steps[name],
                )

            steps_executed += 1
            episode_returns += torch.where(
                active_before,
                rewards.to(torch.float64),
                torch.zeros_like(episode_returns),
            )
            episode_lengths += active_before.to(torch.long)
            done_now = (dones > 0) & active_before
            if bool(done_now.any()):
                term_values = {
                    name: base_env.termination_manager.get_term(name).clone()
                    for name in term_names
                }
                for env_id in torch.where(done_now)[0].detach().cpu().tolist():
                    flags = {
                        name: bool(term_values[name][env_id].item()) for name in term_names
                    }
                    termination_flags[env_id] = flags
                    success[env_id] = clean_motion_end_success(
                        flags, timeout_term_names=timeout_term_names
                    )
                    finish_steps[env_id] = steps_executed
                    witness_motion_id = int(
                        fall_witness["motion_ids"][env_id].item()
                    )
                    if witness_motion_id != env_id:
                        raise RuntimeError(
                            "terminal witness motion mapping changed: "
                            f"env={env_id}, witness_motion={witness_motion_id}"
                        )
                    terminal_witnesses[env_id] = _terminal_transition_witness(
                        env_id=env_id,
                        control_step=steps_executed,
                        termination_flags=flags,
                        fall_witness=fall_witness,
                        divergence_witness=divergence_witness,
                        fall_thresholds=fall_thresholds,
                        divergence_thresholds=divergence_thresholds,
                        actions=actions,
                        raw_action_clip=raw_clip,
                        raw_clip_mask=raw_clip_mask,
                        target_clamped=predicted_target.target_clamped,
                        rate_limited=predicted_target.rate_limited,
                        velocity_target_clamped=(
                            predicted_velocity_target.target_clamped
                        ),
                    )
                finished |= done_now
                active &= ~done_now

        app_stopped = not SIMULATION_APP.is_running()
        wall_seconds = time.perf_counter() - wall_start
        records = _motion_metric_records(
            runtime_motions=runtime_motions,
            step_dt=base_env.step_dt,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            finished=finished,
            success=success,
            termination_flags=termination_flags,
            finish_steps=finish_steps,
            joint_abs_sum=float_accumulators["joint_abs_sum"],
            joint_sq_sum=float_accumulators["joint_sq_sum"],
            joint_samples=count_accumulators["joint_samples"],
            root_distance_sum=float_accumulators["root_distance_sum"],
            root_distance_sq_sum=float_accumulators["root_distance_sq_sum"],
            root_orientation_sum=float_accumulators["root_orientation_sum"],
            root_orientation_sq_sum=float_accumulators["root_orientation_sq_sum"],
            root_samples=count_accumulators["root_samples"],
            body_distance_sum=float_accumulators["body_distance_sum"],
            body_distance_sq_sum=float_accumulators["body_distance_sq_sum"],
            body_samples=count_accumulators["body_samples"],
            contact_match_sum=count_accumulators["contact_match_sum"],
            contact_samples=count_accumulators["contact_samples"],
            contact_tp=contact_tp,
            contact_fp=contact_fp,
            contact_fn=contact_fn,
            action_abs_sum=float_accumulators["action_abs_sum"],
            action_abs_max=float_accumulators["action_abs_max"],
            action_clip_sum=count_accumulators["action_clip_sum"],
            action_target_clamp_sum=count_accumulators["action_target_clamp_sum"],
            action_rate_limit_sum=count_accumulators["action_rate_limit_sum"],
            velocity_target_clamp_sum=count_accumulators[
                "velocity_target_clamp_sum"
            ],
            action_samples=count_accumulators["action_samples"],
            slide_samples=slide_samples,
            timeout_term_names=timeout_term_names,
            max_steps=max_steps,
            app_stopped=app_stopped,
        )
        diagnostic_motion_records: list[dict[str, Any]] = []
        for env_id, (runtime, record) in enumerate(
            zip(runtime_motions, records, strict=True)
        ):
            per_motion_diagnostic = {
                "environment_id": env_id,
                "motion_id": env_id,
                "motion_key": runtime["motion_key"],
                "first_episode_steps": int(episode_lengths[env_id].item()),
                "exact_terminal_state_captured": terminal_witnesses[env_id] is not None,
                "terminal_transition_witness": terminal_witnesses[env_id],
                "episode_extrema": _episode_extrema_record(
                    diagnostic_maxima,
                    diagnostic_maximum_steps,
                    env_id,
                ),
            }
            diagnostic_motion_records.append(per_motion_diagnostic)
            record["first_episode"]["termination_diagnostics"] = {
                "timing": "post_physics_pre_automatic_reset",
                "terminal_transition_witness": terminal_witnesses[env_id],
                "episode_extrema": per_motion_diagnostic["episode_extrema"],
            }
        diagnostics_payload = {
            "schema_version": "accad_g1_first_episode_diagnostics_v1",
            "created_at_utc": utc_now(),
            "action_source": ARGS.action_source,
            "official_acceptance_input": False,
            "timing_contract": {
                "termination_state": (
                    "post_physics_final_decimation_substep_before_Isaac_automatic_reset"
                ),
                "action": "commanded_for_the_same_control_transition",
                "torque": "final_physics_substep_not_peak_over_all_decimation_substeps",
                "first_episode_only": True,
            },
            "fall_thresholds": fall_thresholds,
            "tracking_divergence_thresholds": divergence_thresholds,
            "joint_effort_limits": runtime_contract["joint_effort_diagnostics"],
            "motions": diagnostic_motion_records,
        }
        diagnostics_path = write_json_atomic(
            evaluation_dir / "first_episode_diagnostics.json",
            diagnostics_payload,
        )
        diagnostics_artifact = {
            "path": str(diagnostics_path),
            "checksum_sha256": sha256_file(diagnostics_path),
            "bytes": diagnostics_path.stat().st_size,
            "schema_version": diagnostics_payload["schema_version"],
            "official_acceptance_input": False,
        }

        lengths = [float(record["first_episode"]["length_steps"]) for record in records]
        returns = [float(record["first_episode"]["return"]) for record in records]
        completion_ratios = [
            float(record["first_episode"]["completion_ratio_length_over_archive_frames"])
            for record in records
        ]
        macro_metric_names = (
            "mean_absolute_joint_error_rad",
            "joint_position_rmse_rad",
            "mean_root_position_error_m",
            "root_position_rmse_m",
            "mean_root_orientation_error_rad",
            "root_orientation_rmse_rad",
            "mean_tracking_body_position_error_m",
            "tracking_body_position_rmse_m",
            "foot_contact_match_fraction",
            "stance_or_actual_contact_slide_speed_p95_mps",
            "mean_absolute_raw_action",
            "raw_action_clip_fraction",
            "predicted_target_clamp_fraction",
            "reference_velocity_target_clamp_fraction",
        )
        macro_metrics = {
            name: _safe_mean(
                [
                    float(record["metrics"][name])
                    for record in records
                    if record["metrics"][name] is not None
                ]
            )
            for name in macro_metric_names
        }
        total_joint_samples = int(count_accumulators["joint_samples"].sum().item())
        total_root_samples = int(count_accumulators["root_samples"].sum().item())
        total_body_samples = int(count_accumulators["body_samples"].sum().item())
        total_contact_samples = int(count_accumulators["contact_samples"].sum().item())
        total_action_samples = int(count_accumulators["action_samples"].sum().item())
        weighted_metrics = {
            "mean_absolute_joint_error_rad": _safe_ratio(
                float(float_accumulators["joint_abs_sum"].sum().item()),
                total_joint_samples,
            ),
            "joint_position_rmse_rad": None
            if total_joint_samples == 0
            else math.sqrt(
                float(float_accumulators["joint_sq_sum"].sum().item())
                / total_joint_samples
            ),
            "mean_root_position_error_m": _safe_ratio(
                float(float_accumulators["root_distance_sum"].sum().item()),
                total_root_samples,
            ),
            "root_position_rmse_m": None
            if total_root_samples == 0
            else math.sqrt(
                float(float_accumulators["root_distance_sq_sum"].sum().item())
                / total_root_samples
            ),
            "mean_root_orientation_error_rad": _safe_ratio(
                float(float_accumulators["root_orientation_sum"].sum().item()),
                total_root_samples,
            ),
            "root_orientation_rmse_rad": None
            if total_root_samples == 0
            else math.sqrt(
                float(float_accumulators["root_orientation_sq_sum"].sum().item())
                / total_root_samples
            ),
            "mean_tracking_body_position_error_m": _safe_ratio(
                float(float_accumulators["body_distance_sum"].sum().item()),
                total_body_samples,
            ),
            "tracking_body_position_rmse_m": None
            if total_body_samples == 0
            else math.sqrt(
                float(float_accumulators["body_distance_sq_sum"].sum().item())
                / total_body_samples
            ),
            "foot_contact_match_fraction": _safe_ratio(
                int(count_accumulators["contact_match_sum"].sum().item()),
                total_contact_samples,
            ),
            "mean_absolute_raw_action": _safe_ratio(
                float(float_accumulators["action_abs_sum"].sum().item()),
                total_action_samples,
            ),
            "raw_action_clip_fraction": _safe_ratio(
                int(count_accumulators["action_clip_sum"].sum().item()),
                total_action_samples,
            ),
            "predicted_target_clamp_fraction": _safe_ratio(
                int(count_accumulators["action_target_clamp_sum"].sum().item()),
                total_action_samples,
            ),
            "normalized_action_rate_limit_fraction": _safe_ratio(
                int(count_accumulators["action_rate_limit_sum"].sum().item()),
                total_action_samples,
            ),
            "reference_velocity_target_clamp_fraction": _safe_ratio(
                int(count_accumulators["velocity_target_clamp_sum"].sum().item()),
                total_action_samples,
            ),
        }
        termination_counts = {
            name: sum(bool(record["first_episode"]["termination_flags"].get(name, False)) for record in records)
            for name in term_names
        }
        total_env_steps = int(episode_lengths.sum().item())
        success_count = int(sum(success))
        aggregate_contact = {
            foot_name: contact_precision_recall_f1(
                int(contact_tp[:, foot_id].sum().item()),
                int(contact_fp[:, foot_id].sum().item()),
                int(contact_fn[:, foot_id].sum().item()),
            )
            for foot_id, foot_name in enumerate(("left", "right"))
        }
        aggregate_contact["overall"] = contact_precision_recall_f1(
            int(contact_tp.sum().item()),
            int(contact_fp.sum().item()),
            int(contact_fn.sum().item()),
        )
        aggregate_slide = [
            value
            for env_samples in slide_samples
            for foot_samples in env_samples
            for value in foot_samples
        ]
        aggregate = {
            "motion_count": num_envs,
            "finished_first_episodes": int(finished.sum().item()),
            "successful_motion_end_episodes": success_count,
            "success_rate": float(success_count / num_envs),
            "failed_or_incomplete_motion_ids": [
                record["motion_id"] for record in records if not record["first_episode"]["success"]
            ],
            "termination_counts": termination_counts,
            "return_mean": _safe_mean(returns),
            "return_min": _safe_min(returns),
            "return_max": _safe_max(returns),
            "length_steps_mean": _safe_mean(lengths),
            "completion_ratio_mean": _safe_mean(completion_ratios),
            "completion_ratio_min": _safe_min(completion_ratios),
            "macro_motion_metrics": macro_metrics,
            "weighted_frame_or_element_metrics": weighted_metrics,
            "aggregate_foot_contact": aggregate_contact,
            "aggregate_stance_or_actual_contact_slide_speed_p95_mps": _safe_p95(
                aggregate_slide
            ),
            "total_first_episode_environment_steps": total_env_steps,
            "wall_time_s": wall_seconds,
            "simulated_environment_seconds": total_env_steps * base_env.step_dt,
            "simulated_environment_seconds_per_wall_second": _safe_ratio(
                total_env_steps * base_env.step_dt, wall_seconds
            ),
            "global_control_steps_executed": steps_executed,
            "max_steps": max_steps,
            "simulation_app_stopped_early": app_stopped,
        }
        if official_acceptance_eligible:
            acceptance = build_suite_acceptance(
                aggregate,
                manifest_promotion_verified=bool(
                    initial_report["motion_input"]["manifest_promotion_verified"]
                ),
            )
            passed = acceptance["passed"]
        else:
            if legacy_dynamics_ablation:
                diagnostic_reason = (
                    "checkpoint policy was trained under a different or unverifiable "
                    "actuator dynamics contract"
                )
            else:
                diagnostic_reason = (
                    "zero residual actions diagnose reference-PD feasibility and are not "
                    "a learned-policy acceptance result"
                )
            acceptance = {
                "schema_version": "accad_g1_suite_diagnostic_ineligible_v1",
                "eligible": False,
                "evaluated": False,
                "passed": False,
                "reason": diagnostic_reason,
            }
            passed = False

        wrapped.close()
        wrapped = None
        base_env = None
        report = dict(initial_report)
        report.update(
            {
                "status": "completed",
                "passed": passed,
                "finished_at_utc": utc_now(),
                "runtime_contract": runtime_contract,
                "aggregate": aggregate,
                "acceptance": acceptance,
                "motions": records,
                "first_episode_diagnostics": diagnostics_artifact,
                "exports": [
                    {
                        "path": str(path),
                        "checksum_sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in exports
                ],
                "export_validations": export_validations,
            }
        )
        report_path = write_json_atomic(
            evaluation_dir / "suite_evaluation_summary.json", report
        )
        print(
            f"[INFO] Suite complete: {success_count}/{num_envs} clean motion-end successes"
        )
        print(f"[INFO] Suite report: {report_path}")
    except BaseException as exc:
        failed = dict(initial_report)
        failed.update(
            {
                "status": "failed",
                "passed": False,
                "finished_at_utc": utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json_atomic(evaluation_dir / "suite_evaluation_summary.json", failed)
        raise
    finally:
        if wrapped is not None:
            wrapped.close()
        elif base_env is not None:
            base_env.close()


try:
    main()
finally:
    SIMULATION_APP.close()
