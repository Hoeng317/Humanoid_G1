from __future__ import annotations

import json
import inspect
import math
from pathlib import Path
import sys

import pytest
import torch
from tensordict import TensorDict


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from accad_g1.motion_library import MotionLibrary
from accad_g1.tracking_task import (
    ACCAD_ROOT,
    ACTION_BOUNDARY_ONSET,
    DEFAULT_ACTION_NOISE_STD,
    DEFAULT_ACTOR_OBS_DIM,
    DEFAULT_CRITIC_OBS_DIM,
    DEFAULT_TRAINING_MOTION_ASSIGNMENT,
    G1_ACTION_DIM,
    G1_CONTRACT,
    G1_POLICY_JOINT_NAMES,
    RECOVERY_PPO_DESIRED_KL,
    RECOVERY_PPO_LEARNING_RATE,
    RECOVERY_PPO_SCHEDULE,
    RECOVERY_ALIVE_REWARD_WEIGHT,
    RECOVERY_ACTION_BOUNDARY_REWARD_WEIGHT,
    RECOVERY_ACTION_LIMITER_GAP_REWARD_WEIGHT,
    RECOVERY_DIVERGENCE_RISK_REWARD_WEIGHT,
    RECOVERY_FAILURE_REWARD_WEIGHT,
    RECOVERY_RESET_STATE_NOISE_SCALE,
    TRACKING_POLICY_CONTRACT_VERSION,
    DIVERGENCE_RISK_ONSET_FRACTION,
    TRAINING_OBJECTIVE_SCHEMA_VERSION,
    TRAINING_MOTION_ASSIGNMENT_CHOICES,
    TRAINING_MOTION_ASSIGNMENT_SCHEMA_VERSION,
    _require_path_below_accad,
    action_boundary_margin_l2,
    action_limiter_request_gap_l2,
    advance_motion_time_after_step,
    align_positions_by_env_yaw,
    align_quaternions_by_env_yaw,
    alignment_from_reference_anchor,
    balanced_fixed_motion_ids,
    clean_motion_end_success,
    deterministic_suite_actions,
    fixed_motion_ids_for_envs,
    isaac_tracking_available,
    make_tracking_env_cfg,
    make_tracking_runner_cfg,
    mix_frame_zero_start_times,
    positive_horizon_root_pose_preview,
    quat_apply_wxyz,
    quat_error_angle_wxyz,
    quat_yaw_wxyz,
    raw_action_excess_l2,
    reference_aware_fall_components,
    reference_aware_fall_mask,
    reference_residual_target,
    reference_velocity_target,
    require_isaac_tracking,
    resolve_training_motion_assignment,
    resolved_live_tracking_tensor_contract,
    resolved_policy_action_noise_contract,
    root_frame_vector,
    root_frame_vector_error,
    resolve_recovery_training_objective,
    resolve_action_noise_std,
    rotate_by_env_yaw,
    set_policy_action_noise_std,
    set_ppo_learning_rate_exact,
    terminal_failure_impulse,
    training_motion_assignment_contract,
    tracking_observation_contract,
    tracking_observation_dims,
    tracking_divergence_components,
    tracking_divergence_risk_barrier,
    torque_effort_ratio_components,
    yaw_quat_wxyz,
)


B3_REFERENCE = (
    PIPELINE_ROOT
    / "work"
    / "g1"
    / "Female1Walking_c3d"
    / "B3_-_walk1_g1_50hz.npz"
)


def test_default_action_and_observation_dimensions_are_exact() -> None:
    assert G1_ACTION_DIM == 29
    assert len(G1_POLICY_JOINT_NAMES) == 29
    assert DEFAULT_ACTOR_OBS_DIM == 293
    assert DEFAULT_CRITIC_OBS_DIM == 505
    assert tracking_observation_dims() == (293, 505)
    assert tracking_observation_dims(num_horizons=3) == (253, 465)
    assert DEFAULT_CRITIC_OBS_DIM - DEFAULT_ACTOR_OBS_DIM == 14 * 15 + 2
    with pytest.raises(ValueError, match="positive"):
        tracking_observation_dims(num_horizons=0)


def test_v6_observation_contract_has_exact_names_dimensions_ranges_and_scales() -> None:
    contract = tracking_observation_contract()
    assert contract["schema_version"] == "accad_g1_tracking_observation_contract_v6"
    assert contract["policy_contract_version"] == TRACKING_POLICY_CONTRACT_VERSION
    assert contract["future_horizons_s"] == [0.0, 0.04, 0.08, 0.16]
    assert contract["positive_preview_horizons_s"] == [0.04, 0.08, 0.16]
    assert contract["action"] == {
        "term_names": ["joint_residual"],
        "term_dimensions": [29],
        "dimension": 29,
        "position_target_formula": (
            "clip_limits(q_ref + 0.25 * rate_limit(clip(action,-1,1), "
            "0.15_per_control_step))"
        ),
        "velocity_target_formula": "zeros_like(reference_joint_velocity)",
        "reference_velocity_feedforward_scale": 0.0,
    }

    policy = contract["observation_groups"]["policy"]
    critic = contract["observation_groups"]["critic"]
    expected_policy = [
        ("future_joint_position_error", 116, [0, 116]),
        ("future_root_position_error", 9, [116, 125]),
        ("future_root_orientation_error", 18, [125, 143]),
        ("joint_velocity_error", 29, [143, 172]),
        ("root_orientation_error", 6, [172, 178]),
        ("root_position_error", 3, [178, 181]),
        ("root_linear_velocity_error", 3, [181, 184]),
        ("target_root_linear_velocity", 3, [184, 187]),
        ("phase", 2, [187, 189]),
        ("root_angular_velocity_error", 3, [189, 192]),
        ("target_root_angular_velocity", 3, [192, 195]),
        ("projected_gravity", 3, [195, 198]),
        ("joint_position", 29, [198, 227]),
        ("joint_velocity", 29, [227, 256]),
        ("predicted_contacts", 8, [256, 264]),
        ("last_action", 29, [264, 293]),
    ]
    assert [
        (term["name"], term["dimension"], term["range"])
        for term in policy["terms"]
    ] == expected_policy
    assert policy["dimension"] == 293
    assert critic["dimension"] == 505
    assert [term["name"] for term in critic["terms"][: len(expected_policy)]] == [
        name for name, _, _ in expected_policy
    ]
    assert [
        (term["name"], term["dimension"], term["range"])
        for term in critic["terms"][len(expected_policy) :]
    ] == [
        ("tracking_body_position_error", 42, [293, 335]),
        ("tracking_body_orientation_error", 84, [335, 419]),
        ("tracking_body_linear_velocity_error", 42, [419, 461]),
        ("tracking_body_angular_velocity_error", 42, [461, 503]),
        ("actual_foot_contacts", 2, [503, 505]),
    ]
    scales = {term["name"]: term["scale"] for term in policy["terms"]}
    assert scales["target_root_linear_velocity"] == 0.2
    assert scales["target_root_angular_velocity"] == 0.2
    assert scales["future_root_position_error"] == 1.0
    assert scales["future_root_orientation_error"] == 1.0
    assert all(term["frame"] == "actual_g1_pelvis" for term in policy["terms"][1:3])
    # This is an artifact contract, so it must remain directly JSON serializable.
    assert json.loads(json.dumps(contract, sort_keys=True)) == contract


def test_fixed_lr_recovery_contract_is_explicit() -> None:
    assert TRACKING_POLICY_CONTRACT_VERSION == (
        "accad_g1_tracking_v7_zero_velocity_target_train_selected"
    )
    assert RECOVERY_PPO_LEARNING_RATE == 1.0e-4
    assert RECOVERY_PPO_SCHEDULE == "fixed"
    assert RECOVERY_PPO_DESIRED_KL == 0.01
    assert RECOVERY_RESET_STATE_NOISE_SCALE == 0.0
    assert RECOVERY_ALIVE_REWARD_WEIGHT == 0.25
    assert RECOVERY_FAILURE_REWARD_WEIGHT == -10.0
    assert RECOVERY_DIVERGENCE_RISK_REWARD_WEIGHT == -20.0
    assert RECOVERY_ACTION_BOUNDARY_REWARD_WEIGHT == -2.0
    assert RECOVERY_ACTION_LIMITER_GAP_REWARD_WEIGHT == -1.0
    assert DIVERGENCE_RISK_ONSET_FRACTION == 0.60
    assert ACTION_BOUNDARY_ONSET == 0.90
    assert TRAINING_OBJECTIVE_SCHEMA_VERSION == (
        "accad_g1_training_objective_v4_strong_dense_divergence_action_margins"
    )
    assert DEFAULT_ACTION_NOISE_STD == 0.2


def test_terminal_failure_impulse_cancels_reward_manager_dt_exactly() -> None:
    terminated = torch.tensor([False, True, False, True])
    timeouts = torch.tensor([False, False, True, True])
    raw = terminal_failure_impulse(terminated, 0.02)
    assert torch.equal(raw, torch.tensor([0.0, 50.0, 0.0, 50.0]))
    integrated = raw * RECOVERY_FAILURE_REWARD_WEIGHT * 0.02
    assert torch.equal(integrated, torch.tensor([0.0, -10.0, 0.0, -10.0]))
    # A pure timeout is excluded by TerminationManager.terminated; a timeout
    # simultaneous with a physical failure remains a penalized failure.
    assert torch.equal(
        integrated != 0.0,
        terminated,
    )
    assert torch.equal(timeouts & ~terminated, torch.tensor([False, False, True, False]))


def test_terminal_failure_impulse_is_dt_invariant_after_integration() -> None:
    terminated = torch.tensor([True, False])
    for dt in (0.01, 0.02, 0.04):
        integrated = (
            terminal_failure_impulse(terminated, dt)
            * RECOVERY_FAILURE_REWARD_WEIGHT
            * dt
        )
        assert torch.equal(integrated, torch.tensor([-10.0, 0.0]))
    for invalid in (0.0, -0.02, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="step_dt"):
            terminal_failure_impulse(terminated, invalid)
    with pytest.raises(ValueError, match="rank-one boolean"):
        terminal_failure_impulse(terminated.float(), 0.02)


def test_runner_action_noise_std_is_explicit_and_validated() -> None:
    assert resolve_action_noise_std(0.037) == 0.037
    signature = inspect.signature(make_tracking_runner_cfg)
    assert signature.parameters["init_noise_std"].default == DEFAULT_ACTION_NOISE_STD
    assert signature.parameters["init_noise_std"].kind is inspect.Parameter.KEYWORD_ONLY
    for value in (0.0, -0.1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite positive"):
            resolve_action_noise_std(value)


@pytest.mark.parametrize("noise_std_type", ["scalar", "log"])
def test_loaded_policy_action_noise_override_is_exact(noise_std_type: str) -> None:
    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_dependent_std = False
            self.noise_std_type = noise_std_type
            parameter_name = "std" if noise_std_type == "scalar" else "log_std"
            initial = (
                torch.linspace(0.1, 0.3, 29)
                if noise_std_type == "scalar"
                else torch.linspace(-0.3, 0.3, 29)
            )
            self.register_parameter(
                parameter_name, torch.nn.Parameter(initial)
            )

    policy = Policy()
    witness = set_policy_action_noise_std(policy, 0.037)
    parameter = getattr(policy, witness["after"]["parameter_name"])
    stored = 0.037 if noise_std_type == "scalar" else math.log(0.037)
    assert torch.equal(parameter.detach(), torch.full_like(parameter, stored))
    assert witness["requested_std"] == 0.037
    assert witness["after"] == resolved_policy_action_noise_contract(policy)
    assert witness["after"]["resolved_std_min"] == pytest.approx(0.037)
    assert witness["after"]["resolved_std_max"] == pytest.approx(0.037)


@pytest.mark.parametrize(
    ("state_dependent", "noise_std_type", "parameter_name", "shape"),
    [
        (True, "log", "log_std", (29,)),
        (False, "unknown", "log_std", (29,)),
        (False, "log", "missing", (29,)),
        (False, "scalar", "std", (28,)),
    ],
)
def test_policy_action_noise_contract_rejects_unsupported_layouts(
    state_dependent: bool,
    noise_std_type: str,
    parameter_name: str,
    shape: tuple[int, ...],
) -> None:
    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_dependent_std = state_dependent
            self.noise_std_type = noise_std_type
            if parameter_name != "missing":
                self.register_parameter(
                    parameter_name, torch.nn.Parameter(torch.zeros(shape))
                )

    with pytest.raises(RuntimeError, match="action-noise|noise_std_type|shape"):
        resolved_policy_action_noise_contract(Policy())


def test_ppo_learning_rate_override_updates_algorithm_and_all_optimizer_groups() -> None:
    class Optimizer:
        param_groups = [{"lr": 7.0e-4}, {"lr": 3.0e-4}]

    class Algorithm:
        learning_rate = 7.0e-4
        optimizer = Optimizer()

    algorithm = Algorithm()
    witness = set_ppo_learning_rate_exact(algorithm, 3.0e-5)
    assert witness == {
        "schema_version": "accad_g1_ppo_learning_rate_override_v1",
        "requested_learning_rate": 3.0e-5,
        "before": {
            "algorithm_learning_rate": 7.0e-4,
            "optimizer_param_group_learning_rates": [7.0e-4, 3.0e-4],
        },
        "after": {
            "algorithm_learning_rate": 3.0e-5,
            "optimizer_param_group_learning_rates": [3.0e-5, 3.0e-5],
        },
    }
    assert algorithm.learning_rate == 3.0e-5
    assert [group["lr"] for group in algorithm.optimizer.param_groups] == [
        3.0e-5,
        3.0e-5,
    ]


def test_ppo_learning_rate_override_fails_closed_on_invalid_inputs_and_layouts() -> None:
    class Algorithm:
        learning_rate = 1.0e-4

    for value in (0.0, -1.0e-4, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite positive"):
            set_ppo_learning_rate_exact(Algorithm(), value)
    Algorithm.optimizer = type("Optimizer", (), {"param_groups": []})()
    with pytest.raises(RuntimeError, match="non-empty param_groups"):
        set_ppo_learning_rate_exact(Algorithm(), 1.0e-4)
    Algorithm.optimizer.param_groups = [{}]
    with pytest.raises(RuntimeError, match="does not expose an lr"):
        set_ppo_learning_rate_exact(Algorithm(), 1.0e-4)


def test_recovery_training_objective_validation_fails_closed() -> None:
    assert resolve_recovery_training_objective(0.0, 0.25, -10.0) == (
        0.0,
        0.25,
        -10.0,
    )
    for arguments in (
        (-0.1, 0.25, -10.0),
        (1.1, 0.25, -10.0),
        (0.0, 0.0, -10.0),
        (0.0, 0.25, 0.0),
        (float("nan"), 0.25, -10.0),
    ):
        with pytest.raises(ValueError):
            resolve_recovery_training_objective(*arguments)


def test_fixed_motion_ids_preserve_scalar_and_per_environment_contracts() -> None:
    assert fixed_motion_ids_for_envs(
        num_envs=3, num_motions=5, fixed_motion_id=2
    ) == (2, 2, 2)
    assert fixed_motion_ids_for_envs(
        num_envs=3, num_motions=5, fixed_motion_ids=[4, 0, 3]
    ) == (4, 0, 3)
    assert (
        fixed_motion_ids_for_envs(num_envs=3, num_motions=5) is None
    )


def test_balanced_fixed_training_assignment_has_exact_coverage_contract() -> None:
    assert DEFAULT_TRAINING_MOTION_ASSIGNMENT == "balanced-fixed"
    assert TRAINING_MOTION_ASSIGNMENT_CHOICES == (
        "episode-uniform",
        "balanced-fixed",
    )
    mapping = balanced_fixed_motion_ids(num_envs=8, num_motions=3)
    assert mapping == (0, 1, 2, 0, 1, 2, 0, 1)
    assert resolve_training_motion_assignment(
        "balanced-fixed", num_envs=8, num_motions=3
    ) == mapping

    contract = training_motion_assignment_contract(
        "balanced-fixed",
        num_envs=8,
        motion_keys=("a.npz", "b.npz", "c.npz"),
        num_steps_per_env=4,
        learning_updates=5,
        fixed_motion_ids=mapping,
    )
    assert contract["schema_version"] == TRAINING_MOTION_ASSIGNMENT_SCHEMA_VERSION
    assert contract["mode"] == "balanced-fixed"
    assert contract["production_default"] == "balanced-fixed"
    assert contract["fixed_environment_assignment"] is True
    assert contract["mapping_length"] == 8
    assert len(contract["mapping_digest_sha256"]) == 64
    assert contract["maximum_environment_count_difference"] == 1
    assert contract["all_motions_have_nonzero_environments"] is True
    assert contract["all_motions_nonzero_witness"] == {
        "verified": True,
        "covered_motion_count": 3,
        "minimum_environment_count": 2,
        "missing_motion_ids": [],
    }
    assert contract["theoretical_total_transitions_per_rollout"] == 32
    assert contract["theoretical_total_training_transitions"] == 160
    assert [item["environment_count"] for item in contract["per_motion"]] == [3, 3, 2]
    assert [
        item["theoretical_transitions_per_rollout"]
        for item in contract["per_motion"]
    ] == [12, 12, 8]
    assert [
        item["theoretical_training_transitions"] for item in contract["per_motion"]
    ] == [60, 60, 40]
    assert json.loads(json.dumps(contract, sort_keys=True)) == contract


def test_training_assignment_digest_is_deterministic_and_binds_motion_order() -> None:
    kwargs = {
        "num_envs": 4,
        "num_steps_per_env": 2,
        "learning_updates": 3,
    }
    first = training_motion_assignment_contract(
        "balanced-fixed", motion_keys=("a", "b"), **kwargs
    )
    second = training_motion_assignment_contract(
        "balanced-fixed", motion_keys=("a", "b"), **kwargs
    )
    reordered = training_motion_assignment_contract(
        "balanced-fixed", motion_keys=("b", "a"), **kwargs
    )
    assert first == second
    assert first["mapping_digest_sha256"] != reordered["mapping_digest_sha256"]


def test_episode_uniform_training_assignment_records_no_false_finite_witness() -> None:
    assert resolve_training_motion_assignment(
        "episode-uniform", num_envs=2, num_motions=3
    ) is None
    contract = training_motion_assignment_contract(
        "episode-uniform",
        num_envs=2,
        motion_keys=("a", "b", "c"),
        num_steps_per_env=4,
        learning_updates=5,
    )
    assert contract["fixed_environment_assignment"] is False
    assert contract["mapping_length"] is None
    assert contract["mapping_digest_sha256"] is None
    assert contract["maximum_environment_count_difference"] is None
    assert contract["all_motions_have_nonzero_environments"] is None
    assert contract["all_motions_nonzero_witness"]["verified"] is None
    assert contract["transition_exposure_guarantee"] == "none_episode_resampled"
    assert all(item["environment_count"] is None for item in contract["per_motion"])


def test_training_assignment_rejects_unprovable_or_ambiguous_contracts() -> None:
    with pytest.raises(ValueError, match="num_envs >= num_motions"):
        balanced_fixed_motion_ids(num_envs=2, num_motions=3)
    with pytest.raises(ValueError, match="must be one of"):
        resolve_training_motion_assignment("unknown", num_envs=3, num_motions=3)
    with pytest.raises(ValueError, match="positive integer"):
        resolve_training_motion_assignment("episode-uniform", num_envs=True, num_motions=3)
    with pytest.raises(ValueError, match="unique"):
        training_motion_assignment_contract(
            "balanced-fixed",
            num_envs=3,
            motion_keys=("same", "same"),
            num_steps_per_env=1,
            learning_updates=1,
        )
    with pytest.raises(ValueError, match="exactly follow"):
        training_motion_assignment_contract(
            "balanced-fixed",
            num_envs=3,
            motion_keys=("a", "b"),
            num_steps_per_env=1,
            learning_updates=1,
            fixed_motion_ids=(1, 0, 1),
        )
    with pytest.raises(ValueError, match="cannot provide"):
        training_motion_assignment_contract(
            "episode-uniform",
            num_envs=2,
            motion_keys=("a", "b"),
            num_steps_per_env=1,
            learning_updates=1,
            fixed_motion_ids=(0, 1),
        )


def test_root_frame_vector_error_fixes_sign_and_actual_root_frame() -> None:
    reference = torch.tensor([[2.0, 0.0, 0.5], [1.0, 0.0, 0.0]])
    actual = torch.tensor([[1.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    orientation = torch.stack(
        (
            yaw_quat_wxyz(torch.tensor(0.0)),
            yaw_quat_wxyz(torch.tensor(math.pi / 2.0)),
        )
    )
    error = root_frame_vector_error(reference, actual, orientation)
    torch.testing.assert_close(error[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(
        error[1], torch.tensor([0.0, -1.0, 0.0]), atol=1.0e-6, rtol=0.0
    )
    torch.testing.assert_close(
        root_frame_vector_error(actual, actual, orientation), torch.zeros_like(actual)
    )
    with pytest.raises(ValueError, match="shapes must match"):
        root_frame_vector_error(reference[:1], actual, orientation)


def test_target_root_vector_uses_measured_pelvis_frame_and_is_world_yaw_invariant() -> None:
    vector_w = torch.tensor([[1.0, 0.0, 0.5]], dtype=torch.float64)
    actual_quaternion = yaw_quat_wxyz(
        torch.tensor([math.pi / 2.0], dtype=torch.float64)
    )
    target_b = root_frame_vector(vector_w, actual_quaternion)
    torch.testing.assert_close(
        target_b,
        torch.tensor([[0.0, -1.0, 0.5]], dtype=torch.float64),
        atol=1.0e-7,
        rtol=0.0,
    )

    global_yaw = yaw_quat_wxyz(torch.tensor([0.73], dtype=torch.float64))
    rotated_vector = rotate_by_env_yaw(vector_w, global_yaw)
    rotated_actual = align_quaternions_by_env_yaw(actual_quaternion, global_yaw)
    torch.testing.assert_close(
        root_frame_vector(rotated_vector, rotated_actual),
        target_b,
        atol=1.0e-7,
        rtol=0.0,
    )


def test_positive_horizon_root_preview_excludes_h0_and_uses_actual_pelvis_frame() -> None:
    dtype = torch.float64
    actual_position = torch.tensor([[10.0, 20.0, 0.0]], dtype=dtype)
    actual_quaternion = yaw_quat_wxyz(torch.tensor([math.pi / 2.0], dtype=dtype))
    future_position = torch.tensor(
        [[[999.0, 999.0, 999.0], [11.0, 20.0, 0.0], [10.0, 22.0, 0.0], [10.0, 20.0, 3.0]]],
        dtype=dtype,
    )
    future_quaternion = actual_quaternion[:, None, :].expand(-1, 4, -1).clone()

    position, orientation = positive_horizon_root_pose_preview(
        future_position,
        future_quaternion,
        actual_position,
        actual_quaternion,
    )
    assert position.shape == (1, 3, 3)
    assert orientation.shape == (1, 3, 6)
    torch.testing.assert_close(
        position,
        torch.tensor([[[0.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]]], dtype=dtype),
        atol=1.0e-7,
        rtol=0.0,
    )
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=dtype)
    torch.testing.assert_close(
        orientation,
        identity_6d.reshape(1, 1, 6).expand(1, 3, 6),
        atol=1.0e-7,
        rtol=0.0,
    )


def test_positive_horizon_root_preview_is_global_pose_and_quaternion_sign_invariant() -> None:
    dtype = torch.float64
    actual_position = torch.tensor([[0.3, -0.8, 0.9]], dtype=dtype)
    actual_quaternion = yaw_quat_wxyz(torch.tensor([0.4], dtype=dtype))
    future_position = torch.tensor(
        [[[0.3, -0.8, 0.9], [0.5, -0.7, 1.0], [0.8, -0.4, 1.1], [1.1, 0.2, 1.2]]],
        dtype=dtype,
    )
    future_quaternion = yaw_quat_wxyz(
        torch.tensor([[0.4, 0.5, 0.65, 0.9]], dtype=dtype)
    )
    baseline_position, baseline_orientation = positive_horizon_root_pose_preview(
        future_position,
        future_quaternion,
        actual_position,
        actual_quaternion,
    )

    global_yaw = yaw_quat_wxyz(torch.tensor([-1.1], dtype=dtype))
    translation = torch.tensor([[4.0, -7.0, 0.25]], dtype=dtype)
    transformed_position = align_positions_by_env_yaw(
        future_position, global_yaw, translation
    )
    transformed_actual_position = align_positions_by_env_yaw(
        actual_position, global_yaw, translation
    )
    transformed_quaternion = align_quaternions_by_env_yaw(
        future_quaternion, global_yaw
    )
    transformed_actual_quaternion = align_quaternions_by_env_yaw(
        actual_quaternion, global_yaw
    )
    transformed_result = positive_horizon_root_pose_preview(
        transformed_position,
        transformed_quaternion,
        transformed_actual_position,
        transformed_actual_quaternion,
    )
    torch.testing.assert_close(
        transformed_result[0], baseline_position, atol=1.0e-7, rtol=0.0
    )
    torch.testing.assert_close(
        transformed_result[1], baseline_orientation, atol=1.0e-7, rtol=0.0
    )

    for signed_future, signed_actual in (
        (-future_quaternion, actual_quaternion),
        (future_quaternion, -actual_quaternion),
        (-future_quaternion, -actual_quaternion),
    ):
        signed_result = positive_horizon_root_pose_preview(
            future_position,
            signed_future,
            actual_position,
            signed_actual,
        )
        torch.testing.assert_close(signed_result[0], baseline_position)
        torch.testing.assert_close(signed_result[1], baseline_orientation)


def test_root_command_and_preview_helpers_reject_invalid_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="leading dimensions"):
        root_frame_vector(torch.zeros(2, 3), torch.zeros(1, 4))
    with pytest.raises(ValueError, match="NaN or Inf"):
        root_frame_vector(
            torch.tensor([[float("nan"), 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

    actual_position = torch.zeros(2, 3)
    actual_quaternion = torch.zeros(2, 4)
    actual_quaternion[:, 0] = 1.0
    future_position = torch.zeros(2, 4, 3)
    future_quaternion = torch.zeros(2, 4, 4)
    future_quaternion[..., 0] = 1.0
    with pytest.raises(ValueError, match="at least one positive horizon"):
        positive_horizon_root_pose_preview(
            future_position[:, :1],
            future_quaternion[:, :1],
            actual_position,
            actual_quaternion,
        )
    with pytest.raises(ValueError, match="env and horizon dimensions differ"):
        positive_horizon_root_pose_preview(
            future_position,
            future_quaternion[:, :3],
            actual_position,
            actual_quaternion,
        )
    bad_position = future_position.clone()
    bad_position[0, 2, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        positive_horizon_root_pose_preview(
            bad_position,
            future_quaternion,
            actual_position,
            actual_quaternion,
        )
    bad_quaternion = future_quaternion.clone()
    bad_quaternion[0, 2] = 0.0
    with pytest.raises(ValueError, match="degenerate"):
        positive_horizon_root_pose_preview(
            future_position,
            bad_quaternion,
            actual_position,
            actual_quaternion,
        )


def _fake_live_tracking_env() -> object:
    contract = tracking_observation_contract()

    class ObservationManager:
        active_terms = {
            name: [term["name"] for term in group["terms"]]
            for name, group in contract["observation_groups"].items()
        }
        group_obs_term_dim = {
            name: [(term["dimension"],) for term in group["terms"]]
            for name, group in contract["observation_groups"].items()
        }
        group_obs_dim = {
            name: (group["dimension"],)
            for name, group in contract["observation_groups"].items()
        }
        group_obs_concatenate = {"policy": True, "critic": True}

    class ActionManager:
        active_terms = ["joint_residual"]
        action_term_dim = [29]
        total_action_dim = 29
        action_term = type(
            "ReferenceResidualAction",
            (),
            {
                "cfg": type(
                    "ActionCfg", (), {"reference_velocity_feedforward_scale": 0.0}
                )(),
                "velocity_target_buffer_verified": True,
                "velocity_target_buffer_max_abs_error": 0.0,
            },
        )()

        @classmethod
        def get_term(cls, name: str) -> object:
            assert name == "joint_residual"
            return cls.action_term

    class MotionCommand:
        future_horizons_s = torch.tensor([0.0, 0.04, 0.08, 0.16])

    class CommandManager:
        @staticmethod
        def get_term(name: str) -> object:
            assert name == "motion"
            return MotionCommand()

    class Env:
        observation_manager = ObservationManager()
        action_manager = ActionManager()
        command_manager = CommandManager()

    return Env()


def test_live_tracking_tensor_contract_witnesses_exact_manager_axes() -> None:
    witness = resolved_live_tracking_tensor_contract(_fake_live_tracking_env())
    assert witness["policy_contract_version"] == TRACKING_POLICY_CONTRACT_VERSION
    assert witness["future_horizons_s"] == pytest.approx([0.0, 0.04, 0.08, 0.16])
    assert witness["observation_groups"]["policy"]["dimension"] == 293
    assert witness["observation_groups"]["critic"]["dimension"] == 505
    assert witness["action"] == {
        "term_names": ["joint_residual"],
        "term_dimensions": [29],
        "dimension": 29,
        "position_target_formula": (
            "clip_limits(q_ref + 0.25 * rate_limit(clip(action,-1,1), "
            "0.15_per_control_step))"
        ),
        "velocity_target_formula": "zeros_like(reference_joint_velocity)",
        "reference_velocity_feedforward_scale": 0.0,
        "velocity_target_buffer_verified": True,
        "velocity_target_buffer_max_abs_error": 0.0,
    }
    assert json.loads(json.dumps(witness, sort_keys=True)) == witness


@pytest.mark.parametrize(
    "mutation",
    [
        "term_order",
        "term_shape",
        "group_shape",
        "not_concatenated",
        "action_name",
        "action_dimension",
        "velocity_feedforward_scale",
        "horizons",
    ],
)
def test_live_tracking_tensor_contract_fails_closed_on_axis_drift(mutation: str) -> None:
    env = _fake_live_tracking_env()
    if mutation == "term_order":
        env.observation_manager.active_terms["policy"][:2] = reversed(
            env.observation_manager.active_terms["policy"][:2]
        )
    elif mutation == "term_shape":
        env.observation_manager.group_obs_term_dim["policy"][0] = (115,)
    elif mutation == "group_shape":
        env.observation_manager.group_obs_dim["critic"] = (504,)
    elif mutation == "not_concatenated":
        env.observation_manager.group_obs_concatenate["policy"] = False
    elif mutation == "action_name":
        env.action_manager.active_terms = ["wrong_action"]
    elif mutation == "action_dimension":
        env.action_manager.action_term_dim = [28]
        env.action_manager.total_action_dim = 28
    elif mutation == "velocity_feedforward_scale":
        env.action_manager.action_term.cfg.reference_velocity_feedforward_scale = 1.0
    elif mutation == "horizons":
        env.command_manager.get_term = lambda _: type(
            "Motion", (), {"future_horizons_s": (0.0, 0.04, 0.08, 0.20)}
        )()
    with pytest.raises(RuntimeError):
        resolved_live_tracking_tensor_contract(env)


def test_frame_zero_start_mixture_edges_seed_and_ratio() -> None:
    random_times = torch.linspace(0.1, 5.0, 10_000)
    all_random, zero_mask = mix_frame_zero_start_times(random_times, 0.0)
    torch.testing.assert_close(all_random, random_times)
    assert not bool(zero_mask.any())
    all_zero, zero_mask = mix_frame_zero_start_times(random_times, 1.0)
    torch.testing.assert_close(all_zero, torch.zeros_like(random_times))
    assert bool(zero_mask.all())

    generator_a = torch.Generator().manual_seed(42)
    generator_b = torch.Generator().manual_seed(42)
    mixed_a, mask_a = mix_frame_zero_start_times(
        random_times, 0.5, generator=generator_a
    )
    mixed_b, mask_b = mix_frame_zero_start_times(
        random_times, 0.5, generator=generator_b
    )
    torch.testing.assert_close(mixed_a, mixed_b)
    assert torch.equal(mask_a, mask_b)
    assert abs(float(mask_a.float().mean()) - 0.5) < 0.01
    assert bool(torch.all(mixed_a[~mask_a] > 0.0))
    for probability in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError, match="probability"):
            mix_frame_zero_start_times(random_times, probability)


def test_raw_action_excess_penalty_only_sees_values_outside_safe_clip() -> None:
    inside = torch.tensor([[-1.0, -0.25, 0.5, 1.0]])
    torch.testing.assert_close(raw_action_excess_l2(inside), torch.zeros(1))
    raw = torch.tensor([[-2.0, -1.0, 0.5, 1.5]])
    torch.testing.assert_close(raw_action_excess_l2(raw), torch.tensor([0.3125]))
    with pytest.raises(ValueError, match="positive"):
        raw_action_excess_l2(raw, action_clip=0.0)


def test_action_boundary_margin_warns_before_clip_without_hiding_raw_action() -> None:
    raw = torch.tensor([[-0.89, 0.90, 0.95, 1.0, 4.0]], dtype=torch.float64)
    penalty = action_boundary_margin_l2(raw, onset=0.90, action_clip=1.0)
    torch.testing.assert_close(
        penalty,
        torch.tensor([(0.0 + 0.0 + 0.25 + 1.0 + 1.0) / 5.0], dtype=torch.float64),
    )
    assert raw[0, -1].item() == 4.0
    for onset, clip in ((1.0, 1.0), (-0.1, 1.0), (0.9, 0.8)):
        with pytest.raises(ValueError, match="boundary"):
            action_boundary_margin_l2(raw, onset=onset, action_clip=clip)


def test_action_limiter_request_gap_exposes_rejected_policy_request() -> None:
    raw = torch.tensor([[1.2, 0.5, -0.4]], dtype=torch.float64)
    residual = torch.tensor([[0.85, 0.4, -0.4]], dtype=torch.float64)
    penalty = action_limiter_request_gap_l2(raw, residual, action_clip=1.0)
    torch.testing.assert_close(
        penalty,
        torch.tensor([(0.15**2 + 0.10**2) / 3.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="share"):
        action_limiter_request_gap_l2(raw, residual[:, :2])


def test_motion_time_does_not_advance_on_auto_reset_step() -> None:
    motion_time = torch.tensor([0.0, 1.5, 3.0])
    reset_mask = torch.tensor([True, False, True])
    advanced = advance_motion_time_after_step(motion_time, reset_mask, 0.02)
    torch.testing.assert_close(advanced, torch.tensor([0.0, 1.52, 3.0]))
    torch.testing.assert_close(motion_time, torch.tensor([0.0, 1.5, 3.0]))
    with pytest.raises(ValueError, match="boolean"):
        advance_motion_time_after_step(motion_time, reset_mask.float(), 0.02)


def test_deterministic_suite_zero_actions_never_call_a_checkpoint_policy() -> None:
    observations = torch.ones(3, 7, dtype=torch.float64)

    def forbidden_policy(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("zero action source called the checkpoint policy")

    actions = deterministic_suite_actions(
        "zero",
        observations,
        29,
        checkpoint_policy=forbidden_policy,
    )
    assert actions.shape == (3, 29)
    assert actions.dtype == observations.dtype
    assert not bool(actions.any())
    tensor_dict = TensorDict(
        {"policy": torch.ones(3, 7)},
        batch_size=[3],
    )
    tensor_dict_actions = deterministic_suite_actions(
        "zero",
        tensor_dict,
        29,
        checkpoint_policy=forbidden_policy,
    )
    assert tensor_dict_actions.shape == (3, 29)
    assert tensor_dict_actions.dtype == torch.float32
    assert not bool(tensor_dict_actions.any())


def test_deterministic_suite_checkpoint_actions_require_and_call_policy() -> None:
    observations = torch.ones(2, 5)
    with pytest.raises(ValueError, match="callable policy"):
        deterministic_suite_actions("checkpoint", observations, 29)
    actions = deterministic_suite_actions(
        "checkpoint",
        observations,
        3,
        checkpoint_policy=lambda value: value[:, :3] * 2.0,
    )
    torch.testing.assert_close(actions, torch.full((2, 3), 2.0))
    with pytest.raises(ValueError, match="checkpoint.*zero"):
        deterministic_suite_actions("invalid", observations, 3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"fixed_motion_id": 0, "fixed_motion_ids": [0, 1]},
            "mutually exclusive",
        ),
        ({"fixed_motion_ids": [0]}, "one index per environment"),
        ({"fixed_motion_ids": [0, 1, 5]}, "outside the loaded range"),
        ({"fixed_motion_ids": [0, True, 2]}, "not an integer"),
    ],
)
def test_fixed_motion_ids_reject_ambiguous_or_invalid_mappings(
    kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fixed_motion_ids_for_envs(num_envs=3, num_motions=5, **kwargs)


def test_suite_success_requires_clean_motion_end_without_hard_failure() -> None:
    timeout_names = ("time_out", "motion_end")
    assert clean_motion_end_success(
        {"time_out": False, "motion_end": True, "fall": False},
        timeout_term_names=timeout_names,
    )
    assert clean_motion_end_success(
        {"time_out": True, "motion_end": True, "fall": False},
        timeout_term_names=timeout_names,
    )
    assert not clean_motion_end_success(
        {"time_out": False, "motion_end": True, "fall": True},
        timeout_term_names=timeout_names,
    )
    assert not clean_motion_end_success(
        {"time_out": True, "motion_end": False, "fall": False},
        timeout_term_names=timeout_names,
    )
    with pytest.raises(ValueError, match="classified as a timeout"):
        clean_motion_end_success(
            {"motion_end": True}, timeout_term_names=("time_out",)
        )


def test_xy_yaw_alignment_hits_target_and_preserves_reference_z() -> None:
    reference_position = torch.tensor(
        [[1.0, 2.0, 0.83], [-0.5, 0.25, 1.10]], dtype=torch.float64
    )
    reference_yaw = torch.tensor([0.35, -1.20], dtype=torch.float64)
    reference_quaternion = yaw_quat_wxyz(reference_yaw)
    target_xy = torch.tensor([[10.0, -2.0], [-3.0, 4.0]], dtype=torch.float64)
    target_yaw = torch.tensor([1.10, 0.45], dtype=torch.float64)

    yaw, translation = alignment_from_reference_anchor(
        reference_position, reference_quaternion, target_xy, target_yaw
    )
    aligned_position = align_positions_by_env_yaw(reference_position, yaw, translation)
    aligned_quaternion = align_quaternions_by_env_yaw(reference_quaternion, yaw)

    torch.testing.assert_close(aligned_position[:, :2], target_xy)
    torch.testing.assert_close(aligned_position[:, 2], reference_position[:, 2])
    torch.testing.assert_close(quat_yaw_wxyz(aligned_quaternion), target_yaw)
    assert float(torch.max(quat_error_angle_wxyz(aligned_quaternion, yaw_quat_wxyz(target_yaw)))) < 1.0e-7


def test_alignment_broadcasts_over_future_horizon_and_body_axes() -> None:
    yaw = yaw_quat_wxyz(torch.tensor([math.pi / 2.0, -math.pi / 2.0]))
    translation = torch.tensor([[1.0, 2.0, 0.0], [-1.0, 0.5, 0.0]])
    positions = torch.zeros(2, 3, 4, 3)
    positions[..., 0] = 1.0
    vectors = positions.clone()
    quaternions = torch.zeros(2, 3, 4, 4)
    quaternions[..., 0] = 1.0

    rotated = rotate_by_env_yaw(vectors, yaw)
    aligned = align_positions_by_env_yaw(positions, yaw, translation)
    aligned_quaternion = align_quaternions_by_env_yaw(quaternions, yaw)

    assert rotated.shape == (2, 3, 4, 3)
    assert aligned.shape == (2, 3, 4, 3)
    assert aligned_quaternion.shape == (2, 3, 4, 4)
    torch.testing.assert_close(rotated[0, :, :, 1], torch.ones(3, 4), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(rotated[1, :, :, 1], -torch.ones(3, 4), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(aligned[0, 0, 0], torch.tensor([1.0, 3.0, 0.0]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(
        quat_apply_wxyz(aligned_quaternion[0, 0, 0], torch.tensor([1.0, 0.0, 0.0])),
        torch.tensor([0.0, 1.0, 0.0]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_reference_residual_contract_clips_rate_limits_and_soft_clamps() -> None:
    raw = torch.tensor([[2.0, -2.0, 0.40, 0.10]])
    reference = torch.tensor([[0.0, 0.0, 0.19, -0.19]])
    previous = torch.zeros_like(raw)
    hard = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0], [-0.3, 0.3], [-0.3, 0.3]]])
    soft = torch.tensor([[[-0.8, 0.8], [-0.8, 0.8], [-0.2, 0.2], [-0.2, 0.2]]])

    result = reference_residual_target(
        raw,
        reference,
        previous,
        residual_scale=0.25,
        action_clip=1.0,
        normalized_delta_limit=0.15,
        hard_limits=hard,
        soft_limits=soft,
        target_limit_mode="soft",
    )

    torch.testing.assert_close(result.clipped_action, torch.tensor([[1.0, -1.0, 0.40, 0.10]]))
    torch.testing.assert_close(result.normalized_residual, torch.tensor([[0.15, -0.15, 0.15, 0.10]]))
    torch.testing.assert_close(result.target, torch.tensor([[0.0375, -0.0375, 0.2, -0.165]]))
    assert result.rate_limited.tolist() == [[True, True, True, False]]
    assert result.target_clamped.tolist() == [[False, False, True, False]]


def test_reference_residual_hard_mode_and_optional_rate_limit() -> None:
    raw = torch.tensor([[0.8, -0.8]])
    reference = torch.tensor([[0.9, -0.9]])
    previous = torch.tensor([[-0.5, 0.5]])
    hard = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
    soft = torch.tensor([[[-0.7, 0.7], [-0.7, 0.7]]])
    result = reference_residual_target(
        raw,
        reference,
        previous,
        normalized_delta_limit=None,
        hard_limits=hard,
        soft_limits=soft,
        target_limit_mode="hard",
    )
    torch.testing.assert_close(result.normalized_residual, raw)
    torch.testing.assert_close(result.target, torch.tensor([[1.0, -1.0]]))
    assert not bool(result.rate_limited.any())
    assert bool(result.target_clamped.all())


def test_reference_velocity_target_scales_and_clamps_symmetric_limits() -> None:
    reference = torch.tensor([[2.0, -4.0, 0.5], [-1.0, 3.0, -0.25]])
    limits = torch.tensor([[1.5, 2.0, 1.0], [2.0, 1.0, 0.2]])
    result = reference_velocity_target(
        reference,
        scale=0.5,
        velocity_limits=limits,
    )
    torch.testing.assert_close(result.unclamped_target, reference * 0.5)
    torch.testing.assert_close(
        result.target,
        torch.tensor([[1.0, -2.0, 0.25], [-0.5, 1.0, -0.125]]),
    )
    assert result.target_clamped.tolist() == [
        [False, False, False],
        [False, True, False],
    ]


@pytest.mark.parametrize(
    ("reference", "scale", "limits", "message"),
    [
        (torch.tensor([[float("nan")]]), 1.0, torch.ones(1, 1), "finite tensor"),
        (torch.zeros(1, 1), -0.1, torch.ones(1, 1), "non-negative"),
        (torch.zeros(1, 1), True, torch.ones(1, 1), "non-negative"),
        (torch.zeros(1, 2), 1.0, torch.ones(1, 1), "must match"),
        (torch.zeros(1, 1), 1.0, torch.zeros(1, 1), "strictly positive"),
    ],
)
def test_reference_velocity_target_rejects_unsafe_contracts(
    reference: torch.Tensor,
    scale: float,
    limits: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reference_velocity_target(reference, scale=scale, velocity_limits=limits)


def test_reference_aware_fall_allows_valid_floor_motion_and_rejects_deviation() -> None:
    half = math.sqrt(0.5)
    floor_quat = torch.tensor([[half, half, 0.0, 0.0]], dtype=torch.float64)
    floor_pos = torch.tensor([[0.0, 0.0, 0.22]], dtype=torch.float64)
    high_pelvis_force = torch.tensor([120.0], dtype=torch.float64)
    params = {
        "maximum_root_height_deficit": 0.25,
        "maximum_orientation_error": 1.0,
        "pelvis_contact_threshold": 50.0,
        "pelvis_contact_reference_height": 0.50,
        "maximum_pelvis_contact_reference_tilt": 0.80,
    }

    perfect_floor = reference_aware_fall_mask(
        floor_pos, floor_pos, floor_quat, floor_quat, high_pelvis_force, **params
    )
    assert not bool(perfect_floor.item())

    wrong_orientation = reference_aware_fall_mask(
        floor_pos,
        floor_pos,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        floor_quat,
        torch.zeros(1, dtype=torch.float64),
        **params,
    )
    assert bool(wrong_orientation.item())


def test_reference_aware_fall_rejects_upright_height_loss_and_pelvis_impact() -> None:
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    reference = torch.tensor([[0.0, 0.0, 0.80]], dtype=torch.float64)
    actual_low = torch.tensor([[0.0, 0.0, 0.40]], dtype=torch.float64)
    params = {
        "maximum_root_height_deficit": 0.25,
        "maximum_orientation_error": 1.0,
        "pelvis_contact_threshold": 50.0,
        "pelvis_contact_reference_height": 0.50,
        "maximum_pelvis_contact_reference_tilt": 0.80,
    }

    lost_height = reference_aware_fall_mask(
        actual_low, reference, quat, quat, torch.zeros(1, dtype=torch.float64), **params
    )
    pelvis_impact = reference_aware_fall_mask(
        reference,
        reference,
        quat,
        quat,
        torch.tensor([80.0], dtype=torch.float64),
        **params,
    )
    assert bool(lost_height.item())
    assert bool(pelvis_impact.item())

    # A high but intentionally horizontal/inverted reference is not an
    # upright pelvis-impact case; orientation tracking still governs it.
    half = math.sqrt(0.5)
    horizontal = torch.tensor([[half, half, 0.0, 0.0]], dtype=torch.float64)
    intentional_acrobatics = reference_aware_fall_mask(
        reference,
        reference,
        horizontal,
        horizontal,
        torch.tensor([80.0], dtype=torch.float64),
        **params,
    )
    assert not bool(intentional_acrobatics.item())


def test_reference_aware_fall_components_identify_each_exact_subcause() -> None:
    half = math.sqrt(0.5)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    actual_quaternion = torch.stack(
        (
            identity,
            torch.tensor([half, half, 0.0, 0.0], dtype=torch.float64),
            identity,
        )
    )
    reference_quaternion = identity.repeat(3, 1)
    reference_position = torch.tensor(
        [[0.0, 0.0, 0.8], [0.0, 0.0, 0.8], [0.0, 0.0, 0.8]],
        dtype=torch.float64,
    )
    actual_position = reference_position.clone()
    actual_position[0, 2] = 0.4
    pelvis_force = torch.tensor([0.0, 0.0, 80.0], dtype=torch.float64)
    params = {
        "maximum_root_height_deficit": 0.25,
        "maximum_orientation_error": 1.0,
        "pelvis_contact_threshold": 50.0,
        "pelvis_contact_reference_height": 0.50,
        "maximum_pelvis_contact_reference_tilt": 0.80,
    }

    components = reference_aware_fall_components(
        actual_position,
        reference_position,
        actual_quaternion,
        reference_quaternion,
        pelvis_force,
        **params,
    )

    assert components.height_deficit_exceeded.tolist() == [True, False, False]
    assert components.orientation_error_exceeded.tolist() == [False, True, False]
    assert components.pelvis_contact_applicable.tolist() == [True, True, True]
    assert components.pelvis_force_exceeded.tolist() == [False, False, True]
    assert components.unexpected_pelvis_contact.tolist() == [False, False, True]
    assert components.fall.tolist() == [True, True, True]
    torch.testing.assert_close(
        reference_aware_fall_mask(
            actual_position,
            reference_position,
            actual_quaternion,
            reference_quaternion,
            pelvis_force,
            **params,
        ),
        components.fall,
    )


def test_tracking_divergence_components_preserve_strict_thresholds() -> None:
    actual_root = torch.zeros(4, 3, dtype=torch.float64)
    actual_root[0, 0] = 0.81
    actual_root[3, 0] = 0.80
    reference_root = torch.zeros_like(actual_root)
    actual_joint = torch.zeros(4, 2, dtype=torch.float64)
    actual_joint[1] = 1.10
    actual_joint[3] = 1.00
    reference_joint = torch.zeros_like(actual_joint)
    actual_body = torch.zeros(4, 2, 3, dtype=torch.float64)
    actual_body[2, :, 0] = 0.51
    actual_body[3, :, 0] = 0.50
    reference_body = torch.zeros_like(actual_body)

    components = tracking_divergence_components(
        actual_root,
        reference_root,
        actual_joint,
        reference_joint,
        actual_body,
        reference_body,
        maximum_root_distance=0.80,
        maximum_mean_joint_error=1.0,
        maximum_mean_body_distance=0.50,
    )

    assert components.root_distance_exceeded.tolist() == [True, False, False, False]
    assert components.mean_joint_error_exceeded.tolist() == [False, True, False, False]
    assert components.mean_body_distance_exceeded.tolist() == [False, False, True, False]
    assert components.diverged.tolist() == [True, True, True, False]


def test_tracking_divergence_risk_uses_exact_thresholds_and_max_aggregation() -> None:
    actual_root = torch.zeros(5, 3, dtype=torch.float64)
    actual_root[:, 0] = torch.tensor([0.47, 0.48, 0.64, 0.80, 0.81])
    reference_root = torch.zeros_like(actual_root)
    actual_joint = torch.zeros(5, 2, dtype=torch.float64)
    reference_joint = torch.zeros_like(actual_joint)
    actual_body = torch.zeros(5, 2, 3, dtype=torch.float64)
    reference_body = torch.zeros_like(actual_body)
    components = tracking_divergence_components(
        actual_root,
        reference_root,
        actual_joint,
        reference_joint,
        actual_body,
        reference_body,
        maximum_root_distance=0.80,
        maximum_mean_joint_error=1.0,
        maximum_mean_body_distance=0.50,
    )
    risk = tracking_divergence_risk_barrier(components)
    torch.testing.assert_close(
        risk,
        torch.tensor([0.0, 0.0, 0.25, 1.0, 1.0], dtype=torch.float64),
    )

    # A body component can dominate without summing with the root component.
    actual_body[2, :, 0] = 0.50
    components = tracking_divergence_components(
        actual_root,
        reference_root,
        actual_joint,
        reference_joint,
        actual_body,
        reference_body,
        maximum_root_distance=0.80,
        maximum_mean_joint_error=1.0,
        maximum_mean_body_distance=0.50,
    )
    risk = tracking_divergence_risk_barrier(components)
    assert risk[2].item() == 1.0
    with pytest.raises(ValueError, match="onset_fraction"):
        tracking_divergence_risk_barrier(components, onset_fraction=1.0)


def test_torque_effort_ratios_expose_requested_but_not_applied_saturation() -> None:
    computed = torch.tensor([[15.0, -20.0, 0.0]], dtype=torch.float64)
    applied = torch.tensor([[10.0, -19.0, 5.0]], dtype=torch.float64)
    limits = torch.tensor([10.0, 20.0, 5.0], dtype=torch.float64)

    components = torque_effort_ratio_components(computed, applied, limits)

    torch.testing.assert_close(
        components.computed_abs_ratio,
        torch.tensor([[1.5, 1.0, 0.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        components.applied_abs_ratio,
        torch.tensor([[1.0, 0.95, 1.0]], dtype=torch.float64),
    )
    assert components.computed_limit_exceeded.tolist() == [[True, False, False]]
    assert components.applied_limit_exceeded.tolist() == [[False, False, False]]
    with pytest.raises(ValueError, match="positive"):
        torque_effort_ratio_components(computed, applied, torch.tensor([10.0, 0.0, 5.0]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_limit_mode": "soft", "soft_limits": None}, "requires soft_limits"),
        ({"target_limit_mode": "invalid"}, "must be 'soft' or 'hard'"),
        ({"normalized_delta_limit": 0.0}, "positive or None"),
    ],
)
def test_reference_residual_rejects_ambiguous_contracts(kwargs: dict, message: str) -> None:
    action = torch.zeros(1, 2)
    limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
    defaults = {"hard_limits": limits, "soft_limits": limits}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        reference_residual_target(action, action, action, **defaults)


@pytest.mark.skipif(not B3_REFERENCE.is_file(), reason="B3 Gate-3 reference has not been generated")
def test_actual_b3_reference_satisfies_tracking_action_and_alignment_contract() -> None:
    library = MotionLibrary([B3_REFERENCE], device="cpu")
    assert library.fps == 50.0
    assert library.contract.joint_names == G1_POLICY_JOINT_NAMES
    assert len(library.contract.tracking_body_names) == 14
    assert library.contract.tracking_action_mode == "reference_residual"
    assert library.contract.tracking_residual_scale == pytest.approx(0.25)
    assert library.contract.tracking_action_clip == pytest.approx(1.0)
    assert library.contract.tracking_normalized_delta_limit == pytest.approx(0.15)

    query = library.query(torch.tensor([0]), torch.tensor([0.0]))
    hard_single = torch.tensor(
        list(zip(G1_CONTRACT.lower, G1_CONTRACT.upper, strict=True)), dtype=torch.float32
    )
    hard = hard_single.unsqueeze(0)
    center = 0.5 * (hard[..., 0] + hard[..., 1])
    half_range = 0.45 * (hard[..., 1] - hard[..., 0])
    soft = torch.stack((center - half_range, center + half_range), dim=-1)
    zero = torch.zeros_like(query.joint_pos)
    result = reference_residual_target(
        zero,
        query.joint_pos,
        zero,
        hard_limits=hard,
        soft_limits=soft,
        target_limit_mode="soft",
    )
    torch.testing.assert_close(result.target, query.joint_pos)
    assert not bool(result.target_clamped.any())

    target_xy = torch.tensor([[4.0, -3.0]])
    target_yaw = torch.tensor([0.75])
    yaw, translation = alignment_from_reference_anchor(
        query.root_pos_w, query.root_quat_wxyz, target_xy, target_yaw
    )
    aligned = align_positions_by_env_yaw(query.root_pos_w, yaw, translation)
    torch.testing.assert_close(aligned[:, :2], target_xy)
    torch.testing.assert_close(aligned[:, 2], query.root_pos_w[:, 2])


def test_motion_path_confinement_and_cpu_dependency_message(tmp_path: Path) -> None:
    # ``--basetemp`` intentionally lives inside ACCAD, so tmp_path itself is
    # not an out-of-tree confinement test.  The guard checks confinement before
    # existence; no file needs to be created outside the project-owned tree.
    outside = Path(ACCAD_ROOT).resolve().parent / "outside_tracking_reference.npz"
    with pytest.raises(ValueError, match="must remain below"):
        _require_path_below_accad(outside)
    assert Path(ACCAD_ROOT).is_dir()

    if not isaac_tracking_available():
        with pytest.raises(RuntimeError, match="AppLauncher"):
            require_isaac_tracking()
        with pytest.raises(RuntimeError, match="AppLauncher"):
            make_tracking_env_cfg([B3_REFERENCE])
