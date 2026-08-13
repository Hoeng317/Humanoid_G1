from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from accad_g1.physics_warmup import (
    DEFAULT_PHYSICS_WARMUP_STEPS,
    PHYSICS_WARMUP_SCHEMA_VERSION,
    observation_finiteness_witness,
    post_reset_state_witness,
    run_physics_cold_start_warmup,
    validate_physics_warmup_steps,
)


def test_warmup_step_validation_and_disabled_contract() -> None:
    assert DEFAULT_PHYSICS_WARMUP_STEPS == 100
    assert validate_physics_warmup_steps(0) == 0
    assert validate_physics_warmup_steps(17) == 17
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="non-negative integer"):
            validate_physics_warmup_steps(value)  # type: ignore[arg-type]

    witness = run_physics_cold_start_warmup(object(), 0)
    assert witness == {
        "schema_version": PHYSICS_WARMUP_SCHEMA_VERSION,
        "enabled": False,
        "requested_control_steps": 0,
        "executed_control_steps": 0,
        "policy_inference_used": False,
        "post_reset": None,
        "passed": True,
    }


def test_nested_observation_witness_is_finite_and_json_safe() -> None:
    witness = observation_finiteness_witness(
        {
            "policy": torch.zeros(2, 293),
            "critic": [torch.ones(2, 505)],
        }
    )
    assert witness["tensor_count"] == 2
    assert witness["element_count"] == 2 * (293 + 505)
    assert witness["nonfinite_count"] == 0
    assert witness["all_finite"] is True
    assert json.loads(json.dumps(witness, sort_keys=True)) == witness


def test_observation_witness_rejects_invalid_leaves_and_nonfinite_values() -> None:
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        observation_finiteness_witness({"policy": torch.tensor([[float("nan")]])})
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        observation_finiteness_witness({"policy": "not-a-tensor"})
    with pytest.raises(ValueError, match="no tensors"):
        observation_finiteness_witness({})


def _valid_post_reset_witness() -> dict:
    num_envs = 3
    action_dim = 29
    return post_reset_state_witness(
        observations={"policy": torch.zeros(num_envs, 293)},
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        motion_time=torch.zeros(num_envs),
        raw_action=torch.zeros(num_envs, action_dim),
        normalized_residual=torch.zeros(num_envs, action_dim),
        previous_normalized_residual=torch.zeros(num_envs, action_dim),
        num_envs=num_envs,
        action_dim=action_dim,
        reset_buffers={
            "reset_buf": torch.zeros(num_envs, dtype=torch.bool),
            "reset_terminated": torch.zeros(num_envs, dtype=torch.bool),
            "reset_time_outs": torch.zeros(num_envs, dtype=torch.bool),
        },
        state_errors={
            "joint_position_rad": 0.0,
            "root_position_m": 0.0,
            "root_orientation_quaternion_l2": 0.0,
        },
    )


def test_post_reset_witness_enforces_frame_zero_episode_and_action_state() -> None:
    witness = _valid_post_reset_witness()
    assert witness["authoritative_motion_frame_zero"] is True
    assert witness["episode_counters_zero"] is True
    assert witness["action_state_zero"] is True
    assert witness["passed"] is True
    assert json.loads(json.dumps(witness, sort_keys=True)) == witness


def test_post_reset_witness_fails_closed_on_nonzero_or_wrong_action_shape() -> None:
    with pytest.raises(RuntimeError, match="validation failed"):
        post_reset_state_witness(
            observations={"policy": torch.zeros(1, 293)},
            episode_length_buf=torch.ones(1, dtype=torch.long),
            motion_time=torch.zeros(1),
            raw_action=torch.zeros(1, 29),
            normalized_residual=torch.zeros(1, 29),
            previous_normalized_residual=torch.zeros(1, 29),
            num_envs=1,
            action_dim=29,
        )
    with pytest.raises(RuntimeError, match="shape"):
        post_reset_state_witness(
            observations={"policy": torch.zeros(1, 293)},
            episode_length_buf=torch.zeros(1, dtype=torch.long),
            motion_time=torch.zeros(1),
            raw_action=torch.zeros(1, 28),
            normalized_residual=torch.zeros(1, 29),
            previous_normalized_residual=torch.zeros(1, 29),
            num_envs=1,
            action_dim=29,
        )
