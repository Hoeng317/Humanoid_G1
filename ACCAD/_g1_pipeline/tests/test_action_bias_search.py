from __future__ import annotations

from pathlib import Path
import math
import sys

import pytest
import torch


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from accad_g1.action_bias_search import (
    bias_for_control_step,
    bias_statistics,
    candidate_ranking_key,
    checkpoint_patch_allowed,
    generate_action_bias_candidates,
    generate_temporal_action_bias_schedule,
    locate_actor_output_bias,
    select_best_candidate,
    temporal_bias_statistics,
    tensor_sha256,
    validate_bias_mode,
    validate_target_yaw,
    write_actor_bias_checkpoint,
    write_actor_output_scale_checkpoint,
)
from accad_g1.training_io import sha256_file


def _record(
    candidate_id: int,
    *,
    success: bool,
    completion: float,
    episode_return: float,
    root_rmse: float = 0.2,
    joint_rmse: float = 0.1,
    body_rmse: float = 0.3,
    bias_l2: float = 0.0,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "first_episode": {
            "success": success,
            "completion_ratio_length_over_archive_frames": completion,
            "return": episode_return,
        },
        "metrics": {
            "root_position_rmse_m": root_rmse,
            "joint_position_rmse_rad": joint_rmse,
            "tracking_body_position_rmse_m": body_rmse,
        },
        "bias": {"l2_norm": bias_l2},
    }


def test_seeded_candidates_keep_zero_and_antithetic_pairs():
    first = generate_action_bias_candidates(
        num_candidates=6,
        action_dim=3,
        std=0.005,
        seed=42,
        antithetic=True,
    )
    second = generate_action_bias_candidates(
        num_candidates=6,
        action_dim=3,
        std=0.005,
        seed=42,
        antithetic=True,
    )

    assert first.shape == (6, 3)
    assert first.device.type == "cpu"
    assert first.dtype == torch.float32
    assert torch.equal(first[0], torch.zeros(3))
    assert torch.equal(first[2], -first[1])
    assert torch.equal(first[4], -first[3])
    assert torch.equal(first, second)
    assert not torch.equal(first[5], first[1])


def test_single_candidate_is_always_the_zero_baseline_in_both_modes():
    constant = generate_action_bias_candidates(
        num_candidates=1,
        action_dim=29,
        std=0.005,
        seed=42,
        antithetic=True,
    )
    temporal = generate_temporal_action_bias_schedule(
        num_steps=151,
        num_candidates=1,
        action_dim=29,
        std=0.005,
        seed=42,
        antithetic=True,
    )
    assert constant.shape == (1, 29)
    assert temporal.shape == (151, 1, 29)
    assert torch.count_nonzero(constant).item() == 0
    assert torch.count_nonzero(temporal).item() == 0


def test_target_yaw_accepts_closed_principal_range_and_rejects_nonfinite():
    assert validate_target_yaw(0.0) == 0.0
    assert validate_target_yaw(0.1) == pytest.approx(0.1)
    assert validate_target_yaw(-math.pi) == -math.pi
    assert validate_target_yaw(math.pi) == math.pi
    for value in (math.nextafter(math.pi, math.inf), -math.pi - 1.0e-6, math.inf, math.nan):
        with pytest.raises(ValueError, match=r"\[-pi, pi\]"):
            validate_target_yaw(value)


def test_candidate_generation_validation_and_non_antithetic_sampling():
    values = generate_action_bias_candidates(
        num_candidates=3,
        action_dim=2,
        std=0.1,
        seed=7,
        antithetic=False,
    )
    assert torch.equal(values[0], torch.zeros(2))
    assert not torch.equal(values[2], -values[1])
    with pytest.raises(ValueError, match="num_candidates"):
        generate_action_bias_candidates(
            num_candidates=0, action_dim=2, std=0.1, seed=7
        )
    with pytest.raises(ValueError, match="std"):
        generate_action_bias_candidates(
            num_candidates=2, action_dim=2, std=0.0, seed=7
        )


def test_temporal_schedule_is_seeded_frame_varying_and_full_sequence_antithetic():
    first = generate_temporal_action_bias_schedule(
        num_steps=7,
        num_candidates=6,
        action_dim=3,
        std=0.02,
        seed=43,
        antithetic=True,
    )
    second = generate_temporal_action_bias_schedule(
        num_steps=7,
        num_candidates=6,
        action_dim=3,
        std=0.02,
        seed=43,
        antithetic=True,
    )

    assert first.shape == (7, 6, 3)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert torch.count_nonzero(first[:, 0, :]).item() == 0
    assert torch.equal(first[:, 2, :], -first[:, 1, :])
    assert torch.equal(first[:, 4, :], -first[:, 3, :])
    assert torch.equal(first, second)
    assert not torch.equal(first[0, 1], first[1, 1])
    assert tensor_sha256(first) == tensor_sha256(second)
    assert not torch.equal(first[:, 5, :], first[:, 1, :])


def test_bias_mode_step_resolution_and_checkpoint_patch_contract():
    constant = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    temporal = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    assert torch.equal(
        bias_for_control_step(
            constant,
            bias_mode="constant",
            step=99,
            num_candidates=2,
            action_dim=3,
        ),
        constant,
    )
    assert torch.equal(
        bias_for_control_step(
            temporal,
            bias_mode="temporal_gaussian",
            step=2,
            num_candidates=2,
            action_dim=3,
        ),
        temporal[2],
    )
    assert checkpoint_patch_allowed("constant") is True
    assert checkpoint_patch_allowed("temporal_gaussian") is False
    assert validate_bias_mode("temporal_gaussian") == "temporal_gaussian"
    with pytest.raises(ValueError, match="bias_mode"):
        validate_bias_mode("unknown")
    with pytest.raises(IndexError, match="requested"):
        bias_for_control_step(
            temporal,
            bias_mode="temporal_gaussian",
            step=4,
            num_candidates=2,
            action_dim=3,
        )


def test_temporal_bias_statistics_are_complete_and_finite():
    trajectory = torch.tensor([[0.0, 0.1], [-0.1, 0.2]], dtype=torch.float32)
    stats = temporal_bias_statistics(trajectory)
    assert stats["num_steps"] == 2
    assert stats["dimension"] == 2
    assert stats["max_absolute"] == pytest.approx(0.2)
    assert stats["l2_norm"] == pytest.approx(float(torch.linalg.vector_norm(trajectory)))
    assert len(stats["checksum_sha256"]) == 64
    with pytest.raises(ValueError, match="rank-two"):
        temporal_bias_statistics(torch.zeros(2))


def test_candidate_ranking_requires_success_then_completion_and_stable_ties():
    long_failure = _record(1, success=False, completion=1.0, episode_return=100.0)
    short_success = _record(2, success=True, completion=0.9, episode_return=-100.0)
    assert candidate_ranking_key(short_success) > candidate_ranking_key(long_failure)

    lower_return = _record(3, success=True, completion=1.0, episode_return=1.0)
    higher_return = _record(4, success=True, completion=1.0, episode_return=2.0)
    assert select_best_candidate([lower_return, higher_return]) is higher_return

    zero = _record(0, success=True, completion=1.0, episode_return=2.0, bias_l2=0.0)
    perturbation = _record(5, success=True, completion=1.0, episode_return=2.0, bias_l2=0.1)
    assert select_best_candidate([perturbation, zero]) is zero
    with pytest.raises(ValueError, match="duplicate"):
        select_best_candidate([zero, dict(zero)])


def test_bias_statistics_reject_nonfinite_values():
    stats = bias_statistics(torch.tensor([0.0, -0.25, 0.5]))
    assert stats["dimension"] == 3
    assert stats["max_absolute"] == pytest.approx(0.5)
    assert len(stats["checksum_sha256"]) == 64
    with pytest.raises(ValueError, match="NaN or Inf"):
        bias_statistics(torch.tensor([float("nan")]))


def test_actor_output_bias_resolution_is_topology_and_shape_checked():
    state = {
        "actor.0.weight": torch.zeros(4, 5),
        "actor.0.bias": torch.zeros(4),
        "actor.2.weight": torch.zeros(3, 4),
        "actor.2.bias": torch.zeros(3),
        "critic.0.weight": torch.zeros(1, 5),
        "critic.0.bias": torch.zeros(1),
    }
    spec = locate_actor_output_bias(state, action_dim=3)
    assert spec.bias_key == "actor.2.bias"
    assert spec.weight_key == "actor.2.weight"
    assert spec.input_dim == 4

    hidden_matches_action_but_output_does_not = dict(state)
    hidden_matches_action_but_output_does_not["actor.4.weight"] = torch.zeros(2, 3)
    hidden_matches_action_but_output_does_not["actor.4.bias"] = torch.zeros(2)
    with pytest.raises(ValueError, match="output dimension mismatch"):
        locate_actor_output_bias(hidden_matches_action_but_output_does_not, action_dim=3)

    ambiguous = dict(state)
    ambiguous["target.actor.0.weight"] = torch.zeros(3, 4)
    ambiguous["target.actor.0.bias"] = torch.zeros(3)
    with pytest.raises(ValueError, match="ambiguous actor prefixes"):
        locate_actor_output_bias(ambiguous, action_dim=3)


def test_checkpoint_patch_preserves_source_and_only_changes_final_actor_bias(
    tmp_path: Path,
):
    source = tmp_path / "model_10.pt"
    destination = tmp_path / "derived" / "best_biased_model.pt"
    original_state = {
        "actor.0.weight": torch.arange(20, dtype=torch.float32).reshape(4, 5),
        "actor.0.bias": torch.arange(4, dtype=torch.float32),
        "actor.2.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "actor.2.bias": torch.tensor([0.25, -0.5, 0.75]),
        "critic.0.weight": torch.ones(1, 5),
        "critic.0.bias": torch.zeros(1),
    }
    torch.save(
        {
            "model_state_dict": original_state,
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "iter": 10,
            "infos": None,
        },
        source,
    )
    source_checksum = sha256_file(source)
    delta = torch.tensor([0.01, -0.02, 0.03])

    provenance = write_actor_bias_checkpoint(
        source=source,
        destination=destination,
        action_bias=delta,
        action_dim=3,
        allowed_output_root=tmp_path,
    )

    assert sha256_file(source) == source_checksum
    assert provenance["source"]["preserved"] is True
    assert provenance["actor_output_layer"]["bias_key"] == "actor.2.bias"
    assert provenance["verification"]["non_target_model_values_unchanged"] is True
    patched = torch.load(destination, map_location="cpu", weights_only=True)
    assert torch.equal(
        patched["model_state_dict"]["actor.2.bias"],
        original_state["actor.2.bias"] + delta,
    )
    for key in original_state:
        if key != "actor.2.bias":
            assert torch.equal(patched["model_state_dict"][key], original_state[key])
    assert patched["iter"] == 10
    with pytest.raises(FileExistsError):
        write_actor_bias_checkpoint(
            source=source,
            destination=destination,
            action_bias=delta,
            action_dim=3,
            allowed_output_root=tmp_path,
        )


def test_checkpoint_patch_rejects_destination_escape(tmp_path: Path):
    source = tmp_path / "model_0.pt"
    torch.save(
        {
            "model_state_dict": {
                "actor.0.weight": torch.zeros(2, 3),
                "actor.0.bias": torch.zeros(2),
            }
        },
        source,
    )
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    with pytest.raises(ValueError, match="destination must remain below"):
        write_actor_bias_checkpoint(
            source=source,
            destination=tmp_path / "outside.pt",
            action_bias=torch.zeros(2),
            action_dim=2,
            allowed_output_root=allowed,
        )


def test_actor_output_scale_checkpoint_changes_only_final_affine_layer(
    tmp_path: Path,
):
    source = tmp_path / "model_20.pt"
    destination = tmp_path / "scaled" / "model_scale_075.pt"
    original_state = {
        "actor.0.weight": torch.arange(20, dtype=torch.float32).reshape(4, 5),
        "actor.0.bias": torch.arange(4, dtype=torch.float32),
        "actor.2.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "actor.2.bias": torch.tensor([0.25, -0.5, 0.75]),
        "critic.0.weight": torch.ones(1, 5),
        "critic.0.bias": torch.zeros(1),
    }
    torch.save(
        {"model_state_dict": original_state, "iter": 20, "infos": None}, source
    )
    source_checksum = sha256_file(source)

    provenance = write_actor_output_scale_checkpoint(
        source=source,
        destination=destination,
        output_scale=0.75,
        action_dim=3,
        allowed_output_root=tmp_path,
    )

    assert sha256_file(source) == source_checksum
    assert provenance["output_scale"] == 0.75
    assert provenance["source"]["preserved"] is True
    assert provenance["verification"][
        "only_final_actor_weight_and_bias_changed"
    ] is True
    scaled = torch.load(destination, map_location="cpu", weights_only=True)
    state = scaled["model_state_dict"]
    assert torch.equal(state["actor.2.weight"], original_state["actor.2.weight"] * 0.75)
    assert torch.equal(state["actor.2.bias"], original_state["actor.2.bias"] * 0.75)
    for key in original_state:
        if key not in {"actor.2.weight", "actor.2.bias"}:
            assert torch.equal(state[key], original_state[key])
    with pytest.raises(ValueError, match="output_scale"):
        write_actor_output_scale_checkpoint(
            source=source,
            destination=tmp_path / "bad.pt",
            output_scale=0.0,
            action_dim=3,
            allowed_output_root=tmp_path,
        )
