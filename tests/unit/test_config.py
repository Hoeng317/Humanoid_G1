from copy import deepcopy

import pytest

from humanoid_g1.utils.config import ConfigError, load_experiment, validate_config


def test_composed_experiment_is_valid():
    config = load_experiment("configs/experiments/g1_flat_ppo.yaml")
    assert config.task == "Humanoid-G1-29DoF-Velocity-Flat-v0"
    assert config.data["robot"]["robot"]["body_dof"] == 29
    assert config.data["training"]["policy"]["actor_hidden_dims"] == [512, 256, 128]
    assert config.data["training"]["algorithm"]["num_learning_epochs"] == 5


def test_smoke_uses_small_separate_policy_and_algorithm_presets():
    config = load_experiment("configs/experiments/g1_smoke.yaml")
    assert config.data["training"]["policy"]["actor_hidden_dims"] == [128, 64]
    assert config.data["training"]["algorithm"]["num_learning_epochs"] == 2


def test_invalid_num_envs_fails_before_simulator():
    config = load_experiment("configs/experiments/g1_flat_ppo.yaml").data
    config["overrides"]["num_envs"] = -1
    with pytest.raises(ConfigError, match="num_envs"):
        validate_config(config)


def test_invalid_range_fails_before_simulator():
    config = deepcopy(load_experiment("configs/experiments/g1_flat_ppo.yaml").data)
    config["commands"]["lin_vel_x"] = [1.0, -1.0]
    with pytest.raises(ConfigError, match="minimum"):
        validate_config(config)
