"""Experiment composition, validation, and Isaac configuration mutation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import math

import yaml

from humanoid_g1.assets.joint_contract import load_joint_contract
from humanoid_g1.utils.paths import CONFIGS, ROOT


class ConfigError(ValueError):
    """A configuration is invalid before simulation startup."""


@dataclass
class ExperimentConfig:
    source: Path
    data: dict[str, Any]

    def save(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(self.data, sort_keys=False), encoding="utf-8")

    @property
    def task(self) -> str:
        return str(self.data["task"])

    @property
    def name(self) -> str:
        return str(self.data["experiment"]["name"])


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"configuration must be a mapping: {path}")
    return payload


def _resolve_training_sections(training: dict[str, Any], source: Path) -> dict[str, Any]:
    """Resolve policy/algorithm YAML references inside a training preset.

    Keeping these sections in separate files lets policy architecture and PPO
    optimization settings evolve independently from rollout length and save
    frequency.  Inline mappings remain supported for older experiment files.
    """
    resolved = deepcopy(training)
    for section in ("policy", "algorithm"):
        value = resolved.get(section)
        if isinstance(value, str):
            resolved[section] = _read_yaml(resolve_config_path(value, source.parent))
        elif not isinstance(value, dict):
            raise ConfigError(f"training.{section} must be a mapping or YAML file reference")
    return resolved


def resolve_config_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = []
    if base is not None:
        candidates.append((base / path).resolve())
    candidates.extend([(ROOT / path).resolve(), (CONFIGS / path).resolve()])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_experiment(path: str | Path, cli_overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    source = resolve_config_path(path, ROOT)
    document = _read_yaml(source)
    merged = deepcopy(document)
    for key in ("robot", "simulation", "terrain", "training", "randomization", "deployment"):
        reference = document.get(key)
        if not isinstance(reference, str):
            raise ConfigError(f"{key} must reference a YAML file")
        component_path = resolve_config_path(reference, source.parent)
        component = _read_yaml(component_path)
        if key == "training":
            component = _resolve_training_sections(component, component_path)
        merged[key] = component
    merged.setdefault("overrides", {})
    for key, value in (cli_overrides or {}).items():
        if value is not None:
            merged["overrides"][key] = value
    validate_config(merged)
    return ExperimentConfig(source=source, data=merged)


def _validate_range(label: str, value: Any) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{label} must be [minimum, maximum]")
    if not all(math.isfinite(float(item)) for item in value):
        raise ConfigError(f"{label} contains NaN or Inf")
    if float(value[0]) > float(value[1]):
        raise ConfigError(f"{label} minimum is greater than maximum")


def validate_config(cfg: Mapping[str, Any], real_mode: bool = False) -> None:
    contract = load_joint_contract()
    overrides = cfg.get("overrides", {})
    num_envs = int(overrides.get("num_envs", 1))
    if num_envs <= 0:
        raise ConfigError("num_envs must be positive")
    simulation = cfg["simulation"]
    dt = float(simulation["physics_dt"])
    decimation = int(simulation["decimation"])
    if dt <= 0 or decimation <= 0:
        raise ConfigError("physics_dt and decimation must be positive")
    if abs(dt - contract.physics_dt) > 1.0e-12 or decimation != contract.decimation:
        raise ConfigError("simulation timing differs from deployment joint contract")
    if int(cfg["robot"]["robot"]["body_dof"]) != contract.action_dim:
        raise ConfigError("robot body_dof differs from the 29-DoF joint contract")
    training = cfg["training"]
    if not isinstance(training.get("policy"), Mapping):
        raise ConfigError("training.policy must resolve to a mapping")
    if not isinstance(training.get("algorithm"), Mapping):
        raise ConfigError("training.algorithm must resolve to a mapping")
    for name in ("num_steps_per_env", "max_iterations", "save_interval"):
        if int(training[name]) <= 0:
            raise ConfigError(f"training.{name} must be positive")
    for name, weight in cfg.get("rewards", {}).items():
        if not math.isfinite(float(weight)):
            raise ConfigError(f"reward {name} is not finite")
    for name, value in cfg.get("commands", {}).items():
        _validate_range(f"commands.{name}", value)
    randomization = cfg["randomization"]
    _validate_range("randomization.friction.static", randomization["friction"]["static"])
    _validate_range("randomization.friction.dynamic", randomization["friction"]["dynamic"])
    _validate_range("randomization.base_mass_add", randomization["base_mass_add"])
    _validate_range("randomization.joint_reset_velocity", randomization["joint_reset_velocity"])
    _validate_range("randomization.push.interval_s", randomization["push"]["interval_s"])
    _validate_range("randomization.push.velocity_xy", randomization["push"]["velocity_xy"])
    deployment = cfg["deployment"]
    if int(deployment["actor_observation_dim"]) != 480 or int(deployment["critic_observation_dim"]) != 495:
        raise ConfigError("observation schema dimensions must be actor=480 and critic=495")
    if abs(float(deployment["policy_dt"]) - contract.policy_dt) > 1.0e-12:
        raise ConfigError("deployment policy_dt differs from physics_dt * decimation")
    if real_mode:
        hardware = cfg["robot"]["robot"]["hardware"]
        if not hardware.get("verified"):
            raise ConfigError("real mode requires a verified hardware profile")
        interface = deployment.get("real_interface")
        if not interface or interface == "lo":
            raise ConfigError("real mode requires an explicit non-loopback interface")


def apply_to_isaac_cfg(exp: ExperimentConfig, env_cfg: Any, agent_cfg: Any) -> tuple[Any, Any]:
    """Apply composed settings after Isaac registry/Hydra config construction."""
    cfg = exp.data
    overrides = cfg["overrides"]
    simulation = cfg["simulation"]
    training = cfg["training"]
    env_cfg.scene.num_envs = int(overrides.get("num_envs", env_cfg.scene.num_envs))
    env_cfg.sim.dt = float(simulation["physics_dt"])
    env_cfg.decimation = int(simulation["decimation"])
    env_cfg.episode_length_s = float(simulation["episode_length_s"])
    env_cfg.sim.device = str(overrides.get("device", simulation.get("device", env_cfg.sim.device)))
    env_cfg.seed = int(overrides.get("seed", cfg["experiment"]["seed"]))
    command = env_cfg.commands.base_velocity
    command.ranges.lin_vel_x = tuple(cfg["commands"]["lin_vel_x"])
    command.ranges.lin_vel_y = tuple(cfg["commands"]["lin_vel_y"])
    command.ranges.ang_vel_z = tuple(cfg["commands"]["ang_vel_z"])
    for name, weight in cfg.get("rewards", {}).items():
        term = getattr(env_cfg.rewards, name, None)
        if term is not None:
            term.weight = float(weight)
    randomization = cfg["randomization"]
    if not randomization["enabled"]:
        env_cfg.events.physics_material = None
        env_cfg.events.add_base_mass = None
        env_cfg.events.base_com = None
        env_cfg.events.actuator_gains = None
        env_cfg.events.joint_friction = None
        env_cfg.events.push_robot = None
        env_cfg.observations.policy.enable_corruption = False
    else:
        env_cfg.events.physics_material.params["static_friction_range"] = tuple(randomization["friction"]["static"])
        env_cfg.events.physics_material.params["dynamic_friction_range"] = tuple(randomization["friction"]["dynamic"])
        env_cfg.events.physics_material.params["restitution_range"] = tuple(randomization["friction"]["restitution"])
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = tuple(randomization["base_mass_add"])
        com = randomization.get("center_of_mass_offset_m")
        if com:
            env_cfg.events.base_com.params["com_range"] = {
                axis: tuple(com[axis]) for axis in ("x", "y", "z")
            }
        else:
            env_cfg.events.base_com = None
        kp_scale = randomization.get("kp_scale")
        kd_scale = randomization.get("kd_scale")
        if kp_scale and kd_scale:
            env_cfg.events.actuator_gains.params["stiffness_distribution_params"] = tuple(kp_scale)
            env_cfg.events.actuator_gains.params["damping_distribution_params"] = tuple(kd_scale)
        else:
            env_cfg.events.actuator_gains = None
        friction_add = randomization.get("joint_friction_add")
        if friction_add:
            env_cfg.events.joint_friction.params["friction_distribution_params"] = tuple(friction_add)
        else:
            env_cfg.events.joint_friction = None
        env_cfg.events.reset_robot_joints.params["velocity_range"] = tuple(
            randomization["joint_reset_velocity"]
        )
        if not randomization["push"]["enabled"]:
            env_cfg.events.push_robot = None
        else:
            env_cfg.events.push_robot.interval_range_s = tuple(randomization["push"]["interval_s"])
            velocity = tuple(randomization["push"]["velocity_xy"])
            env_cfg.events.push_robot.params["velocity_range"] = {"x": velocity, "y": velocity}
    agent_cfg.seed = env_cfg.seed
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.num_steps_per_env = int(training["num_steps_per_env"])
    agent_cfg.max_iterations = int(overrides.get("max_iterations", training["max_iterations"]))
    agent_cfg.save_interval = int(training["save_interval"])
    agent_cfg.empirical_normalization = bool(training["empirical_normalization"])
    agent_cfg.clip_actions = float(training["clip_actions"])
    for key, value in training["policy"].items():
        setattr(agent_cfg.policy, key, value)
    for key, value in training["algorithm"].items():
        setattr(agent_cfg.algorithm, key, value)
    return env_cfg, agent_cfg
