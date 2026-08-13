"""G1 velocity locomotion task registrations."""

import gymnasium as gym


_COMMON = {
    "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
    "disable_env_checker": True,
}

gym.register(
    id="Humanoid-G1-29DoF-Velocity-Flat-v0",
    **_COMMON,
    kwargs={
        "env_cfg_entry_point": "humanoid_g1.tasks.locomotion.env_cfg:RobotFlatEnvCfg",
        "play_env_cfg_entry_point": "humanoid_g1.tasks.locomotion.env_cfg:RobotFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "humanoid_g1.tasks.locomotion.rsl_rl_cfg:G1PPORunnerCfg",
    },
)

gym.register(
    id="Humanoid-G1-29DoF-Velocity-Rough-v0",
    **_COMMON,
    kwargs={
        "env_cfg_entry_point": "humanoid_g1.tasks.locomotion.env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": "humanoid_g1.tasks.locomotion.env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "humanoid_g1.tasks.locomotion.rsl_rl_cfg:G1PPORunnerCfg",
    },
)

gym.register(
    id="Humanoid-G1-29DoF-Velocity-Play-v0",
    **_COMMON,
    kwargs={
        "env_cfg_entry_point": "humanoid_g1.tasks.locomotion.env_cfg:RobotFlatPlayEnvCfg",
        "play_env_cfg_entry_point": "humanoid_g1.tasks.locomotion.env_cfg:RobotFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "humanoid_g1.tasks.locomotion.rsl_rl_cfg:G1PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity",
    **_COMMON,
    kwargs={
        "env_cfg_entry_point": "humanoid_g1.tasks.locomotion.env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": "humanoid_g1.tasks.locomotion.env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "humanoid_g1.tasks.locomotion.rsl_rl_cfg:G1PPORunnerCfg",
    },
)
