"""Isaac Lab articulation configuration for the official Unitree ROS G1 URDF."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from humanoid_g1.utils.paths import URDF_PATH


G1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(URDF_PATH),
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness={".*_hip_.*": 100.0, "waist_yaw_joint": 200.0},
            damping={".*_hip_.*": 2.0, "waist_yaw_joint": 5.0},
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=139.0,
            velocity_limit_sim=20.0,
            stiffness={".*_hip_roll_.*": 100.0, ".*_knee_.*": 150.0},
            damping={".*_hip_roll_.*": 2.0, ".*_knee_.*": 4.0},
            armature=0.01,
        ),
        "N5020-16-arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_roll.*"],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
        "N5020-16-ankle-waist": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_.*", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=35.0,
            velocity_limit_sim=30.0,
            stiffness=40.0,
            damping={".*_ankle_.*": 2.0, "waist_.*_joint": 5.0},
            armature=0.01,
        ),
        "W4010-25": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            effort_limit_sim=5.0,
            velocity_limit_sim=22.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)

