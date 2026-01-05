# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Defines the Kuka-Allegro robot configuration for simulation with Isaac Sim.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

module_path = os.path.dirname(__file__)
root_path = os.path.dirname(module_path)
tiangong2pro_usd_path = os.path.join(root_path, "tiangong/tiangong2pro.usd")

TIANGONG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=tiangong2pro_usd_path,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # right arm
            "shoulder_pitch_r_joint": 0.0,
            "shoulder_roll_r_joint": 0.0,
            "shoulder_yaw_r_joint": 0.0,
            "elbow_pitch_r_joint": 0.0,
            "elbow_yaw_r_joint": 0.0,
            "wrist_pitch_r_joint": 0.0,
            "wrist_roll_r_joint": 0.0,

            # right hand (A/B/C/D chains)
            "Joint_A01_R": 0.0,
            "Joint_B01_R": 0.0,
            "Joint_C01_R": 0.0,
            "Joint_D01_R": 0.0,
        },
    ),
    actuators={
        "tiangong2pro_actuators": ImplicitActuatorCfg(
            # 依据 URDF 关节名称配置匹配表达式
            joint_names_expr=[

                # right arm
                "shoulder_pitch_r_joint",
                "shoulder_roll_r_joint",
                "shoulder_yaw_r_joint",
                "elbow_pitch_r_joint",
                "elbow_yaw_r_joint",
                "wrist_pitch_r_joint",
                "wrist_roll_r_joint",

                # right hand
                "Joint_A01_R",
                "Joint_B01_R",
                "Joint_C01_R",
                "Joint_D01_R",
            ],
            # 力矩上限
            effort_limit_sim={

                # right arm
                "shoulder_pitch_r_joint": 91.0,
                "shoulder_roll_r_joint": 95.0,
                "shoulder_yaw_r_joint": 35.0,
                "elbow_pitch_r_joint": 35.0,
                "elbow_yaw_r_joint": 24.0,
                "wrist_(pitch|roll)_r_joint": 6.3,

                # right hand
                "Joint_A01_R": 0.05,
                "Joint_B01_R": 0.05,
                "Joint_C01_R": 0.05,
                "Joint_D01_R": 0.05,
            },
            # 刚度/阻尼配置
            stiffness={
                # right arm
                "shoulder_(pitch|roll|yaw)_r_joint": 300.0,
                "elbow_(pitch|yaw)_r_joint": 200.0,
                "wrist_(pitch|roll)_r_joint": 60.0,

                # hands
                "Joint_A01_R": 5.0,
                "Joint_B01_R": 5.0,
                "Joint_C01_R": 5.0,
                "Joint_D01_R": 5.0,
            },
            damping={
                # right arm
                "shoulder_(pitch|roll|yaw)_r_joint": 45.0,
                "elbow_(pitch|yaw)_r_joint": 25.0,
                "wrist_(pitch|roll)_r_joint": 10.0,

                # hands
                "Joint_A01_R": 0.5,
                "Joint_B01_R": 0.5,
                "Joint_C01_R": 0.5,
                "Joint_D01_R": 0.5,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Kuka Allegro robot."""
