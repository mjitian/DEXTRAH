# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import pathlib
import numpy as np
import warp as wp
import math
from dextrah_lab.assets.tiangong.tiangong2pro import TIANGONG_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
# from dextrah_lab.assets.kuka_allegro.kuka_allegro import KUKA_ALLEGRO_CFG

@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (1., 1.),
            "damping_distribution_params": (1., 1.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0., 0.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1., 1.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class TiangongEnvCfg(DirectRLEnvCfg):
    # Placeholder for objects_dir which targets the directory of objects for training
    objects_dir = "visdex_objects"
    valid_objects_dir = ["visdex_objects"]

    # Toggle for using cuda graph
    use_cuda_graph = False

    # env
    sim_dt = 1 / 120.
    fabrics_dt = 1 / 60.
    decimation = 2  # 60 Hz
    episode_length_s = 10.
    fabric_decimation = 2  # number of fabric steps per physics step
    num_sim_steps_to_render = 2  # renders every 4 sim steps, so 60 Hz
    num_actions = 10 # 6维位姿(XYZ+RPY) + 4维手指 = 10维动作空间
    success_timeout = 2.
    distillation = False
    num_student_observations = 0
    num_teacher_observations = 0
    num_observations = 0
    num_states = 0

    state_space = 0
    observation_space = 0
    action_space = 0

    asymmetric_obs = True
    obs_type = "full"
    simulate_stereo = False
    stereo_baseline = 55 / 1000

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=num_sim_steps_to_render,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=4 * 5 * 2 ** 15
        ),
    )

    # robot: 9自由度（手臂7+手指2）
    # 机器人初始位置和姿态配置
    robot_cfg: ArticulationCfg = TIANGONG_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  # 基于天工机身基准位置设置
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                # 右臂关节（7个DOF）
                "shoulder_pitch_r_joint": -np.pi / 6,  # 肩关节俯仰：±170°（±2.9671 rad）
                "shoulder_roll_r_joint": 0.,  # 肩关节翻滚：-150°~+15°（-2.618~+0.2618 rad）
                "shoulder_yaw_r_joint": 0.,  # 肩关节偏航：±170°（±2.9671 rad）
                # 初始姿态设为手臂水平伸直向前，以防止与桌面碰撞
                "elbow_pitch_r_joint": -np.pi / 2,  # 肘关节俯仰：-150°~+15°（-2.618~+0.2618 rad）
                "elbow_yaw_r_joint": 2.9,  # 肘关节偏航：±170°（±2.9671 rad）
                "wrist_pitch_r_joint": 0.,  # 腕关节俯仰：-45°~+60°（-0.7854~+1.0472 rad）
                "wrist_roll_r_joint": 0.,  # 腕关节翻滚：-75°~+95°（-1.309~+1.6581 rad）
                # 手指关节（4个DOF）
                "Joint_A01_R": 0.0,
                "Joint_B01_R": 0.0,
                "Joint_C01_R": 0.0,
                "Joint_D01_R": 0.0,

            },
        )
    )

    # 驱动关节列表：严格对应11个自由度（手臂7+手指4）
    actuated_joint_names = [
        # 右臂关节
        "shoulder_pitch_r_joint",
        "shoulder_roll_r_joint",
        "shoulder_yaw_r_joint",
        "elbow_pitch_r_joint",
        "elbow_yaw_r_joint",
        "wrist_pitch_r_joint",
        "wrist_roll_r_joint",
        # 手指关节（4个DOF）
        "Joint_A01_R",
        "Joint_B01_R",
        "Joint_C01_R",
        "Joint_D01_R",
    ]

    # 手部关键链接：基于URDF手部基座定义
    # 在urdf转usd过程中，固定关节连接的连杆融合，手部基座链接名称改变
    hand_body_names = [
        "wrist_roll_r_link",  # 右手基座
    ]

    module_path = os.path.dirname(__file__)
    root_path = os.path.dirname(os.path.dirname(module_path))
    scene_objects_usd_path = os.path.join(root_path, "assets/scene_objects/")

    table_texture_dir = os.path.join(
        root_path, "assets", "curated_table_textures"
    )
    dome_light_dir = os.path.join(
        root_path, "assets", "dome_light_textures"
    )
    metropolis_asset_dir = os.path.join(
        root_path, "assets", "object_textures"
    )

    # TODO：修改桌面初始位置以适配天工机器人
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=scene_objects_usd_path + "table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3 + 0.725 / 2,
                 0.668 - 1.16 / 2,
                 0.93 - 0.03 / 2),
            rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # TODO：根据手眼标定结果调整相机初始位姿
    tf = np.array([
        7.416679444534866883e-02, -9.902696855667120213e-01, 1.177507386359286923e-01, -7.236400044878017468e-01,
        -1.274026398887237732e-01, 1.076995435286611930e-01, 9.859864987275952508e-01, -6.886495877727516479e-01,
        -9.890742408692511090e-01, -8.812921292808308105e-02, -1.181752422362273985e-01, 6.366771698474239516e-01,
        0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00
    ]).reshape(4, 4)
    camera_pos = tf[:3, 3].tolist()
    camera_rot = [0.51567701, -0.52073085, 0.53658829, 0.41831759]
    del tf
    # 略微随机化相机的位置和朝向
    camera_rand_rot_range = 3
    camera_rand_pos_range = 0.03

    # TODO 或许需要根据奥比中光 Gemini 336 相机调整参数
    # 这段代码配置了仿真环境中的视觉传感器（相机），定义了其光学参数、图像分辨率以及在仿真引擎中的渲染方式。

    horizontal_aperture = 21.02
    focal_length = 23.59
    img_width = int(160 * 2) #320
    img_height = int(120 * 2) #240
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=camera_pos, rot=camera_rot, convention="ros"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length, focus_distance=400.0, horizontal_aperture=horizontal_aperture,
            clipping_range=(0.01, 2.)
        ),
        width=img_width,
        height=img_height,
    )
    fov = 2 * math.atan(horizontal_aperture / (2 * focal_length))
    focal_px = img_width * 0.5 / math.tan(fov / 2)
    a = focal_px
    b = img_width * 0.5
    c = focal_px
    d = img_height * 0.5
    intrinsic_matrix = [
        [a, 0., b],
        [0., c, d],
        [0., 0., 1.]
    ]

    # 可视化标记：保持原配置
    pred_pos_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pos_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )

    gt_pos_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pos_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )

    # scene：保持原配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2., replicate_physics=False)

    # 奖励权重：保持原配置，可根据天工动作特性后续调整
    hand_to_object_weight = 1.
    hand_to_object_sharpness = 10.
    object_to_goal_weight = 5.
    in_success_region_at_rest_weight = 10.
    lift_sharpness = 8.5

    # 目标达成参数：保持原配置
    object_goal_tol = 0.1  # m
    success_for_adr = 0.4
    min_steps_for_dr_change = 5 * int(episode_length_s / (decimation * sim_dt))

    # 抓取标准：保持原配置
    min_num_episode_steps = 60
    object_height_thresh = 0.15

    # 物体生成参数：保持原配置
    x_center = 0.55
    x_width = 0.5
    y_center = -0.1
    y_width = 0.8

    # DR控制：保持原配置
    enable_adr = True
    num_adr_increments = 50
    starting_adr_increments = 0

    # 关节摩擦系数：
    starting_robot_dof_friction_coefficients = [
        # 右臂关节
        1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8,
        # 手指关节（4个DOF）
        0.1, 0.1, 0.1, 0.1,
    ]

    # 领域随机化配置：保持原配置
    events: EventCfg = EventCfg()

    adr_cfg_dict = {
        "num_increments": num_adr_increments,
        "robot_physics_material": {
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.8, 1.0)
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (0.5, 2.),
            "damping_distribution_params": (0.5, 2.),
        },
        "robot_joint_friction": {
            "friction_distribution_params": (0., 5.),
        },
        "object_physics_material": {
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.8, 1.0)
        },
        "object_scale_mass": {
            "mass_distribution_params": (0.5, 3.),
        },
    }

    # 物体干扰参数：保持原配置
    wrench_trigger_every = int(1. / (decimation * sim_dt))  # 1 sec
    torsional_radius = 0.01  # m
    hand_to_object_dist_threshold = .3  # m

    # 物体缩放：保持原配置
    object_scale_max = 1.75
    object_scale_min = 0.5
    deactivate_object_scaling = True

    aux_coeff = 1.

    # 自定义ADR参数：保持原配置
    adr_custom_cfg_dict = {
        "object_wrench": {
            "max_linear_accel": (0., 10.)
        },
        "object_spawn": {
            "x_width_spawn": (0., x_width),
            "y_width_spawn": (0., y_width),
            "rotation": (0., 1.)
        },
        "object_state_noise": {
            "object_pos_noise": (0.0, 0.03),  # m
            "object_pos_bias": (0.0, 0.02),  # m
            "object_rot_noise": (0.0, 0.1),  # rad
            "object_rot_bias": (0.0, 0.08),  # rad
        },
        "robot_spawn": {
            "joint_pos_noise": (0., 0.35),
            "joint_vel_noise": (0., 1.)
        },
        "robot_state_noise": {
            "robot_joint_pos_noise": (0.0, 0.08),  # rad
            "robot_joint_pos_bias": (0.0, 0.08),  # rad
            "robot_joint_vel_noise": (0.0, 0.18),  # rad
            "robot_joint_vel_bias": (0.0, 0.08),  # rad
        },
        "reward_weights": {
            "finger_curl_reg": (-0.01, -0.005),
            "object_to_goal_sharpness": (-15., -20.),
            "lift_weight": (5., 0.)
        },
        "pd_targets": {
            "velocity_target_factor": (1., 0.)
        },
        "fabric_damping": {
            "gain": (10., 20.)
        },
        "observation_annealing": {
            "coefficient": (0., 0.)
        },
    }

    # File "/home/dodo/DEXTRAH/dextrah_lab/tasks/tiangong/tiangong_env.py", line 88, in __init__
    # raise ValueError('Max pose angle must be positive')
    # 动作空间参数：要求大于零
    max_pose_angle = 30.#?

    # 深度图随机化：保持原配置
    img_aug_type = "rgb"
    aug_depth = True
    cam_matrix = wp.mat44f()
    cam_matrix[0, 0] = 2.2460368
    cam_matrix[1, 1] = 2.9947157
    cam_matrix[2, 3] = -1.
    cam_matrix[3, 2] = 1.e-3
    d_min = 0.5
    d_max = 1.3
    depth_randomization_cfg_dict = {
        "pixel_dropout_and_randu": {
            "p_dropout": 0.0125 / 4,
            "p_randu": 0.0125 / 4,
            "d_max": d_min,
            "d_min": d_max,
        },
        "sticks": {
            "p_stick": 0.001 / 4,
            "max_stick_len": 18.,
            "max_stick_width": 3.,
            "d_max": d_min,
            "d_min": d_max,
        },
        "correlated_noise": {
            "sigma_s": 1. / 2,
            "sigma_d": 1. / 6,
            "d_max": d_min,
            "d_min": d_max,
        },
        "normal_noise": {
            "sigma_theta": 0.01,
            "cam_matrix": cam_matrix,
            "d_max": d_min,
            "d_min": d_max,
        }
    }

    # 超时终止配置：保持原配置
    disable_out_of_reach_done = False
    disable_dome_light_randomization = False
    disable_arm_randomization = False
