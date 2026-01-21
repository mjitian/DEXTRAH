#!/usr/bin/env python
#
# Copyright (c) 2024, Nvidia.  All rights reserved.

from threading import Lock, Thread
import math
import time
import argparse
import sys

# ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from std_msgs.msg import Float32MultiArray

from bodyctrl_msgs.msg import CmdSetMotorPosition
from bodyctrl_msgs.msg import SetMotorPosition
from bodyctrl_msgs.msg import MotorStatusMsg
from bodyctrl_msgs.msg import MotorStatus

# Numpy and torch imports
import numpy as np
import torch

# Fabrics imports
from fabrics_sim.fabrics.tiangong2pro_pose_fabric import TiangongPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
# 说明：获取双臂关节的状态信息，其中包含关节的当前位置、速度、电流和温度。
# 控制方式：topic
# 话题名称：/arm/status
# 数据定义位置：bodyctrl_msgs::msg::MotorStatusMsg.msg

# 说明：关节的位置控制接口，需要提供期望位置、期望速度、最大电流。
# 控制方式：topic
# 话题名称：/arm/cmd_pos
# 数据定义位置：bodyctrl_msgs::msg::CmdSetMotorPosition.msg
SPEED = 0.0  # rad/s
CURRENT = 0.0  # A

class TiangongFabricNode(Node):
    def __init__(self, speed_mode):
        """
        Creates a tiangong fabric for controlling the robot, and hooks to
        the correct ROS 2 topics with publishers and subscribers.
        """
        # Initialize ROS 2 node
        print("initializing TiangongFabricNode ROS2 node")
        start_time = time.time()
        super().__init__('tiangong_fabric')
        elapse_time = time.time() - start_time
        print(f"<done> elapse: {elapse_time}")

        self.iters_per_cycle = 1  # default to "normal" speed
        if speed_mode == "normal":
            self.iters_per_cycle = 1
        elif speed_mode == "fast":
            self.iters_per_cycle = 2

        # Set the GPU device
        self.device = 'cpu'
        self.cuda_graph = False

        # Set the warp cache directory based on device
        warp_cache_dir = ""
        initialize_warp(self.device)

        # TODO 定义关节名称
        # arm-right 20 + id
        self.arm_controlled_joints = [21, 22, 23, 24, 25, 26, 27]
        self.hand_controlled_joints = [
            "Joint_A01_R",
            "Joint_B01_R",
        ]


        # Timestep for publishing fabric commands out
        self.rate_hz = 60.
        self.publish_dt = 1. / self.rate_hz  # sec
        # Timestep for integrating the fabric state
        self.fabric_dt = 1. / self.rate_hz  # sec

        # Declare position command signals for the arm PD controller and boolean for gripper
        self._tiangong_joint_position_command_lock = Lock()
        self._hand_joint_position_command_lock = Lock()

        self._tiangong_joint_position_command = None
        self._tiangong_joint_velocity_command = None
        self._hand_joint_position_command = None
        self._hand_joint_velocity_command = None

        # Declare measured feedback signals of robot
        self._tiangong_joint_position_lock = Lock()
        self._hand_joint_position_lock = Lock()
        self._tiangong_joint_position = [0.] * len(self.arm_controlled_joints)
        self._hand_joint_position = None

        # Declaration of commands going into fabric
        self._hand_target_lock = Lock()
        self._palm_target_lock = Lock()
        self.palm_target = None
        self.hand_target = None

        # Robot feedback health monitoring
        self.robot_synced = False
        self.tiangong_feedback_time = time.time()
        self.hand_feedback_time = time.time()
        self.robot_feedback_time_elapsed = 0.  # sec
        self.heartbeat_time_threshold = .1  # sec

        # Set up pub/sub for tiangong
        self._tiangong_pub = self.create_publisher(CmdSetMotorPosition, "/arm/cmd_pos", 1)
        self._tiangong_timer = self.create_timer(self.publish_dt, self._tiangong_pub_callback)
        self._tiangong_sub = self.create_subscription(
            MotorStatusMsg,
            '/arm/status',
            self._tiangong_sub_callback,
            1)

        # Set up pub/sub for hand
        self._hand_pub = self.create_publisher(JointState, "/hand/joint_commands", 1)
        self._hand_timer = self.create_timer(self.publish_dt, self._hand_pub_callback)
        self._hand_sub = self.create_subscription(
            JointState,
            '/hand/joint_states',
            self._hand_sub_callback,
            1)

        # Set up sub for receiving commands for the fabric
        # Subscriber for getting pose commands
        self._tiangong_pose_command_sub = self.create_subscription(
            Float32MultiArray,
            '/tiangong_fabric/pose_commands',
            self._tiangong_fabric_pose_command_sub_callback,
            1)

        # Subscriber for getting hand commands
        self._tiangong_hand_command_sub = self.create_subscription(
            JointState,
            '/tiangong_fabric/hand_commands',
            self._tiangong_fabric_hand_command_sub_callback,
            1)

        # Set up publisher for broadcasting fabric state as feedback
        self._tiangong_fabric_states_lock = Lock()
        self.tiangong_fabric_states_msg = JointState()
        
        self.tiangong_fabric_states_msg.name = [
            "shoulder_pitch_r_joint",
            "shoulder_roll_r_joint",
            "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint",
            "elbow_yaw_r_joint",
            "wrist_pitch_r_joint",
            "wrist_roll_r_joint",
            "Joint_A01_R",
            "Joint_B01_R",
        ]

        self._tiangong_fabric_pub = \
            self.create_publisher(JointState, "/tiangong_fabric/joint_states", 1)
        self._tiangong_fabric_timer = \
            self.create_timer(self.publish_dt, self._tiangong_fabric_pub_callback)


    def _tiangong_pub_callback(self):
        """
        Publishes latest tiangong joint position command onto ROS 2 topic that
        goes to PD controller.
        """
        with self._tiangong_joint_position_command_lock:
            if self._tiangong_joint_position_command is not None and \
                    self._tiangong_joint_velocity_command is not None:
                msg = CmdSetMotorPosition()
                for name in self.arm_controlled_joints:
                    set_motor_pos = SetMotorPosition()
                    set_motor_pos.name = name
                    set_motor_pos.pos = self._tiangong_joint_position_command[name - 21]
                    set_motor_pos.spd = SPEED
                    set_motor_pos.cur = CURRENT
                    msg.cmds.append(set_motor_pos)
                self._tiangong_pub.publish(msg)

    def _tiangong_sub_callback(self, msg):
        """
        Acquires the feedback time, sets the measured joint position for the
        tiangong, and also sets the command for the tiangong to this measured position
        if a command does not yet exist.
        ------------------------------------------
        :param msg: ROS 2 MotorStatusMsg message type
        """
        with self._tiangong_joint_position_lock:
            self.tiangong_feedback_time = time.time()
            for motor_statu in msg.status:
                if motor_statu.name in self.arm_controlled_joints:
                    index = self.arm_controlled_joints.index(motor_statu.name)
                    self._tiangong_joint_position[index] = motor_statu.pos
        with self._tiangong_joint_position_command_lock:
            if self._tiangong_joint_position_command is None or self._tiangong_joint_velocity_command is None:
                self._tiangong_joint_position_command = list(self._tiangong_joint_position)
                self._tiangong_joint_velocity_command = len(msg.status) * [0.]

    def _hand_pub_callback(self):
        """
        Publishes latest hand joint position command onto ROS 2 topic that
        goes to controller.
        """
        with self._hand_joint_position_command_lock:
            if self._hand_joint_position_command is not None and \
                    self._hand_joint_velocity_command is not None:
                msg = JointState()
                msg.name = self.hand_controlled_joints
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.position = self._hand_joint_position_command
                msg.velocity = self._hand_joint_velocity_command
                msg.effort = []
                self._hand_pub.publish(msg)

    def _hand_sub_callback(self, msg):
        """
        Acquires the feedback time, sets the measured joint position for the
        hand, and also sets the command for the hand to this measured position
        if a command does not yet exist.
        ------------------------------------------
        :param msg: ROS 2 JointState message type
        """
        with self._hand_joint_position_lock:
            self.hand_feedback_time = time.time()
            self._hand_joint_position = msg.position
        with self._hand_joint_position_command_lock:
            if self._hand_joint_position_command is None or self._hand_joint_velocity_command is None:
                self._hand_joint_position_command = msg.position
                self._hand_joint_velocity_command = len(msg.velocity) * [0.]

    def _tiangong_fabric_pose_command_sub_callback(self, msg):
        """
        Sets the palm pose target coming in from the ROS topic.
        ------------------------------------------
        :param msg: ROS 2 Float32MultiArray message type
        """
        with self._palm_target_lock:
            self.palm_target.copy_(torch.tensor([list(msg.data)], device=self.device))

    def _tiangong_fabric_hand_command_sub_callback(self, msg):
        """
        Sets the PCA position target coming in from the ROS topic.
        ------------------------------------------
        :param msg: ROS 2 CmdSetMotorPosition message type
        """
        with self._hand_target_lock:
            self.hand_target.copy_(torch.tensor([list(msg.position)], device=self.device))

    def _tiangong_fabric_pub_callback(self):
        """
        Writes out the full fabric state on the ROS topic.
        """
        with self._tiangong_fabric_states_lock:
            if len(self.tiangong_fabric_states_msg.position) > 0 and \
                    len(self.tiangong_fabric_states_msg.velocity) > 0:
                self.tiangong_fabric_states_msg.header.stamp = self.get_clock().now().to_msg()
                self._tiangong_fabric_pub.publish(self.tiangong_fabric_states_msg)

    def robot_feedback_heartbeat(self):
        """
        Calculates the heartbeat to the real robot such that processes can be
        stopped if heartbeat not detected.
        """
        # Check to see if joint measurements are fresh.
        tiangong_feedback_time_elapsed = None
        hand_feedback_time_elapsed = None

        with self._tiangong_joint_position_lock:
            tiangong_feedback_time_elapsed = time.time() - self.tiangong_feedback_time
        with self._hand_joint_position_lock:
            hand_feedback_time_elapsed = time.time() - self.hand_feedback_time
        # Find the largest elapsed feedback time between the the arm and hand and
        # set that to the robot's feedback time
        if tiangong_feedback_time_elapsed > hand_feedback_time_elapsed:
            self.robot_feedback_time_elapsed = tiangong_feedback_time_elapsed
        else:
            self.robot_feedback_time_elapsed = hand_feedback_time_elapsed

    def set_joint_commands(self, q, qd, qdd):
        """
        Writes the fabric state as commands destined for PD control and
        also as feedback.
        ------------------------------------------
        :param q: bxn Numpy array of fabric position. b=batch_size, n=number of joints
        :param qd: bxn Numpy array of fabric velocity. b=batch_size, n=number of joints
        :param qdd: bxn Numpy array of fabric acceleration. b=batch_size, n=number of joints
        """
        # Set the fabric position as the command position for the robot
        with self._tiangong_joint_position_command_lock:
            self._tiangong_joint_position_command = list(q[0, :7])
            self._tiangong_joint_velocity_command = list(qd[0, :7])
        with self._hand_joint_position_command_lock:
            self._hand_joint_position_command = list(q[0, 7:])
            self._hand_joint_velocity_command = list(qd[0, 7:])

        # Pack the full fabric state into a ROS 2 CmdSetMotorPosition message as feedback
        with self._tiangong_fabric_states_lock:
            self.tiangong_fabric_states_msg.position = list(q[0, :])
            self.tiangong_fabric_states_msg.velocity = list(qd[0, :])
            self.tiangong_fabric_states_msg.effort = list(qdd[0, :])

    def run(self):
        # Initialize fabric-----------------------
        # Declare batch size and number of joints
        batch_size = 1
        num_dof = 9  # 7 for tiangong arm + 2 for hand PCA

        # Provide initial commands for the fabric
        
        # 根据天工机器人实际情况修改后的初始值
        # Palm pose target
        # 旋转顺序为 euler_zyx
        self.palm_target = \
            torch.tensor([[0.449, -0.222, 1.196, -2.895, 1.035, 0.335]], device=self.device)

        # Hand PCA target
        self.hand_target = \
            torch.tensor([[0., 0.]], device=self.device)

        # This creates a world model that book keeps all the meshes
        # in the world, their pose, name, etc.
        print('Importing world')
        world_filename = 'tiangong_boxes'
        max_objects_per_env = 20
        world_model = WorldMeshesModel(batch_size=batch_size,
                                       max_objects_per_env=max_objects_per_env,
                                       device=self.device,
                                       world_filename=world_filename)

        # This reports back handles to the meshes which is consumed
        # by the fabric for collision avoidance
        object_ids, object_indicator = world_model.get_object_ids()

        # Create Tiangong pose-pca fabric
        tiangong_fabric = TiangongPoseFabric(
            batch_size, device=self.device, timestep=self.fabric_dt,
            graph_capturable=self.cuda_graph
        )

        # Create integrator for the fabric
        tiangong_integrator = DisplacementIntegrator(tiangong_fabric)

        # Allocate for fabric state
        q = torch.zeros(batch_size, num_dof, device=self.device)
        qd = torch.zeros(batch_size, num_dof, device=self.device)
        qdd = torch.zeros(batch_size, num_dof, device=self.device)

        # Create CUDA graph
        g = None
        q_new = None
        qd_new = None
        qdd_new = None
        if self.cuda_graph:
            with self._hand_target_lock:
                with self._palm_target_lock:
                    # NOTE: elements of inputs must be in the same order as expected in the
                    # set_features function of the fabric
                    inputs = [self.hand_target, self.palm_target, "euler_zyx",
                            q.detach(), qd.detach(), object_ids, object_indicator]
                    g, q_new, qd_new, qdd_new = capture_fabric(
                        tiangong_fabric, q, qd, qdd, self.fabric_dt,
                        tiangong_integrator, inputs, self.device)
                    print('Captured fabric.')

        # Sleep a little to ensure feedback subs have received feedback
        time.sleep(self.heartbeat_time_threshold + 0.2)

        # Query feedback heartbeat
        self.robot_feedback_heartbeat()
        print(self.robot_feedback_time_elapsed)

        # Indicate if heartbeat time out.
        if self.robot_feedback_time_elapsed > self.heartbeat_time_threshold:
            print('Heartbeat timed out')

        # Main loop. While feedback from real robot meets heartbeat interval and ROS 2
        # has not been signaled for shutdown, keep cycling the fabric and publishing
        # its state as commands to the PD controller and as feedback
        control_iter = 0
        print_iter = 60
        loop_time_filtered = 0.
        while self.robot_feedback_time_elapsed < self.heartbeat_time_threshold and rclpy.ok():
            # First send robot joint commands to where it is now for about 1 second to get
            # communications flowing.
            if not self.robot_synced:
                # Copy over the kuka joint positions
                q[0, :7].copy_(torch.tensor(self._tiangong_joint_position, device=self.device))
                # Copy over the allegro joint_positions
                q[0, 7:].copy_(torch.tensor(self._hand_joint_position, device=self.device))

                # Set joint commands, which will be published over ROS
                self.set_joint_commands(q.detach().cpu().numpy().astype('float'),
                                        qd.detach().cpu().numpy().astype('float'),
                                        qdd.detach().cpu().numpy().astype('float'))

                print('sending at curr')
                time.sleep(1.)

                self.robot_synced = True

            # Set start time
            start = time.time()

            # Set features/actions
            with self._hand_target_lock:
                with self._palm_target_lock:
                    # Integrate fabric forward
                    # NOTE: iters_per_cycle greater than 1 steps faster through
                    # the fabric integral curve                    
                    for i in range(self.iters_per_cycle):
                        if self.cuda_graph:
                            # Replay the fabric graph
                            g.replay()
                            # Update states with graph output
                            q.copy_(q_new)
                            qd.copy_(qd_new)
                            qdd.copy_(qdd_new)
                        else:
                            # Set fabric features
                            tiangong_fabric.set_features(
                                self.hand_target, self.palm_target, "euler_zyx",
                                q.detach(), qd.detach(), object_ids, object_indicator)
                            # Step the fabric integrator
                            tiangong_integrator.step(q, qd, qdd, self.fabric_dt)
            # Set joint commands, which will be published over ROS
            self.set_joint_commands(q.detach().cpu().numpy().astype('float'),
                                    qd.detach().cpu().numpy().astype('float'),
                                    qdd.detach().cpu().numpy().astype('float'))

            # Keep 30 Hz tick rate
            while (time.time() - start) < self.publish_dt:
                time.sleep(.00001)

            # Query feedback heartbeat
            self.robot_feedback_heartbeat()

            # Print control loop frequencies
            loop_time = time.time() - start
            alpha = 0.5
            if control_iter == 0:
                loop_time_filtered = loop_time
            else:
                loop_time_filtered = alpha * loop_time + (1. - alpha) * loop_time_filtered
            if (control_iter % print_iter) == 0:
                print('avg control rate', 1. / loop_time_filtered)

            control_iter += 1


if __name__ == '__main__':
    # Parse the fabrics speed mode
    parser = argparse.ArgumentParser()
    parser.add_argument('speed_mode', type=str,
                        help='Set the fabrics motion speed: normal, fast')
    args = None
    # Check to see the speed_mode argument was actually passed
    try:
        args = parser.parse_args()
    except:
        print('--------------------------')
        print('Ensure you set a speed mode')
        print('Please set it to "normal", "fast"')
        sys.exit()

    # Check for validity of set mode
    speed_mode = args.speed_mode
    speed_modes = ["normal", "fast"]
    if speed_mode not in speed_modes:
        print('Invalid speed mode. Please set it to "slow", "normal", "fast", or "superfast"')
        sys.exit()

    print("Starting Tian Gong fabric node")
    rclpy.init()

    # Create the fabric
    tiangong_fabric_node = TiangongFabricNode(speed_mode=speed_mode)

    # Spawn separate thread that spools the fabric
    spin_thread = Thread(target=rclpy.spin, args=(tiangong_fabric_node,), daemon=True)
    spin_thread.start()

    time.sleep(1.)

    tiangong_fabric_node.run()

    tiangong_fabric_node.destroy_node()
    rclpy.shutdown()

    print('Fabric closed.')

