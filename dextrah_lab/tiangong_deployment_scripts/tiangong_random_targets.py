import os

import rclpy
from rclpy.node import Node
import threading
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

import numpy as np
import time
import copy
from threading import Thread, Lock
import torch
from dextrah_lab.tasks.tiangong.tiangong_constants import (
    HAND_PCA_MINS,
    HAND_PCA_MAXS,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
)



class TiangongRandomTargets(Node):
    def __init__(self):
        super().__init__("tiangong_random_commander")
        # Create mutex lock
        self.mutex = Lock()

        # Init node
        command_rate = 60. #0.5
        self.rate = self.create_rate(frequency=command_rate, clock=self.get_clock())

        # Some settings for cspace targets
        # HAND_PCA_MINS = [0.2475, -0.3286, -0.7238, -0.0192, -0.5532]
        # HAND_PCA_MAXS = [3.8336, 3.0025, 0.8977, 1.0243, 0.0629]
        PALM_POSE_MINS = PALM_POSE_MINS_FUNC(max_pose_angle=45.)
        PALM_POSE_MAXS = PALM_POSE_MAXS_FUNC(max_pose_angle=45.)
        padding = 0.075
        self.upper_pose_limits = np.array(PALM_POSE_MAXS) - padding
        self.lower_pose_limits = np.array(PALM_POSE_MINS) + padding
        self.upper_pca_limits = np.array(HAND_PCA_MAXS) - padding
        self.lower_pca_limits = np.array(HAND_PCA_MINS) + padding

        # Joint command publisher
        self.pose_command_msg = JointState()
        self.hand_command_msg = JointState()
        self.tiangong_fabric_pose_commands_pub = self.create_publisher(
            topic="/tiangong_fabric/pose_commands",
            msg_type=JointState,
            qos_profile=1,
        )
        self.tiangong_fabric_hand_commands_pub = self.create_publisher(
            topic="/tiangong_fabric/hand_commands",
            msg_type=JointState,
            qos_profile=1,
        )

        self.nominal_pose =\
            np.array([0.449, -0.222, 1.196, -2.895, 1.035, 0.335])
            
        self.nominal_pca = np.array([0.0, 0.0])

    def run(self):
        # While ROS is fine, keep sending random cspace targets
        action_step = 0
        action_switched = 0
        pose_command = copy.deepcopy(self.nominal_pose)
        hand_command = copy.deepcopy(self.nominal_pca)

        while rclpy.ok():
            # Generate target and send
#            pose_command = (
#                self.upper_pose_limits - self.lower_pose_limits
#            ) * np.random.rand(self.upper_pose_limits.shape[0]) + self.lower_pose_limits

            if action_step % 30 == 0:
                pose_command = copy.deepcopy(self.nominal_pose)
                if action_switched == 0:
                    #pose_command[1] -= .1
                    pose_command[2] -= .15
                    action_switched = 1
                else:
                    #pose_command[1] += .1
                    pose_command[2] += .15
                    action_switched = 0

                # pose_command = np.array(
                #     [-0.6868, 0.0320, 0.685, -2.3873, -0.0824, 3.1301]
                # )
#                pca_command = (
#                    self.upper_pca_limits - self.lower_pca_limits
#                ) * np.random.rand(self.upper_pca_limits.shape[0]) + self.lower_pca_limits
                hand_command = self.nominal_pca + np.random.rand(2)
                print(pose_command, hand_command)

            timestamp = self.get_clock().now().to_msg()
            self.pose_command_msg.header.stamp = timestamp
            self.hand_command_msg.header.stamp = timestamp
            self.pose_command_msg.position = pose_command.tolist()
            self.hand_command_msg.position = hand_command.tolist()

            print("Sending random command.")
            self.tiangong_fabric_pose_commands_pub.publish(self.pose_command_msg)
            self.tiangong_fabric_hand_commands_pub.publish(self.hand_command_msg)

            # Keep the 2 hz tick rate
            self.rate.sleep()

            action_step += 1


if __name__ == "__main__":
    rclpy.init()
    print("Sending random cspace position targets...")
    random_commander = TiangongRandomTargets()
    thread = threading.Thread(target=rclpy.spin, args=(random_commander,), daemon=True)
    thread.start()
    time.sleep(1.)
    random_commander.run()

    random_commander.destroy_node()
    rclpy.shutdown()

