from threading import Lock, Thread
import math
import time
import argparse
import sys

# ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class AmazingHandController(Node):
    def __init__(self):
        super().__init__('amazing_hand_controller')
        
        self.publish_dt = 1.0 / 100.0  # 60 Hz
        # Initialize joint state
        self.joint_state = JointState()
        # self.joint_state.name = [
        #     'thumb_joint_1_right', 'index_joint_1_right', 'middle_joint_1_right', 
        #     'ring_joint_1_right'
        # ]
        self.joint_state.name = [
            'Joint_A01_R', 'Joint_B01_R'
        ]

        self.joint_state.position = [0.0] * len(self.joint_state.name)
        
        # Mutex for thread safety
        self.joint_state_lock = Lock()
        
        # Publisher for joint states
        self.joint_state_pub = self.create_publisher(JointState, '/hand/joint_states', 1)
        self.pub_timer = self.create_timer(self.publish_dt, self.publish_joint_states)
        
    def publish_joint_states(self):
        with self.joint_state_lock:
            self.joint_state.header.stamp = self.get_clock().now().to_msg()
            self.joint_state_pub.publish(self.joint_state)
    
    def set_joint_positions(self, positions):
        with self.joint_state_lock:
            if len(positions) != len(self.joint_state.position):
                self.get_logger().error("Invalid number of joint positions provided.")
                return
            self.joint_state.position = positions
    def run(self):
        while rclpy.ok():
            print('amzing_hand_controller running...')
            time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    
    amazing_hand_controller = AmazingHandController()

    # Spawn separate thread that spools the fabric
    spin_thread = Thread(target=rclpy.spin, args=(amazing_hand_controller, ), daemon=True)
    spin_thread.start()

    time.sleep(1.)

    amazing_hand_controller.run()
    amazing_hand_controller.destroy_node()
    rclpy.shutdown()

    print('Fabric closed.')


if __name__ == '__main__':
    main()