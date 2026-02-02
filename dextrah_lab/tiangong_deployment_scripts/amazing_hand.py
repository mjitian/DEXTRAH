from threading import Lock, Thread
import time
from rustypot import Scs0009PyController

# ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState



class AmazingHandController(Node):
    def __init__(self):
        super().__init__('amazing_hand_controller')

        #Speed
        self.MaxSpeed = 5

        self.controller = Scs0009PyController(
                serial_port="/dev/ttyACM0",
                baudrate=1000000,
                timeout=0.5,
            )


        self.publish_dt = 1.0 / 60.0  # 60 Hz
        # Initialize joint state
        self.joint_state = JointState()
        self.joint_state.name = [
            'Joint_A01_R', 'Joint_B01_R'
        ]

        self.motor_ids = {
            'Joint_A01_R': [1, 2],
            'Joint_B01_R': [7, 8],
        }

        self.joint_state.position = [0.0] * len(self.joint_state.name)
        self.joint_state.velocity = [0.0] * len(self.joint_state.name)
        # Mutex for thread safety
        # self.joint_state_lock = Lock()
        
        # Publisher for joint states
        self.joint_state_pub = self.create_publisher(JointState, '/hand/joint_states', 1)
        self.pub_timer = self.create_timer(self.publish_dt, self.publish_joint_states)
        
        # self.joint_command_lock = Lock()
        self.joint_command_sub = self.create_subscription(
            JointState,
            '/hand/joint_commands',
            self.joint_command_callback,
            1
        )

    def joint_command_callback(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.joint_state.name:
                target_pos = msg.position[i] - 0.6108652381980153  # Adjust for offset
                motor_ids = self.motor_ids[name]
                # Send command to both motors for the joint
                self.controller.write_goal_speed(
                    motor_ids[0],
                    self.MaxSpeed,
                )
                time.sleep(0.0002)
                self.controller.write_goal_speed(
                    motor_ids[1],
                    self.MaxSpeed,
                )
                time.sleep(0.0002)
                self.controller.write_goal_position(
                    motor_ids[0],
                    target_pos,
                )
                self.controller.write_goal_position(
                    motor_ids[1],
                    -target_pos,
                )
                time.sleep(0.005)

    
    def publish_joint_states(self):
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.read_status()
        self.joint_state_pub.publish(self.joint_state)
    
    def read_status(self):
        pos = self.controller.read_present_position(1)
        speed = self.controller.read_present_speed(1)
        self.get_logger().info(f'Positions 1: {pos[0] + 0.6108652381980153}')
        self.get_logger().info(f'Speeds: {speed}')
        self.joint_state.position[0] = pos[0] + 0.6108652381980153  # Offset to match zero position
        self.joint_state.velocity[0] = speed[0]

        pos = self.controller.read_present_position(7)
        speed = self.controller.read_present_speed(7)
        self.get_logger().info(f'Positions 7: {pos[0] + 0.6108652381980153}')
        self.get_logger().info(f'Speeds: {speed}')
        self.joint_state.position[1] = pos[0] + 0.6108652381980153  # Offset to match zero position
        self.joint_state.velocity[1] = speed[0]            

def main(args=None):
    rclpy.init(args=args)
    
    amazing_hand_controller = AmazingHandController()
    amazing_hand_controller.get_logger().info('Amazing Hand Controller Node has started.')
    rclpy.spin(amazing_hand_controller)
    amazing_hand_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()