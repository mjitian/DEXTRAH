import rclpy
from rclpy.node import Node
from bodyctrl_msgs.msg import CmdSetMotorPosition
from bodyctrl_msgs.msg import SetMotorPosition
import time

def main():
    rclpy.init()
    node = Node("move_head_node")
    publisher = node.create_publisher(CmdSetMotorPosition, "/head/cmd_pos", 1)
    motor_names = [1, 2, 3]
    msg = CmdSetMotorPosition()
    msg.cmds = [SetMotorPosition() for _ in motor_names]
    for i in motor_names:
        msg.cmds[i - 1].name = i
        msg.cmds[i - 1].pos = 0.  # Set desired yaw position in radians
        msg.cmds[i - 1].spd = 0.1  # Set speed
        msg.cmds[i - 1].cur = 3.0  # Set current limit
    msg.cmds[1].pos = 0.2

    while rclpy.ok():
        msg.header.stamp = node.get_clock().now().to_msg()
        publisher.publish(msg)
        node.get_logger().info("Published head position command.")
        time.sleep(1/60)  # Publish at 60 Hz

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()