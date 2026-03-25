import numpy as np
import rclpy
from constants import MASS, g
from nav_msgs.msg import Odometry
from propagator import UAVPropagator
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from uav import UAVModel


class UAVNode(Node):
    def __init__(self):
        super().__init__('uav_sim_node')

        dt = 0.01

        initial_state = np.zeros(13)
        initial_state[0:3] = [10, 10, 10]
        initial_state[6] = 1

        self.state = initial_state
        self.model = UAVModel(self.state)
        self.propagator = UAVPropagator(dt, self.model)

        self.control = np.array([MASS * g + 0.2, 0.0, 0.0, 0.0])

        self.odom_pub = self.create_publisher(Odometry, 'uav/odom', 10)

        self.control_sub = self.create_subscription(
            Float32MultiArray, 'uav/control', self.control_callback, 10
        )

        self.timer = self.create_timer(dt, self.update)

    def control_callback(self, msg):
        self.control = np.array(msg.data)

    def update(self):
        self.state = self.propagator.rk4(self.state, self.control)

        odom_msg = Odometry()

        # ================= HEADER =================
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'world'
        odom_msg.child_frame_id = 'base_link'  # standard ROS

        # ================= POSE =================
        odom_msg.pose.pose.position.x = float(self.state[0])
        odom_msg.pose.pose.position.y = float(self.state[1])
        odom_msg.pose.pose.position.z = float(self.state[2])

        odom_msg.pose.pose.orientation.w = float(self.state[6])
        odom_msg.pose.pose.orientation.x = float(self.state[7])
        odom_msg.pose.pose.orientation.y = float(self.state[8])
        odom_msg.pose.pose.orientation.z = float(self.state[9])

        # ================= VELOCITY =================
        # liniowa
        odom_msg.twist.twist.linear.x = float(self.state[3])
        odom_msg.twist.twist.linear.y = float(self.state[4])
        odom_msg.twist.twist.linear.z = float(self.state[5])

        # kątowa
        odom_msg.twist.twist.angular.x = float(self.state[10])
        odom_msg.twist.twist.angular.y = float(self.state[11])
        odom_msg.twist.twist.angular.z = float(self.state[12])

        self.odom_pub.publish(odom_msg)

        # DEBUG
        self.get_logger().info(f'Pos: {self.state[0:3]}')
        self.get_logger().info(f'Lin Vel: {self.state[3:6]}')
        self.get_logger().info(f'Ang Vel: {self.state[10:13]}')


def main(args=None):
    rclpy.init(args=args)

    node = UAVNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
