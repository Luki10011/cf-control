
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy import init, spin, shutdown
from UAV.rk4 import RK4Propagator
from UAV.uav_state import UAVState, UAVParameters
from nav_msgs.msg import Odometry
from cf_control_msgs.msg import ThrustAndTorque

class UAVModelNode(Node):

    def __init__(self):
        super().__init__('uav_model_node')
    
        # Declaring parameters
        self._declare_parameters()

        params = self._load_parameters()
        self.params = {
            'mass': params.mass,
            'gravity': params.gravity,
            'inertia': params.inertia_tensor,
            'dt': params.dt
        }

        # Initializing state and propagator
        self.state = UAVState()
        self.propagator = RK4Propagator(self.params)
        self.control = np.zeros(4)

        # Publishers and subscribers 

        self.timer = self.create_timer(self.params['dt'], self.next_state)
        
        self.state_publisher = self.create_publisher(Odometry, 'state', 10)

        self.control_subscriber = self.create_subscription(
            ThrustAndTorque,
            'control_inputs',
            self.control_callback,
            10
        )

        self.get_logger().info('UAV Model Node has been started with parameters: {}'.format(self.params))

    def next_state(self):
        current_state = self.state.get_state()
        integrated_state = self.propagator.propagate(
            current_state,
            self.control
        )
        self.state.update_state(
            position = integrated_state[0:3],
            linear_velocity = integrated_state[3:6],
            orientation = integrated_state[6:10],
            angular_velocity = integrated_state[10:13]
        )
        self.publish_state()

    def _declare_parameters(self):
        self.declare_parameter('mass', 1.0)
        self.declare_parameter('gravity', 9.81)
        self.declare_parameter('inertia_tensor', np.eye(3).flatten().tolist())
        self.declare_parameter('dt', 0.01)
    
    def _load_parameters(self) -> UAVParameters:
        parameters = UAVParameters()
        parameters.mass = self.get_parameter('mass').value
        parameters.gravity = self.get_parameter('gravity').value
        inertia_tensor = list(self.get_parameter('inertia_tensor').value)
        
        parameters.inertia_tensor = np.array(inertia_tensor).reshape((3, 3))
        parameters.dt = self.get_parameter('dt').value

        return parameters
    
    def control_callback(self, msg):
        self.control = np.array([
            msg.collective_thrust,
            msg.torque.x,
            msg.torque.y,
            msg.torque.z,
        ])


    def publish_state(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        msg.pose.pose.position.x = self.state.position[0]
        msg.pose.pose.position.y = self.state.position[1]
        msg.pose.pose.position.z = self.state.position[2]

        msg.pose.pose.orientation.w = self.state.orientation[0]
        msg.pose.pose.orientation.x = self.state.orientation[1]
        msg.pose.pose.orientation.y = self.state.orientation[2]
        msg.pose.pose.orientation.z = self.state.orientation[3]

        msg.twist.twist.linear.x = self.state.linear_velocity[0]
        msg.twist.twist.linear.y = self.state.linear_velocity[1]
        msg.twist.twist.linear.z = self.state.linear_velocity[2]

        msg.twist.twist.angular.x = self.state.angular_velocity[0]
        msg.twist.twist.angular.y = self.state.angular_velocity[1]
        msg.twist.twist.angular.z = self.state.angular_velocity[2]

        self.state_publisher.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = UAVModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if node.context.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
