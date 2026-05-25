
import numpy as np
import rclpy
from cf_control_msgs.msg import ThrustAndTorque
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node

from UAV.uav_state import UAVParameters, UAVState
from controller.mellinger_controller import MellingerController


class ControllerNode(Node):
    """ROS2 node that publishes thrust and torque commands for the UAV model."""

    def __init__(self):
        super().__init__('controller_node')

        self._declare_parameters()
        self._load_controller_parameters()

        self.current_state = None
        self._last_debug_log_ns = 0

        self.state_subscription = self.create_subscription(
            Odometry,
            '/crazyflie/odom',
            self._state_callback,
            10,
        )
        self.command_publisher = self.create_publisher(
            ThrustAndTorque,
            '/cf_control/control_command',
            10,
        )

        control_rate = float(self.get_parameter('control_rate').value)
        self.create_timer(1.0 / control_rate, self._control_loop)

        self.get_logger().info(
            f'ControllerNode ready. Publishing to "control_inputs" at {control_rate:.1f} Hz.'
        )

    def _declare_parameters(self):
        self.declare_parameter('control_rate', 100.0)
        self.declare_parameter('mass', 1.0)
        self.declare_parameter('gravity', 9.81)
        self.declare_parameter('inertia_tensor', np.eye(3).flatten().tolist())
        self.declare_parameter('hover_position', [0.0, 0.0, 1.0])
        self.declare_parameter('Kp', np.eye(3).flatten().tolist())
        self.declare_parameter('Kv', np.eye(3).flatten().tolist())
        self.declare_parameter('KR', np.eye(3).flatten().tolist())
        self.declare_parameter('KOmega', np.eye(3).flatten().tolist())

    def _load_controller_parameters(self):
        params = UAVParameters()
        params.mass = float(self.get_parameter('mass').value)
        params.gravity = float(self.get_parameter('gravity').value)

        inertia_tensor = np.asarray(self.get_parameter('inertia_tensor').value, dtype=float)
        if inertia_tensor.size == 9:
            params.inertia_tensor = inertia_tensor.reshape((3, 3))
        elif inertia_tensor.size == 3:
            params.inertia_tensor = np.diag(inertia_tensor)

        self._params = params
        self._hover_thrust = float(params.mass * params.gravity)
        self._controller = MellingerController(
            params,
            self._matrix_from_parameter('Kp'),
            self._matrix_from_parameter('Kv'),
            self._matrix_from_parameter('KR'),
            self._matrix_from_parameter('KOmega'),
        )
        self.get_logger().info(
            f'Loaded controller params: mass={params.mass:.4f} kg, '
            f'gravity={params.gravity:.3f} m/s^2, hover_thrust={self._hover_thrust:.4f} N'
        )

    def _matrix_from_parameter(self, name):
        values = np.asarray(self.get_parameter(name).value, dtype=float)
        if values.size == 9:
            return values.reshape((3, 3))
        if values.size == 3:
            return np.diag(values)
        return np.eye(3)

    def _state_callback(self, msg: Odometry):
        state = UAVState()
        state.position = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            dtype=float,
        )
        state.linear_velocity = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
            dtype=float,
        )
        state.orientation = np.array(
            [
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
            ],
            dtype=float,
        )
        state.angular_velocity = np.array(
            [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z],
            dtype=float,
        )
        self.current_state = state

    def _target_state(self):
        hover_position = np.asarray(self.get_parameter('hover_position').value, dtype=float)
        if hover_position.size != 3:
            hover_position = np.array([1.0, 1.0, 1.0])

        return {
            'pos': hover_position,
            'vel': np.zeros(3),
            'acc': np.zeros(3),
            'quat': np.array([1.0, 0.0, 0.0, 0.0]),
            'omega': np.zeros(3),
            'w_dot': np.zeros(3),
        }

    def _control_loop(self):
        if self.current_state is None:
            return

        target_state = self._target_state()
        thrust, torque = self._controller.compute_control(self.current_state, target_state)
        # thrust = float(np.clip(thrust, 0.0, 0.56))

        current_z = float(self.current_state.position[2])
        current_vz = float(self.current_state.linear_velocity[2])
        target_z = float(target_state['pos'][2])
		
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_debug_log_ns >= 1_000_000_000:
            self.get_logger().info(
                f'hover_debug: z={current_z:.3f} vz={current_vz:.3f} '
                f'hover_position={target_state["pos"]:.3f} '
                f'target_z={target_z:.3f} hover_thrust={self._hover_thrust:.4f} '
                f'thrust={thrust:.4f}'
            )
            self._last_debug_log_ns = now_ns

        msg = ThrustAndTorque()
        msg.timestamp = self.get_clock().now().nanoseconds
        msg.collective_thrust =  thrust
        msg.torque = Vector3(
            x=float(torque[0]),
            y=float(torque[1]),
            z=float(torque[2]),
        )
        self.command_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
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