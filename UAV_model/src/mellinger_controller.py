import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from cf_control_msgs.msg import ThrustAndTorque # Typ wiadomości z Twojego przykładu

# Importy Twoich modułów
from model import UAVModel
from test_trajectory import Trajectory
from constants import g 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class MellingerControllerNode(Node):
    def __init__(self):
        super().__init__('mellinger_controller_node')
        
        # 1. PARAMETRY MODELU (z SDF dla masy i inercji)
        self.mass = 0.025
        self.inertia = np.array([
            [16.571710e-06, 0.830806e-06, 0.718277e-06],
            [0.830806e-06, 16.655602e-06, 1.800197e-06],
            [1.800197e-06, 0.718277e-06, 29.261652e-06],
        ])

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # To jest kluczowe!
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        init_state = np.zeros(13)
        init_state[6] = 1.0 # qw
        self.uav_model = UAVModel(mass=self.mass, inertia_tensor=self.inertia, initial_conditions=init_state)
        
        # Logika trajektorii
        self.traj_logic = Trajectory(type='circle', R=1.0, model=self.uav_model)

        # 2. KOMUNIKACJA
        self.create_subscription(
            Odometry,
            '/crazyflie/odom',
            self.odom_callback,
            qos_profile
        )
        
        # Zmieniony Publisher na nowy topik i typ wiadomości
        self.control_pub = self.create_publisher(
            ThrustAndTorque, 
            '/cf_control/control_command', 
            10
        )
        
        self.odom_received = False
        self.takeoff_height = 1.0
        self.takeoff_duration = 5.0
        
        self.timer = self.create_timer(0.01, self.control_loop) # 100Hz
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.get_logger().info("Kontroler gotowy. Wysyłam ThrustAndTorque na /cf_control/control_command")

    def odom_callback(self, msg):
        self.odom_received = True
        
        # Pozycja i prędkość liniowa
        self.uav_model.position = np.array([
            msg.pose.pose.position.x, 
            msg.pose.pose.position.y, 
            msg.pose.pose.position.z])
        
        self.uav_model.linear_velocity = np.array([
            msg.twist.twist.linear.x, 
            msg.twist.twist.linear.y, 
            msg.twist.twist.linear.z])

        # Kwaternion: ROS [x, y, z, w] -> Model [w, x, y, z]
        self.uav_model.quternions_orientation = np.array([
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z])
        
        self.uav_model.angular_velocity = np.array([
            msg.twist.twist.angular.x, 
            msg.twist.twist.angular.y, 
            msg.twist.twist.angular.z])

    def control_loop(self):
        if not self.odom_received:
            return

        t = (self.get_clock().now().nanoseconds / 1e9) - self.start_time
        
        # DEFINICJA PUNKTU DOCELOWEGO (Target Point)
        # Przez pierwsze 5s leć do góry, potem przesuń się w bok
        if t < 5.0:
            p_ref = np.array([0.0, 0.0, 1.0])
        else:
            p_ref = np.array([1.0, 0.0, 1.0]) # Punkt (x=1, y=0, z=1)

        # W locie do punktu prędkość i przyspieszenie referencyjne to 0
        v_ref = np.zeros(3)
        acc_ref = np.zeros(3)

        # Obliczanie stanu referencyjnego (Flat Outputs dla zawisu)
        target_state = self.traj_logic.calculate_state_from_flat_inputs(
            acc_ref, np.zeros(3), np.zeros(3), yaw=0.0, yaw_rate=0.0, yaw_acc=0.0
        )
        target_state.update({'pos': p_ref, 'vel': v_ref, 'acc': acc_ref})

        # Mellinger
        u1, tau = self.traj_logic.mellinger_controll(target_state)

        # Diagnostyka błędu
        z_error = p_ref[2] - self.uav_model.position[2]
        if int(t * 10) % 50 == 0: # Loguj co 5 sekund
             self.get_logger().info(f"Wysokość: {self.uav_model.position[2]:.2f}m | Błąd Z: {z_error:.3f}")

        # Wysyłanie do Mixera C++
        msg = ThrustAndTorque()
        msg.collective_thrust = float(u1)
        msg.torque.x = float(tau[0])
        msg.torque.y = float(tau[1])
        msg.torque.z = float(tau[2])
        self.control_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MellingerControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()