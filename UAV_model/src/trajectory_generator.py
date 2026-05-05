import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
# Zakładamy, że masz własne wiadomości dla thrust/torque lub używasz standardowych
from std_msgs.msg import Float64MultiArray 

# Import Twoich klas (upewnij się, że są w PYTHONPATH)
# from your_package.trajectory_logic import Trajectory 

class MellingerTrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('mellinger_traj_gen')
        
        # Parametry trajektorii
        self.declare_parameter('radius', 7.0)
        self.declare_parameter('omega_traj', 0.2) # rad/s
        self.declare_parameter('height', 1.0)
        
        self.R = self.get_parameter('radius').value
        self.w_t = self.get_parameter('omega_traj').value
        self.z_h = self.get_parameter('height').value
        
        # Publishery
        self.publisher_ = self.create_publisher(PoseStamped, '/cf1/reference/pose', 10)
        self.full_state_pub = self.create_publisher(Float64MultiArray, '/cf1/reference/full_state', 10)
        
        # Timer (np. 50Hz)
        self.timer_period = 0.2
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.start_time = self.get_clock().now().nanoseconds / 1e9

    def get_circular_flat_inputs(self, t):
        """Generuje parametry ruchu po okręgu i ich pochodne."""
        # Pozycja
        pos = np.array([self.R * np.cos(self.w_t * t), 
                        self.R * np.sin(self.w_t * t), 
                        self.z_h])
        
        # Prędkość (1. pochodna)
        vel = np.array([-self.R * self.w_t * np.sin(self.w_t * t),
                         self.R * self.w_t * np.cos(self.w_t * t),
                         0.0])
        
        # Przyspieszenie (2. pochodna)
        acc = np.array([-self.R * self.w_t**2 * np.cos(self.w_t * t),
                        -self.R * self.w_t**2 * np.sin(self.w_t * t),
                        0.0])
        
        # Jerk (3. pochodna)
        jerk = np.array([self.R * self.w_t**3 * np.sin(self.w_t * t),
                         -self.R * self.w_t**3 * np.cos(self.w_t * t),
                         0.0])
        
        # Snap (4. pochodna)
        snap = np.array([self.R * self.w_t**4 * np.cos(self.w_t * t),
                         self.R * self.w_t**4 * np.sin(self.w_t * t),
                         0.0])
        
        # Yaw - ustawiamy np. zawsze w stronę lotu lub stały
        yaw = 0.0 
        yaw_dot = 0.0
        yaw_acc = 0.0
        
        return pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_acc

    def timer_callback(self):
        t = (self.get_clock().now().nanoseconds / 1e9) - self.start_time
        
        # 1. Pobierz flat outputs dla czasu t
        pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_acc = self.get_circular_flat_inputs(t)
        
        # 2. Wykorzystaj swoją funkcję calculate_state_from_flat_inputs
        # Tutaj musisz podpiąć swój obiekt klasy Trajectory
        # results = self.trajectory_calculator.calculate_state_from_flat_inputs(acc, jerk, snap, yaw, yaw_dot, yaw_acc)
        
        # 3. Publikacja (Przykład dla Pose)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        # msg.pose.orientation = ... (tutaj kwaternion z results)
        
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MellingerTrajectoryGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()