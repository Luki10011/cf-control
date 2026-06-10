import numpy as np
import rclpy
from cf_control_msgs.msg import ThrustAndTorque
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node

from UAV.uav_state import UAVParameters, UAVState
from controller.mellinger_controller import MellingerController
from controller.trajectory_planner import TrajectoryServer
from controller.mpc_controller import MPCPositionController


class ControllerNode(Node):
    """ROS2 node that publishes thrust and torque commands for the UAV model."""

    def __init__(self):
        super().__init__('controller_node')

        self._declare_parameters()
        self._load_controller_parameters()

        # Inicjalizacja managera trajektorii
        self._trajectory_server = TrajectoryServer()
        self._trajectory_generated = False  # Flaga zabezpieczająca przed ponownym generowaniem

        self.current_state = None
        self._last_debug_log_ns = 0

        # Subskrypcja stanu aktualnego (Upewnij się, że ten topic pasuje do ros2 topic list!)
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

        # --- TELEMETRIA DO WYKRESÓW GROUND TRUTH ---
        self._time_history = []
        self._pos_actual_history = []
        self._pos_target_history = []
        self._plots_generated = False

        self._last_valid_target_state = None
        
        self.get_logger().info(
            f'ControllerNode ready. Hardcoded Minimum Snap trajectory waiting for odometry.'
        )

    def _declare_parameters(self):
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('mass', 0.0282)
        self.declare_parameter('gravity', 9.81)
        self.declare_parameter('inertia_tensor', np.eye(3).flatten().tolist())
        self.declare_parameter('hover_position', [0.0, 0.0, 1.0])
        self.declare_parameter('max_tilt_angle_deg', 35.0)  # Maximum tilt angle in degrees
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
        
        max_tilt_angle_deg = float(self.get_parameter('max_tilt_angle_deg').value)
        max_tilt_angle_rad = np.deg2rad(max_tilt_angle_deg)
        
        self._controller = MellingerController(
            params,
            self._matrix_from_parameter('Kp'),
            self._matrix_from_parameter('Kv'),
            self._matrix_from_parameter('KR'),
            self._matrix_from_parameter('KOmega'),
            max_tilt_angle=max_tilt_angle_rad,
        )

        # Initialize MPC outer-loop for translational feedforward
        dt_param = float(self.get_parameter('dt').value) if self.has_parameter('dt') else 0.1
        self._mpc = MPCPositionController(dt=dt_param, horizon=12, a_max=5.0)

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

        # JEŚLI OTRZYMALIŚMY PIERWSZĄ ODOMETRIĘ -> Generujemy trajektorię na sztywno
        if not self._trajectory_generated:
            self._setup_hardcoded_trajectory()

    def _calculate_smooth_durations(self, waypoints, v_avg=0.4):
        durations = []
        for i in range(len(waypoints) - 1):
            # Oblicz dystans geometryczny między punktami
            distance = np.linalg.norm(waypoints[i+1] - waypoints[i])
            
            # Czas podstawowy wynikający z prędkości
            t_segment = distance / v_avg
            
            # DODATKOWY ZAPAS: Jeśli kąt między segmentami jest ostry (nawrót), 
            # dajemy solverowi dodatkowy czas na wyhamowanie pędu
            if i > 0:
                v1 = waypoints[i] - waypoints[i-1]
                v2 = waypoints[i+1] - waypoints[i]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                if cos_angle < 0: # Zwrot o więcej niż 90 stopni (nawrót)
                    t_segment *= 1.5 # Daj 50% więcej czasu na ten manewr
                    
            durations.append(max(2.5, t_segment)) # Segment nie może być krótszy niż 2.5s
        return durations

    def _setup_hardcoded_trajectory(self):
        """Generuje i uruchamia prostą trajektorię testową (Krok po kroku)."""
        self._trajectory_generated = True
        
        start_pos = self.current_state.position
        self.get_logger().info(f"Wykryto pozycję startową: {start_pos}. Generowanie prostej trajektorii weryfikacyjnej...")

        # --- PROSTY ZESTAW WAYPOINTÓW ---
        # Dron startuje z pozycji obecnej, leci w górę, w bok i wraca.
        waypoints = np.array([
            start_pos,                # Punkt 0: Start
            [-0.5, 1.0, 1.2],
            [-1.5, 2.0, 1.2],          # Punkt 1: Dystans = 2.44m -> Duży skok, potrzebuje czasu
            [-2.0, 2.0, 1.2],          # Punkt 2: Dystans = 1.63m -> Zmiana kierunku i wznoszenie
            [-3.3, 1.2, 1.2],          # Punkt 3: Dystans = 0.73m -> Mały powrót, krótki czas!
            [-2.5, 0.8, 1.2],          # Punkt 2: Dystans = 1.63m -> Zmiana kierunku i wznoszenie
            [-1.8, 0.4, 1.2],          # Punkt 1: Dystans = 2.44m -> Duży skok, potrzebuje czasu
            [-1.2, 0.0, 1.2],
            
            # [1.0, 1.0, 1.2]           # Punkt 4: Dystans = 0.53m -> Krótkie doprecyzowanie
        ])

        # Czasy idealnie skorelowane z fizyczną długością każdego segmentu:
        segment_durations = [9, 7, 7, 7, 7, 7, 7] # Każdy segment ma 7 sekund (łącznie 49s trajektorii)

        total_traj_time = sum(segment_durations)
        self.get_logger().info(f"Całkowity czas prostej trajektorii: {total_traj_time:.2f} sekund.")

        # Wywołanie solvera QP Minimum Snap
        success = self._trajectory_server.generate_from_waypoints(waypoints, segment_durations)
        
        if success:
            now_sec = self.get_clock().now().nanoseconds / 1e9
            self._trajectory_server.start(now_sec)
            self.get_logger().info("🔥 PROSTA TRAJEKTORIA WERYFIKACYJNA URUCHOMIONA!")
        else:
            self.get_logger().error("Solver Minimum Snap nie zdołał wyznaczyć nawet prostej trajektorii. Sprawdź konfigurację bibliotek.")

    def _get_default_hover_state(self):
        hover_position = np.asarray(self.get_parameter('hover_position').value, dtype=float)
        if hover_position.size != 3:
            hover_position = np.array([0.0, 0.0, 1.0])

        return {
            'pos': hover_position,
            'vel': np.zeros(3),
            'acc': np.zeros(3),
            'quat': np.array([1.0, 0.0, 0.0, 0.0]),
            'omega': np.zeros(3),
            'w_dot': np.zeros(3),
            'yaw': 0.0
        }
    
    def compute_reference_quaternion(self, acc_des: np.ndarray, yaw_des: float, gravity: float = 9.81) -> np.ndarray:
        """
        Wyznacza pełny kwaternion referencyjny [w, x, y, z] na podstawie
        żądanego przyspieszenia liniowego oraz zadanego kąta Yaw.
        """
        # Całkowita żądana siła (przyspieszenie + kompensacja grawitacji)
        # Zakładamy, że jednostkowa masa m=1, bo interesuje nas tylko kierunek
        z_g = np.array([0.0, 0.0, gravity])
        f_des = acc_des + z_g
        
        f_norm = np.linalg.norm(f_des)
        if f_norm < 1e-6:
            z_B_des = np.array([0.0, 0.0, 1.0])
        else:
            z_B_des = f_des / f_norm
            
        # Rzut osi X na płaszczyznę poziomą przy zadanym Yaw
        x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
        
        # Budowa macierzy rotacji R_des
        y_B_des = np.cross(z_B_des, x_c)
        y_B_norm = np.linalg.norm(y_B_des)
        if y_B_norm < 1e-6:
            # Przypadek osobliwy - dron leci idealnie pionowo, bazujemy na czystym obrocie Z
            y_B_des = np.array([-np.sin(yaw_des), np.cos(yaw_des), 0.0])
        else:
            y_B_des = y_B_des / y_B_norm
            
        x_B_des = np.cross(y_B_des, z_B_des)
        R_des = np.column_stack([x_B_des, y_B_des, z_B_des])
        
        # Konwersja macierzy rotacji 3x3 na kwaternion [w, x, y, z]
        # Używamy bezpiecznej numerycznie metody Shepperda lub standardowej konwersji:
        t = np.trace(R_des)
        if t > 0:
            M = np.sqrt(t + 1.0) * 2.0
            qw = 0.25 * M
            qx = (R_des[2, 1] - R_des[1, 2]) / M
            qy = (R_des[0, 2] - R_des[2, 0]) / M
            qz = (R_des[1, 0] - R_des[0, 1]) / M
        else:
            # Bezpieczne fallbacki dla dużych obrotów
            if (R_des[0, 0] > R_des[1, 1]) and (R_des[0, 0] > R_des[2, 2]):
                M = np.sqrt(1.0 + R_des[0, 0] - R_des[1, 1] - R_des[2, 2]) * 2.0
                qw = (R_des[2, 1] - R_des[1, 2]) / M
                qx = 0.25 * M
                qy = (R_des[0, 1] + R_des[1, 0]) / M
                qz = (R_des[0, 2] + R_des[2, 0]) / M
            elif R_des[1, 1] > R_des[2, 2]:
                M = np.sqrt(1.0 + R_des[1, 1] - R_des[0, 0] - R_des[2, 2]) * 2.0
                qw = (R_des[0, 2] - R_des[2, 0]) / M
                qx = (R_des[0, 1] + R_des[1, 0]) / M
                qy = 0.25 * M
                qz = (R_des[1, 2] + R_des[2, 1]) / M
            else:
                M = np.sqrt(1.0 + R_des[2, 2] - R_des[0, 0] - R_des[1, 1]) * 2.0
                qw = (R_des[1, 0] - R_des[0, 1]) / M
                qx = (R_des[0, 2] + R_des[2, 0]) / M
                qy = (R_des[1, 2] + R_des[2, 1]) / M
                qz = 0.25 * M

        return np.array([qw, qx, qy, qz])

    def _control_loop(self):
        # Log ratunkowy w przypadku braku odometrii
        if self.current_state is None:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_debug_log_ns >= 2_000_000_000:
                self.get_logger().warn("⚠️ Oczekiwanie na odometrię... Sprawdź poprawność topicu /crazyflie/odom")
                self._last_debug_log_ns = now_ns
            return
        if (self.get_clock().now().nanoseconds >= 6_000_000_000):
            now_sec = self.get_clock().now().nanoseconds / 1e9 - 6.0
            flat_state = self._trajectory_server.update(now_sec)
            
            if flat_state is not None:
                ref_quaternion = self.compute_reference_quaternion(
                    acc_des=flat_state['acc'], 
                    yaw_des=flat_state['yaw'],
                    gravity=9.81
                )

                # 2. Przypisz go poprawnie do struktury danych
                target_state = {
                    'pos': flat_state['pos'],
                    'vel': flat_state['vel'],
                    'acc': flat_state['acc'],
                    'quat': ref_quaternion,  # <--- TUTAJ! Koniec z oszukiwaniem kontrolera!
                    'omega': np.array([0.0, 0.0, flat_state['yaw_rate']]),
                    'w_dot': np.array([0.0, 0.0, flat_state['yaw_acc']]),
                    'yaw': flat_state['yaw']
                }
                mode_string = "TRAJECTORY"

                # NOWOŚĆ: Ciągle zapamiętujemy ostatni stan z aktywny trajektorii
                self._last_valid_target_state = target_state.copy()

                # Rejestrujemy dane do wykresu porównawczego
                self._time_history.append(now_sec)
                self._pos_actual_history.append(self.current_state.position.copy())
                self._pos_target_history.append(target_state['pos'].copy())

            else:
                # NOWOŚĆ: Jeśli mamy zapisany ostatni stan trajektorii, wykonujemy
                # krótki manewr final-approach: obliczamy żądaną prędkość i przyspieszenie
                # tak, aby po upływie krótkiego czasu `tf` osiągnąć punkt końcowy.
                if self._last_valid_target_state is not None:
                    # Parametr czasu podejścia - jak szybko spróbujemy dotrzeć do punktu
                    tf = 1.5  # [s] - krótki final approach

                    pos_target = np.asarray(self._last_valid_target_state['pos'], dtype=float)
                    pos_curr = np.asarray(self.current_state.position, dtype=float)
                    vel_curr = np.asarray(self.current_state.linear_velocity, dtype=float)

                    # Żądana prędkość do osiągnięcia punktu w czasie tf
                    vel_des = (pos_target - pos_curr) / tf
                    max_vel = 2.0
                    vnorm = float(np.linalg.norm(vel_des))
                    if vnorm > max_vel and vnorm > 1e-6:
                        vel_des *= (max_vel / vnorm)

                    # Feedforward przyspieszenie potrzebne do osiągnięcia punktu w czasie tf
                    acc_des = 2.0 * (pos_target - pos_curr - vel_curr * tf) / (tf**2)
                    max_acc = 5.0
                    anorm = float(np.linalg.norm(acc_des))
                    if anorm > max_acc and anorm > 1e-6:
                        acc_des *= (max_acc / anorm)

                    target_state = {
                        'pos': pos_target,        # Trzymaj ostatnią pozycję X, Y, Z
                        'vel': vel_des,           # Zaimplementowana żądana prędkość podejścia
                        'acc': acc_des,           # Feedforward przyspieszenie końcowe
                        'quat': np.array([1.0, 0.0, 0.0, 0.0]),
                        'omega': np.zeros(3),     # Zeruj prędkości kątowe
                        'w_dot': np.zeros(3),
                        'yaw': self._last_valid_target_state['yaw']
                    }
                    mode_string = "FINAL_APPROACH"
                else:
                    # Wypadek awaryjny (jeśli trajektoria w ogóle nie wystartowała)
                    target_state = self._get_default_hover_state()
                    mode_string = "HOVER_DEFAULT"
                
                # Jeśli trajektoria właśnie się skończyła, a my zebraliśmy dane -> Generujemy raport
                if len(self._pos_target_history) > 0 and not self._plots_generated:
                    self._plots_generated = True
                    self._generate_ground_truth_plots()

            

            # Obliczenie komend sterujących Mellingera
            thrust, torque = self._controller.compute_control(self.current_state, target_state)

            # Log diagnostyczny (1 Hz)
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_debug_log_ns >= 1_000_000_000:
                self.get_logger().info(
                    f'Czas: {now_sec:.2f}'
                    f'[{mode_string}] pos=[{self.current_state.position[0]:.2f}, {self.current_state.position[1]:.2f}, {self.current_state.position[2]:.2f}] '
                    f'target=[{target_state["pos"][0]:.2f}, {target_state["pos"][1]:.2f}, {target_state["pos"][2]:.2f}] thrust={thrust:.4f}'
                )
                self._last_debug_log_ns = now_ns

            

            # Publikacja
            # thrust = np.clip(thrust, 0.05, 0.4)

            # Publikacja bezpiecznych sterowań
            msg = ThrustAndTorque()
            msg.collective_thrust = float(np.clip(thrust, 0.05, 1))
            msg.torque = Vector3(x=float(torque[0]), y=float(torque[1]), z=float(torque[2]))
            self.command_publisher.publish(msg)

    def _generate_ground_truth_plots(self):
        """Generuje pliki PNG z porównaniem trajektorii i zapisuje je w folderze pakietu."""
        self.get_logger().info("📊 Generowanie wykresów porównawczych Ground Truth...")
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os

        # --- WYMUSZENIE ŚCIEŻKI NA SZTYWNO ---
        # Wskazujemy dokładnie ten folder, o który prosiłeś
        output_dir = "/home/developer/ros2_ws/src/controller/plots"
        os.makedirs(output_dir, exist_ok=True)

        # Definiujemy pełne ścieżki do plików PNG
        path_geo = os.path.join(output_dir, 'tracking_geometric_comparison.png')
        path_time = os.path.join(output_dir, 'tracking_time_comparison.png')

        # --- LOGIKA GENEROWANIA WYKRESÓW (zostaje bez zmian) ---
        t_hist = np.array(self._time_history) - self._time_history[0]
        pos_act = np.array(self._pos_actual_history)
        pos_tar = np.array(self._pos_target_history)

        # Wykres 1: Rzuty płaskie 2D
        fig_geo, axs = plt.subplots(1, 2, figsize=(14, 6))
        fig_geo.suptitle("Porównanie śladu przestrzennego lotu z Trajektorią Zadaną", fontsize=14, fontweight='bold')
        
        axs[0].plot(pos_tar[:, 0], pos_tar[:, 1], 'g--', label='Zadana (Minimum Snap)', linewidth=2)
        axs[0].plot(pos_act[:, 0], pos_act[:, 1], 'b-', label='Rzeczywista (Odom Ground Truth)', linewidth=1.5)
        axs[0].set_xlabel("X [m]")
        axs[0].set_ylabel("Y [m]")
        axs[0].grid(True, alpha=0.5)
        axs[0].legend()

        axs[1].plot(pos_tar[:, 0], pos_tar[:, 2], 'g--', label='Zadana (Minimum Snap)', linewidth=2)
        axs[1].plot(pos_act[:, 0], pos_act[:, 2], 'b-', label='Rzeczywista (Odom Ground Truth)', linewidth=1.5)
        axs[1].set_xlabel("X [m]")
        axs[1].set_ylabel("Z [m]")
        axs[1].grid(True, alpha=0.5)
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(path_geo, dpi=300)
        plt.close(fig_geo)

        # Wykres 2: Profile czasowe X, Y, Z
        fig_time, axs_t = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig_time.suptitle("Analiza uchybu śledzenia w czasie", fontsize=14, fontweight='bold')
        labels = ['Pozycja X [m]', 'Pozycja Y [m]', 'Pozycja Z [m]']
        colors_act = ['darkred', 'darkgreen', 'darkblue']

        for i in range(3):
            axs_t[i].plot(t_hist, pos_tar[:, i], color='gray', linestyle='--', linewidth=2, label='Zadane')
            axs_t[i].plot(t_hist, pos_act[:, i], color=colors_act[i], linewidth=1.5, label='Rzeczywiste')
            axs_t[i].set_ylabel(labels[i])
            axs_t[i].grid(True, alpha=0.3)
            axs_t[i].legend(loc='upper right')

        axs_t[2].set_xlabel("Czas lotu [s]")
        
        plt.tight_layout()
        plt.savefig(path_time, dpi=300)
        plt.close(fig_time)

        self.get_logger().info(f"✅ Wykresy zostały pomyślnie zapisane w: {output_dir}")

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