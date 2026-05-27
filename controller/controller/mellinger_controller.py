
from UAV.uav_state import UAVParameters, UAVState
import numpy as np


class MellingerController:
    """Geometric Mellinger controller for the controller package."""

    def __init__(
        self,
        parameters: UAVParameters,
        Kp: np.ndarray,
        Kv: np.ndarray,
        KR: np.ndarray,
        KOmega: np.ndarray,
        max_tilt_angle: float = np.deg2rad(45.0),  # Max 45 degrees by default
    ):
        self.mass = parameters.mass
        self.inertia_tensor = parameters.inertia_tensor
        self.gravity = parameters.gravity
        self.Kp = np.asarray(Kp, dtype=float)
        self.Kv = np.asarray(Kv, dtype=float)
        self.KR = np.asarray(KR, dtype=float)
        self.KOmega = np.asarray(KOmega, dtype=float)
        self.max_tilt_angle = max_tilt_angle
        self.max_tilt_sin = np.sin(max_tilt_angle)  # Cache for efficiency

    def compute_control(self, current_state: UAVState, target_state):
        """Udoskonalona metoda compute_control z pełnym feedforwardem i dynamicznym yaw."""
        pos_curr = np.asarray(current_state.position, dtype=float)
        vel_curr = np.asarray(current_state.linear_velocity, dtype=float)
        quat_curr = np.asarray(current_state.orientation, dtype=float)
        omega_curr = np.asarray(current_state.angular_velocity, dtype=float)

        pos_des = np.asarray(target_state['pos'], dtype=float)
        vel_des = np.asarray(target_state['vel'], dtype=float)
        acc_des = np.asarray(target_state.get('acc', np.zeros(3)), dtype=float)
        omega_des = np.asarray(target_state.get('omega', np.zeros(3)), dtype=float)
        alpha_des = np.asarray(target_state.get('w_dot', np.zeros(3)), dtype=float)
        
        # Pobieramy ref_thrust bezpośrednio ze zmiennej wejściowej (jeśli Twój serwer trajektorii go zwraca)
        # Jeśli nie, wyliczamy go klasycznie: m * ||acc_des + g||
        g_vec = np.array([0.0, 0.0, self.gravity])
        ref_thrust = float(target_state.get('ref_thrust', self.mass * np.linalg.norm(acc_des + g_vec)))

        # 1. Pętla Pozycji z poprawnym Feed-Forward
        ep = pos_curr - pos_des
        ev = vel_curr - vel_des
        
        # f_des z uwzględnieniem przyspieszenia z trajektorii
        F_des = -self.Kp @ ep - self.Kv @ ev + self.mass * g_vec + self.mass * acc_des

        from UAV.utils import quaternion_to_rotation_matrix
        R_curr = quaternion_to_rotation_matrix(quat_curr)
        z_B = R_curr[:, 2]
        
        # Wypadkowy ciąg na osie silników
        u1 = float(np.dot(F_des, z_B))

        # 2. Rekonstrukcja R_des (Zgodna z kodem znajomej)
        f_des_norm = float(np.linalg.norm(F_des))
        if f_des_norm < 1e-6:
            z_des = np.array([0.0, 0.0, 1.0])
        else:
            z_des = F_des / f_des_norm

        # CONSTRAINT: Limit maximum tilt angle to prevent extreme attitudes
        # If tilt angle exceeds limit, rescale horizontal components while keeping vertical
        z_des_norm = np.linalg.norm(z_des)
        if z_des_norm > 0:
            # Compute tilt angle from the z-component (cos(tilt_angle) = z_des[2] / norm)
            # For a unit vector z_des, tilt_angle = acos(z_des[2])
            # sin(tilt_angle) = ||[z_des[0], z_des[1]]|| / norm
            horizontal_component = np.sqrt(z_des[0]**2 + z_des[1]**2)
            
            # If horizontal component is too large relative to max_tilt_angle, rescale
            # sin(max_tilt_angle) = horizontal / ||z_des||
            if horizontal_component > self.max_tilt_sin:
                # Rescale: z_des_new = [x, y, z] such that ||[x, y]|| = max_tilt_sin
                # and z = cos(max_tilt_angle)
                scale_factor = self.max_tilt_sin / horizontal_component
                z_des[0] *= scale_factor
                z_des[1] *= scale_factor
                z_des[2] = np.sqrt(1.0 - self.max_tilt_sin**2)  # cos(max_tilt_angle)
                z_des = z_des / np.linalg.norm(z_des)  # Renormalize to ensure unit vector

        # 3. Wyznaczenie pełnej macierzy rotacji zadanej R_des (Stabilna numerycznie)
        from UAV.utils import quaternion_to_rotation_matrix
        R_curr = quaternion_to_rotation_matrix(quat_curr)

        psi_des = float(target_state.get('yaw', 0.0))
        
        # Tworzymy pożądany wektor osi X drona w płaszczyźnie poziomej
        x_c = np.array([np.cos(psi_des), np.sin(psi_des), 0.0], dtype=float)

        # Bezpieczne wyznaczenie osi Y (ortogonalnej do z_des oraz x_c)
        y_des_dir = np.cross(z_des, x_c)
        y_des_norm = np.linalg.norm(y_des_dir)
        
        if y_des_norm < 1e-4:
            # W punkcie osobliwym (gdy dron leci pionowo w bok), rzutujemy na domyślną oś Y
            y_des = np.array([-np.sin(psi_des), np.cos(psi_des), 0.0], dtype=float)
        else:
            y_des = y_des_dir / y_des_norm

        # Rekonstruujemy oś X tak, aby cały układ był idealnie ortogonalny
        x_des = np.cross(y_des, z_des)
        
        # Składamy finalną macierz orientacji docelowej
        R_des = np.column_stack((x_des, y_des, z_des))

        # 3. Pętla Orientacji (Zwróć uwagę na uchyb eR - u znajomej jest R_des.T @ R)
        error_mat = R_des.T @ R_curr - R_curr.T @ R_des
        eR = 0.5 * np.array([
            error_mat[2, 1] - error_mat[1, 2],
            error_mat[0, 2] - error_mat[2, 0],
            error_mat[1, 0] - error_mat[0, 1]
        ])

        # Uchyb prędkości kątowej uwzględniający rzutowanie macierzy rotacji
        eOmega = omega_curr - R_curr.T @ R_des @ omega_des
        
        gyro_component = np.cross(omega_curr, self.inertia_tensor @ omega_curr)
        accelerating_component = self.inertia_tensor @ alpha_des
        tau = -self.KR @ eR - self.KOmega @ eOmega + gyro_component + accelerating_component

        return u1, tau
    
