
from UAV.uav_state import UAVParameters, UAVState
import numpy as np


class MellingerController:
    """Geometric Mellinger controller adapted to pure-NumPy mathematical safety."""

    def __init__(
        self,
        parameters: UAVParameters,
        Kp: np.ndarray,
        Kv: np.ndarray,
        KR: np.ndarray,
        KOmega: np.ndarray,
        max_tilt_angle: float = np.deg2rad(45.0),
    ):
        self.mass = parameters.mass
        self.inertia_tensor = parameters.inertia_tensor
        self.gravity = parameters.gravity
        self.Kp = np.asarray(Kp, dtype=float)
        self.Kv = np.asarray(Kv, dtype=float)
        self.KR = np.asarray(KR, dtype=float)
        self.KOmega = np.asarray(KOmega, dtype=float)
        self.max_tilt_angle = max_tilt_angle
        self.max_tilt_sin = np.sin(max_tilt_angle)

    def _vee(self, S: np.ndarray) -> np.ndarray:
        """Extract the axial vector from a skew-symmetric matrix (inverse of skew)."""
        return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)

    def compute_control(self, current_state: UAVState, target_state):
        """
        Udoskonalona metoda compute_control z poprawnym mapowaniem SO(3) 
        oraz pełnym feedforwardem z zewnętrznego szablonu.
        """
        # 1. Pobranie i rzutowanie stanów na tablice numpy
        pos_curr = np.asarray(current_state.position, dtype=float)
        vel_curr = np.asarray(current_state.linear_velocity, dtype=float)
        quat_curr = np.asarray(current_state.orientation, dtype=float)
        omega_curr = np.asarray(current_state.angular_velocity, dtype=float)
        pos_des = np.asarray(target_state['pos'], dtype=float)
        vel_des = np.asarray(target_state['vel'], dtype=float)
        acc_des = np.asarray(target_state.get('acc', np.zeros(3)), dtype=float)
        omega_des = np.asarray(target_state.get('omega', np.zeros(3)), dtype=float)
        alpha_des = np.asarray(target_state.get('w_dot', np.zeros(3)), dtype=float)
        

        g_vec = np.array([0.0, 0.0, self.gravity], dtype=float)

        # 2. Pętla Pozycji (Wyznaczenie siły żądanej F_des)
        ep = pos_curr - pos_des
        ev = vel_curr - vel_des
        
        # Wzór: F_des = -Kp*ep - Kv*ev + m*g*e3 + m*acc_des
        F_des = -self.Kp @ ep - self.Kv @ ev + self.mass * g_vec + self.mass * acc_des

        from UAV.utils import quaternion_to_rotation_matrix
        R_curr = quaternion_to_rotation_matrix(quat_curr)
        z_B = R_curr[:, 2]
        
        # Ciąg wypadkowy to rzut siły na aktualną oś pionową drona
        u1 = float(np.dot(F_des, z_B))

        # 3. Rekonstrukcja pożądanej macierzy rotacji R_des
        f_des_norm = float(np.linalg.norm(F_des))
        if f_des_norm < 1e-6:
            z_des = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            z_des = F_des / f_des_norm

        # Zabezpieczenie przed ekstremalnym kątem pochylenia (Tilt Limit)
        horizontal_component = np.sqrt(z_des[0]**2 + z_des[1]**2)
        if horizontal_component > self.max_tilt_sin:
            scale_factor = self.max_tilt_sin / horizontal_component
            z_des[0] *= scale_factor
            z_des[1] *= scale_factor
            z_des[2] = np.sqrt(1.0 - self.max_tilt_sin**2)
            z_des = z_des / np.linalg.norm(z_des)

        # Pobranie Yaw bezpośrednio w radianach z target_state
        psi_des = float(target_state.get('yaw'))
        
        if psi_des > 2.0 * np.pi or psi_des < -2.0 * np.pi:
            # Zabezpieczenie: jeśli Twój planner podaje yaw w stopniach, konwertujemy na radiany
            psi_des = np.deg2rad(psi_des)

        # Tworzenie rzutu osi X na płaszczyznę poziomą na podstawie zadanego Yaw
        x_c = np.array([np.cos(psi_des), np.sin(psi_des), 0.0], dtype=float)

        # Budowa ortogonalnej bazy R_des
        y_des_dir = np.cross(z_des, x_c)
        y_des_norm = float(np.linalg.norm(y_des_dir))
        
        if y_des_norm < 1e-6:
            # Przypadek osobliwy - rzut awaryjny na czystą oś Y z Yaw
            y_des = np.array([-np.sin(psi_des), np.cos(psi_des), 0.0], dtype=float)
        else:
            y_des = y_des_dir / y_des_norm

        x_des = np.cross(y_des, z_des)
        R_des = np.column_stack((x_des, y_des, z_des))

        # 4. Pętla Orientacji (Zastosowanie eleganckiego i poprawnego vee-map)
        eR_mat = R_des.T @ R_curr - R_curr.T @ R_des
        eR = 0.5 * self._vee(eR_mat)

        # Uchyb prędkości kątowej
        eOmega = omega_curr - R_curr.T @ R_des @ omega_des
        
        # Kompensacja żyroskopowa i feedforward momentów obrotowych
        gyro_component = np.cross(omega_curr, self.inertia_tensor @ omega_curr)
        accelerating_component = self.inertia_tensor @ alpha_des
        
        # Wyznaczenie wektora momentów (Torque)
        tau = -self.KR @ eR - self.KOmega @ eOmega + gyro_component + accelerating_component

        # 5. Bezpieczne nasycenie momentów (zapobiega dzikiemu koziołkowaniu drona)
        tau[0] = np.clip(tau[0], -0.011, 0.011)
        tau[1] = np.clip(tau[1], -0.011, 0.011)
        tau[2] = np.clip(tau[2], -0.015, 0.015)  # Poluzowane Yaw dla lepszej responsywności

        return u1, tau
    
