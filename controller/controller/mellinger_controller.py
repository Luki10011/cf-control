
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
    ):
        self.mass = parameters.mass
        self.inertia_tensor = parameters.inertia_tensor
        self.gravity = parameters.gravity
        self.Kp = np.asarray(Kp, dtype=float)
        self.Kv = np.asarray(Kv, dtype=float)
        self.KR = np.asarray(KR, dtype=float)
        self.KOmega = np.asarray(KOmega, dtype=float)

    def compute_control(self, current_state: UAVState, target_state):
        """Compute collective thrust and body torques for the current state."""
        pos_curr = np.asarray(current_state.position, dtype=float)
        vel_curr = np.asarray(current_state.linear_velocity, dtype=float)
        quat_curr = np.asarray(current_state.orientation, dtype=float)
        omega_curr = np.asarray(current_state.angular_velocity, dtype=float)

        pos_des = np.asarray(target_state['pos'], dtype=float)
        vel_des = np.asarray(target_state['vel'], dtype=float)
        quat_des = np.asarray(target_state.get('quat', np.array([1.0, 0.0, 0.0, 0.0])), dtype=float)
        omega_des = np.asarray(target_state.get('omega', np.zeros(3)), dtype=float)
        acc_des = np.asarray(target_state.get('acc', np.zeros(3)), dtype=float)
        alpha_des = np.asarray(
            target_state.get('w_dot', target_state.get('alpha', np.zeros(3))),
            dtype=float,
        )

        ep = pos_curr - pos_des
        ev = vel_curr - vel_des
        g_vec = np.array([0.0, 0.0, self.gravity])
        F_des = -self.Kp @ ep - self.Kv @ ev + self.mass * g_vec + self.mass * acc_des

        from UAV.utils import quaternion_to_rotation_matrix

        R_curr = quaternion_to_rotation_matrix(quat_curr)
        z_B = R_curr[:, 2]
        u1 = float(np.dot(F_des, z_B))

        if np.linalg.norm(F_des) < 1e-6:
            z_des = np.array([0.0, 0.0, 1.0])
        else:
            z_des = F_des / np.linalg.norm(F_des)

        R_des_flat = quaternion_to_rotation_matrix(quat_des)
        x_c = R_des_flat[:, 0]

        y_des_dir = np.cross(z_des, x_c)
        if np.linalg.norm(y_des_dir) < 1e-6:
            y_des = np.array([0.0, 1.0, 0.0])
        else:
            y_des = y_des_dir / np.linalg.norm(y_des_dir)

        x_des = np.cross(y_des, z_des)
        R_des = np.column_stack((x_des, y_des, z_des))

        error_mat = 0.5 * (R_des.T @ R_curr - R_curr.T @ R_des)
        eR = np.array(
            [
                error_mat[2, 1] - error_mat[1, 2],
                error_mat[0, 2] - error_mat[2, 0],
                error_mat[1, 0] - error_mat[0, 1],
            ]
        ) * 0.5

        eOmega = omega_curr - omega_des
        gyro_component = np.cross(omega_curr, self.inertia_tensor @ omega_curr)
        accelerating_component = self.inertia_tensor @ alpha_des
        tau = -self.KR @ eR - self.KOmega @ eOmega + gyro_component + accelerating_component

        return u1, tau

    def compute_Control(self, current_state: UAVState, target_state):
        """Backward-compatible wrapper for the original method name."""
        return self.compute_control(current_state, target_state)
