import numpy as np

from UAV.utils import quaternion_to_rotation_matrix


class UAVParameters:
    """Container for the UAV physical parameters used by the controllers."""

    def __init__(self):
        self.mass = 1.0
        self.gravity = 9.81
        self.dt = 0.01
        self.inertia_tensor = np.eye(3)

    def load_from_yaml(self, data):
        """Populate parameters from a dictionary-like source."""
        if not isinstance(data, dict):
            return

        if 'mass' in data:
            self.mass = float(data['mass'])
        if 'gravity' in data:
            self.gravity = float(data['gravity'])
        if 'dt' in data:
            self.dt = float(data['dt'])

        for key in ('inertia_tensor', 'J'):
            if key in data:
                inertia = np.array(data[key], dtype=float)
                if inertia.size == 9:
                    self.inertia_tensor = inertia.reshape((3, 3))
                elif inertia.size == 3:
                    self.inertia_tensor = np.diag(inertia)
                break


class UAVState:
    """Simple state container compatible with the controller interfaces."""

    def __init__(
        self,
        position=None,
        linear_velocity=None,
        orientation=None,
        angular_velocity=None,
    ):
        self.position = np.zeros(3) if position is None else np.asarray(position, dtype=float)
        self.linear_velocity = (
            np.zeros(3) if linear_velocity is None else np.asarray(linear_velocity, dtype=float)
        )
        self.orientation = (
            np.array([1.0, 0.0, 0.0, 0.0])
            if orientation is None
            else np.asarray(orientation, dtype=float)
        )
        self.angular_velocity = (
            np.zeros(3)
            if angular_velocity is None
            else np.asarray(angular_velocity, dtype=float)
        )

    def get_state(self):
        return np.concatenate(
            [
                self.position,
                self.linear_velocity,
                self.orientation,
                self.angular_velocity,
            ]
        )

    def update_state(
        self,
        position=None,
        linear_velocity=None,
        orientation=None,
        angular_velocity=None,
    ):
        if position is not None:
            self.position = np.asarray(position, dtype=float)
        if linear_velocity is not None:
            self.linear_velocity = np.asarray(linear_velocity, dtype=float)
        if orientation is not None:
            self.orientation = np.asarray(orientation, dtype=float)
        if angular_velocity is not None:
            self.angular_velocity = np.asarray(angular_velocity, dtype=float)


class MellingerController:
    """Geometric Mellinger controller for a quadrotor UAV."""

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

    def compute_control(self, current_state, target_state):
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
