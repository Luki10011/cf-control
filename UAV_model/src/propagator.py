import numpy as np
from constants import g
from uav import UAVModel
from utils import multiply_quaternions, multiply_vector_by_quaternion


class UAVPropagator:
    def __init__(self, timestep, uavModel):
        self.model: UAVModel = uavModel
        self.ts = timestep

    def state_equation(self, control):
        T, tx, ty, tz = control

        # Force and torque
        T_vec = np.array([0.0, 0.0, T])
        tau = np.array([tx, ty, tz])

        # State unpack
        r_dot = self.model.linear_velocity

        v_dot = -np.array([0, 0, g]) + (1 / self.model.mass) * multiply_vector_by_quaternion(
            self.model.quternions_orientation, T_vec
        )

        q = self.model.quternions_orientation
        omega = self.model.angular_velocity

        omega_quat = np.array([0.0, omega[0] / 2, omega[1] / 2, omega[2] / 2])

        q_dot = multiply_quaternions(q, omega_quat)

        omega_dot = np.linalg.inv(self.model.inertia_tensor) @ (
            tau - np.cross(omega, self.model.inertia_tensor @ omega)
        )

        return np.concatenate([r_dot, v_dot, q_dot, omega_dot])

    def rk4(self, state_vector, control):

        dt = self.ts

        def set_state(x):
            self.model.position = x[0:3]
            self.model.linear_velocity = x[3:6]
            self.model.quternions_orientation = x[6:10]
            self.model.angular_velocity = x[10:13]

        # RK4 steps
        set_state(state_vector)
        k1 = self.state_equation(control)

        set_state(state_vector + 0.5 * dt * k1)
        k2 = self.state_equation(control)

        set_state(state_vector + 0.5 * dt * k2)
        k3 = self.state_equation(control)

        set_state(state_vector + dt * k3)
        k4 = self.state_equation(control)

        next_state = state_vector + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        next_state[6:10] /= np.linalg.norm(next_state[6:10])

        return next_state
