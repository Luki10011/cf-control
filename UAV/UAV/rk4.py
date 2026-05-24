
import numpy as np
from UAV.utils import multiply_vector_by_quaternion, multiply_quaternions


class RK4Propagator():
    def __init__(self, uav_parameters):
        self.mass = uav_parameters['mass']
        self.inertia_tensor = uav_parameters['inertia']
        self.gravity = uav_parameters['gravity']
        self.dt = uav_parameters['dt']

    def dynamics(self, state, control_inputs):
        
        # Extracting data from state and control inputs
        r = state[0:3]
        v = state[3:6]
        q = state[6:10]
        omega = state[10:13]

        T, tx, ty, tz = control_inputs

        tau = np.array([tx, ty, tz])


        # calulating derivatives

        r_dot = v

        v_dot = -np.array([0, 0, self.gravity]) + (1 / self.mass) * multiply_vector_by_quaternion(
            q, np.array([0.0, 0.0, T])
        )


        omega_quat = np.array([0.0, omega[0] / 2, omega[1] / 2, omega[2] / 2])

        q_dot = multiply_quaternions(q, omega_quat)

        omega_dot = np.linalg.inv(self.inertia_tensor) @ (
            tau - np.cross(omega, self.inertia_tensor @ omega)
        )

        return np.concatenate([r_dot, v_dot, q_dot, omega_dot])


    def propagate(self, state, control_inputs):

        k1 = self.dynamics(state, control_inputs)
        k2 = self.dynamics(state + 0.5 * self.dt * k1, control_inputs)
        k3 = self.dynamics(state + 0.5 * self.dt * k2, control_inputs)
        k4 = self.dynamics(state + self.dt * k3, control_inputs)

        next_state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return next_state