import numpy as np
from constants import INERTIA_TENSOR, MASS, g
from utils import multiply_quaternions, multiply_vector_by_quaternion


class UAVModel:
    def __init__(self, initial_conditions: np.ndarray):
        self.mass = MASS
        self.inertia_tensor = INERTIA_TENSOR
        self.position = initial_conditions[0:3]
        self.linear_velocity = initial_conditions[3:6]
        self.quternions_orientation = initial_conditions[6:10]
        self.angular_velocity = initial_conditions[10:13]


class UAVPropagator:
    def __init__(self, timestep, uavModel):
        self.model = uavModel
        self.ts = timestep

    def state_equation(self, control):
        T, tx, ty, tz = control

        # Force and torque
        T_vec = np.array([0.0, 0.0, T])
        tau = np.array([tx, ty, tz])

        # State unpack
        r_dot = self.model.linear_velocity

        v_dot = -g + (1 / MASS) * multiply_vector_by_quaternion(
            self.model.quternions_orientation, T_vec
        )

        q = self.model.quternions_orientation
        omega = self.model.angular_velocity

        omega_quat = np.array([0.0, omega[0] / 2, omega[1] / 2, omega[2] / 2])

        q_dot = multiply_quaternions(q, omega_quat)

        omega_dot = np.linalg.inv(INERTIA_TENSOR) @ (tau - np.cross(omega, INERTIA_TENSOR @ omega))

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

        # Normalize quaternion
        next_state[6:10] /= np.linalg.norm(next_state[6:10])

        return next_state


def main():
    initial_state = np.zeros(13)
    initial_state[2] = 10
    initial_state[6] = 1.0

    dt = 0.01
    model = UAVModel(initial_state)
    propagator = UAVPropagator(dt, model)

    state = initial_state.copy()

    control = [10, 0.0, 0.0, 0.0]

    for i in range(100):
        state = propagator.rk4(state, control)
        print(f'Step {i}: position = {state[0:3]}')

    print('Simulation complete.')


if __name__ == '__main__':
    main()
