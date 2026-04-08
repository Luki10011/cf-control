import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from constants import g
from model import UAVModel
from propagator import UAVPropagator


class Trajectory:
    def __init__(self, type, R, initial_conditions, ts):
        self.type = type
        self.R = R
        self.model = UAVModel(initial_conditions)
        self.propagator = UAVPropagator(ts, self.model)

    def flat_outputs(self, tk):
        if self.type == 'lemniscate':
            t = np.arange(0, tk, self.propagator.ts).astype(np.float32)
            self.x = self.R * np.cos(t) / (1 + np.sin(t) ** 2)
            self.y = self.R * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2)
            self.z = self.model.position[2]
            self.yaw = np.tan(self.y / self.x)

        axis_len = max(max(self.x), max(self.y)) + 10

        plt.figure()
        plt.plot(self.x, self.y)
        plt.arrow(0, 0, axis_len, 0, head_width=2, head_length=4, fc='r', ec='r')

        plt.arrow(0, 0, 0, axis_len, head_width=2, head_length=4, fc='g', ec='g')
        plt.savefig('test.jpg')

    def calculate_state_from_flat_inputs(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        acc: np.ndarray,
        jerk: np.ndarray,
        snap: np.ndarray,
        yaw,
        yaw_rate,
        yaw_acc,
    ):
        p = pos
        v = vel
        m = self.model.mass
        J = self.model.inertia_tensor
        ag = np.array([[acc[0], acc[1], acc[2] + g]]).T
        T = m * np.linalg.norm(ag)

        z_B = ag / np.linalg.norm(ag)
        x_c = np.array([[np.cos(yaw), np.sin(yaw), 0]]).T

        y_B = np.cross(z_B, x_c) / (np.linalg.norm(np.cross(z_B, x_c)))
        x_B = np.cross(y_B, z_B)

        R = np.column_stack((x_B, y_B, z_B))

        h_omega = (m / T) * (jerk - (np.dot(np.dot(jerk, z_B), z_B)))

        z_W = R @ z_B

        w_x = -np.dot(h_omega, y_B)
        w_y = np.dot(h_omega, x_B)
        w_z = yaw_rate * np.dot(z_W, z_B)

        w = np.array([[w_x, w_y, w_z]]).T

        w_dot_x = (m / T) * snap[0] - 2 * (m / T) * snap[0] * w_z - w_x * w_z
        w_dot_y = (m / T) * snap[1] + 2 * (m / T) * snap[0] * w_z - w_y * w_z
        w_dot_z = yaw_acc * np.dot(z_W, z_B)

        w_dot = np.array([[w_dot_x, w_dot_y, w_dot_z]]).T

        tau = J @ w_dot + np.cross(w, np.dot(J, w))

        return (
            p,
            v,
        )


def main():
    initial_state = np.zeros(13)
    initial_state[0:3] = [10, 10, 10]
    initial_state[6] = 1

    trajectory = Trajectory('lemniscate', 100, initial_state, 0.01)
    trajectory.flat_outputs(2 * 3.41)


if __name__ == '__main__':
    main()
