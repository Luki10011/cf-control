import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from model import UAVModel
from propagator import UAVPropagator


class Trajectory:
    def __init__(self, type, R, initial_conditions, ts):
        self.type = type
        self.R = R
        self.model = UAVModel(initial_conditions)
        self.propagator = UAVPropagator(ts, self.model)

    def calculate_state_based_on_flat(self):
        pass

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


def main():
    initial_state = np.zeros(13)
    initial_state[0:3] = [10, 10, 10]
    initial_state[6] = 1

    trajectory = Trajectory('lemniscate', 100, initial_state, 0.01)
    trajectory.flat_outputs(2 * 3.41)


if __name__ == '__main__':
    main()
