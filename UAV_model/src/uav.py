import numpy as np
from constants import INERTIA_TENSOR, MASS


class UAVModel:
    def __init__(self, initial_conditions: np.ndarray):
        self.mass = MASS
        self.inertia_tensor = INERTIA_TENSOR
        self.position = initial_conditions[0:3]
        self.linear_velocity = initial_conditions[3:6]
        self.quternions_orientation = initial_conditions[6:10]
        self.angular_velocity = initial_conditions[10:13]
