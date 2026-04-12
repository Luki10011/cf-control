import numpy as np


class UAVModel:
    def __init__(self, mass, inertia_tensor, initial_conditions: np.ndarray):
        self.mass = mass
        self.inertia_tensor = inertia_tensor
        self.position = initial_conditions[0:3]
        self.linear_velocity = initial_conditions[3:6]
        self.quternions_orientation = initial_conditions[6:10]
        self.angular_velocity = initial_conditions[10:13]
