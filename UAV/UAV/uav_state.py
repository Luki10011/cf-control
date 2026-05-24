
import numpy as np

class UAVParameters:
    def __init__(self):

        # Some default values 
        self.mass = 1.0 
        self.inertia_tensor = np.eye(3)
        self.gravity = 9.81
        self.dt = 0.01

    def load_from_yaml(self, cfg):
        
        self.mass = cfg.get('mass', self.mass)
        self.gravity = cfg.get('gravity', self.gravity)
        inertia_tensor = list(cfg.get('inertia_tensor', self.inertia_tensor))
        self.dt = cfg.get('dt', self.dt)
        self.inertia_tensor = np.array(inertia_tensor).reshape((3, 3))
        


class UAVState():
    
    def __init__(self):
        # Default state values
        self.position = np.zeros(3)
        self.linear_velocity = np.zeros(3)
        self.orientation = np.array([1, 0, 0, 0])
        self.angular_velocity = np.zeros(3)   

    def update_state(self, position, linear_velocity, orientation, angular_velocity):
        self.position = position
        self.linear_velocity = linear_velocity
        self.orientation = orientation
        self.normalize_quaternion()
        self.angular_velocity = angular_velocity

    def normalize_quaternion(self):
        norm = np.linalg.norm(self.orientation)
        self.orientation /= norm
        
    def get_state(self):
        return np.concatenate([
            self.position,
            self.linear_velocity,
            self.orientation,
            self.angular_velocity
        ])
    
