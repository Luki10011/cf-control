
from UAV.uav_state import UAVParameters, UAVState
import numpy as np

class MellingerController:

    def __init__(self, 
                 parameters : UAVParameters,
                 Kp : np.ndarray,
                 Kv : np.ndarray,
                 KR : np.ndarray,
                 KOmega : np.ndarray):
        
        self.mass = parameters.mass
        self.inertia_tensor = parameters.inertia_tensor
        self.gravity = parameters.gravity
        self.Kp = Kp
        self.Kv = Kv
        self.KR = KR 
        self.KOmega = KOmega

    def compute_Control(self,
                        current_state : UAVState,
                        target_state : UAVState
                        ):

    
