

from UAV.uav_state import UAVParameters
from UAV.utils import rotation_matrix_to_quaternion
import numpy as np

import numpy as np

def calculate_state_from_flat_inputs(
        position : np.ndarray,
        linear_velocity : np.ndarray,
        acc : np.ndarray,
        jerk : np.ndarray,
        snap : np.ndarray,
        yaw,
        yaw_rate,
        yaw_acc,
        parameters : UAVParameters
    ):
        p = position
        v = linear_velocity
        m = parameters.mass
        J = parameters.inertia_tensor
        g = parameters.gravity 


        a_total = acc + np.array([0.0, 0.0, g])
        thrust_val = m * np.linalg.norm(a_total)
        
        if thrust_val < 1e-6:
            z_B = np.array([0.0, 0.0, 1.0])
        else:
            z_B = a_total / np.linalg.norm(a_total)

        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        
        y_B_direction = np.cross(z_B, x_c)
        y_B = y_B_direction / np.linalg.norm(y_B_direction)
        x_B = np.cross(y_B, z_B)

        R_mat = np.column_stack((x_B, y_B, z_B))
        q = rotation_matrix_to_quaternion(R_mat)

        w_x = - (m / thrust_val) * np.dot(jerk, y_B)
        w_y = (m / thrust_val) * np.dot(jerk, x_B)
        
        z_W = np.array([0.0, 0.0, 1.0]) # Globalna oś Z
        w_z = yaw_rate * np.dot(z_W, z_B)

        w = np.array([w_x, w_y, w_z])

        w_dot_x = -((m / thrust_val) * snap[1] + 2 * (m / thrust_val) * jerk[2] * w_x - w_y * w_z)
        w_dot_y = (m / thrust_val) * snap[0] - 2 * (m / thrust_val) * jerk[2] * w_y - w_x * w_z
        
        w_dot_z = yaw_acc * np.dot(z_W, z_B)
        
        w_dot = np.array([w_dot_x, w_dot_y, w_dot_z])

        tau = J @ w_dot + np.cross(w, J @ w)

        return {
            'pos': p,
            'vel': v,
            'quat': q, 
            'omega': w,
            'thrust': thrust_val,
            'torque': tau
        }