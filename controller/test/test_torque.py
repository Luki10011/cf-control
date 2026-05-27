import numpy as np
from controller.mellinger_controller import MellingerController
from UAV.uav_state import UAVParameters, UAVState

# Build parameters
params = UAVParameters()
params.mass = 0.0282
params.gravity = 9.81
params.inertia_tensor = np.diag([21.09144e-6, 21.57533e-6, 35.86538e-5])

# Gains (read from config -- match what node uses)
Kp = np.diag([1.4,1.4,1.0])
Kv = np.diag([0.45,0.45,0.55])
KR = np.diag([1.0e-4,1.0e-4,2.5e-5])
KOmega = np.diag([2.0e-4,2.0e-4,8.0e-5])

mc = MellingerController(params,Kp,Kv,KR,KOmega, max_tilt_angle=np.deg2rad(35.0))

# create current state nearly level
state = UAVState()
state.position = np.array([1.8,1.6,2.0])
state.linear_velocity = np.array([0.5, -0.3, 0.2])
state.orientation = np.array([1.0,0.0,0.0,0.0])
state.angular_velocity = np.zeros(3)

# aggressive target far away
target_state = {
    'pos': np.array([3.0,1.5,1.2]),
    'vel': np.zeros(3),
    'acc': np.array([0.0,0.0,0.0]),
    'omega': np.zeros(3),
    'w_dot': np.zeros(3),
    'yaw': 0.0
}

u1, tau = mc.compute_control(state, target_state)
print('raw u1, tau:', u1, tau)
# emulate clipping in node
max_torque = np.array([0.02,0.02,0.02])
clamped = np.clip(tau, -max_torque, max_torque)
print('clamped tau:', clamped)
