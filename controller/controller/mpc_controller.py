import numpy as np
from scipy.optimize import minimize

class MPCPositionController:
    def __init__(self, dt=0.1, horizon=20, a_max=0.01, q_pos=3e-8, q_vel=3e-3, r_acc=2):
        self.dt = float(dt)
        self.N = int(horizon)
        self.a_max = float(a_max)
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.r_acc = float(r_acc)
        self.last_solution = None # Dla warm startu

    def solve(self, pos0, vel0, pos_target, vel_target=None, acc_ff=None):
        pos0 = np.asarray(pos0, dtype=float)
        vel0 = np.asarray(vel0, dtype=float)
        pos_target = np.asarray(pos_target, dtype=float)
        vel_target = np.zeros(3) if vel_target is None else np.asarray(vel_target, dtype=float)
        acc_ff = np.zeros(3) if acc_ff is None else np.asarray(acc_ff, dtype=float)

        # Warm start
        if self.last_solution is not None:
            x0 = np.concatenate([self.last_solution[3:], self.last_solution[-3:]])
        else:
            x0 = np.tile(acc_ff, self.N)

        Q = self.q_pos * np.eye(3)
        Qv = self.q_vel * np.eye(3)
        R = self.r_acc * np.eye(3)

        def rollout(accs_flat):
            accs = accs_flat.reshape((self.N, 3))
            p, v, cost = pos0.copy(), vel0.copy(), 0.0
            for k in range(self.N):
                a = accs[k]
                p = p + v * self.dt + 0.5 * a * self.dt**2
                v = v + a * self.dt
                
                ep = p - pos_target
                ev = v - vel_target
                da = a - acc_ff
                cost += ep @ (Q @ ep) + ev @ (Qv @ ev) + da @ (R @ da)
            return cost

        bounds = [(-self.a_max, self.a_max)] * (3 * self.N)
        res = minimize(rollout, x0, method='SLSQP', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-4})

        if res.success:
            self.last_solution = res.x
            return res.x[0:3]
        else:
            return -2.0 * (pos0 - pos_target) - 1.0 * (vel0 - vel_target)