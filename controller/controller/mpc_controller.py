import numpy as np
from scipy.optimize import minimize


class MPCPositionController:
    """Simple receding-horizon MPC that outputs desired acceleration.

    Solves a small QP over accelerations to drive position to a target while
    respecting acceleration bounds. Uses scipy.optimize.SLSQP (no external QP solver).
    """

    def __init__(self, dt: float = 0.1, horizon: int = 10, a_max: float = 5.0, q_pos: float = 10.0, r_acc: float = 0.1):
        self.dt = float(dt)
        self.N = int(horizon)
        self.a_max = float(a_max)
        self.q_pos = float(q_pos)
        self.r_acc = float(r_acc)

    def solve(self, pos0: np.ndarray, vel0: np.ndarray, pos_target: np.ndarray, vel_target: np.ndarray = None, acc_ff: np.ndarray = None):
        """Solve MPC and return first-step acceleration command.

        pos0, vel0, pos_target are 3-element arrays. vel_target and acc_ff are optional.
        """
        pos0 = np.asarray(pos0, dtype=float)
        vel0 = np.asarray(vel0, dtype=float)
        pos_target = np.asarray(pos_target, dtype=float)
        if vel_target is None:
            vel_target = np.zeros(3)
        if acc_ff is None:
            acc_ff = np.zeros(3)

        N = self.N
        dt = self.dt

        # decision variable: stacked accelerations [a0x,a0y,a0z, a1x,...]
        x0 = np.tile(acc_ff, N)

        Q = self.q_pos * np.eye(3)
        R = self.r_acc * np.eye(3)

        def rollout(accs_flat):
            accs = accs_flat.reshape((N, 3))
            pos = pos0.copy()
            vel = vel0.copy()
            cost = 0.0
            for k in range(N):
                a = accs[k]
                # dynamics
                pos = pos + vel * dt + 0.5 * a * dt * dt
                vel = vel + a * dt
                # stage cost: pos error to final target + control effort around feedforward
                e = pos - pos_target
                da = a - acc_ff
                cost += e @ (Q @ e) + da @ (R @ da)
            return cost

        # bounds for accelerations per axis
        bounds = [(-self.a_max, self.a_max)] * (3 * N)

        res = minimize(rollout, x0, method='SLSQP', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-3})

        if not res.success:
            # fallback: simple PD-like acceleration
            a_fb = -2.0 * (pos0 - pos_target) - 1.0 * (vel0 - vel_target)
            a_fb = np.clip(a_fb, -self.a_max, self.a_max)
            return a_fb

        a_seq = res.x.reshape((N, 3))
        return a_seq[0]
