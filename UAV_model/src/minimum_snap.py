import numpy as np

class MinimumSnapTrajectory:
    def __init__(self):
        self.coefficients = None
        self.T = 0.0
        self.p_start = None
        self.p_end = None

    def generate_trajectory(self, p_start, p_end, duration):
        """
        Generuje współczynniki wielomianu 7. stopnia dla osi X, Y, Z
        p_start, p_end: np.array([x, y, z])
        duration: czas przelotu w sekundach (T)
        """
        self.T = duration
        self.p_start = p_start
        self.p_end = p_end
        self.coefficients = {}

        # Macierz warunków dla wielomianu: p(t), v(t), a(t), j(t) w t=0 oraz t=T
        # Wiersze odpowiadają równaniom dla t=0 (p,v,a,j) i t=T (p,v,a,j)
        T = duration
        A = np.array([
            [1, 0, 0,   0,   0,    0,     0,      0],          # p(0)
            [0, 1, 0,   0,   0,    0,     0,      0],          # v(0)
            [0, 0, 2,   0,   0,    0,     0,      0],          # a(0)
            [0, 0, 0,   6,   0,    0,     0,      0],          # j(0)
            [1, T, T**2, T**3, T**4, T**5, T**6,  T**7],       # p(T)
            [0, 1, 2*T,  3*T**2, 4*T**3, 5*T**4, 6*T**5, 7*T**6], # v(T)
            [0, 0, 2,    6*T,    12*T**2,20*T**3,30*T**4,42*T**5],# a(T)
            [0, 0, 0,    6,      24*T,   60*T**2,120*T**3,210*T**4]# j(T)
        ])

        # Liczymy współczynniki niezależnie dla X, Y, Z
        for axis in range(3):
            # Warunki brzegowe: na starcie i końcu prędkość, acc i jerk są zero
            B = np.array([p_start[axis], 0.0, 0.0, 0.0,  # t = 0
                          p_end[axis],   0.0, 0.0, 0.0]) # t = T
            
            # Rozwiązujemy układ równań A * c = B
            c = np.linalg.solve(A, B)
            self.coefficients[axis] = c

    def evaluate(self, t):
        """
        Zwraca pozycję, prędkość, przyspieszenie, jerk i snap w danym momencie czasu t
        """
        if t < 0: t = 0.0
        if t > self.T: t = self.T

        pos = np.zeros(3)
        vel = np.zeros(3)
        acc = np.zeros(3)
        jerk = np.zeros(3)
        snap = np.zeros(3)

        for axis in range(3):
            c = self.coefficients[axis]
            
            # POPRAWIONE: c[5]*t**5 zamiast c[5]*t**4
            pos[axis] = c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5 + c[6]*t**6 + c[7]*t**7
            
            # Prędkość
            vel[axis] = c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4 + 6*c[6]*t**5 + 7*c[7]*t**6
            
            # Przyspieszenie
            acc[axis] = 2*c[2] + 6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3 + 30*c[6]*t**4 + 42*c[7]*t**5
            
            # Dodatkowo: Jerk (trzecia pochodna)
            jerk[axis] = 6*c[3] + 24*c[4]*t + 60*c[5]*t**2 + 120*c[6]*t**3 + 210*c[7]*t**4
            
            # Dodatkowo: Snap (czwarta pochodna)
            snap[axis] = 24*c[4] + 120*c[5]*t + 360*c[6]*t**2 + 840*c[7]*t**3

        return pos, vel, acc, jerk, snap