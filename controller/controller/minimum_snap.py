import numpy as np
from scipy.optimize import minimize
from controller.trajectory_generator import PolynomialTrajectory

class MinimumSnapGenerator:
    def __init__(self):
        pass

    def _compute_single_segment_H(self, T):
        """Generuje macierz kosztu Snap (8x8) dla jednego segmentu o czasie T."""
        H = np.zeros((8, 8))
        # Całkowanie kwadratu 4. pochodnej wielomianu 7. stopnia:
        # p^(4)(t) = 24*c4 + 120*c5*t + 360*c6*t^2 + 840*c7*t^3
        # Współczynniki wynikają bezpośrednio z analitycznego wyznaczenia całek wektorowych.
        H[4, 4] = 576 * T
        H[4, 5] = H[5, 4] = 1440 * T**2
        H[4, 6] = H[6, 4] = 2880 * T**3
        H[4, 7] = H[7, 4] = 5040 * T**4
        
        H[5, 5] = 4800 * T**3
        H[5, 6] = H[6, 5] = 10800 * T**4
        H[5, 7] = H[7, 5] = 20160 * T**5
        
        H[6, 6] = 25920 * T**5
        H[6, 7] = H[7, 6] = 50400 * T**6
        
        H[7, 7] = 100800 * T**7
        return H

    def _generate_derivatives_vector(self, order, t):
        """Pomocniczy wektor pochodnych dla wielomianu 7. stopnia w czasie t."""
        if order == 0: return np.array([1.0, t, t**2, t**3, t**4, t**5, t**6, t**7])
        if order == 1: return np.array([0.0, 1.0, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5, 7*t**6])
        if order == 2: return np.array([0.0, 0.0, 2.0, 6*t, 12*t**2, 20*t**3, 30*t**4, 42*t**5])
        if order == 3: return np.array([0.0, 0.0, 0.0, 6.0, 24*t, 60*t**2, 120*t**3, 210*t**4])
        if order == 4: return np.array([0.0, 0.0, 0.0, 0.0, 24.0, 120*t, 360*t**2, 840*t**3])
        return np.zeros(8)

    def generate_trajectory(self, waypoints, segment_durations):
        """
        waypoints: tablica np.ndarray o wymiarze (M, 3) zawierająca punkty X, Y, Z
        segment_durations: lista/tablica czasów trwania (M-1) segmentów
        """
        num_waypoints = len(waypoints)
        num_segments = num_waypoints - 1
        num_coefs_per_segment = 8
        total_coefs = num_segments * num_coefs_per_segment

        # 1. Zbuduj globalną macierz kosztu H (blokowo-diagonalną)
        H_global = np.zeros((total_coefs, total_coefs))
        for i, T in enumerate(segment_durations):
            H_seg = self._compute_single_segment_H(T)
            start_idx = i * num_coefs_per_segment
            H_global[start_idx:start_idx+8, start_idx:start_idx+8] = H_seg

        # Definicja funkcji celu dla solvera: 1/2 * c^T * H * c
        def cost_function(c):
            return 0.5 * np.dot(c, np.dot(H_global, c))

        # Obliczamy trajektorię niezależnie dla każdej osi (X, Y, Z)
        coefficients_output = {'x': [], 'y': [], 'z': []}

        for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
            A_eq = []
            b_eq = []

            # A. Ograniczenia pozycji na punktach węzłowych (Waypoints)
            for i in range(num_segments):
                T = segment_durations[i]
                start_idx = i * num_coefs_per_segment
                
                # Początek segmentu i musi być w punkcie waypoints[i]
                row_start = np.zeros(total_coefs)
                row_start[start_idx:start_idx+8] = self._generate_derivatives_vector(0, 0.0)
                A_eq.append(row_start)
                b_eq.append(waypoints[i, axis_idx])
                
                # Koniec segmentu i musi być w punkcie waypoints[i+1]
                row_end = np.zeros(total_coefs)
                row_end[start_idx:start_idx+8] = self._generate_derivatives_vector(0, T)
                A_eq.append(row_end)
                b_eq.append(waypoints[i+1, axis_idx])

            # B. Ograniczenia ciągłości pochodnych (Vel, Acc, Jerk, Snap) na złączeniach segmentów
            for i in range(num_segments - 1):
                T_curr = segment_durations[i]
                idx_curr = i * num_coefs_per_segment
                idx_next = (i + 1) * num_coefs_per_segment
                
                # Ciągłość dla pochodnych rzędu 1 do 4 (Prędkość, Przyspieszenie, Jerk, Snap)
                for order in range(1, 5):
                    row_continuity = np.zeros(total_coefs)
                    # pochodna na końcu obecnego segmentu minus pochodna na początku następnego ma dać 0
                    row_continuity[idx_curr:idx_curr+8] = self._generate_derivatives_vector(order, T_curr)
                    row_continuity[idx_next:idx_next+8] = -self._generate_derivatives_vector(order, 0.0)
                    A_eq.append(row_continuity)
                    b_eq.append(0.0)

            # C. Ograniczenia brzegowe (Dron startuje i kończy w bezruchu)
            # Start (segment 0, t=0): vel=0, acc=0
            for order in [1, 2]:
                row_bound = np.zeros(total_coefs)
                row_bound[0:8] = self._generate_derivatives_vector(order, 0.0)
                A_eq.append(row_bound)
                b_eq.append(0.0)
                
            # Koniec (ostatni segment, t=T_last): vel=0, acc=0
            T_last = segment_durations[-1]
            idx_last = (num_segments - 1) * num_coefs_per_segment
            for order in [1, 2]:
                row_bound = np.zeros(total_coefs)
                row_bound[idx_last:idx_last+8] = self._generate_derivatives_vector(order, T_last)
                A_eq.append(row_bound)
                b_eq.append(0.0)

            # Konwersja list ograniczeń do tablic numpy
            A_eq = np.array(A_eq)
            b_eq = np.array(b_eq)

            # Definicja ograniczeń równościowych dla scipy.optimize
            constraints = {'type': 'eq', 'fun': lambda c: np.dot(A_eq, c) - b_eq}
            
            # Punkt startowy optymalizacji (same zera)
            c0 = np.zeros(total_coefs)
            
            # Uruchomienie solvera QP
            res = minimize(cost_function, c0, method='SLSQP', constraints=constraints, options={'maxiter': 1000})
            
            if not res.success:
                raise RuntimeError(f"Optymalizacja Minimum Snap nie powiodła się dla osi {axis_name}: {res.message}")
            
            # Zapisanie wyników i przeformatowanie do macierzy (N, 8)
            coefficients_output[axis_name] = res.x.reshape((num_segments, num_coefs_per_segment))

        # Zwracamy gotowy obiekt trajektorii wielomianowej
        return PolynomialTrajectory(coefficients_output, segment_durations)