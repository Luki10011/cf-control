import numpy as np
from scipy.optimize import minimize
from controller.trajectory_generator import PolynomialTrajectory

class MinimumSnapGenerator:
    def __init__(self):
        pass

    def _compute_single_segment_H(self, T):
        """Znormalizowana macierz kosztu Snap (8x8) dla jednego segmentu."""
        H = np.zeros((8, 8))
        # Obliczamy analityczne stałe dla znormalizowanego czasu tau in [0, 1] (czyli T=1.0)
        H[4, 4] = 576
        H[4, 5] = H[5, 4] = 1440
        H[4, 6] = H[6, 4] = 2880
        H[4, 7] = H[7, 4] = 5040
        
        H[5, 5] = 4800
        H[5, 6] = H[6, 5] = 10800
        H[5, 7] = H[7, 5] = 20160
        
        H[6, 6] = 25920
        H[6, 7] = H[7, 6] = 50400
        
        H[7, 7] = 100800
        
        # Skalowanie macierzy - klucz do stabilności numerycznej
        scale_factor = 1.0 / (T**7)
        return H * scale_factor

    def _generate_derivatives_vector(self, order, tau, T):
        """
        Znormalizowany wektor pochodnych.
        tau: znormalizowany czas w [0, 1] (0.0 dla startu, 1.0 dla końca)
        T: rzeczywisty czas trwania segmentu (do skalowania)
        """
        vec = np.zeros(8)
        if order == 0: vec = np.array([1.0, tau, tau**2, tau**3, tau**4, tau**5, tau**6, tau**7])
        elif order == 1: vec = np.array([0.0, 1.0, 2*tau, 3*tau**2, 4*tau**3, 5*tau**4, 6*tau**5, 7*tau**6])
        elif order == 2: vec = np.array([0.0, 0.0, 2.0, 6*tau, 12*tau**2, 20*tau**3, 30*tau**4, 42*tau**5])
        elif order == 3: vec = np.array([0.0, 0.0, 0.0, 6.0, 24*tau, 60*tau**2, 120*tau**3, 210*tau**4])
        elif order == 4: vec = np.array([0.0, 0.0, 0.0, 0.0, 24.0, 120*tau, 360*tau**2, 840*tau**3])
        
        # Reguła łańcuchowa dla czasu rzeczywistego
        scale = 1.0 / (T**order)
        return vec * scale

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
                row_start[start_idx:start_idx+8] = self._generate_derivatives_vector(0, 0.0, T)
                A_eq.append(row_start)
                b_eq.append(waypoints[i, axis_idx])
                
                # Koniec segmentu i musi być w punkcie waypoints[i+1]
                row_end = np.zeros(total_coefs)
                row_end[start_idx:start_idx+8] = self._generate_derivatives_vector(0, 1.0, T)
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
                    row_continuity[idx_curr:idx_curr+8] = self._generate_derivatives_vector(order, 1.0, T_curr)
                    row_continuity[idx_next:idx_next+8] = -self._generate_derivatives_vector(order, 0.0, segment_durations[i+1])
                    A_eq.append(row_continuity)
                    b_eq.append(0.0)

            # C. Ograniczenia brzegowe (Dron startuje i kończy w bezruchu)
            # Start (segment 0, t=0): vel=0, acc=0
            for order in [1, 2]:
                row_bound = np.zeros(total_coefs)
                row_bound[0:8] = self._generate_derivatives_vector(order, 0.0, segment_durations[0])
                A_eq.append(row_bound)
                b_eq.append(0.0)
                
            # Koniec (ostatni segment, t=T_last): vel=0, acc=0
            T_last = segment_durations[-1]
            idx_last = (num_segments - 1) * num_coefs_per_segment
            for order in [1, 2]:
                row_bound = np.zeros(total_coefs)
                row_bound[idx_last:idx_last+8] = self._generate_derivatives_vector(order, 1.0, T_last)
                A_eq.append(row_bound)
                b_eq.append(0.0)

            # Konwersja list ograniczeń do tablic numpy
            A_eq = np.array(A_eq)
            b_eq = np.array(b_eq)

            # Definicja ograniczeń równościowych dla scipy.optimize
            num_constraints = A_eq.shape[0]

            # Zbudowanie macierzy KKT (lewa strona układu równań)
            # Używamy np.block do stworzenia macierzy blokowej:
            # [ H_global   A_eq.T ]
            # [ A_eq       0      ]
            KKT_left = np.block([
                [H_global, A_eq.T],
                [A_eq, np.zeros((num_constraints, num_constraints))]
            ])

            # Zbudowanie wektora prawej strony:
            # [ 0    ] (rozmiaru total_coefs)
            # [ b_eq ] (rozmiaru num_constraints)
            KKT_right = np.concatenate([np.zeros(total_coefs), b_eq])

            try:
                # Rozwiązanie układu równań liniowych
                solution = np.linalg.solve(KKT_left, KKT_right)
                
                # Wyciągnięcie tylko współczynników 'c' (pierwsze total_coefs elementów)
                c_optimal = solution[:total_coefs]
                
                # Zapisanie wyników i przeformatowanie do macierzy (N, 8)
                coefficients_output[axis_name] = c_optimal.reshape((num_segments, num_coefs_per_segment))
                
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"Rozwiązanie Minimum Snap nie powiodło się dla osi {axis_name}. Macierz KKT jest osobliwa. {e}")
            
            # if not res.success:
            #     raise RuntimeError(f"Optymalizacja Minimum Snap nie powiodła się dla osi {axis_name}: {res.message}")
            
            # Zapisanie wyników i przeformatowanie do macierzy (N, 8)
            # coefficients_output[axis_name] = res.x.reshape((num_segments, num_coefs_per_segment))

        # Zwracamy gotowy obiekt trajektorii wielomianowej
        return PolynomialTrajectory(coefficients_output, segment_durations)