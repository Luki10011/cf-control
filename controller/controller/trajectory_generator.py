import numpy as np

class PolynomialTrajectory:
    def __init__(self, coefficients, segment_durations, yaw_coefficients=None):
        """
        coefficients: słownik {'x': macierz, 'y': macierz, 'z': macierz}
                      gdzie każda macierz ma wymiar (N, 8) dla N segmentów i wielomianów 7. stopnia.
        segment_durations: lista lub tablica o długości N zawierająca czasy trwania każdego segmentu.
        yaw_coefficients: opcjonalna macierz (N, 4) dla kąta Yaw (zazwyczaj wystarczy wielomian 3. stopnia)
        """
        self.coefs = coefficients
        self.durations = np.array(segment_durations)
        self.num_segments = len(segment_durations)
        
        # Obliczamy globalne czasy startowe każdego segmentu (skumulowana suma)
        # Np. dla czasów [2.0, 3.0, 1.5] starty to [0.0, 2.0, 5.0]
        self.start_times = np.insert(np.cumsum(self.durations)[:-1], 0, 0.0)
        self.total_time = np.sum(self.durations)
        
        # Jeśli nie podano współczynników dla Yaw, dron domyślnie patrzy przed siebie (0.0)
        if yaw_coefficients is not None:
            self.yaw_coefs = yaw_coefficients
        else:
            self.yaw_coefs = np.zeros((self.num_segments, 4)) # Wielomian 3. stopnia dla Yaw

    def _get_segment_index_and_local_time(self, t):
        """Znajduje indeks segmentu i czas lokalny dla danego czasu globalnego t."""
        # Zabezpieczenie przed wyjściem poza czas trajektorii
        if t < 0.0:
            return 0, 0.0
        if t >= self.total_time:
            return self.num_segments - 1, self.durations[-1]

        # Szukamy ostatniego czasu startowego, który jest mniejszy bądź równy t
        idx = np.searchsorted(self.start_times, t, side='right') - 1
        local_t = t - self.start_times[idx]
        return idx, local_t

    def evaluate(self, t):
        """
        Zwraca pełny stan trajektorii w czasie t, przeliczony ze znormalizowanego czasu tau.
        """
        idx, lt = self._get_segment_index_and_local_time(t)
        T = self.durations[idx] # Pobieramy czas trwania obecnego segmentu
        tau = lt / T            # Czas znormalizowany tau in [0, 1]
        
        # Pobieramy współczynniki wielomianów 7. stopnia dla tego segmentu
        cx = self.coefs['x'][idx]
        cy = self.coefs['y'][idx]
        cz = self.coefs['z'][idx]
        
        # --- OBLICZENIA DLA POZYCJI X, Y, Z (Wielomian 7. stopnia) ---
        # Używamy tau do potęg
        t_pos  = np.array([1.0, tau, tau**2, tau**3, tau**4, tau**5, tau**6, tau**7])
        t_vel  = np.array([0.0, 1.0, 2*tau, 3*tau**2, 4*tau**3, 5*tau**4, 6*tau**5, 7*tau**6])
        t_acc  = np.array([0.0, 0.0, 2.0, 6*tau, 12*tau**2, 20*tau**3, 30*tau**4, 42*tau**5])
        t_jerk = np.array([0.0, 0.0, 0.0, 6.0, 24*tau, 60*tau**2, 120*tau**3, 210*tau**4])
        t_snap = np.array([0.0, 0.0, 0.0, 0.0, 24.0, 120*tau, 360*tau**2, 840*tau**3])
        
        # Iloczyn skalarny ORAZ skalowanie wynikające z reguły łańcuchowej (1/T^k)
        pos  = np.array([np.dot(cx, t_pos),  np.dot(cy, t_pos),  np.dot(cz, t_pos)])
        vel  = np.array([np.dot(cx, t_vel),  np.dot(cy, t_vel),  np.dot(cz, t_vel)])  * (1.0 / T)
        acc  = np.array([np.dot(cx, t_acc),  np.dot(cy, t_acc),  np.dot(cz, t_acc)])  * (1.0 / T**2)
        jerk = np.array([np.dot(cx, t_jerk), np.dot(cy, t_jerk), np.dot(cz, t_jerk)]) * (1.0 / T**3)
        snap = np.array([np.dot(cx, t_snap), np.dot(cy, t_snap), np.dot(cz, t_snap)]) * (1.0 / T**4)

        # --- OBLICZENIA DLA YAW ---
        # Jeśli yaw jest w czasie znormalizowanym, traktujemy je analogicznie
        cyaw = self.yaw_coefs[idx]
        t_yaw      = np.array([1.0, tau, tau**2, tau**3])
        t_yaw_dot  = np.array([0.0, 1.0, 2*tau, 3*tau**2])
        t_yaw_ddot = np.array([0.0, 0.0, 2.0, 6*tau])
        
        yaw      = np.dot(cyaw, t_yaw)
        yaw_rate = np.dot(cyaw, t_yaw_dot) * (1.0 / T)
        yaw_acc  = np.dot(cyaw, t_yaw_ddot) * (1.0 / T**2)

        return {
            'pos': pos, 'vel': vel, 'acc': acc, 'jerk': jerk, 'snap': snap,
            'yaw': yaw, 'yaw_rate': yaw_rate, 'yaw_acc': yaw_acc
        }