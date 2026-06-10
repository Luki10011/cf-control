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
        Zwraca pełny stan trajektorii w czasie t z bezpiecznym, 
        dynamicznym wyliczaniem kąta Yaw na podstawie wektora prędkości.
        """
        idx, lt = self._get_segment_index_and_local_time(t)
        
        # Pobieramy współczynniki wielomianów 7. stopnia dla tego segmentu
        cx = self.coefs['x'][idx]
        cy = self.coefs['y'][idx]
        cz = self.coefs['z'][idx]
        
        # --- OBLICZENIA DLA POZYCJI X, Y, Z (Wielomian 7. stopnia) ---
        t_pos   = np.array([1.0, lt, lt**2, lt**3, lt**4, lt**5, lt**6, lt**7])
        t_vel   = np.array([0.0, 1.0, 2*lt, 3*lt**2, 4*lt**3, 5*lt**4, 6*lt**5, 7*lt**6])
        t_acc   = np.array([0.0, 0.0, 2.0, 6*lt, 12*lt**2, 20*lt**3, 30*lt**4, 42*lt**5])
        t_jerk  = np.array([0.0, 0.0, 0.0, 6.0, 24*lt, 60*lt**2, 120*lt**3, 210*lt**4])
        t_snap  = np.array([0.0, 0.0, 0.0, 0.0, 24.0, 120*lt, 360*lt**2, 840*lt**3])
        
        pos  = np.array([np.dot(cx, t_pos),  np.dot(cy, t_pos),  np.dot(cz, t_pos)])
        vel  = np.array([np.dot(cx, t_vel),  np.dot(cy, t_vel),  np.dot(cz, t_vel)])
        acc  = np.array([np.dot(cx, t_acc),  np.dot(cy, t_acc),  np.dot(cz, t_acc)])
        jerk = np.array([np.dot(cx, t_jerk), np.dot(cy, t_jerk), np.dot(cz, t_jerk)])
        snap = np.array([np.dot(cx, t_snap), np.dot(cy, t_snap), np.dot(cz, t_snap)])

        # --- BEZPIECZNE DYNAMICZNE OBLICZENIA DLA YAW ---
        vel_xy_norm_sq = vel[0]**2 + vel[1]**2
        vel_xy_norm = np.sqrt(vel_xy_norm_sq)
        
        # Próg prędkości (0.15 m/s) zapobiega wariowaniu arctan2 w bezruchu
        if vel_xy_norm < 0.15:
            # Jeśli dron prawie się nie porusza w XY, zachowaj ostatni znany kąt yaw
            if not hasattr(self, 'last_yaw'):
                self.last_yaw = 0.0
            yaw = self.last_yaw
            yaw_rate = 0.0
            yaw_acc = 0.0
        else:
            # Oblicz bieżący yaw z kierunku prędkości postępowej
            yaw = np.arctan2(vel[1], vel[0])
            
            # Zapamiętujemy udany stan yaw
            self.last_yaw = yaw
            
            # Analityczna pochodna yaw_rate = (vx*ay - vy*ax) / (vx^2 + vy^2)
            yaw_rate = (vel[0] * acc[1] - vel[1] * acc[0]) / vel_xy_norm_sq
            
            # Twarde nasycenie (saturacja) prędkości obrotu dla ochrony pętli KOmega
            yaw_rate = np.clip(yaw_rate, -1.0, 1.0) # max 1 radian na sekundę
            
            # Dla wygładzenia trajektorii i stabilizacji kaskady pomijamy szumiącą drugą pochodną
            yaw_acc = 0.0

        return {
            'pos': pos,
            'vel': vel,
            'acc': acc,
            'jerk': jerk,
            'snap': snap,
            'yaw': yaw,
            'yaw_rate': yaw_rate,
            'yaw_acc': yaw_acc
        }