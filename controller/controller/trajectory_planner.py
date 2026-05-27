import numpy as np
from controller.minimum_snap import MinimumSnapGenerator

class TrajectoryServer:
    def __init__(self):
        """
        Menedżer trajektorii minimalnoudarowej.
        Odpowiada za przechowywanie wygenerowanej trajektorii, zarządzanie czasem
        oraz próbkowanie stanów (Flat Outputs) dla kontrolera.
        """
        self.generator = MinimumSnapGenerator()
        self.trajectory = None
        
        # Zmienne stanowe zarządzania czasem
        self.start_time = None
        self.is_running = False

    def generate_from_waypoints(self, waypoints: np.ndarray, segment_durations: list) -> bool:
        """
        Generuje i zapisuje trajektorię wielomianową na podstawie zadanych punktów.
        
        waypoints: tablica np.ndarray o wymiarze (M, 3) [X, Y, Z]
        segment_durations: lista zawierająca czasy trwania poszczególnych odcinków
        """
        try:
            self.trajectory = self.generator.generate_trajectory(waypoints, segment_durations)
            self.is_running = False
            self.start_time = None
            return True
        except Exception as e:
            # W środowisku produkcyjnym można tu podpiąć logger lub przekazać błąd wyżej
            print(f"[TrajectoryServer] Błąd podczas generowania trajektorii: {str(e)}")
            self.trajectory = None
            self.is_running = False
            return False

    def start(self, current_timestamp: float):
        """
        Inicjalizuje start odtwarzania trajektorii.
        
        current_timestamp: Aktualny czas systemowy (np. z clock.now().nanoseconds / 1e9)
        """
        if self.trajectory is None:
            print("[TrajectoryServer] Nie można wystartować - brak wygenerowanej trajektorii.")
            return
        
        self.start_time = current_timestamp
        self.is_running = True

    def stop(self):
        """Zatrzymuje odtwarzanie trajektorii."""
        self.is_running = False
        self.start_time = None

    def update(self, current_timestamp: float) -> dict:
        """
        Pobiera stan trajektorii dla aktualnego momentu czasowego.
        Ta metoda powinna być wywoływana w pętli timera węzła ROS 2 (np. z częstotliwością 100 Hz).
        
        current_timestamp: Aktualny czas systemowy w sekundach.
        Zwraca: Słownik z Flat Outputs lub None, jeśli trajektoria się zakończyła.
        """
        if not self.is_running or self.start_time is None:
            return None

        # Obliczamy czas lokalny, który upłynął od startu trajektorii
        elapsed_time = current_timestamp - self.start_time

        # Jeśli osiągnęliśmy lub przekroczyliśmy całkowity czas trajektorii,
        # zwróć dokładny stan końcowy (t = total_time) raz, aby kontroler
        # dostał precyzyjne zadane wartości końcowe, potem zakończ odtwarzanie.
        if elapsed_time >= self.trajectory.total_time:
            # Pobierz ostatni stan trajektorii (w czasie total_time) i zatrzymaj odtwarzanie
            flat_state = self.trajectory.evaluate(self.trajectory.total_time)
            self.stop()
            return flat_state

        # Próbkowanie wielomianów z klasy PolynomialTrajectory
        flat_state = self.trajectory.evaluate(elapsed_time)
        return flat_state