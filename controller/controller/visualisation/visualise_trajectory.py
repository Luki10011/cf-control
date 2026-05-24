import numpy as np
import matplotlib.pyplot as plt

# Poprawny import wewnątrz Twojego pakietu
from controller.minimum_snap import MinimumSnapGenerator

def plot_trajectory_test():
    # 1. Definiujemy testowe punkty trasy (Waypoints) dla drona
    waypoints = np.array([
        [0.0, 0.0, 0.0],  # Start
        [1.0, 2.0, 1.5],  # Punkt 1
        [3.0, 2.5, 2.0],  # Punkt 2
        [4.0, 0.0, 1.0]   # Meta
    ])
    
    segment_durations = [2.0, 2.5, 2.0]
    total_time = sum(segment_durations)

    # 2. Generujemy trajektorię za pomocą Twojego generatora QP
    generator = MinimumSnapGenerator()
    trajectory = generator.generate_trajectory(waypoints, segment_durations)

    # 3. Próbkujemy wygenerowaną trajektorię
    time_steps = np.linspace(0.0, total_time, 500)
    
    positions = []
    velocities = []
    accelerations = []
    jerks = []
    snaps = []

    for t in time_steps:
        state = trajectory.evaluate(t)
        positions.append(state['pos'])
        velocities.append(state['vel'])
        accelerations.append(state['acc'])
        jerks.append(state['jerk'])
        snaps.append(state['snap'])

    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    jerks = np.array(jerks)
    snaps = np.array(snaps)

    # ==========================================
    # WIZUALIZACJA 1: Rzuty płaskie 2D (Zastępstwo za uszkodzone 3D)
    # ==========================================
    fig_2d, axs_geo = plt.subplots(1, 3, figsize=(15, 5))
    fig_2d.suptitle("Profil przestrzenny trajektorii drona (Rzuty 2D)", fontsize=14, fontweight='bold')
    
    # Rzut z góry: Płaszczyzna X-Y
    axs_geo[0].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajektoria')
    axs_geo[0].scatter(waypoints[:, 0], waypoints[:, 1], c='r', marker='o', s=80, label='Waypoints')
    axs_geo[0].set_xlabel("Pozycja X [m]")
    axs_geo[0].set_ylabel("Pozycja Y [m]")
    axs_geo[0].set_title("Widok z góry (X-Y)")
    axs_geo[0].grid(True)
    axs_geo[0].legend()
    
    # Rzut z boku: Płaszczyzna X-Z
    axs_geo[1].plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2)
    axs_geo[1].scatter(waypoints[:, 0], waypoints[:, 2], c='r', marker='o', s=80)
    axs_geo[1].set_xlabel("Pozycja X [m]")
    axs_geo[1].set_ylabel("Pozycja Z [m]")
    axs_geo[1].set_title("Widok z boku (X-Z)")
    axs_geo[1].grid(True)
    
    # Rzut od przodu: Płaszczyzna Y-Z
    axs_geo[2].plot(positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    axs_geo[2].scatter(waypoints[:, 1], waypoints[:, 2], c='r', marker='o', s=80)
    axs_geo[2].set_xlabel("Pozycja Y [m]")
    axs_geo[2].set_ylabel("Pozycja Z [m]")
    axs_geo[2].set_title("Widok od przodu (Y-Z)")
    axs_geo[2].grid(True)

    # ==========================================
    # WIZUALIZACJA 2: Pochodne wyższych rzędów w czasie
    # ==========================================
    fig_plots, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    
    axs[0].plot(time_steps, positions[:, 0], 'r-', label='X')
    axs[0].plot(time_steps, positions[:, 1], 'g-', label='Y')
    axs[0].plot(time_steps, positions[:, 2], 'b-', label='Z')
    axs[0].set_ylabel('Pozycja [m]')
    axs[0].title.set_text('Profile czasowe pochodnych stanu płaskiego (Flat Outputs)')
    axs[0].legend(loc='upper right')
    
    axs[1].plot(time_steps, velocities[:, 0], 'r-')
    axs[1].plot(time_steps, velocities[:, 1], 'g-')
    axs[1].plot(time_steps, velocities[:, 2], 'b-')
    axs[1].set_ylabel('Prędkość [m/s]')
    
    axs[2].plot(time_steps, accelerations[:, 0], 'r-')
    axs[2].plot(time_steps, accelerations[:, 1], 'g-')
    axs[2].plot(time_steps, accelerations[:, 2], 'b-')
    axs[2].set_ylabel('Przyspieszenie [m/s²]')
    
    axs[3].plot(time_steps, jerks[:, 0], 'r-')
    axs[3].plot(time_steps, jerks[:, 1], 'g-')
    axs[3].plot(time_steps, jerks[:, 2], 'b-')
    axs[3].set_ylabel('Jerk [m/s³]')
    
    axs[4].plot(time_steps, snaps[:, 0], 'r-')
    axs[4].plot(time_steps, snaps[:, 1], 'g-')
    axs[4].plot(time_steps, snaps[:, 2], 'b-')
    axs[4].set_ylabel('Snap [m/s⁴]')
    axs[4].set_xlabel('Czas [s]')
    
    current_time_border = 0.0
    for duration in segment_durations:
        current_time_border += duration
        for ax in axs:
            ax.axvline(x=current_time_border, color='gray', linestyle='--', alpha=0.7)

    for ax in axs:
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_trajectory_test()