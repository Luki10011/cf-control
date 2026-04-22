import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

# Zakładam, że te moduły są w Twoim projekcie
from constants import g
from model import UAVModel
from utils import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix


class Trajectory:
    def __init__(self, type, R, model):
        self.type = type
        self.R = R
        self.model : UAVModel = model # Przyjmuje gotowy obiekt UAVModel

    def calculate_state_from_flat_inputs(
        self,
        acc: np.ndarray,
        jerk: np.ndarray,
        snap: np.ndarray,
        yaw,
        yaw_rate,
        yaw_acc,
    ):
        p = self.model.position
        v = self.model.linear_velocity
        m = self.model.mass
        J = self.model.inertia_tensor
        
        # Obliczenia siły ciągu (Thrust)
        ag = np.array([acc[0], acc[1], acc[2] + g])
        thrust_val = m * np.linalg.norm(ag)

        z_B = ag / np.linalg.norm(ag)
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0])

        y_B = np.cross(z_B, x_c) / np.linalg.norm(np.cross(z_B, x_c))
        x_B = np.cross(y_B, z_B)

        # Macierz rotacji i kwaternion
        R_mat = np.column_stack((x_B, y_B, z_B))
        q = rotation_matrix_to_quaternion(R_mat)

        # Prędkość kątowa (Omega)
        # Rzutujemy jerk na płaszczyznę prostopadłą do z_B
        h_omega = (m / thrust_val) * (jerk - np.dot(jerk, z_B) * z_B)

        z_W = R_mat @ z_B  
        w_x = -np.dot(h_omega, y_B)
        w_y = np.dot(h_omega, x_B)
        w_z = yaw_rate * np.dot(z_W, z_B)   


        w = np.array([w_x, w_y, w_z])

        # To dziala
        w_dot_x = -((m / thrust_val) * snap[1] + 2 * (m / thrust_val) * jerk[2] * w_x - w_y * w_z)
        w_dot_y = (m / thrust_val) * snap[0] - 2 * (m / thrust_val) * jerk[2] * w_y - w_x * w_z
        w_dot_z = yaw_acc * np.dot(z_W, z_B)
        w_dot = np.array([w_dot_x, w_dot_y, w_dot_z])

        tau = J @ w_dot + np.cross(w, J @ w)

        return {
            'pos': p,
            'vel': v,
            'quat': q, 
            'omega': w,
            'thrust': thrust_val,
            'torque': tau
        }
    
    def mellinger_controll(self, target_state):
        # target_state to słownik zwrócony przez calculate_state_from_flat_inputs
        # current_state powinien zawierać: pos, vel, quat, omega
        
        # Parametry (powinny być w self.model)
        m = self.model.mass
        g_vec = np.array([0, 0, 9.81])
        Kp = 6 * np.eye(3) # Przykładowe wzmocnienia pozycyjne
        Kv = 3 * np.eye(3)  # Przykładowe wzmocnienia prędkości
        KR = 2 * np.eye(3)  # Wzmocnienia orientacji
        Kw = 4 * np.eye(3)  # Wzmocnienia prędkości kątowej

        # 1. Błąd pozycji i prędkości [cite: 478]
        ep = self.model.position - target_state['pos']
        ev = self.model.linear_velocity - target_state['vel']

        # F_des = -Kp*ep - Kv*ev + m*g*zW + m*acc_target
        # Uwaga: acc_target można wyciągnąć z logiki flat outputs
        acc_target = target_state['acc'] 
        F_des = -Kp @ ep - Kv @ ev + m * g_vec + m * acc_target
        
        # Aktualna oś z body (zB) pobrana z kwaternionu
        R_curr = quaternion_to_rotation_matrix(self.model.quternions_orientation)
        zB = R_curr[:, 2]
        u1 = np.dot(F_des, zB) 

        # 3. Błąd rotacji (eR) [cite: 486]
        R_des = quaternion_to_rotation_matrix(target_state['quat'])
        # eR = 1/2 * (R_des^T * R - R^T * R_des)^vee
        error_mat = 0.5 * (R_des.T @ R_curr - R_curr.T @ R_des)
        eR = np.array([error_mat[2, 1], error_mat[0, 2], error_mat[1, 0]]) # Operacja vee [cite: 486]

        # 4. Błąd prędkości kątowej (ew) [cite: 488]
        ew = self.model.angular_velocity - target_state['omega']

        # 5. Momenty sterujące (u2, u3, u4) [cite: 489]
        # Artykuł sugeruje prosty feedback, ale dla stabilności można dodać kompensację gyroskopową
        tau = -KR @ eR - Kw @ ew
        
        return u1, tau


def extract_data_from_file():
    df = pd.read_csv('trajectory_from_flat_output_test_data.csv')    
    df.set_index('test_name', inplace=True)
    return df

    

def test_trajectory(df, test_name):
    # 1. Pobranie wiersza danych dla konkretnego przypadku
    row = df.loc[test_name]
    
    # 2. Przygotowanie wejść (Flat Outputs)
    in_pos = row[['in_pos_x', 'in_pos_y', 'in_pos_z']].to_numpy()
    in_vel = row[['in_vel_x', 'in_vel_y', 'in_vel_z']].to_numpy()
    in_acc = row[['in_acc_x', 'in_acc_y', 'in_acc_z']].to_numpy()
    in_jerk = row[['in_jerk_x', 'in_jerk_y', 'in_jerk_z']].to_numpy()
    in_snap = row[['in_snap_x', 'in_snap_y', 'in_snap_z']].to_numpy()
    
    # 3. Parametry modelu i warunki początkowe
    mass = row['in_mass']
    inertia = np.diag([row['in_I_xx'], row['in_I_yy'], row['in_I_zz']])
    
    # Inicjalizacja wektora stanu [pos, vel, quat, omega] - 13 elementów
    init_cond = np.zeros(13)
    init_cond[0:3] = in_pos
    init_cond[3:6] = in_vel
    init_cond[6] = 1.0  # q_w = 1.0 (orientacja początkowa)

    # 4. Inicjalizacja obiektów i obliczenia
    uav = UAVModel(mass=mass, inertia_tensor=inertia, initial_conditions=init_cond)
    traj = Trajectory(type='test', R=1.0, model=uav)

    results = traj.calculate_state_from_flat_inputs(
        in_acc, in_jerk, in_snap, 
        row['in_yaw'], row['in_yaw_rate'], row['in_yaw_acceleration']
    )

    # 5. WERYFIKACJA WSZYSTKICH PARAMETRÓW
    print(f"\n=========== Weryfikacja testu: {test_name} ===========")
    
    # Tolerancja dla obliczeń (1e-4 dla kwaternionów/sił jest bezpieczne)
    tol = 1e-4

    # POS: out_pos_x, out_pos_y, out_pos_z
    expected_pos = row[['out_pos_x', 'out_pos_y', 'out_pos_z']].to_numpy()
    np.testing.assert_allclose(results['pos'], expected_pos, atol=tol, err_msg=f"Błąd pozycji w {test_name}")
    print(f"Pozycja jest poprawna: {results['pos']} ≈ {expected_pos}")

    # VEL: out_vel_x, out_vel_y, out_vel_z
    expected_vel = row[['out_vel_x', 'out_vel_y', 'out_vel_z']].to_numpy()
    np.testing.assert_allclose(results['vel'], expected_vel, atol=tol, err_msg=f"Błąd prędkości liniowej w {test_name}")
    print(f"Prędkość liniowa jest poprawna: {results['vel']} ≈ {expected_vel}")

    # QUAT: out_quat_w, out_quat_x, out_quat_y, out_quat_z
    expected_quat = row[['out_quat_w', 'out_quat_x', 'out_quat_y', 'out_quat_z']].to_numpy()
    # Uwaga: Kwaterniony q i -q reprezentują tę samą rotację
    actual_quat = results['quat'].flatten()
    np.testing.assert_allclose(actual_quat, expected_quat, atol=tol, err_msg=f"Błąd kwaternionu w {test_name}")
    print(f"Kwaternion jest poprawny: {actual_quat} ≈ {expected_quat} (lub -{expected_quat})")

    # OMEGA: out_omega_x, out_omega_y, out_omega_z
    expected_omega = row[['out_omega_x', 'out_omega_y', 'out_omega_z']].to_numpy()
    np.testing.assert_allclose(results['omega'].flatten(), expected_omega, atol=tol, err_msg=f"Błąd prędkości kątowej w {test_name}")
    print(f"Prędkość kątowa jest poprawna: {results['omega'].flatten()} ≈ {expected_omega}")

    # THRUST: out_thrust
    expected_thrust = row['out_thrust']
    np.testing.assert_allclose(results['thrust'], expected_thrust, atol=tol, err_msg=f"Błąd siły ciągu w {test_name}")
    print(f"Siła ciągu jest poprawna: {results['thrust']} ≈ {expected_thrust}")

    # TORQUE: out_torque_x, out_torque_y, out_torque_z
    expected_torque = row[['out_torque_x', 'out_torque_y', 'out_torque_z']].to_numpy()
    np.testing.assert_allclose(results['torque'].flatten(), expected_torque, atol=tol, err_msg=f"Błąd momentu obrotowego w {test_name}")
    print(f"Moment obrotowy jest poprawny: {results['torque'].flatten()} ≈ {expected_torque}")

    print(f"=========== Wszystkie parametry ({len(row[24:])} kolumn out_) są poprawne! ===========")

def main():
    try:
        data = extract_data_from_file()
        for name in data.index:
            test_trajectory(data, name)

    except FileNotFoundError:
        print("Błąd: Nie znaleziono pliku CSV z danymi testowymi.")

if __name__ == '__main__':
    main()