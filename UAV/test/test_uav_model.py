
from UAV.uav_state import UAVParameters
import numpy as np
from unittest.mock import patch

from UAV.rk4 import RK4Propagator

import pytest

@pytest.fixture
def default_uav_params():
    return {
        'mass': 2.0,
        'inertia': np.diag([0.05, 0.05, 0.1]),
        'gravity': 9.81,
        'dt': 0.1
    }

@pytest.fixture
def initial_state_hover():
    """
    Zwraca stan początkowy uav:
    Pozycja=[0,0,0], Prędkość Liniowa=[0,0,0], 
    Orientacja (Quat)=[1,0,0,0] (brak obrotu), Prędkość Kątowa=[0,0,0]
    """
    state = np.zeros(13)
    state[6] = 1.0  # q_w = 1.0 (neutralny kwaternion)
    return state


class TestUAVParameters:

    def test_load_from_dict_scalars(self):
        p = UAVParameters()
        p.load_from_yaml({'mass': 2.0, 'gravity': 9.0, 'dt': 0.005})
        assert p.mass == 2.0
        assert p.gravity == 9.0
        assert p.dt == 0.005

    def test_load_from_dict_flat_inertia_tensor(self):
        p = UAVParameters()
        flat = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
        p.load_from_yaml({'inertia_tensor': flat})
        assert p.inertia_tensor.shape == (3, 3)
        assert np.isclose(p.inertia_tensor[0, 0], 1.0)
        assert np.isclose(p.inertia_tensor[1, 1], 2.0)
        assert np.isclose(p.inertia_tensor[2, 2], 3.0)


    def test_missing_keys_keep_defaults(self):
        p = UAVParameters()
        original_mass = p.mass
        original_dt = p.dt
        original_inertia = p.inertia_tensor.copy()
        
        p.load_from_yaml({}) 
        
        assert p.mass == original_mass
        assert p.dt == original_dt
        assert np.array_equal(p.inertia_tensor, original_inertia)




# --- TESTY JEDNOSTKOWE ---

class TestRK4Propagator:

    @patch('UAV.rk4.multiply_vector_by_quaternion')
    def test_free_fall_dynamics(self, mock_mult_vec, default_uav_params, initial_state_hover):
        """
        Testuje spadek swobodny (brak sterowania, ciąg = 0).
        Dla zerowego ciągu przyspieszenie powinno wynosić dokładnie [0, 0, -g].
        """
        mock_mult_vec.return_value = np.array([0.0, 0.0, 0.0])
        
        propagator = RK4Propagator(default_uav_params)
        control_inputs = np.array([0.0, 0.0, 0.0, 0.0])  # T=0, tau=[0,0,0]
        
        derivatives = propagator.dynamics(initial_state_hover, control_inputs)
        
        assert np.allclose(derivatives[0:3], 0.0)
        assert np.allclose(derivatives[3:6], [0.0, 0.0, -default_uav_params['gravity']])
        assert np.allclose(derivatives[10:13], 0.0)

    @patch('UAV.rk4.multiply_quaternions')
    @patch('UAV.rk4.multiply_vector_by_quaternion')
    def test_hover_equilibrium_propagation(self, mock_mult_vec, mock_mult_quat, default_uav_params, initial_state_hover):
        m = default_uav_params['mass']
        g = default_uav_params['gravity']
        hover_thrust = m * g
        
        mock_mult_vec.return_value = np.array([0.0, 0.0, hover_thrust])
        mock_mult_quat.return_value = np.array([0.0, 0.0, 0.0, 0.0]) 
        
        propagator = RK4Propagator(default_uav_params)
        control_inputs = np.array([hover_thrust, 0.0, 0.0, 0.0])
        
        next_state = propagator.propagate(initial_state_hover, control_inputs)
        
        assert np.allclose(next_state[0:3], initial_state_hover[0:3])
        assert np.allclose(next_state[3:6], initial_state_hover[3:6])
        assert np.allclose(next_state[6:10], initial_state_hover[6:10])

    def test_pure_rotational_acceleration(self, default_uav_params, initial_state_hover):
        propagator = RK4Propagator(default_uav_params)
        
        tx, ty, tz = 0.5, 0.0, 0.0
        control_inputs = np.array([0.0, tx, ty, tz]) # Ciąg=0, Moment w osi X = 0.5
        
        derivatives = propagator.dynamics(initial_state_hover, control_inputs)
        omega_dot = derivatives[10:13]
        
        expected_omega_dot_x = tx / default_uav_params['inertia'][0, 0]
        
        assert np.isclose(omega_dot[0], expected_omega_dot_x)
        assert np.isclose(omega_dot[1], 0.0)
        assert np.isclose(omega_dot[2], 0.0)

    @patch('UAV.rk4.multiply_quaternions')
    @patch('UAV.rk4.multiply_vector_by_quaternion')
    def test_rk4_integration_accuracy(self, mock_mult_vec, mock_mult_quat, default_uav_params, initial_state_hover):
        m = default_uav_params['mass']
        g = default_uav_params['gravity']
        dt = default_uav_params['dt']
        
        required_thrust = m * (2.0 + g)
        mock_mult_vec.return_value = np.array([0.0, 0.0, required_thrust])
        mock_mult_quat.return_value = np.array([0.0, 0.0, 0.0, 0.0])
        
        propagator = RK4Propagator(default_uav_params)
        control_inputs = np.array([required_thrust, 0.0, 0.0, 0.0])
        
        next_state = propagator.propagate(initial_state_hover, control_inputs)
        expected_v_z = 0.0 + 2.0 * dt
        assert np.isclose(next_state[5], expected_v_z)
        
        # Oczekiwana pozycja po czasie dt: z = z_0 + v_0*dt + 0.5 * a * dt^2
        expected_z = 0.0 + 0.5 * 2.0 * (dt ** 2)
        assert np.isclose(next_state[2], expected_z)