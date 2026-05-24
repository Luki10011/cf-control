import csv
import pathlib
import numpy as np
import pytest

# Dostosowane do Twojej struktury modułów
from UAV.uav_state import UAVParameters
from UAV.flatness import calculate_state_from_flat_inputs

_CSV_PATH = pathlib.Path(__file__).parent / 'trajectory_from_flat_output_test_data.csv'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cases():
    """Return list of (test_name, inputs, expected) tuples from the CSV."""
    cases = []
    with _CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['test_name']
            inputs = {
                'position': np.array([
                    float(row['in_pos_x']),
                    float(row['in_pos_y']),
                    float(row['in_pos_z']),
                ]),
                'linear_velocity': np.array([
                    float(row['in_vel_x']),
                    float(row['in_vel_y']),
                    float(row['in_vel_z']),
                ]),
                'acc': np.array([
                    float(row['in_acc_x']),
                    float(row['in_acc_y']),
                    float(row['in_acc_z']),
                ]),
                'jerk': np.array([
                    float(row['in_jerk_x']),
                    float(row['in_jerk_y']),
                    float(row['in_jerk_z']),
                ]),
                'snap': np.array([
                    float(row['in_snap_x']),
                    float(row['in_snap_y']),
                    float(row['in_snap_z']),
                ]),
                'yaw': float(row['in_yaw']),
                'yaw_rate': float(row['in_yaw_rate']),
                'yaw_acc': float(row['in_yaw_acceleration']),
                'mass': float(row['in_mass']),
                'gravity': float(row['in_gravity']),
                'I_xx': float(row['in_I_xx']),
                'I_yy': float(row['in_I_yy']),
                'I_zz': float(row['in_I_zz']),
            }
            expected = {
                'pos': np.array([
                    float(row['out_pos_x']),
                    float(row['out_pos_y']),
                    float(row['out_pos_z']),
                ]),
                'quat': np.array([
                    float(row['out_quat_w']),
                    float(row['out_quat_x']),
                    float(row['out_quat_y']),
                    float(row['out_quat_z']),
                ]),
                'vel': np.array([
                    float(row['out_vel_x']),
                    float(row['out_vel_y']),
                    float(row['out_vel_z']),
                ]),
                'omega': np.array([
                    float(row['out_omega_x']),
                    float(row['out_omega_y']),
                    float(row['out_omega_z']),
                ]),
                'thrust': float(row['out_thrust']),
                'torque': np.array([
                    float(row['out_torque_x']),
                    float(row['out_torque_y']),
                    float(row['out_torque_z']),
                ]),
            }
            cases.append((name, inputs, expected))
    return cases


def _make_params(inp):
    """Build UAVParameters according to project convention."""
    p = UAVParameters()
    p.mass = inp['mass']
    p.gravity = inp['gravity']
    # Zmiana z p.J na p.inertia_tensor (zgodnie z Twoją klasą)
    p.inertia_tensor = np.diag([inp['I_xx'], inp['I_yy'], inp['I_zz']])
    return p


def _run(inp):
    """Call calculate_state_from_flat_inputs and return the output dictionary."""
    return calculate_state_from_flat_inputs(
        position=inp['position'],
        linear_velocity=inp['linear_velocity'],
        acc=inp['acc'],
        jerk=inp['jerk'],
        snap=inp['snap'],
        yaw=inp['yaw'],
        yaw_rate=inp['yaw_rate'],
        yaw_acc=inp['yaw_acc'],
        parameters=_make_params(inp)
    )


# ---------------------------------------------------------------------------
# Parametrised test class
# ---------------------------------------------------------------------------

_CASES = _load_cases()


@pytest.mark.parametrize('name,inp,exp', _CASES, ids=[c[0] for c in _CASES])
class TestFlatnessFromCSV:
    """Verify calculate_state_from_flat_inputs against reference CSV data."""

    def test_position_passthrough(self, name, inp, exp):
        """Position is passed through unchanged."""
        res = _run(inp)
        assert np.allclose(res['pos'], exp['pos'], atol=1e-9), (
            f'[{name}] position mismatch: got {res["pos"]}, expected {exp["pos"]}'
        )

    def test_velocity_passthrough(self, name, inp, exp):
        """Velocity is passed through unchanged."""
        res = _run(inp)
        assert np.allclose(res['vel'], exp['vel'], atol=1e-9), (
            f'[{name}] velocity mismatch: got {res["vel"]}, expected {exp["vel"]}'
        )

    def test_quaternion_unit_norm(self, name, inp, exp):
        """Output quaternion must be a unit quaternion."""
        res = _run(inp)
        q = res['quat']
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-9), (
            f'[{name}] quaternion is not unit: |q| = {np.linalg.norm(q)}'
        )

    def test_quaternion_value(self, name, inp, exp):
        """Output quaternion matches the reference (q and -q are both accepted)."""
        res = _run(inp)
        q = res['quat']
        match = (
            np.allclose(q, exp['quat'], atol=1e-7)
            or np.allclose(q, -exp['quat'], atol=1e-7)
        )
        assert match, (
            f'[{name}] quaternion mismatch: got {q}, expected ±{exp["quat"]}'
        )

    def test_angular_velocity(self, name, inp, exp):
        """Body-frame angular velocity matches the reference."""
        res = _run(inp)
        omega = res['omega']
        assert np.allclose(omega, exp['omega'], atol=1e-7), (
            f'[{name}] omega mismatch: got {omega}, expected {exp["omega"]}'
        )

    def test_thrust(self, name, inp, exp):
        """Collective thrust matches the reference."""
        res = _run(inp)
        assert np.isclose(res['thrust'], exp['thrust'], atol=1e-6), (
            f'[{name}] thrust mismatch: got {res["thrust"]}, expected {exp["thrust"]}'
        )

    def test_torque(self, name, inp, exp):
        """Body torques match the reference within tolerance."""
        res = _run(inp)
        assert np.allclose(res['torque'], exp['torque'], atol=3e-4), (
            f'[{name}] torque mismatch: got {res["torque"]}, expected {exp["torque"]}'
        )