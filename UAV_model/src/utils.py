import numpy as np


def multiply_vector_by_quaternion(q, v):
    qw, qx, qy, qz = q
    vx, vy, vz = v

    # q * v
    tw = -qx * vx - qy * vy - qz * vz
    tx = qw * vx + qy * vz - qz * vy
    ty = qw * vy + qz * vx - qx * vz
    tz = qw * vz + qx * vy - qy * vx

    # result * q_conjugate
    res_x = tw * -qx + tx * qw + ty * -qz - tz * -qy
    res_y = tw * -qy + ty * qw + tz * -qx - tx * -qz
    res_z = tw * -qz + tz * qw + tx * -qy - ty * -qx

    return np.array([res_x, res_y, res_z])


def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def rotation_matrix_to_quaternion(R):
    """
    R: macierz 3x3
    zwraca: (w, x, y, z)
    """
    r00, r01, r02 = R[0]
    r10, r11, r12 = R[1]
    r20, r21, r22 = R[2]

    trace = r00 + r11 + r22

    if trace > 0:
        S = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * S
        x = (r21 - r12) / S
        y = (r02 - r20) / S
        z = (r10 - r01) / S

    elif (r00 > r11) and (r00 > r22):
        S = 2.0 * np.sqrt(1.0 + r00 - r11 - r22)
        w = (r21 - r12) / S
        x = 0.25 * S
        y = (r01 + r10) / S
        z = (r02 + r20) / S

    elif r11 > r22:
        S = 2.0 * np.sqrt(1.0 + r11 - r00 - r22)
        w = (r02 - r20) / S
        x = (r01 + r10) / S
        y = 0.25 * S
        z = (r12 + r21) / S

    else:
        S = 2.0 * np.sqrt(1.0 + r22 - r00 - r11)
        w = (r10 - r01) / S
        x = (r02 + r20) / S
        y = (r12 + r21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)  # normalizacja
