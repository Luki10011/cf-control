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
