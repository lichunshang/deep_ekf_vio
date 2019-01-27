import numpy as np
from log import logger
import transformations


def skew(v):
    m = np.zeros([3, 3])
    m[0, 1] = -v[2]
    m[0, 2] = v[1]
    m[1, 0] = v[2]

    m[1, 2] = -v[0]
    m[2, 0] = -v[1]
    m[2, 1] = v[0]

    return m


def unskew(m):
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def log_map_SO3(R):
    assert (len(R.shape) == 2 and R.shape[0] == 3 and R.shape[1] == 3)

    arccos = (np.trace(R) - 1) / 2

    if np.abs(arccos) > 1:
        phi = 0.0
        logger.print("WARNING: invalid arccos: %f\n" % arccos)
        logger.print("%s\n" % str(R))
    else:
        phi = np.arccos((np.trace(R) - 1) / 2)

    if abs(phi) > 1e-12:
        u = unskew(R - np.transpose(R)) / (2 * np.sin(phi))
        theta = phi * u
    else:
        theta = 0.5 * unskew(R - R.transpose())

    return theta


# reorthogonalize the SO(3) part of SE(3) by normalizing a quaternion
def reorthogonalize_SE3(T):
    # ensure the rotational matrix is orthogonal
    q = transformations.quaternion_from_matrix(T)
    n = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    q = q / n
    T_new = transformations.quaternion_matrix(q)
    T_new[0:3, 3] = T[0:3, 3]
    return T_new
