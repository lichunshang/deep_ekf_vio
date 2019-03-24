import numpy as np
from log import logger
import transformations
import scipy.linalg


def C_from_T(T):
    return T[0:3, 0:3]


def r_from_T(T):
    return T[0:3, 3]


def T_from_Ct(C, r):
    T = np.eye(4, 4)
    T[0:3, 0:3] = C
    T[0:3, 3] = r

    return T


def skew3(v):
    assert (type(v) == np.ndarray and v.size == 3) or (type(v) == type([]) and len(v) == 3)

    m = np.zeros([3, 3])
    m[0, 1] = -v[2]
    m[0, 2] = v[1]
    m[1, 0] = v[2]

    m[1, 2] = -v[0]
    m[2, 0] = -v[1]
    m[2, 1] = v[0]

    return m


def unskew3(m):
    assert (m.shape[0] == 3 and m.shape[1] == 3)
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


# have problem with values close to zero it seems
def log_SO3_eigen(C):
    assert (len(C.shape) == 2 and C.shape[0] == 3 and C.shape[1] == 3)

    phi_norm = np.arccos(np.clip((np.trace(C) - 1) / 2, -1.0, 1.0))
    w, v = np.linalg.eig(C)
    e_sel_idx = np.argmin(np.abs(w - 1))  # pick the eigen value closest to 1
    a = np.real(v[:, e_sel_idx])

    # pick the phi that produces the closest rotation matrix
    phis = [phi_norm * a, -phi_norm * a]
    m_sel_idx = np.argmin([np.linalg.norm(exp_SO3(phis[0]) - C),
                           np.linalg.norm(exp_SO3(phis[1]) - C)])
    return phis[m_sel_idx]


def log_SO3(C):
    assert (len(C.shape) == 2 and C.shape[0] == 3 and C.shape[1] == 3)

    phi_norm = np.arccos(np.clip((np.trace(C) - 1) / 2, -1.0, 1.0))

    assert (phi_norm >= 0 and np.sin(phi_norm) >= 0)

    if phi_norm < np.pi / 2 and np.sin(phi_norm) > 1e-6:
        u = unskew3(C - np.transpose(C)) / (2 * np.sin(phi_norm))
        phi = phi_norm * u
    elif phi_norm < np.pi / 2:
        phi = 0.5 * unskew3(C - C.transpose())
    else:
        phi = log_SO3_eigen(C)
        # phi = unskew3(scipy.linalg.logm(C))

    return phi


def left_jacobi_SO3(phi):
    phi = np.reshape(phi, [3, 1])
    phi_norm = np.linalg.norm(phi)
    if np.abs(phi_norm) > 1e-8:
        a = phi / phi_norm
        J = (np.sin(phi_norm) / phi_norm) * np.eye(3, 3) + (1 - (np.sin(phi_norm) / phi_norm)) * a.dot(
                a.transpose()) + ((1 - np.cos(phi_norm)) / phi_norm) * skew3(a)
    else:
        J = np.eye(3, 3)
    return J


def left_jacobi_SO3_inv(phi):
    phi = np.reshape(phi, [3, 1])
    phi_norm = np.linalg.norm(phi)
    if np.abs(phi_norm) > 1e-8:
        a = phi / phi_norm
        cot_half_phi_norm = 1 / np.tan(phi_norm / 2)
        J_inv = (phi_norm / 2) * cot_half_phi_norm * np.eye(3, 3) + \
                (1 - (phi_norm / 2) * cot_half_phi_norm) * (a.dot(a.transpose())) - (phi_norm / 2) * skew3(a)
    else:
        J_inv = np.eye(3, 3)
    return J_inv


def log_SE3(T):
    C = C_from_T(T)
    r = r_from_T(T)
    phi = log_SO3(C)
    rou = left_jacobi_SO3_inv(phi).dot(r)
    return np.concatenate([rou, phi])


def exp_SO3(phi):
    phi_norm = np.linalg.norm(phi)
    if np.abs(phi_norm) > 1e-8:
        unit_phi = phi / phi_norm
        unit_phi_skewed = skew3(unit_phi)
        m = np.eye(3, 3) + np.sin(phi_norm) * unit_phi_skewed + \
            (1 - np.cos(phi_norm)) * unit_phi_skewed.dot(unit_phi_skewed)
    else:
        phi_skewed = skew3(phi)
        m = np.eye(3, 3) + phi_skewed + 0.5 * phi_skewed.dot(phi_skewed)

    return m


def interpolate_SO3(C1, C2, alpha):
    C_interp = scipy.linalg.fractional_matrix_power(C2.dot(C1.transpose()), alpha).dot(C1)
    if np.linalg.norm(np.imag(C_interp)) > 1e-10:
        logger.print("Bad SO(3) interp:")
        logger.print(C_interp)
    return np.real(C_interp)


def interpolate_SE3(T1, T2, alpha):
    T_interp = scipy.linalg.fractional_matrix_power(T2.dot(np.linalg.inv(T1)), alpha).dot(T1)
    if np.linalg.norm(np.imag(T_interp)) > 1e-10:
        logger.print("Bad SE(3) interp:")
        logger.print(T_interp)
    return np.real(T_interp)


# reorthogonalize the SO(3) part of SE(3) by normalizing a quaternion
def reorthogonalize_SE3(T):
    # ensure the rotational matrix is orthogonal
    q = transformations.quaternion_from_matrix(T)
    n = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    q = q / n
    T_new = transformations.quaternion_matrix(q)
    T_new[0:3, 3] = T[0:3, 3]
    return T_new
