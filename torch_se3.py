import torch
from log import logger
import sys
import traceback


def exp_SO3(phi):
    phi_norm = torch.norm(phi)

    if phi_norm > 1e-8:
        unit_phi = phi / phi_norm
        unit_phi_skewed = skew3(unit_phi)
        C = torch.eye(3, 3, device=phi.device) + torch.sin(phi_norm) * unit_phi_skewed + \
            (1 - torch.cos(phi_norm)) * torch.mm(unit_phi_skewed, unit_phi_skewed)
    else:
        phi_skewed = skew3(phi)
        C = torch.eye(3, 3, device=phi.device) + phi_skewed + 0.5 * torch.mm(phi_skewed, phi_skewed)

    return C


# assumes small rotations
def log_SO3(C):
    phi_norm = torch.acos(torch.clamp((torch.trace(C) - 1) / 2, -1.0, 1.0))
    if torch.sin(phi_norm) > 1e-6:
        phi = phi_norm * unskew3(C - C.transpose(0, 1)) / (2 * torch.sin(phi_norm))
    else:
        phi = 0.5 * unskew3(C - C.transpose(0, 1))

    return phi


def log_SO3_eigen(C):  # no autodiff
    phi_norm = torch.acos(torch.clamp((torch.trace(C) - 1) / 2, -1.0, 1.0))

    # eig is not very food for C close to identity, will only keep around 3 decimals places
    w, v = torch.eig(C, eigenvectors=True)
    a = torch.tensor([0., 0., 0.], device=C.device)
    for i in range(0, w.size(0)):
        if torch.abs(w[i, 0] - 1.0) < 1e-6 and torch.abs(w[i, 1] - 0.0) < 1e-6:
            a = v[:, i]

    assert (torch.abs(torch.norm(a) - 1.0) < 1e-6)

    if torch.allclose(exp_SO3(phi_norm * a), C, atol=1e-3):
        return phi_norm * a
    elif torch.allclose(exp_SO3(-phi_norm * a), C, atol=1e-3):
        return -phi_norm * a
    else:
        raise ValueError("Invalid logarithmic mapping")


def skew3(v):
    m = torch.zeros(3, 3, device=v.device)
    m[0, 1] = -v[2]
    m[0, 2] = v[1]
    m[1, 0] = v[2]

    m[1, 2] = -v[0]
    m[2, 0] = -v[1]
    m[2, 1] = v[0]

    return m


def unskew3(m):
    return torch.stack([m[2, 1], m[0, 2], m[1, 0]])


def J_left_SO3_inv(phi):
    phi = phi.view(3, 1)
    phi_norm = torch.norm(phi)
    if torch.abs(phi_norm) > 1e-6:
        a = phi / phi_norm
        cot_half_phi_norm = 1.0 / torch.tan(phi_norm / 2)
        J_inv = (phi_norm / 2) * cot_half_phi_norm * torch.eye(3, 3, device=phi.device) + \
                (1 - (phi_norm / 2) * cot_half_phi_norm) * \
                torch.mm(a, a.transpose(0, 1)) - (phi_norm / 2) * skew3(a)
    else:
        J_inv = torch.eye(3, 3, device=phi.device) - 0.5 * skew3(phi)
    return J_inv


def J_left_SO3(phi):
    phi = phi.view(3, 1)
    phi_norm = torch.norm(phi)
    if torch.abs(phi_norm) > 1e-6:
        a = phi / phi_norm
        J = (torch.sin(phi_norm) / phi_norm) * torch.eye(3, 3, device=phi.device) + \
            (1 - (torch.sin(phi_norm) / phi_norm)) * torch.mm(a, a.transpose(0, 1)) + \
            ((1 - torch.cos(phi_norm)) / phi_norm) * skew3(a)
    else:
        J = torch.eye(3, 3, device=phi.device) + 0.5 * skew3(phi)
    return J


# ============================= Batched Methods =============================
def skew3_b(v):
    m = torch.zeros([v.size(0), 3, 3], device=v.device)
    m[..., 0, 1] = -v[..., 2, 0]
    m[..., 0, 2] = v[..., 1, 0]
    m[..., 1, 0] = v[..., 2, 0]

    m[..., 1, 2] = -v[..., 0, 0]
    m[..., 2, 0] = -v[..., 1, 0]
    m[..., 2, 1] = v[..., 0, 0]

    return m


def unskew3_b(m):
    return torch.unsqueeze(torch.stack([m[..., 2, 1], m[..., 0, 2], m[..., 1, 0]], -1), -1)


def exp_SO3_b(phi):
    eps = 1e-8
    C = torch.zeros(phi.size(0), 3, 3, device=phi.device)

    phi_norm = torch.norm(phi, dim=1, keepdim=True)
    sel = torch.squeeze(phi_norm > eps)

    phi_norm_sel = phi_norm[sel]
    phi_no_sel = phi[~sel]

    if phi_norm_sel.size(0):
        unit_phi_sel = phi[sel] / phi_norm_sel
        unit_phi_skewed_sel = skew3_b(unit_phi_sel)
        C[sel] = torch.eye(3, 3, device=phi.device).repeat([phi_norm_sel.size(0), 1, 1]) + \
                 torch.sin(phi_norm_sel) * unit_phi_skewed_sel + \
                 (1 - torch.cos(phi_norm_sel)) * torch.matmul(unit_phi_skewed_sel, unit_phi_skewed_sel)

    if phi_no_sel.size(0):
        phi_skewed_no_sel = skew3_b(phi_no_sel)
        C[~sel] = torch.eye(3, 3, device=phi.device).repeat([phi_no_sel.size(0), 1, 1]) + phi_skewed_no_sel

    return C


# assumes small rotations, does not handle case when phi is close to pi
# supports more than one batch dimensions
def log_SO3_b(C, raise_exeption=True):
    eps = 1e-6
    eps_pi = 1e-4  # strict eps_pi

    ret_sz = list(C.shape[:-2]) + [3, 1]
    phi = torch.zeros(*ret_sz, device=C.device)
    trace = torch.sum(torch.diagonal(C, dim1=-2, dim2=-1), dim=-1, keepdim=True)
    acos_ratio = torch.unsqueeze((trace - 1) / 2, -1)

    if torch.any(acos_ratio + 1.0 < eps_pi):
        sel_invalid = torch.sum(acos_ratio + 1.0 < eps_pi, (-2, -1)) > 0
        logger.print(sel_invalid)
        logger.print(C[sel_invalid])
        logger.print("Warn: log_SO3_b acos_ratio close to -1")
        if raise_exeption:
            raise ValueError("Warn: log_SO3_b acos_ratio close to -1")

    sel = ((acos_ratio - 1.0 < -eps) & ~(acos_ratio + 1.0 < eps_pi)).view(ret_sz[:-2])
    not_sel = (~(acos_ratio - 1.0 < -eps) & ~(acos_ratio + 1.0 < eps_pi)).view(ret_sz[:-2])
    phi_norm_sel = torch.acos(acos_ratio[sel])
    C_sel = C[sel]
    C_not_sel = C[not_sel]

    phi[sel] = phi_norm_sel * unskew3_b(C_sel - C_sel.transpose(-2, -1)) / (2 * torch.sin(phi_norm_sel))
    phi[not_sel] = 0.5 * unskew3_b(C_not_sel - C_not_sel.transpose(-2, -1))

    return phi


def J_left_SO3_inv_b(phi):
    eps = 1e-6
    J_inv = torch.zeros(phi.size(0), 3, 3, device=phi.device)
    phi_norm = torch.norm(phi, dim=1, keepdim=True)
    sel = torch.squeeze(phi_norm > eps)

    phi_norm_sel = phi_norm[sel]
    if phi_norm_sel.size(0):
        unit_phi_sel = phi[sel] / phi_norm_sel
        cot_half_phi_norm_sel = 1.0 / torch.tan(phi_norm_sel / 2)
        J_inv[sel] = (phi_norm_sel / 2) * cot_half_phi_norm_sel * \
                     torch.eye(3, 3, device=phi.device).repeat(phi_norm_sel.size(0), 1, 1) + \
                     (1 - (phi_norm_sel / 2) * cot_half_phi_norm_sel) * \
                     torch.matmul(unit_phi_sel, unit_phi_sel.transpose(-2, -1)) - \
                     (phi_norm_sel / 2) * skew3_b(unit_phi_sel)

    phi_no_sel = phi[~sel]
    if phi_no_sel.size(0):
        J_inv[~sel] = torch.eye(3, 3, device=phi.device).repeat(phi_no_sel.size(0), 1, 1) - 0.5 * skew3_b(phi_no_sel)

    return J_inv
