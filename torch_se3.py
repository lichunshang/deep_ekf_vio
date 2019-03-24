import torch


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
