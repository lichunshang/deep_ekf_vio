import torch
from model import TorchSE3


@torch.jit.script
def skew3(v):
    v = v.view(3)
    zero = torch.zeros(1, device=v.device)[0]
    row0 = torch.stack([zero, -v[2], v[1]])
    row1 = torch.stack([v[2], zero, -v[0]])
    row2 = torch.stack([-v[1], v[0], zero])
    m = torch.stack([row0, row1, row2])
    # m = torch.zeros(3, 3, device=v.device)
    # m[0, 1] = -v[2]
    # m[0, 2] = v[1]
    # m[1, 0] = v[2]
    #
    # m[1, 2] = -v[0]
    # m[2, 0] = -v[1]
    # m[2, 1] = v[0]

    return m


@torch.jit.script
def unskew3(m):
    return torch.stack([m[2, 1], m[0, 2], m[1, 0]])


@torch.jit.script
def exp_SO3(phi):
    phi_norm = torch.norm(phi)

    if bool(phi_norm > 1e-8):
        unit_phi = phi / phi_norm
        unit_phi_skewed = skew3(unit_phi)

        C = torch.eye(3, 3, device=phi.device) + torch.sin(phi_norm) * unit_phi_skewed + \
            (1 - torch.cos(phi_norm)) * torch.mm(unit_phi_skewed, unit_phi_skewed)

    else:
        phi_skewed = skew3(phi)
        C = torch.eye(3, 3, device=phi.device) + phi_skewed + 0.5 * torch.mm(phi_skewed, phi_skewed)

    return C


@torch.jit.script
def exp_SO3_over_batch(phis):
    Cs = []
    for i in range(phis.shape[0]):
        Cs.append(exp_SO3(phis[i]))

    return torch.stack(Cs)


def exp_SO3_over_batch2(phis):
    Cs = []
    for i in range(phis.shape[0]):
        Cs.append(TorchSE3.exp_SO3(phis[i]))

    return torch.stack(Cs)
