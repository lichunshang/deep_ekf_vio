import torch
import time
import torch_se3
import numpy as np
import se3
from tests.torch_se3_tests import over_batch


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


def skew3_batches(v):
    sz = list(v.size()[:-1]) + [3, 3, ]
    m = torch.zeros(*sz, device=v.device)
    m[..., 0, 1] = -v[..., 2]
    m[..., 0, 2] = v[..., 1]
    m[..., 1, 0] = v[..., 2]

    m[..., 1, 2] = -v[..., 0]
    m[..., 2, 0] = -v[..., 1]
    m[..., 2, 1] = v[..., 0]

    return m


def exp_SO3_batched(phi):
    phi_norm = torch.norm(phi, dim=1, keepdim=True)

    unit_phi = phi / phi_norm
    phi_norm = torch.unsqueeze(phi_norm, -1)
    unit_phi_skewed = skew3_batches(unit_phi)
    unit_phi_skewed2 = torch.matmul(unit_phi_skewed, unit_phi_skewed)

    C1 = torch.eye(3, 3, device=phi.device).repeat([phi.size(0), 1, 1]) + \
         torch.sin(phi_norm) * unit_phi_skewed + \
         (1 - torch.cos(phi_norm)) * unit_phi_skewed2

    C2 = torch.eye(3, 3, device=phi.device) + unit_phi_skewed * phi_norm + 0.5 * unit_phi_skewed2 * phi_norm * phi_norm

    mask = phi_norm > 1e-8

    C = mask.float() * C1 + (1 - mask.float()) * C2

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
        Cs.append(torch_se3.exp_SO3(phis[i]))

    return torch.stack(Cs)


if __name__ == '__main__':
    # start_t = time.time()
    # C = exp_SO3_over_batch(torch.ones([1000000, 3]).cuda())
    # print("Took %.2f" % (time.time() - start_t))
    #
    # start_t = time.time()
    # C = exp_SO3_over_batch2(torch.ones([1000000, 3]).cuda())
    # print("Took %.2f" % (time.time() - start_t))
    n = 100000
    v = torch.cat([torch.zeros(n, 3), torch.rand([n, 3]), torch.rand([n, 3]) * 1e-10, torch.rand([n, 3])], dim=0).cuda()
    # v = torch.rand([100000, 3], requires_grad=True).cuda()

    # start_t = time.time()
    # for i in range(0, 10):
    #     C = over_batch(torch_se3.exp_SO3, v)
    # print("Took %.5f" % (time.time() - start_t))

    start_t = time.time()
    for i in range(0, 1):
        C = torch_se3.exp_SO3_b(torch.unsqueeze(v, -1))
    print("Took %.5f" % (time.time() - start_t))

    start_t = time.time()
    for i in range(0, 1):
        C = exp_SO3_batched(v)
    print("Took %.5f" % (time.time() - start_t))

    # v1 = torch.rand(4, 18, 18).cuda()
    # v2 = torch.rand(4, 18, 18).cuda()
    #
    # start_t = time.time()
    # for i in range(0, 15500):
    #     batches = []
    #     for i in range(0, 4):
    #         batches.append(torch.mm(v1[i], v2[i]))
    #     a = torch.stack(batches)
    # print("Took %.5f" % (time.time() - start_t))
    #
    # start_t = time.time()
    # for i in range(0, 15500):
    #     a = torch.matmul(v1, v2)
    # print("Took %.5f" % (time.time() - start_t))
