import unittest
import torch as tr
import scipy.linalg as slinalg
import se3
import numpy as np
import torch_se3

tr.manual_seed(0)


def over_batch(fn, param):
    batches = []

    for i in range(0, len(param)):
        batches.append(fn(param[i]))

    return tr.stack(batches)


class Test_torch_se3(unittest.TestCase):
    def test_skew_unskew(self):
        self.assertTrue(tr.allclose(torch_se3.unskew3(torch_se3.skew3(tr.tensor([1., 2., 3.]))),
                                    tr.tensor([1., 2., 3.]), atol=1e-8))
        self.assertTrue(tr.allclose(torch_se3.skew3(tr.tensor([1., 2., 3.])).transpose(0, 1),
                                    -torch_se3.skew3(tr.tensor([1., 2., 3.])), atol=1e-8))
        self.assertTrue(tr.allclose(tr.mm(torch_se3.skew3(tr.tensor([1., 2., 3.])), tr.tensor([[1.], [2.], [3.]])),
                                    tr.tensor([0., 0., 0.]), atol=1e-8))

    def test_exp_SO3(self):
        self.assertTrue(tr.allclose(torch_se3.exp_SO3(tr.tensor([1., 2., 3.])),
                                    tr.tensor(np.array(slinalg.expm(se3.skew3([1, 2, 3]))),
                                              dtype=tr.float32), atol=1e-8))
        self.assertTrue(tr.allclose(torch_se3.exp_SO3(tr.tensor([1e-8, 2e-8, 3e-8])),
                                    tr.tensor(np.array(slinalg.expm(se3.skew3([1e-8, 2e-8, 3e-8]))),
                                              dtype=tr.float32), atol=1e-8))
        self.assertTrue(tr.allclose(torch_se3.exp_SO3(tr.tensor([0., 0., 0.])), tr.eye(3, 3), atol=1e-8))

        C = torch_se3.exp_SO3(tr.tensor([-1, 0.2, -0.75]))
        u = tr.tensor([[1, -2, 0.1]]).transpose(0, 1)
        self.assertTrue(tr.allclose(torch_se3.skew3(tr.mm(C, u)),
                                    tr.mm(tr.mm(C, torch_se3.skew3(u)), C.transpose(0, 1)), atol=1e-7))

        exp_Cu = torch_se3.exp_SO3(tr.mm(C, u))
        C_expu_Ct = tr.mm(tr.mm(C, torch_se3.exp_SO3(u)), C.transpose(0, 1))
        self.assertTrue(tr.allclose(exp_Cu, C_expu_Ct, atol=1e-7))

    def test_log_SO3(self):
        C = tr.tensor(se3.exp_SO3([-100., 200., -80]), dtype=tr.float32)
        log_C_scipy = torch_se3.unskew3(tr.tensor(np.real(slinalg.logm(C)), dtype=tr.float32))
        log_C = torch_se3.log_SO3(C)
        self.assertTrue(tr.allclose(log_C, log_C_scipy, atol=1e-6))

        C = tr.tensor(se3.exp_SO3([-1., 2., -0.8]), dtype=tr.float32)
        log_C_scipy = torch_se3.unskew3(tr.tensor(np.real(slinalg.logm(C)), dtype=tr.float32))
        log_C = torch_se3.log_SO3(C)
        self.assertTrue(tr.allclose(log_C, log_C_scipy, atol=1e-8))

        C = tr.tensor(se3.exp_SO3([1e-8, -2e-8, 3e-8]), dtype=tr.float32)
        log_C = torch_se3.log_SO3(C)
        self.assertTrue(tr.allclose(log_C, tr.tensor([1e-8, -2e-8, 3e-8]), atol=1e-10))

        C = tr.tensor(se3.exp_SO3([1e-9, -2e-9, 3e-9]), dtype=tr.float32)
        log_C = torch_se3.log_SO3(C)
        self.assertTrue(tr.allclose(log_C, tr.tensor([1e-9, -2e-9, 3e-9]), atol=1e-10))

        C = tr.tensor(se3.exp_SO3([0., 0., 0.]), dtype=tr.float32)
        log_C = torch_se3.log_SO3(C)
        self.assertTrue(tr.allclose(log_C, tr.zeros(3, ), atol=1e-10))

    def test_log_SO3_eigen(self):
        C = tr.tensor(se3.exp_SO3([-100., 200., -80]), dtype=tr.float32)
        log_C_scipy = torch_se3.unskew3(tr.tensor(np.real(slinalg.logm(C)), dtype=tr.float32))
        log_C = torch_se3.log_SO3_eigen(C)
        self.assertTrue(tr.allclose(log_C, log_C_scipy, atol=1e-6))

        C = tr.tensor(se3.exp_SO3([-1., 2., -0.8]), dtype=tr.float32)
        log_C_scipy = torch_se3.unskew3(tr.tensor(np.real(slinalg.logm(C)), dtype=tr.float32))
        log_C = torch_se3.log_SO3_eigen(C)
        self.assertTrue(tr.allclose(log_C, log_C_scipy, atol=1e-8))

        C = tr.tensor(se3.exp_SO3([1e-3, -2e-3, 3e-3]), dtype=tr.float32)
        log_C = torch_se3.log_SO3_eigen(C)
        self.assertTrue(tr.allclose(log_C, tr.tensor([1e-3, -2e-3, 3e-3]), atol=1e-5))

        C = tr.tensor(se3.exp_SO3([0., 0., 0.]), dtype=tr.float32)
        log_C = torch_se3.log_SO3_eigen(C)
        self.assertTrue(tr.allclose(log_C, tr.zeros(3, ), atol=1e-10))

        C = tr.tensor(se3.exp_SO3([0.1, -0.2, 3.133624802220163]), dtype=tr.float32)
        log_C = torch_se3.log_SO3_eigen(C)
        self.assertTrue(tr.allclose(log_C, tr.tensor([0.1, -0.2, 3.133624802220163]), atol=1e-6) or
                        tr.allclose(-log_C, tr.tensor([0.1, -0.2, 3.133624802220163]), atol=1e-6))

        C = tr.tensor(se3.exp_SO3([-np.pi, 0, 0]), dtype=tr.float32)
        log_C = torch_se3.log_SO3_eigen(C)
        self.assertTrue(tr.allclose(log_C, tr.tensor([-np.pi, 0, 0]), atol=1e-6) or
                        tr.allclose(-log_C, tr.tensor([-np.pi, 0, 0]), atol=1e-6))

    def test_J_left_SO3_inv(self):
        phi = tr.tensor([-0.2, 1.5, -2])
        J = torch_se3.J_left_SO3(phi)
        J_inv = torch_se3.J_left_SO3_inv(phi)
        self.assertTrue(tr.allclose(J.inverse(), J_inv, atol=1e-6))

        self.assertTrue(tr.allclose(torch_se3.J_left_SO3(tr.tensor([0., 0., 0.])), tr.eye(3, 3), atol=1e-15))
        self.assertTrue(tr.allclose(torch_se3.J_left_SO3_inv(tr.tensor([0., 0., 0.])), tr.eye(3, 3), atol=1e-15))

        phi = tr.tensor([1e-7, -2e-7, 3e-7])
        J = torch_se3.J_left_SO3(phi)
        J_inv = torch_se3.J_left_SO3_inv(phi)
        self.assertTrue(tr.allclose(J.inverse(), J_inv, atol=1e-10))

        phi = tr.tensor([2, -0.5, -1.1])
        lhs = tr.eye(3, 3) + tr.mm(torch_se3.skew3(phi), torch_se3.J_left_SO3(phi))
        rhs = torch_se3.exp_SO3(phi)
        self.assertTrue(tr.allclose(lhs, rhs, atol=1e-6))

        lhs = torch_se3.J_left_SO3(phi)
        rhs = tr.mm(torch_se3.exp_SO3(phi), torch_se3.J_left_SO3(-phi))
        self.assertTrue(tr.allclose(lhs, rhs, atol=1e-6))

    def test_cuda_device_and_grad(self):
        C = torch_se3.exp_SO3(tr.tensor([0.1, 0.2, 0.3], requires_grad=True).cuda())
        self.assertTrue(C.requires_grad)
        self.assertTrue(C.is_cuda)

        C = torch_se3.exp_SO3(tr.tensor([0.0, 0.0, 0.0], requires_grad=True).cuda())
        self.assertTrue(C.requires_grad)
        self.assertTrue(C.is_cuda)

        phi = torch_se3.log_SO3(C)
        self.assertTrue(phi.requires_grad)
        self.assertTrue(phi.is_cuda)

        phi = torch_se3.log_SO3(tr.eye(3, 3, requires_grad=True).cuda())
        self.assertTrue(phi.requires_grad)
        self.assertTrue(phi.is_cuda)

        phi = torch_se3.log_SO3_eigen(C)
        self.assertTrue(phi.requires_grad)
        self.assertTrue(phi.is_cuda)

        phi = torch_se3.log_SO3_eigen(tr.eye(3, 3, requires_grad=True).cuda())
        self.assertTrue(phi.requires_grad)
        self.assertTrue(phi.is_cuda)

        v_skewed = torch_se3.skew3(tr.tensor([0.1, 0.2, 0.3], requires_grad=True).cuda())
        self.assertTrue(v_skewed.requires_grad)
        self.assertTrue(v_skewed.is_cuda)

        v = torch_se3.unskew3(v_skewed)
        self.assertTrue(v.requires_grad)
        self.assertTrue(v.is_cuda)

        J = torch_se3.J_left_SO3(tr.tensor([0.1, 0.2, 0.3], requires_grad=True).cuda())
        self.assertTrue(J.requires_grad)
        self.assertTrue(J.is_cuda)

        J = torch_se3.J_left_SO3(tr.tensor([0.0, 0.0, 0.0], requires_grad=True).cuda())
        self.assertTrue(J.requires_grad)
        self.assertTrue(J.is_cuda)

        J = torch_se3.J_left_SO3_inv(tr.tensor([0.1, 0.2, 0.3], requires_grad=True).cuda())
        self.assertTrue(J.requires_grad)
        self.assertTrue(J.is_cuda)

        J = torch_se3.J_left_SO3_inv(tr.tensor([0.0, 0.0, 0.0], requires_grad=True).cuda())
        self.assertTrue(J.requires_grad)
        self.assertTrue(J.is_cuda)

    def test_no_cuda_device_and_grad(self):
        C = torch_se3.exp_SO3(tr.tensor([0.1, 0.2, 0.3], requires_grad=True))
        self.assertTrue(C.requires_grad)
        self.assertFalse(C.is_cuda)

        C = torch_se3.exp_SO3(tr.tensor([0.0, 0.0, 0.0], requires_grad=True))
        self.assertTrue(C.requires_grad)
        self.assertFalse(C.is_cuda)

        phi = torch_se3.log_SO3(C)
        self.assertTrue(phi.requires_grad)
        self.assertFalse(phi.is_cuda)

        phi = torch_se3.log_SO3(tr.eye(3, 3, requires_grad=True))
        self.assertTrue(phi.requires_grad)
        self.assertFalse(phi.is_cuda)

        phi = torch_se3.log_SO3_eigen(C)
        self.assertTrue(phi.requires_grad)
        self.assertFalse(phi.is_cuda)

        phi = torch_se3.log_SO3_eigen(tr.eye(3, 3, requires_grad=True))
        self.assertTrue(phi.requires_grad)
        self.assertFalse(phi.is_cuda)

        v_skewed = torch_se3.skew3(tr.tensor([0.1, 0.2, 0.3], requires_grad=True))
        self.assertTrue(v_skewed.requires_grad)
        self.assertFalse(v_skewed.is_cuda)

        v = torch_se3.unskew3(v_skewed)
        self.assertTrue(v.requires_grad)
        self.assertFalse(v.is_cuda)

        J = torch_se3.J_left_SO3(tr.tensor([0.1, 0.2, 0.3], requires_grad=True))
        self.assertTrue(J.requires_grad)
        self.assertFalse(J.is_cuda)

        J = torch_se3.J_left_SO3(tr.tensor([0.0, 0.0, 0.0], requires_grad=True))
        self.assertTrue(J.requires_grad)
        self.assertFalse(J.is_cuda)

        J = torch_se3.J_left_SO3_inv(tr.tensor([0.1, 0.2, 0.3], requires_grad=True))
        self.assertTrue(J.requires_grad)
        self.assertFalse(J.is_cuda)

        J = torch_se3.J_left_SO3_inv(tr.tensor([0.0, 0.0, 0.0], requires_grad=True))
        self.assertTrue(J.requires_grad)
        self.assertFalse(J.is_cuda)

    def test_simple_grad(self):
        v = tr.tensor([0.1, 0.2, 0.3], requires_grad=True)
        C = torch_se3.exp_SO3(v)
        v2 = torch_se3.log_SO3(C)
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[0], v, retain_graph=True)[0],
                                    tr.tensor([1., 0., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[1], v, retain_graph=True)[0],
                                    tr.tensor([0., 1., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[2], v, retain_graph=True)[0],
                                    tr.tensor([0., 0., 1.]), atol=1e-7))

        v = tr.tensor([0.0, 0.0, 0.0], requires_grad=True)
        C = torch_se3.exp_SO3(v)
        v2 = torch_se3.log_SO3(C)
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[0], v, retain_graph=True)[0],
                                    tr.tensor([1., 0., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[1], v, retain_graph=True)[0],
                                    tr.tensor([0., 1., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[2], v, retain_graph=True)[0],
                                    tr.tensor([0., 0., 1.]), atol=1e-7))

        v = tr.tensor([0.1, 0.2, 0.3], requires_grad=True)
        C = torch_se3.exp_SO3(v)
        v2 = torch_se3.unskew3(tr.mm(C - tr.eye(3, 3), torch_se3.J_left_SO3_inv(v)))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[0], v, retain_graph=True)[0],
                                    tr.tensor([1., 0., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[1], v, retain_graph=True)[0],
                                    tr.tensor([0., 1., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[2], v, retain_graph=True)[0],
                                    tr.tensor([0., 0., 1.]), atol=1e-7))

        v = tr.tensor([0.1, 0.2, 0.3], requires_grad=True)
        C = torch_se3.exp_SO3(v)
        v2 = torch_se3.unskew3(tr.mm(C - tr.eye(3, 3), torch_se3.J_left_SO3(v).inverse()))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[0], v, retain_graph=True)[0],
                                    tr.tensor([1., 0., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[1], v, retain_graph=True)[0],
                                    tr.tensor([0., 1., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[2], v, retain_graph=True)[0],
                                    tr.tensor([0., 0., 1.]), atol=1e-7))

    def test_se3_grad(self):
        pass


class Test_batched_torch_se3(unittest.TestCase):
    def close(self, t1, t2, rtol=1e-05, atol=1e-08):
        self.assertTrue(tr.allclose(t1, t2, atol=atol, rtol=rtol))

    def test_skew_unskew(self):
        dat = tr.rand([100, 3])
        ret = over_batch(torch_se3.skew3, dat)
        self.close(ret, torch_se3.skew3_b(dat), atol=0)
        self.close(over_batch(torch_se3.unskew3, ret), torch_se3.unskew3_b(ret), atol=0)

    def test_exp_SO3(self):
        n = 100
        dat = tr.cat([tr.rand([n, 3]), tr.rand([n, 3]) * 1e-10], dim=0)
        dat = dat[tr.randperm(len(dat))]
        ret = over_batch(torch_se3.exp_SO3, dat)
        self.close(ret, torch_se3.exp_SO3_b(dat), atol=0)

    def test_log_SO3(self):
        n = 100
        dat = over_batch(torch_se3.exp_SO3, tr.cat([tr.rand([n, 3]),
                                                    tr.rand([n, 3]) * 1e-10,
                                                    tr.rand([n, 3]) * 1e-8], dim=0))
        dat = dat[tr.randperm(len(dat))]
        ret = over_batch(torch_se3.log_SO3, dat)
        self.close(ret, torch_se3.log_SO3_b(dat), atol=0)

        dat.requires_grad = True
        dat = dat.cuda()
        ret = torch_se3.log_SO3_b(dat)
        self.assertTrue(ret.is_cuda)
        self.assertTrue(ret.requires_grad)

    def test_exp_SO3_and_log_SO3_grad(self):
        n = 5
        dat = tr.cat([tr.rand([n, 3]), tr.rand([n, 3]) * 1e-10], dim=0)
        dat = dat[tr.randperm(len(dat))]
        dat.requires_grad = True
        dat = dat.cuda()

        C = torch_se3.exp_SO3_b(dat)
        phi = torch_se3.log_SO3_b(C)

        print("hello")

    def test_J_left_SO3_inv(self):
        pass

    def test_cuda_and_grad(self):
        pass

    def test_gradient(self):
        pass


if __name__ == '__main__':
    Test_batched_torch_se3().test_exp_SO3_and_log_SO3_grad()
    # unittest.main()
