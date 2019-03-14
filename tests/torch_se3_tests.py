import unittest
import torch as tr
import scipy.linalg as slinalg
import se3_math as se3
import numpy as np
from model import TorchSE3 as trse3


class Test_TorchSE3(unittest.TestCase):
    def test_skew_unskew(self):
        self.assertTrue(tr.allclose(trse3.unskew3(trse3.skew3(tr.tensor([1., 2., 3.]))),
                                    tr.tensor([1., 2., 3.]), atol=1e-8))
        self.assertTrue(tr.allclose(trse3.skew3(tr.tensor([1., 2., 3.])).transpose(0, 1),
                                    -trse3.skew3(tr.tensor([1., 2., 3.])), atol=1e-8))
        self.assertTrue(tr.allclose(tr.mm(trse3.skew3(tr.tensor([1., 2., 3.])), tr.tensor([[1.], [2.], [3.]])),
                                    tr.tensor([0., 0., 0.]), atol=1e-8))

    def test_exp_SO3(self):
        self.assertTrue(tr.allclose(trse3.exp_SO3(tr.tensor([1., 2., 3.])),
                                    tr.tensor(np.array(slinalg.expm(se3.skew3([1, 2, 3]))),
                                              dtype=tr.float32), atol=1e-8))
        self.assertTrue(tr.allclose(trse3.exp_SO3(tr.tensor([1e-8, 2e-8, 3e-8])),
                                    tr.tensor(np.array(slinalg.expm(se3.skew3([1e-8, 2e-8, 3e-8]))),
                                              dtype=tr.float32), atol=1e-8))
        self.assertTrue(tr.allclose(trse3.exp_SO3(tr.tensor([0., 0., 0.])), tr.eye(3, 3), atol=1e-8))

        C = trse3.exp_SO3(tr.tensor([-1, 0.2, -0.75]))
        u = tr.tensor([[1, -2, 0.1]]).transpose(0, 1)
        self.assertTrue(tr.allclose(trse3.skew3(tr.mm(C, u)),
                                    tr.mm(tr.mm(C, trse3.skew3(u)), C.transpose(0, 1)), atol=1e-7))

        exp_Cu = trse3.exp_SO3(tr.mm(C, u))
        C_expu_Ct = tr.mm(tr.mm(C, trse3.exp_SO3(u)), C.transpose(0, 1))
        self.assertTrue(tr.allclose(exp_Cu, C_expu_Ct, atol=1e-7))

    def test_log_SO3(self):
        C = tr.tensor(se3.exp_SO3([-100., 200., -80]), dtype=tr.float32)
        log_C_scipy = trse3.unskew3(tr.tensor(np.real(slinalg.logm(C)), dtype=tr.float32))
        log_C = trse3.log_SO3(C)
        self.assertTrue(tr.allclose(log_C, log_C_scipy, atol=1e-6))

        C = tr.tensor(se3.exp_SO3([-1., 2., -0.8]), dtype=tr.float32)
        log_C_scipy = trse3.unskew3(tr.tensor(np.real(slinalg.logm(C)), dtype=tr.float32))
        log_C = trse3.log_SO3(C)
        self.assertTrue(tr.allclose(log_C, log_C_scipy, atol=1e-8))

        C = tr.tensor(se3.exp_SO3([1e-8, -2e-8, 3e-8]), dtype=tr.float32)
        log_C = trse3.log_SO3(C)
        self.assertTrue(tr.allclose(log_C, tr.tensor([1e-8, -2e-8, 3e-8]), atol=1e-10))

        C = tr.tensor(se3.exp_SO3([1e-9, -2e-9, 3e-9]), dtype=tr.float32)
        log_C = trse3.log_SO3(C)
        self.assertTrue(tr.allclose(log_C, tr.tensor([1e-9, -2e-9, 3e-9]), atol=1e-10))

        C = tr.tensor(se3.exp_SO3([0., 0., 0.]), dtype=tr.float32)
        log_C = trse3.log_SO3(C)
        self.assertTrue(tr.allclose(log_C, tr.zeros(3, ), atol=1e-10))

    def test_log_SO3_eigen(self):
        C = tr.tensor(se3.exp_SO3([-100., 200., -80]), dtype=tr.float32)
        log_C_scipy = trse3.unskew3(tr.tensor(np.real(slinalg.logm(C)), dtype=tr.float32))
        log_C = trse3.log_SO3_eigen(C)
        self.assertTrue(tr.allclose(log_C, log_C_scipy, atol=1e-6))

        C = tr.tensor(se3.exp_SO3([-1., 2., -0.8]), dtype=tr.float32)
        log_C_scipy = trse3.unskew3(tr.tensor(np.real(slinalg.logm(C)), dtype=tr.float32))
        log_C = trse3.log_SO3_eigen(C)
        self.assertTrue(tr.allclose(log_C, log_C_scipy, atol=1e-8))

        C = tr.tensor(se3.exp_SO3([1e-3, -2e-3, 3e-3]), dtype=tr.float32)
        log_C = trse3.log_SO3_eigen(C)
        self.assertTrue(tr.allclose(log_C, tr.tensor([1e-3, -2e-3, 3e-3]), atol=1e-5))

        C = tr.tensor(se3.exp_SO3([0., 0., 0.]), dtype=tr.float32)
        log_C = trse3.log_SO3_eigen(C)
        self.assertTrue(tr.allclose(log_C, tr.zeros(3, ), atol=1e-10))

        C = tr.tensor(se3.exp_SO3([0.1, -0.2, 3.133624802220163]), dtype=tr.float32)
        log_C = trse3.log_SO3_eigen(C)
        self.assertTrue(tr.allclose(log_C, tr.tensor([0.1, -0.2, 3.133624802220163]), atol=1e-6) or
                        tr.allclose(-log_C, tr.tensor([0.1, -0.2, 3.133624802220163]), atol=1e-6))

        C = tr.tensor(se3.exp_SO3([-np.pi, 0, 0]), dtype=tr.float32)
        log_C = trse3.log_SO3_eigen(C)
        self.assertTrue(tr.allclose(log_C, tr.tensor([-np.pi, 0, 0]), atol=1e-6) or
                        tr.allclose(-log_C, tr.tensor([-np.pi, 0, 0]), atol=1e-6))

    def test_J_left_SO3_inv(self):
        phi = tr.tensor([-0.2, 1.5, -2])
        J = trse3.J_left_SO3(phi)
        J_inv = trse3.J_left_SO3_inv(phi)
        self.assertTrue(tr.allclose(J.inverse(), J_inv, atol=1e-6))

        self.assertTrue(tr.allclose(trse3.J_left_SO3(tr.tensor([0., 0., 0.])), tr.eye(3, 3), atol=1e-15))
        self.assertTrue(tr.allclose(trse3.J_left_SO3_inv(tr.tensor([0., 0., 0.])), tr.eye(3, 3), atol=1e-15))

        phi = tr.tensor([1e-7, -2e-7, 3e-7])
        J = trse3.J_left_SO3(phi)
        J_inv = trse3.J_left_SO3_inv(phi)
        self.assertTrue(tr.allclose(J.inverse(), J_inv, atol=1e-10))

        phi = tr.tensor([2, -0.5, -1.1])
        lhs = tr.eye(3, 3) + tr.mm(trse3.skew3(phi), trse3.J_left_SO3(phi))
        rhs = trse3.exp_SO3(phi)
        self.assertTrue(tr.allclose(lhs, rhs, atol=1e-6))

        lhs = trse3.J_left_SO3(phi)
        rhs = tr.mm(trse3.exp_SO3(phi), trse3.J_left_SO3(-phi))
        self.assertTrue(tr.allclose(lhs, rhs, atol=1e-6))

    def test_cuda_device_and_grad(self):
        C = trse3.exp_SO3(tr.tensor([0.1, 0.2, 0.3], requires_grad=True).cuda())
        self.assertTrue(C.requires_grad)
        self.assertTrue(C.is_cuda)

        C = trse3.exp_SO3(tr.tensor([0.0, 0.0, 0.0], requires_grad=True).cuda())
        self.assertTrue(C.requires_grad)
        self.assertTrue(C.is_cuda)

        phi = trse3.log_SO3(C)
        self.assertTrue(phi.requires_grad)
        self.assertTrue(phi.is_cuda)

        phi = trse3.log_SO3(tr.eye(3, 3, requires_grad=True).cuda())
        self.assertTrue(phi.requires_grad)
        self.assertTrue(phi.is_cuda)

        phi = trse3.log_SO3_eigen(C)
        self.assertTrue(phi.requires_grad)
        self.assertTrue(phi.is_cuda)

        phi = trse3.log_SO3_eigen(tr.eye(3, 3, requires_grad=True).cuda())
        self.assertTrue(phi.requires_grad)
        self.assertTrue(phi.is_cuda)

        v_skewed = trse3.skew3(tr.tensor([0.1, 0.2, 0.3], requires_grad=True).cuda())
        self.assertTrue(v_skewed.requires_grad)
        self.assertTrue(v_skewed.is_cuda)

        v = trse3.unskew3(v_skewed)
        self.assertTrue(v.requires_grad)
        self.assertTrue(v.is_cuda)

        J = trse3.J_left_SO3(tr.tensor([0.1, 0.2, 0.3], requires_grad=True).cuda())
        self.assertTrue(J.requires_grad)
        self.assertTrue(J.is_cuda)

        J = trse3.J_left_SO3(tr.tensor([0.0, 0.0, 0.0], requires_grad=True).cuda())
        self.assertTrue(J.requires_grad)
        self.assertTrue(J.is_cuda)

        J = trse3.J_left_SO3_inv(tr.tensor([0.1, 0.2, 0.3], requires_grad=True).cuda())
        self.assertTrue(J.requires_grad)
        self.assertTrue(J.is_cuda)

        J = trse3.J_left_SO3_inv(tr.tensor([0.0, 0.0, 0.0], requires_grad=True).cuda())
        self.assertTrue(J.requires_grad)
        self.assertTrue(J.is_cuda)

    def test_no_cuda_device_and_grad(self):
        C = trse3.exp_SO3(tr.tensor([0.1, 0.2, 0.3], requires_grad=True))
        self.assertTrue(C.requires_grad)
        self.assertFalse(C.is_cuda)

        C = trse3.exp_SO3(tr.tensor([0.0, 0.0, 0.0], requires_grad=True))
        self.assertTrue(C.requires_grad)
        self.assertFalse(C.is_cuda)

        phi = trse3.log_SO3(C)
        self.assertTrue(phi.requires_grad)
        self.assertFalse(phi.is_cuda)

        phi = trse3.log_SO3(tr.eye(3, 3, requires_grad=True))
        self.assertTrue(phi.requires_grad)
        self.assertFalse(phi.is_cuda)

        phi = trse3.log_SO3_eigen(C)
        self.assertTrue(phi.requires_grad)
        self.assertFalse(phi.is_cuda)

        phi = trse3.log_SO3_eigen(tr.eye(3, 3, requires_grad=True))
        self.assertTrue(phi.requires_grad)
        self.assertFalse(phi.is_cuda)

        v_skewed = trse3.skew3(tr.tensor([0.1, 0.2, 0.3], requires_grad=True))
        self.assertTrue(v_skewed.requires_grad)
        self.assertFalse(v_skewed.is_cuda)

        v = trse3.unskew3(v_skewed)
        self.assertTrue(v.requires_grad)
        self.assertFalse(v.is_cuda)

        J = trse3.J_left_SO3(tr.tensor([0.1, 0.2, 0.3], requires_grad=True))
        self.assertTrue(J.requires_grad)
        self.assertFalse(J.is_cuda)

        J = trse3.J_left_SO3(tr.tensor([0.0, 0.0, 0.0], requires_grad=True))
        self.assertTrue(J.requires_grad)
        self.assertFalse(J.is_cuda)

        J = trse3.J_left_SO3_inv(tr.tensor([0.1, 0.2, 0.3], requires_grad=True))
        self.assertTrue(J.requires_grad)
        self.assertFalse(J.is_cuda)

        J = trse3.J_left_SO3_inv(tr.tensor([0.0, 0.0, 0.0], requires_grad=True))
        self.assertTrue(J.requires_grad)
        self.assertFalse(J.is_cuda)

    def test_simple_grad(self):
        v = tr.tensor([0.1, 0.2, 0.3], requires_grad=True)
        C = trse3.exp_SO3(v)
        v2 = trse3.log_SO3(C)
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[0], v, retain_graph=True)[0],
                                    tr.tensor([1., 0., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[1], v, retain_graph=True)[0],
                                    tr.tensor([0., 1., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[2], v, retain_graph=True)[0],
                                    tr.tensor([0., 0., 1.]), atol=1e-7))

        v = tr.tensor([0.0, 0.0, 0.0], requires_grad=True)
        C = trse3.exp_SO3(v)
        v2 = trse3.log_SO3(C)
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[0], v, retain_graph=True)[0],
                                    tr.tensor([1., 0., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[1], v, retain_graph=True)[0],
                                    tr.tensor([0., 1., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[2], v, retain_graph=True)[0],
                                    tr.tensor([0., 0., 1.]), atol=1e-7))

        v = tr.tensor([0.1, 0.2, 0.3], requires_grad=True)
        C = trse3.exp_SO3(v)
        v2 = trse3.unskew3(tr.mm(C - tr.eye(3, 3), trse3.J_left_SO3_inv(v)))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[0], v, retain_graph=True)[0],
                                    tr.tensor([1., 0., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[1], v, retain_graph=True)[0],
                                    tr.tensor([0., 1., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[2], v, retain_graph=True)[0],
                                    tr.tensor([0., 0., 1.]), atol=1e-7))

        v = tr.tensor([0.1, 0.2, 0.3], requires_grad=True)
        C = trse3.exp_SO3(v)
        v2 = trse3.unskew3(tr.mm(C - tr.eye(3, 3), trse3.J_left_SO3(v).inverse()))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[0], v, retain_graph=True)[0],
                                    tr.tensor([1., 0., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[1], v, retain_graph=True)[0],
                                    tr.tensor([0., 1., 0.]), atol=1e-7))
        self.assertTrue(tr.allclose(tr.autograd.grad(v2[2], v, retain_graph=True)[0],
                                    tr.tensor([0., 0., 1.]), atol=1e-7))

    def test_se3_grad(self):
        pass


if __name__ == '__main__':
    unittest.main()
