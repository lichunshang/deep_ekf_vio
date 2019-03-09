import unittest
import torch
import scipy.linalg
import se3_math
import numpy as np
from model import TorchSE3


class Test_TorchSE3(unittest.TestCase):
    def test_skew_unskew(self):
        self.assertTrue(torch.allclose(TorchSE3.unskew3(TorchSE3.skew3(torch.tensor([1., 2., 3.]))),
                                       torch.tensor([1., 2., 3.]), atol=1e-8))
        self.assertTrue(torch.allclose(TorchSE3.skew3(torch.tensor([1., 2., 3.])).transpose(0, 1),
                                       -TorchSE3.skew3(torch.tensor([1., 2., 3.])), atol=1e-8))
        self.assertTrue(torch.allclose(torch.mm(TorchSE3.skew3(torch.tensor([1., 2., 3.])),
                                                torch.tensor([[1.], [2.], [3.]])),
                                       torch.tensor([0., 0., 0.]), atol=1e-8))

    def test_exp_SO3(self):
        self.assertTrue(torch.allclose(TorchSE3.exp_SO3(torch.tensor([1., 2., 3.])),
                                       torch.tensor(np.array(scipy.linalg.expm(se3_math.skew3([1, 2, 3]))),
                                                    dtype=torch.float32), atol=1e-8))
        self.assertTrue(torch.allclose(TorchSE3.exp_SO3(torch.tensor([1e-8, 2e-8, 3e-8])),
                                       torch.tensor(np.array(scipy.linalg.expm(se3_math.skew3([1e-8, 2e-8, 3e-8]))),
                                                    dtype=torch.float32), atol=1e-8))
        self.assertTrue(torch.allclose(TorchSE3.exp_SO3(torch.tensor([0., 0., 0.])), torch.eye(3, 3), atol=1e-8))

        C = TorchSE3.exp_SO3(torch.tensor([-1, 0.2, -0.75]))
        u = torch.tensor([[1, -2, 0.1]]).transpose(0, 1)
        self.assertTrue(torch.allclose(TorchSE3.skew3(torch.mm(C, u)),
                                       torch.mm(torch.mm(C, TorchSE3.skew3(u)), C.transpose(0, 1)), atol=1e-7))

        exp_Cu = TorchSE3.exp_SO3(torch.mm(C, u))
        C_expu_Ct = torch.mm(torch.mm(C, TorchSE3.exp_SO3(u)), C.transpose(0, 1))
        self.assertTrue(torch.allclose(exp_Cu, C_expu_Ct, atol=1e-7))

    def test_log_SO3(self):
        C = torch.tensor(se3_math.exp_SO3([-100., 200., -80]), dtype=torch.float32)
        log_C_scipy = TorchSE3.unskew3(torch.tensor(np.real(scipy.linalg.logm(C)), dtype=torch.float32))
        log_C = TorchSE3.log_SO3(C)
        self.assertTrue(torch.allclose(log_C, log_C_scipy, atol=1e-6))

        C = torch.tensor(se3_math.exp_SO3([-1., 2., -0.8]), dtype=torch.float32)
        log_C_scipy = TorchSE3.unskew3(torch.tensor(np.real(scipy.linalg.logm(C)), dtype=torch.float32))
        log_C = TorchSE3.log_SO3(C)
        self.assertTrue(torch.allclose(log_C, log_C_scipy, atol=1e-8))

        C = torch.tensor(se3_math.exp_SO3([1e-8, -2e-8, 3e-8]), dtype=torch.float32)
        log_C = TorchSE3.log_SO3(C)
        self.assertTrue(torch.allclose(log_C, torch.tensor([1e-8, -2e-8, 3e-8]), atol=1e-10))

        C = torch.tensor(se3_math.exp_SO3([1e-9, -2e-9, 3e-9]), dtype=torch.float32)
        log_C = TorchSE3.log_SO3(C)
        self.assertTrue(torch.allclose(log_C, torch.tensor([1e-9, -2e-9, 3e-9]), atol=1e-10))

        C = torch.tensor(se3_math.exp_SO3([0., 0., 0.]), dtype=torch.float32)
        log_C = TorchSE3.log_SO3(C)
        self.assertTrue(torch.allclose(log_C, torch.zeros(3, ), atol=1e-10))

    def test_log_SO3_eigen(self):
        C = torch.tensor(se3_math.exp_SO3([-100., 200., -80]), dtype=torch.float32)
        log_C_scipy = TorchSE3.unskew3(torch.tensor(np.real(scipy.linalg.logm(C)), dtype=torch.float32))
        log_C = TorchSE3.log_SO3_eigen(C)
        self.assertTrue(torch.allclose(log_C, log_C_scipy, atol=1e-6))

        C = torch.tensor(se3_math.exp_SO3([-1., 2., -0.8]), dtype=torch.float32)
        log_C_scipy = TorchSE3.unskew3(torch.tensor(np.real(scipy.linalg.logm(C)), dtype=torch.float32))
        log_C = TorchSE3.log_SO3_eigen(C)
        self.assertTrue(torch.allclose(log_C, log_C_scipy, atol=1e-8))

        C = torch.tensor(se3_math.exp_SO3([1e-3, -2e-3, 3e-3]), dtype=torch.float32)
        log_C = TorchSE3.log_SO3_eigen(C)
        self.assertTrue(torch.allclose(log_C, torch.tensor([1e-3, -2e-3, 3e-3]), atol=1e-5))

        C = torch.tensor(se3_math.exp_SO3([0., 0., 0.]), dtype=torch.float32)
        log_C = TorchSE3.log_SO3_eigen(C)
        self.assertTrue(torch.allclose(log_C, torch.zeros(3, ), atol=1e-10))

        C = torch.tensor(se3_math.exp_SO3([0.1, -0.2, 3.133624802220163]), dtype=torch.float32)
        log_C = TorchSE3.log_SO3_eigen(C)
        self.assertTrue(torch.allclose(log_C, torch.tensor([0.1, -0.2, 3.133624802220163]), atol=1e-6) or
                        torch.allclose(-log_C, torch.tensor([0.1, -0.2, 3.133624802220163]), atol=1e-6))

        C = torch.tensor(se3_math.exp_SO3([-np.pi, 0, 0]), dtype=torch.float32)
        log_C = TorchSE3.log_SO3_eigen(C)
        self.assertTrue(torch.allclose(log_C, torch.tensor([-np.pi, 0, 0]), atol=1e-6) or
                        torch.allclose(-log_C, torch.tensor([-np.pi, 0, 0]), atol=1e-6))

    def test_J_left_SO3_inv(self):
        pass


if __name__ == '__main__':
    unittest.main()
