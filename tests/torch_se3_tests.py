import unittest
import torch as tr
import scipy.linalg as slinalg
import se3
import numpy as np
import torch_se3
import time

tr.manual_seed(0)
tr.set_printoptions(precision=10)


def over_batch(fn, param):
    batches = []

    for i in range(0, len(param)):
        batches.append(fn(param[i]))

    if tr.is_tensor(param):
        return tr.stack(batches)
    return np.stack(batches)


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
        self.assertEqual(t1.shape, t2.shape)
        self.assertTrue(tr.allclose(t1, t2, atol=atol, rtol=rtol))

    def test_skew_unskew(self):
        dat = tr.rand([10000, 3])
        ret = over_batch(torch_se3.skew3, dat)
        self.close(ret, torch_se3.skew3_b(tr.unsqueeze(dat, -1)), atol=0)
        self.close(tr.unsqueeze(over_batch(torch_se3.unskew3, ret), -1), torch_se3.unskew3_b(ret), atol=0)

    def test_exp_SO3(self):
        n = 10000
        dat0 = tr.cat([tr.zeros(n, 3, 1), tr.rand([n, 3, 1]), tr.rand([n, 3, 1]) * 1e-10], dim=0)
        dat0 = dat0[tr.randperm(len(dat0))]
        self.close(over_batch(torch_se3.exp_SO3, dat0), torch_se3.exp_SO3_b(dat0), atol=1e-7)

        # of one either near or far from zero
        dat = tr.zeros(n, 3, 1)
        self.close(over_batch(torch_se3.exp_SO3, dat), torch_se3.exp_SO3_b(dat), atol=0)
        dat = tr.rand([n, 3, 1])
        self.close(over_batch(torch_se3.exp_SO3, dat), torch_se3.exp_SO3_b(dat), atol=1e-7)

        # one item
        dat = tr.zeros(1, 3, 1)
        self.close(over_batch(torch_se3.exp_SO3, dat), torch_se3.exp_SO3_b(dat), atol=0)
        dat = tr.rand([1, 3, 1])
        self.close(over_batch(torch_se3.exp_SO3, dat), torch_se3.exp_SO3_b(dat), atol=1e-7)

        # grad and cuda
        dat0.requires_grad = True
        dat0 = dat0.cuda()
        ret = torch_se3.exp_SO3_b(dat0)
        self.assertTrue(ret.is_cuda)
        self.assertTrue(ret.requires_grad)

    def test_log_SO3(self):
        n = 10000
        dat0 = over_batch(se3.exp_SO3, np.concatenate([np.zeros([n, 3]),
                                                       np.random.rand(n, 3),
                                                       np.random.rand(n, 3) * 1e-2,
                                                       np.random.rand(n, 3) * 1e-3,
                                                       np.random.rand(n, 3) * 1e-4,
                                                       np.random.rand(n, 3) * 1e-5,
                                                       np.random.rand(n, 3) * 1e-6], 0))
        dat0 = tr.tensor(dat0, dtype=tr.float32)
        dat0 = dat0[tr.randperm(len(dat0))]
        ret = tr.unsqueeze(over_batch(torch_se3.log_SO3, dat0), -1)
        self.close(ret, torch_se3.log_SO3_b(dat0), atol=0)

        # of one either near or far from zero
        dat = over_batch(torch_se3.exp_SO3, tr.rand([n, 3]))
        self.close(tr.unsqueeze(over_batch(torch_se3.log_SO3, dat), -1), torch_se3.log_SO3_b(dat), atol=0)
        dat = tr.eye(3, 3).repeat(n, 1, 1)
        self.close(tr.zeros(n, 3, 1), torch_se3.log_SO3_b(dat), atol=0)

        # one item
        dat = over_batch(torch_se3.exp_SO3, tr.rand([1, 3]))
        self.close(tr.unsqueeze(over_batch(torch_se3.log_SO3, dat), -1), torch_se3.log_SO3_b(dat), atol=0)
        dat = tr.eye(3, 3).repeat(1, 1, 1)
        self.close(tr.zeros(1, 3, 1), torch_se3.log_SO3_b(dat), atol=0)

        # grad and cuda
        dat0.requires_grad = True
        dat0 = dat0.cuda()
        ret = torch_se3.log_SO3_b(dat0)
        self.assertTrue(ret.is_cuda)
        self.assertTrue(ret.requires_grad)

    def test_log_SO3_close_to_pi(self):
        dat = over_batch(torch_se3.exp_SO3, tr.tensor([0.1, -0.2, 3.13]).view(1, 3, 1))
        # torch_se3.log_SO3_b(dat)
        self.assertRaises(ValueError, torch_se3.log_SO3_b, dat)

    def test_log_SO3_double_batch_dims(self):
        n1 = 100
        n2 = 500
        dat0 = []
        for i in range(0, n2):
            d = over_batch(se3.exp_SO3, np.concatenate([np.zeros([n1, 3]),
                                                        np.random.rand(n1, 3),
                                                        np.random.rand(n1, 3) * 1e-2,
                                                        np.random.rand(n1, 3) * 1e-3,
                                                        np.random.rand(n1, 3) * 1e-4,
                                                        np.random.rand(n1, 3) * 1e-5,
                                                        np.random.rand(n1, 3) * 1e-6], 0))
            np.random.shuffle(d)
            dat0.append(d)
        dat0 = np.stack(dat0)
        np.random.shuffle(dat0)
        dat0 = tr.tensor(dat0, dtype=tr.float32)
        print("test_log_SO3_double_batch_dims Finished computing dat0")

        ret = []
        for i in range(0, dat0.shape[0]):
            ret.append(over_batch(torch_se3.log_SO3, dat0[i]))
        ret = tr.unsqueeze(tr.stack(ret), -1)
        print("test_log_SO3_double_batch_dims Finished computing ret")

        start_time = time.time()
        self.close(ret, torch_se3.log_SO3_b(dat0), atol=0)
        print("test_log_SO3_double_batch_dims elapsed time %.5f" % (time.time() - start_time))

        dat = over_batch(torch_se3.exp_SO3, tr.rand([1, 3])).view(1, 1, 3, 3)
        self.close(torch_se3.log_SO3(dat[0, 0]).view(1, 1, 3, 1), torch_se3.log_SO3_b(dat), atol=0)
        dat = tr.eye(3, 3).view(1, 1, 3, 3)
        self.close(tr.zeros(1, 1, 3, 1), torch_se3.log_SO3_b(dat), atol=0)

    def test_exp_SO3_and_log_SO3_grad(self):
        n = 100
        dat = tr.cat([tr.rand([n, 3, 1]), tr.rand([n, 3, 1]) * 1e-10], dim=0)
        dat = dat[tr.randperm(len(dat))]
        dat.requires_grad = True
        dat = dat.cuda()

        C = torch_se3.exp_SO3_b(dat)
        phi = torch_se3.log_SO3_b(C)

        for i in range(0, phi.shape[0]):
            for j in range(0, phi.shape[1]):
                grad = tr.autograd.grad(phi[i, j], dat, retain_graph=True)[0]
                cmp = tr.zeros_like(phi)
                cmp[i, j] = 1
                self.close(grad[i], cmp[i], atol=1e-6)
                not_i = list(range(0, phi.shape[0]))
                not_i.remove(i)
                self.close(grad[not_i], cmp[not_i], atol=0)

    def test_J_left_SO3_inv(self):
        n = 10000
        dat0 = tr.cat([tr.zeros(n, 3, 1), tr.rand([n, 3, 1]), tr.rand([n, 3, 1]) * 1e-10], dim=0)
        dat0 = dat0[tr.randperm(len(dat0))]
        self.close(over_batch(torch_se3.J_left_SO3_inv, dat0), torch_se3.J_left_SO3_inv_b(dat0), atol=1e-7)

        # of one either near or far from zero
        dat = tr.zeros(n, 3, 1)
        self.close(over_batch(torch_se3.J_left_SO3_inv, dat), torch_se3.J_left_SO3_inv_b(dat), atol=0)
        dat = tr.rand([n, 3, 1])
        self.close(over_batch(torch_se3.J_left_SO3_inv, dat), torch_se3.J_left_SO3_inv_b(dat), atol=1e-7)

        # one item
        dat = tr.zeros(1, 3, 1)
        self.close(over_batch(torch_se3.J_left_SO3_inv, dat), torch_se3.J_left_SO3_inv_b(dat), atol=0)
        dat = tr.rand([1, 3, 1])
        self.close(over_batch(torch_se3.J_left_SO3_inv, dat), torch_se3.J_left_SO3_inv_b(dat), atol=1e-7)

        # grad and cuda
        dat0.requires_grad = True
        dat0 = dat0.cuda()
        ret = torch_se3.J_left_SO3_inv_b(dat0)
        self.assertTrue(ret.is_cuda)
        self.assertTrue(ret.requires_grad)

    def test_test_J_left_SO3_inv_grad(self):
        n = 100
        dat = tr.cat([tr.rand([n, 3, 1]), tr.rand([n, 3, 1]) * 1e-10], dim=0)
        dat = dat[tr.randperm(len(dat))]
        dat.requires_grad = True
        dat = dat.cuda()

        C = tr.eye(3, 3, device=dat.device).repeat(dat.size(0), 1, 1) + \
            tr.matmul(torch_se3.skew3_b(dat), torch_se3.J_left_SO3_inv_b(dat).inverse())
        phi = torch_se3.log_SO3_b(C)

        for i in range(0, phi.shape[0]):
            for j in range(0, phi.shape[1]):
                grad = tr.autograd.grad(phi[i, j], dat, retain_graph=True)[0]
                cmp = tr.zeros_like(phi)
                cmp[i, j] = 1
                self.close(grad[i], cmp[i], atol=1e-6)
                not_i = list(range(0, phi.shape[0]))
                not_i.remove(i)
                self.close(grad[not_i], cmp[not_i], atol=0)


if __name__ == '__main__':
    # Test_batched_torch_se3().test_exp_SO3()
    # Test_batched_torch_se3().test_log_SO3()
    # Test_batched_torch_se3().test_log_SO3_double_batch_dims()
    # Test_batched_torch_se3().test_exp_SO3_and_log_SO3_grad()
    # Test_batched_torch_se3().test_J_left_SO3_inv()
    # Test_batched_torch_se3().test_test_J_left_SO3_inv_grad()
    # Test_batched_torch_se3().test_skew_unskew()
    Test_batched_torch_se3().test_log_SO3_close_to_pi()
    # unittest.main(verbosity=10)
