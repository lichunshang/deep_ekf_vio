import unittest
from data_loader import SequenceData
from model import IMUKalmanFilter
import torch_se3
import torch
import numpy as np
from params import par
from log import logger
from se3 import log_SO3
import os
import scipy.linalg
from utils import Plotter


class Test_EKF(unittest.TestCase):

    @staticmethod
    def sub_block3x3(M, i, j):
        assert (len(M.shape) == 2 and M.shape[0] % 3 == 0 and M.shape[1] % 3 == 0)
        return M[i * 3:i * 3 + 3, j * 3:j * 3 + 3]

    # enforce non zero if in list and zero other wise
    def check_non_zero_3x3_sub_blocks(self, M, non_zero_sub_blocks, offset_i=0, offset_j=0):
        assert (len(M.shape) == 2 and M.shape[0] % 3 == 0 and M.shape[1] % 3 == 0)
        n_rows = M.shape[0] // 3
        n_cols = M.shape[1] // 3

        for i in range(0, n_rows):
            for j in range(0, n_cols):
                sub_block = Test_EKF.sub_block3x3(M, i, j)

                if (i + offset_i, j + offset_j,) in non_zero_sub_blocks:
                    self.assertFalse(torch.allclose(sub_block, torch.zeros(3, 3), atol=0, rtol=0))
                else:
                    self.assertTrue(torch.allclose(sub_block, torch.zeros(3, 3), atol=0, rtol=0))

    # def check_3x3_sub_blocks_property(self, M, sel_sub_blocks, fn, offset_i=0, offset_j=0):
    #     assert (len(M.shape) == 2 and M.shape[0] % 3 == 0 and M.shape[1] % 3 == 0)
    #     n_rows = M.shape[0] // 3
    #     n_cols = M.shape[1] // 3
    #
    #     for i in range(0, n_rows):
    #         for j in range(0, n_cols):
    #             sub_block = Test_EKF.sub_block3x3(M, i, j)
    #             if (i + offset_i, j + offset_j,) in sel_sub_blocks:
    #                 self.assertTrue(fn(sub_block))
    #
    # # enforce zero if in list and non zero otherwise
    # def check_zero_3x3_sub_blocks(self, M, zero_sub_blocks, offset_i=0, offset_j=0):
    #     assert (len(M.shape) == 2 and M.shape[0] % 3 == 0 and M.shape[1] % 3 == 0)
    #     n_rows = M.shape[0] // 3
    #     n_cols = M.shape[1] // 3
    #
    #     non_zero_sub_blocks = []
    #     for i in range(0, n_rows):
    #         for j in range(0, n_cols):
    #             if (i + offset_i, i + offset_j,) not in zero_sub_blocks:
    #                 non_zero_sub_blocks.append((i + offset_i, i + offset_j,))
    #
    #     return self.check_non_zero_3x3_sub_blocks(M, non_zero_sub_blocks, offset_i=0, offset_j=0)

    def plot_ekf_data(self, output_dir, timestamps, gt_poses, gt_vels, est_poses, est_states, g_const=9.80665):

        # convert all to np
        timestamps = np.array([np.array(item) for item in timestamps], dtype=np.float64)
        gt_poses = np.array([np.array(item) for item in gt_poses], dtype=np.float64)
        gt_vels = np.array([np.array(item) for item in gt_vels], dtype=np.float64)
        est_poses = np.array([np.array(item) for item in est_poses], dtype=np.float64)
        est_states = np.array([np.array(item) for item in est_states], dtype=np.float64)

        est_positions = []
        est_vels = []
        est_rots = []
        est_gravities = []
        est_ba = []
        est_bw = []

        for i in range(0, len(est_poses)):
            pose = np.linalg.inv(est_poses[i])
            g, C, r, v, bw, ba = IMUKalmanFilter.decode_state(torch.tensor(est_states[i]))
            est_positions.append(pose[0:3, 3])
            est_rots.append(log_SO3(pose[0:3, 0:3]))
            est_vels.append(np.array(v))
            est_gravities.append(np.array(g))
            est_bw.append(np.array(bw))
            est_ba.append(np.array(ba))

        est_positions = np.squeeze(est_positions)
        est_vels = np.squeeze(est_vels)
        est_rots = np.squeeze(est_rots)
        est_gravities = np.squeeze(est_gravities)
        est_bw = np.squeeze(est_bw)
        est_ba = np.squeeze(est_ba)

        gt_rots = np.array([log_SO3(p[:3, :3]) for p in gt_poses])
        gt_gravities = np.array([gt_poses[i, 0:3, 0:3].transpose().dot([0, 0, g_const])
                                 for i in range(0, len(gt_poses))])

        est_rots[:, 0] = np.unwrap(est_rots[:, 0])
        est_rots[:, 1] = np.unwrap(est_rots[:, 1])
        est_rots[:, 2] = np.unwrap(est_rots[:, 2])
        gt_rots[:, 0] = np.unwrap(gt_rots[:, 0])
        gt_rots[:, 1] = np.unwrap(gt_rots[:, 1])
        gt_rots[:, 2] = np.unwrap(gt_rots[:, 2])

        plotter = Plotter(output_dir)
        plotter.plot(([gt_poses[:, 0, 3], gt_poses[:, 1, 3]],
                      [est_positions[:, 0], est_positions[:, 1]],),
                     "x [m]", "y [m]", "XY Plot", labels=["gt_poses", "imu_int_pose"], equal_axes=True)
        plotter.plot(([gt_poses[:, 0, 3], gt_poses[:, 2, 3]],
                      [est_positions[:, 0], est_positions[:, 2]],),
                     "x [m]", "z [m]", "XZ Plot", labels=["gt_poses", "imu_int_pose"], equal_axes=True)
        plotter.plot(([gt_poses[:, 1, 3], gt_poses[:, 2, 3]],
                      [est_positions[:, 1], est_positions[:, 2]],),
                     "y [m]", "z [m]", "YZ Plot", labels=["gt_poses", "imu_int_pose"], equal_axes=True)

        plotter.plot(([timestamps, gt_poses[:, 0, 3]], [timestamps, est_positions[:, 0]],),
                     "t [s]", "p [m]", "Pos X", labels=["gt", "est"])
        plotter.plot(([timestamps, gt_poses[:, 1, 3]], [timestamps, est_positions[:, 1]],),
                     "t [s]", "p [m]", "Pos Y", labels=["gt", "est"])
        plotter.plot(([timestamps, gt_poses[:, 2, 3]], [timestamps, est_positions[:, 2]],),
                     "t [s]", "p [m]", "Pos Z", labels=["gt", "est"])

        plotter.plot(([timestamps, gt_vels[:, 0]], [timestamps, est_vels[:, 0]],),
                     "t [s]", "v [m/s]", "Vel X", labels=["gt", "est"])
        plotter.plot(([timestamps, gt_vels[:, 1]], [timestamps, est_vels[:, 1]],),
                     "t [s]", "v [m/s]", "Vel Y", labels=["gt", "est"])
        plotter.plot(([timestamps, gt_vels[:, 2]], [timestamps, est_vels[:, 2]],),
                     "t [s]", "v [m/s]", "Vel Z", labels=["gt", "est"])

        plotter.plot(([timestamps, gt_rots[:, 0]], [timestamps, est_rots[:, 0]],),
                     "t [s]", "rot [rad]", "Rot X", labels=["gt", "est"])
        plotter.plot(([timestamps, gt_rots[:, 1]], [timestamps, est_rots[:, 1]],),
                     "t [s]", "rot [rad]", "Rot Y", labels=["gt", "est"])
        plotter.plot(([timestamps, gt_rots[:, 2]], [timestamps, est_rots[:, 2]],),
                     "t [s]", "rot [rad]", "Rot Z", labels=["gt", "set"])

        plotter.plot(([timestamps, gt_gravities[:, 0]], [timestamps, est_gravities[:, 0]],),
                     "t [s]", "accel [m/s^2]", "Gravity X", labels=["gt", "est"])
        plotter.plot(([timestamps, gt_gravities[:, 1]], [timestamps, est_gravities[:, 1]],),
                     "t [s]", "accel [m/s^2]", "Gravity Y", labels=["gt", "est"])
        plotter.plot(([timestamps, gt_gravities[:, 2]], [timestamps, est_gravities[:, 2]],),
                     "t [s]", "accel [m/s^2]", "Gravity Z", labels=["gt", "est"])

        plotter.plot(([timestamps, est_bw[:, 0]],), "t [s]", "w [rad/s]", "Gyro Bias X")
        plotter.plot(([timestamps, est_bw[:, 1]],), "t [s]", "w [rad/s]", "Gyro Bias Y")
        plotter.plot(([timestamps, est_bw[:, 2]],), "t [s]", "w [rad/s]", "Gyro Bias Z")

        plotter.plot(([timestamps, est_ba[:, 0]],), "t [s]", "a [m/s^2]", "Accel Bias X")
        plotter.plot(([timestamps, est_ba[:, 1]],), "t [s]", "a [m/s^2]", "Accel Bias Y")
        plotter.plot(([timestamps, est_ba[:, 2]],), "t [s]", "a [m/s^2]", "Accel Bias Z")

    def test_process_model_F_G_Q_covar(self):
        device = "cuda"
        imu_noise = torch.eye(12, 12).to(device)
        ekf = IMUKalmanFilter()

        covar = torch.zeros(18, 18).to(device).to(device)

        for i in range(0, 5):
            t_accum, C_accum, r_accum, v_accum, covar, F, G, Phi, Q = \
                ekf.predict_one_step(C_accum=torch_se3.exp_SO3(torch.tensor([1., 2., 3.])).to(device),
                                     r_accum=torch.tensor([-2, 0.25, -0.1]).view(3, 1).to(device),
                                     v_accum=torch.tensor([0.1, -0.1, 0.2]).view(3, 1).to(device),
                                     t_accum=torch.tensor(10.).to(device),
                                     dt=torch.tensor(0.05).to(device),
                                     g_k=torch.tensor([-0.1, 0.2, 11]).view(3, 1).to(device),
                                     v_k=torch.tensor([5, -2, 1.]).view(3, 1).to(device),
                                     bw_k=torch.tensor([0.10, -0.11, 0.12]).view(3, 1).to(device),
                                     ba_k=torch.tensor([-0.13, 0.14, -0.15]).view(3, 1).to(device),
                                     covar=covar,
                                     gyro_meas=torch.tensor([1.0, -11, -1.2]).view(3, 1).to(device),
                                     accel_meas=torch.tensor([-.5, 4, 6]).view(3, 1).to(device),
                                     imu_noise_covar=imu_noise)

            self.assertTrue(torch.allclose(t_accum.detach().cpu(), torch.tensor(10.05), atol=1e-8))

            # check proper blocks to be zero
            self.check_non_zero_3x3_sub_blocks(F.detach().cpu(),
                                               [(1, 1,), (1, 4,), (2, 1,),
                                                (2, 3,), (3, 0,), (3, 1,),
                                                (3, 3,), (3, 4,), (3, 5,), ])

            self.check_non_zero_3x3_sub_blocks(G.detach().cpu(),
                                               [(1, 0,), (3, 0,), (3, 2,), (4, 1,), (5, 3,), ])
            self.check_non_zero_3x3_sub_blocks(Phi.detach().cpu(),
                                               [(0, 0,), (1, 1,), (1, 4,),
                                                (2, 0,), (2, 1,), (2, 2,), (2, 3,), (2, 5),
                                                (3, 0,), (3, 1,), (3, 3,), (3, 4,), (3, 5,), (4, 4,), (5, 5)])

            F_np = F.detach().cpu().numpy().astype(np.float64)
            F_exp = scipy.linalg.expm(F_np * 0.05)

            self.assertTrue(np.allclose(Test_EKF.sub_block3x3(F_exp, 1, 1),
                                        Test_EKF.sub_block3x3(Phi, 1, 1).detach().cpu().numpy(), atol=1e-7))
            self.assertTrue(np.allclose(Test_EKF.sub_block3x3(F_exp, 3, 3),
                                        Test_EKF.sub_block3x3(Phi, 3, 3).detach().cpu().numpy(), atol=1e-7))

            # check symmetrical
            self.assertTrue(torch.allclose(covar, covar.transpose(0, 1), atol=1e-9))

        # no correlation between gravity and any other stuff (yet)
        self.assertTrue(torch.allclose(covar[0:3, 0:18].detach().cpu(), torch.zeros(3, 18), atol=0, rtol=0))
        self.assertTrue(torch.allclose(covar[0:18, 0:3].detach().cpu(), torch.zeros(18, 3), atol=0, rtol=0))

        # no correlation between rotation and accelerometer bias
        self.assertTrue(torch.allclose(covar[15:18, 3:6].detach().cpu(), torch.zeros(3, 3), atol=0, rtol=0))
        self.assertTrue(torch.allclose(covar[3:6, 15:18].detach().cpu(), torch.zeros(3, 3), atol=0, rtol=0))

    def test_predict_K06_plotted(self):
        output_dir = os.path.join(par.results_coll_dir, "ekf_test_predict_K06_plotted")
        logger.initialize(output_dir, use_tensorboard=False)
        timestamps, gt_poses, gt_vels, poses, states, covars, precomp_covars = self.predict_test_case("K06", "cpu",
                                                                                                      False)
        self.plot_ekf_data(output_dir, timestamps, gt_poses, gt_vels, poses, states)

    def test_pred_K04_cuda_graph(self):
        timestamps, gt_poses, gt_vels, poses, states, covars, precomp_covars = self.predict_test_case("K04", "cuda",
                                                                                                      True)
        self.assertTrue(states[1].requires_grad)
        self.assertTrue(states[1].is_cuda)
        self.assertTrue(covars[1].requires_grad)
        self.assertTrue(covars[1].is_cuda)
        self.assertTrue(poses[1].requires_grad)
        self.assertTrue(poses[1].is_cuda)

        self.assertTrue(states[-1].requires_grad)
        self.assertTrue(states[-1].is_cuda)
        self.assertTrue(covars[-1].requires_grad)
        self.assertTrue(covars[-1].is_cuda)
        self.assertTrue(poses[-1].requires_grad)
        self.assertTrue(poses[-1].is_cuda)

    def predict_test_case(self, seq, device, req_grad):
        df = SequenceData(seq).df
        # df = df.loc[0:30, :]

        imu_noise = torch.eye(12, 12).to(device)
        ekf = IMUKalmanFilter()

        timestamps = list(df.loc[:, "timestamp"].values)
        imu_timestamps = list(df.loc[:, "imu_timestamps"].values)
        gyro_measurements = list(df.loc[:, "gyro_measurements"].values)
        accel_measurements = list(df.loc[:, "accel_measurements"].values)
        gt_poses = np.array(list(df.loc[:, "T_i_vk"].values))
        gt_vels = np.array(list(df.loc[:, "v_vk_i_vk"].values))

        self.assertEqual(len(imu_timestamps), len(gyro_measurements))
        self.assertEqual(len(imu_timestamps), len(accel_measurements))
        self.assertEqual(len(imu_timestamps), len(gt_poses))

        g = np.array([0, 0, 9.808679801065017])
        states = [IMUKalmanFilter.encode_state(torch.tensor(gt_poses[0, 0:3, 0:3].transpose().dot(g),
                                                            dtype=torch.float32),  # g
                                               torch.eye(3, 3),  # C
                                               torch.zeros(3),  # r
                                               torch.tensor(gt_vels[0], dtype=torch.float32),  # v
                                               torch.zeros(3),  # bw
                                               torch.zeros(3)).to(device)]  # ba
        covars = [torch.zeros(18, 18).to(device)]
        precomp_covars = [torch.zeros(18, 18).to(device)]
        poses = [torch.tensor(np.linalg.inv(gt_poses[0]), dtype=torch.float32).to(device), ]

        states[0].requires_grad = req_grad
        covars[0].requires_grad = req_grad
        poses[0].requires_grad = req_grad

        for i in range(0, len(imu_timestamps) - 1):
            imu_data = torch.tensor(np.concatenate([np.expand_dims(imu_timestamps[i], 1),
                                                    gyro_measurements[i], accel_measurements[i]], axis=1),
                                    dtype=torch.float32).to(device)
            state, covar = ekf.predict(imu_data, imu_noise, states[-1], covars[-1])

            precomp_covars.append(covar)

            pose, state, covar = ekf.composition(poses[-1], state, covar)

            states.append(state)
            covars.append(covar)
            poses.append(pose)

        return timestamps, gt_poses, gt_vels, poses, states, covars, precomp_covars

    def test_meas_jacobian_numerically(self):
        torch.set_default_tensor_type('torch.DoubleTensor')
        device = "cpu"
        T_imu_cam = torch.eye(4, 4)
        T_imu_cam[0:3, 0:3] = torch_se3.exp_SO3(torch.tensor([1., -2., 3.]))
        T_imu_cam[0:3, 3] = torch.tensor([-3., 2., 1.])
        T_imu_cam = T_imu_cam.to(device)
        ekf = IMUKalmanFilter()

        C_pred = torch_se3.exp_SO3(torch.tensor([0.1, 3, -0.3])).to(device)
        r_pred = torch.tensor([-1.05, 20, -1.]).view(3, 1).to(device)
        vis_meas = torch.tensor([4., -6., 8., -7., 5., -9.]).to(device).view(6, 1)
        residual, H = ekf.meas_residual_and_jacobi(C_pred, r_pred, vis_meas, T_imu_cam)

        e = 1e-6
        p = torch.eye(3, 3) * e
        H_C_numerical = torch.zeros(6, 3)
        H_r_numerical = torch.zeros(6, 3)

        for i in range(0, 3):
            pb = p[:, i]
            residual_minus_pb, _ = ekf.meas_residual_and_jacobi(torch.mm(C_pred, torch_se3.exp_SO3(-pb)),
                                                                r_pred, vis_meas, T_imu_cam)
            residual_plus_pb, _ = ekf.meas_residual_and_jacobi(torch.mm(C_pred, torch_se3.exp_SO3(pb)),
                                                               r_pred, vis_meas, T_imu_cam)
            H_C_numerical[:, i] = (residual_plus_pb - residual_minus_pb).view(6) / (2 * e)

        for i in range(0, 3):
            pb = p[:, i].view(3, 1)
            residual_minus_pb, _ = ekf.meas_residual_and_jacobi(C_pred, r_pred - pb, vis_meas, T_imu_cam)
            residual_plus_pb, _ = ekf.meas_residual_and_jacobi(C_pred, r_pred + pb, vis_meas, T_imu_cam)
            H_r_numerical[:, i] = (residual_plus_pb - residual_minus_pb).view(6) / (2 * e)

        self.assertTrue(torch.allclose(H_C_numerical, H[:, 3:6], atol=1e-7))
        self.assertTrue(torch.allclose(H_r_numerical, H[:, 6:9], atol=1e-7))

        torch.set_default_tensor_type('torch.FloatTensor')

    def test_ekf_K04_cuda_graph(self):
        imu_covar = torch.diag(torch.tensor([1e-5, 1e-5, 1e-5,
                                             1e-8, 1e-8, 1e-8,
                                             1e-1, 1e-1, 1e-1,
                                             1e-3, 1e-3, 1e-3])).to("cuda")
        vis_meas_covar = torch.diag(torch.tensor([1e-2, 1e-2, 1e-2,
                                                  1e0, 1e0, 1e0])).to("cuda")
        init_covar = np.eye(18, 18)
        _, _, _, poses, states, covars = \
            self.ekf_test_case("K04", init_covar, imu_covar, vis_meas_covar, "cuda", req_grad=True)

        self.assertTrue(states[1].requires_grad)
        self.assertTrue(states[1].is_cuda)
        self.assertTrue(covars[1].requires_grad)
        self.assertTrue(covars[1].is_cuda)
        self.assertTrue(poses[1].requires_grad)
        self.assertTrue(poses[1].is_cuda)

        self.assertTrue(states[-1].requires_grad)
        self.assertTrue(states[-1].is_cuda)
        self.assertTrue(covars[-1].requires_grad)
        self.assertTrue(covars[-1].is_cuda)
        self.assertTrue(poses[-1].requires_grad)
        self.assertTrue(poses[-1].is_cuda)

    def test_ekf_all_plotted(self):
        output_dir = os.path.join(par.results_coll_dir, "test_ekf_all_plotted")
        logger.initialize(output_dir, use_tensorboard=False)

        # seqs = ["K06", ]
        seqs = ["K01", "K04", "K06", "K07", "K08", "K09", "K10"]

        device = "cpu"
        req_grad = False
        imu_covar = torch.diag(torch.tensor([1e-5, 1e-5, 1e-5,
                                             1e-8, 1e-8, 1e-8,
                                             1e-1, 1e-1, 1e-1,
                                             1e-3, 1e-3, 1e-3])).to(device)
        vis_meas_covar = torch.diag(torch.tensor([1e-2, 1e-2, 1e-2,
                                                  1e0, 1e0, 1e0])).to(device)
        init_covar = np.eye(18, 18)
        init_covar[0:3, 0:3] = np.eye(3, 3) * 1e-4  # g
        init_covar[3:9, 3:9] = np.zeros([6, 6])  # C,r
        init_covar[9:12, 9:12] = np.eye(3, 3) * 1e-2  # v
        init_covar[12:15, 12:15] = np.eye(3, 3) * 1e-8  # bw
        # init_covar[15:18, 15:18] = np.eye(3, 3) * 1e-2  # ba

        for seq in seqs:
            print("Running seq %s" % seq)
            timestamps, gt_poses, gt_vels, poses, states, covars = \
                self.ekf_test_case(seq, init_covar, imu_covar, vis_meas_covar, device, req_grad)
            self.plot_ekf_data(os.path.join(output_dir, seq), timestamps, gt_poses, gt_vels, poses, states)

    def test_ekf_K06_with_artificial_biases_plotted(self):
        output_dir = os.path.join(par.results_coll_dir, "test_ekf_K06_with_artificial_biases_plotted")
        logger.initialize(output_dir, use_tensorboard=False)

        device = "cpu"
        req_grad = False
        imu_covar = torch.diag(torch.tensor([1e-5, 1e-5, 1e-5,
                                             1e-8, 1e-8, 1e-8,
                                             1e-1, 1e-1, 1e-1,
                                             1e-3, 1e-3, 1e-3])).to(device)
        vis_meas_covar = torch.diag(torch.tensor([1e-2, 1e-2, 1e-2,
                                                  1e0, 1e0, 1e0])).to(device)
        init_covar = np.eye(18, 18)
        init_covar[0:3, 0:3] = np.eye(3, 3) * 1e-8  # g
        init_covar[3:9, 3:9] = np.zeros([6, 6])  # C,r
        init_covar[9:12, 9:12] = np.eye(3, 3) * 1e-2  # v
        init_covar[12:15, 12:15] = np.eye(3, 3) * 1e-8  # bw
        init_covar[15:18, 15:18] = np.eye(3, 3) * 1e2  # ba

        timestamps, gt_poses, gt_vels, poses, states, covars = \
            self.ekf_test_case("K06", init_covar, imu_covar, vis_meas_covar, device, req_grad,
                               accel_bias_inject=np.array([0.1, -0.2, 0.3]))
        self.plot_ekf_data(os.path.join(output_dir, "accel_bias"), timestamps, gt_poses, gt_vels, poses, states)

        init_covar = np.eye(18, 18)
        init_covar[0:3, 0:3] = np.eye(3, 3) * 1e-8  # g
        init_covar[3:9, 3:9] = np.zeros([6, 6])  # C,r
        init_covar[9:12, 9:12] = np.eye(3, 3) * 1e-2  # v
        init_covar[12:15, 12:15] = np.eye(3, 3) * 1e2  # bw
        init_covar[15:18, 15:18] = np.eye(3, 3) * 1e-2  # ba

        timestamps, gt_poses, gt_vels, poses, states, covars = \
            self.ekf_test_case("K06", init_covar, imu_covar, vis_meas_covar, device, req_grad,
                               gyro_bias_inject=np.array([-0.1, 0.2, -0.3]))
        self.plot_ekf_data(os.path.join(output_dir, "gyro_bias"), timestamps, gt_poses, gt_vels, poses, states)

    def ekf_test_case(self, seq, init_covar, imu_covar, vis_meas_covar, device, req_grad,
                      accel_bias_inject=np.zeros(3), gyro_bias_inject=np.zeros(3)):
        seq_data = SequenceData(seq)
        df = seq_data.df
        # df = df.loc[0:30, :]

        timestamps = list(df.loc[:, "timestamp"].values)
        imu_timestamps = list(df.loc[:, "imu_timestamps"].values)
        gyro_measurements = list(df.loc[:, "gyro_measurements"].values)
        accel_measurements = list(df.loc[:, "accel_measurements"].values)
        gt_poses = np.array(list(df.loc[:, "T_i_vk"].values))
        gt_vels = np.array(list(df.loc[:, "v_vk_i_vk"].values))
        T_imu_cam = np.linalg.inv(seq_data.T_cam_imu)
        # T_imu_cam = np.eye(4, 4)

        ekf = IMUKalmanFilter()

        self.assertEqual(len(imu_timestamps), len(gyro_measurements))
        self.assertEqual(len(imu_timestamps), len(accel_measurements))
        self.assertEqual(len(imu_timestamps), len(gt_poses))

        g = np.array([0, 0, 9.808679801065017])
        # g = np.array([0, 0, 9.80665])
        init_state = IMUKalmanFilter.encode_state(torch.tensor(gt_poses[0, 0:3, 0:3].transpose().dot(g),
                                                               dtype=torch.float32),  # g
                                                  torch.eye(3, 3),  # C
                                                  torch.zeros(3),  # r
                                                  torch.tensor(gt_vels[0], dtype=torch.float32),  # v
                                                  torch.zeros(3),  # bw
                                                  torch.zeros(3)).to(device)  # ba
        init_pose = torch.tensor(np.linalg.inv(gt_poses[0]), dtype=torch.float32).to(device)
        init_covar = torch.tensor(init_covar, dtype=torch.float32).to(device)

        # collect the data
        imu_data = []
        imu_data_idxs = [0]
        for i in range(0, len(imu_timestamps)):
            imu_data.append(np.concatenate([np.expand_dims(imu_timestamps[i], 1),
                                            gyro_measurements[i] + gyro_bias_inject,
                                            accel_measurements[i] + accel_bias_inject], axis=1))
            imu_data_idxs.append(imu_data_idxs[-1] + len(imu_timestamps[i]))
        imu_data = torch.tensor(np.concatenate(imu_data), dtype=torch.float32).to(device)
        imu_data_idxs = torch.tensor(imu_data_idxs, dtype=torch.int16)

        vis_meas = []
        for i in range(0, len(imu_timestamps) - 1):
            T_rel = np.linalg.inv(gt_poses[i]).dot(gt_poses[i + 1])
            T_rel_vis = np.linalg.inv(T_imu_cam).dot(T_rel).dot(T_imu_cam)
            vis_meas.append(torch.tensor(np.concatenate([log_SO3(T_rel_vis[0:3, 0:3]), T_rel_vis[0:3, 3]]),
                                         dtype=torch.float32, requires_grad=req_grad).to(device))
        vis_meas = torch.stack(vis_meas)
        vis_meas_covars = vis_meas_covar.repeat(vis_meas.shape[0], 1, 1).to(device)
        vis_meas_covars.requires_grad = req_grad

        poses, states, covars = ekf.forward(torch.unsqueeze(imu_data_idxs, dim=0),
                                            torch.unsqueeze(imu_data, dim=0),
                                            torch.tensor(imu_covar, dtype=torch.float32).to(device),
                                            torch.unsqueeze(init_pose, dim=0),
                                            torch.unsqueeze(init_state, dim=0),
                                            torch.unsqueeze(init_covar, dim=0),
                                            torch.unsqueeze(vis_meas, dim=0),
                                            torch.unsqueeze(vis_meas_covars, dim=0),
                                            torch.unsqueeze(torch.tensor(T_imu_cam, dtype=torch.float32), 0).to(device))

        return timestamps, gt_poses, gt_vels, poses[0], states[0], covars[0]


if __name__ == '__main__':
    # Test_EKF().test_pred_K04_cuda_graph()
    # Test_EKF().test_predict_K06_plotted()
    # Test_EKF().test_process_model_F_G_Q_covar()
    # Test_EKF().test_meas_jacobian_numerically()
    # Test_EKF().test_ekf_all_plotted()
    # Test_EKF().test_ekf_K04_cuda_graph()
    # Test_EKF().test_ekf_K06_with_artificial_biases_plotted()
    unittest.main()
