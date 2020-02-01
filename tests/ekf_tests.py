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
import time
from eval import kitti_eval_pyimpl
from eval import plot_ekf_data
from eval import EurocErrorCalc, KittiErrorCalc


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

    def test_process_model_F_G_Q_covar(self):
        device = "cuda"
        imu_noise = torch.eye(12, 12).to(device)
        ekf = IMUKalmanFilter()

        covar = torch.zeros(1, 18, 18).to(device).to(device)

        for i in range(0, 5):
            t_accum, C_accum, r_accum, v_accum, covar, F, G, Phi, Q = \
                ekf.predict_one_step(C_accum=torch_se3.exp_SO3(torch.tensor([1., 2., 3.])).view(1, 3, 3).to(device),
                                     r_accum=torch.tensor([-2, 0.25, -0.1]).view(1, 3, 1).to(device),
                                     v_accum=torch.tensor([0.1, -0.1, 0.2]).view(1, 3, 1).to(device),
                                     t_accum=torch.tensor(10.).view(1, 1, 1).to(device),
                                     dt=torch.tensor(0.05).view(1, 1, 1).to(device),
                                     g_k=torch.tensor([-0.1, 0.2, 11]).view(1, 3, 1).to(device),
                                     v_k=torch.tensor([5, -2, 1.]).view(1, 3, 1).to(device),
                                     bw_k=torch.tensor([0.10, -0.11, 0.12]).view(1, 3, 1).to(device),
                                     ba_k=torch.tensor([-0.13, 0.14, -0.15]).view(1, 3, 1).to(device),
                                     covar=covar,
                                     gyro_meas=torch.tensor([1.0, -11, -1.2]).view(1, 3, 1).to(device),
                                     accel_meas=torch.tensor([-.5, 4, 6]).view(1, 3, 1).to(device),
                                     imu_noise_covar=imu_noise)

            self.assertTrue(torch.allclose(t_accum[-1].detach().cpu(), torch.tensor(10.05), atol=1e-8))

            # check proper blocks to be zero
            self.check_non_zero_3x3_sub_blocks(F[-1].detach().cpu(),
                                               [(1, 1,), (1, 4,), (2, 1,),
                                                (2, 3,), (3, 0,), (3, 1,),
                                                (3, 3,), (3, 4,), (3, 5,), ])

            self.check_non_zero_3x3_sub_blocks(G[-1].detach().cpu(),
                                               [(1, 0,), (3, 0,), (3, 2,), (4, 1,), (5, 3,), ])
            self.check_non_zero_3x3_sub_blocks(Phi[-1].detach().cpu(),
                                               [(0, 0,), (1, 1,), (1, 4,),
                                                (2, 0,), (2, 1,), (2, 2,), (2, 3,), (2, 5),
                                                (3, 0,), (3, 1,), (3, 3,), (3, 4,), (3, 5,), (4, 4,), (5, 5)])

            F_np = F[-1].detach().cpu().numpy().astype(np.float64)
            F_exp = scipy.linalg.expm(F_np * 0.05)

            self.assertTrue(np.allclose(Test_EKF.sub_block3x3(F_exp, 1, 1),
                                        Test_EKF.sub_block3x3(Phi[-1], 1, 1).detach().cpu().numpy(), atol=1e-7))
            self.assertTrue(np.allclose(Test_EKF.sub_block3x3(F_exp, 3, 3),
                                        Test_EKF.sub_block3x3(Phi[-1], 3, 3).detach().cpu().numpy(), atol=1e-7))

            # check symmetrical
            self.assertTrue(torch.allclose(covar[-1], covar[-1].transpose(0, 1), atol=1e-9))

        # no correlation between gravity and any other stuff (yet)
        self.assertTrue(torch.allclose(covar[-1][0:3, 0:18].detach().cpu(), torch.zeros(3, 18), atol=0, rtol=0))
        self.assertTrue(torch.allclose(covar[-1][0:18, 0:3].detach().cpu(), torch.zeros(18, 3), atol=0, rtol=0))

        # no correlation between rotation and accelerometer bias
        self.assertTrue(torch.allclose(covar[-1][15:18, 3:6].detach().cpu(), torch.zeros(3, 3), atol=0, rtol=0))
        self.assertTrue(torch.allclose(covar[-1][3:6, 15:18].detach().cpu(), torch.zeros(3, 3), atol=0, rtol=0))

    def test_predict_plotted(self):
        output_dir = os.path.join(par.results_coll_dir, "ekf_test_predict_plotted")
        logger.initialize(output_dir, use_tensorboard=False)

        seqs = ["K01", "K06", ]
        timestamps, gt_poses, gt_vels, poses, states, covars, precomp_covars = \
            self.predict_test_case(seqs, (0, 1091,), "cpu", False)

        for i in range(0, len(seqs)):
            plot_ekf_data(os.path.join(output_dir, seqs[i]),
                          timestamps[i], gt_poses[i], gt_vels[i], poses[i], states[i])

    def imu_predict_kitti(self):
        output_dir = os.path.join(par.results_coll_dir, "imu_predict_kitti")
        seqs = ["K01", "K04", "K06", "K07", "K08", "K09", "K10"]
        error_calc = KittiErrorCalc(seqs)

        for i in range(0, len(seqs)):
            logger.initialize(os.path.join(output_dir, seqs[i]), use_tensorboard=False)

            timestamps, gt_poses, gt_vels, poses, states, covars, precomp_covars = \
                self.predict_test_case([seqs[i]], None, "cpu", False)

            poses_np = np.linalg.inv(poses[0].numpy().astype(np.float64))
            np.save(logger.ensure_file_dir_exists(os.path.join(output_dir, seqs[i], "est.npy")), poses_np)

            # plot_ekf_data(os.path.join(output_dir, seqs[i]),
            #               timestamps[0], gt_poses[0], gt_vels[0], poses[0], states[0])

            logger.print("Processed seq %s" % seqs[i])
            err = kitti_eval_pyimpl.calc_kitti_seq_errors(gt_poses[0],
                                                          np.linalg.inv(poses[0].numpy().astype(np.float64)))
            err = err[0]
            err = np.array(err)
            logger.print("Errors trans & rot")
            logger.print("%.6f" % np.average(err[:, 0]))
            logger.print("%.6f" % (np.average(err[:, 1]) * 180 / np.pi))

            error_calc.accumulate_error(seqs[i], poses_np)

        logger.print("Ave Trans Errors: ")
        logger.print(np.average(np.array(error_calc.errors)[:, 0]))
        logger.print("Ave Rot Errors: ")
        logger.print(np.average(np.array(error_calc.errors)[:, 1]))

    def imu_predict_euroc(self):
        output_dir = os.path.join(par.results_coll_dir, "imu_predict_kitti")
        seqs = ['MH_05', 'V1_03', 'V2_02']
        error_calc = EurocErrorCalc(seqs)

        for i in range(0, len(seqs)):
            logger.initialize(os.path.join(output_dir, seqs[i]), use_tensorboard=False)

            timestamps, gt_poses, gt_vels, poses, states, covars, precomp_covars = \
                self.predict_test_case([seqs[i]], None, "cpu", False)

            poses_np = np.linalg.inv(poses[0].numpy().astype(np.float64))
            np.save(logger.ensure_file_dir_exists(os.path.join(output_dir, seqs[i], "est.npy")), poses_np)

            # plot_ekf_data(os.path.join(output_dir, seqs[i]),
            #               timestamps[0], gt_poses[0], gt_vels[0], poses[0], states[0])

            err = error_calc.accumulate_error(seqs[i], poses_np)
            logger.print("Processed seq %s" % seqs[i])
            logger.print("Errors trans & rot")
            logger.print("%.6f" % err)

        logger.print("Ave Trans Errors: ")
        logger.print(np.average(np.array(error_calc.errors)))

    def test_ekf_predict_cuda_graph(self):
        timestamps, gt_poses, gt_vels, poses, states, covars, precomp_covars = \
            self.predict_test_case(["K04", "K06", "K08", "K01", "K07"], (0, 100,), "cuda", True)
        self.assertTrue(states.requires_grad)
        self.assertTrue(states.is_cuda)
        self.assertTrue(covars.requires_grad)
        self.assertTrue(covars.is_cuda)
        self.assertTrue(poses.requires_grad)
        self.assertTrue(poses.is_cuda)

    def predict_test_case(self, seqs, seqs_range, device, req_grad):

        seqs_data = [SequenceData(seq) for seq in seqs]
        if seqs_range is None:
            data_frames = [d.df for d in seqs_data]
        else:
            data_frames = [d.df[seqs_range[0]:seqs_range[1]] for d in seqs_data]
        data_frames_lengths = [len(d) for d in data_frames]
        assert (all(data_frames_lengths[0] == l for l in data_frames_lengths))

        timestamps = np.array([list(df.loc[:, "timestamp"].values) for df in data_frames])
        gt_poses = np.array([list(df.loc[:, "T_i_vk"].values) for df in data_frames])
        gt_vels = np.array([list(df.loc[:, "v_vk_i_vk"].values) for df in data_frames])

        imu_timestamps = np.array([df.loc[:, "imu_timestamps"].values for df in data_frames])
        gyro_measurements = np.array([df.loc[:, "gyro_measurements"].values for df in data_frames])
        accel_measurements = np.array([df.loc[:, "accel_measurements"].values for df in data_frames])

        ekf = IMUKalmanFilter()

        self.assertEqual(timestamps.shape[0:2], gt_poses.shape[0:2])
        self.assertEqual(timestamps.shape[0:2], gt_vels.shape[0:2])

        g = np.array([0, 0, 9.808679801065017])
        init_state = []
        for i in range(0, len(gt_poses)):
            init_state.append(IMUKalmanFilter.encode_state(torch.tensor(gt_poses[i, 0, 0:3, 0:3].transpose().dot(g),
                                                                        dtype=torch.float32),  # g
                                                           torch.eye(3, 3),  # C
                                                           torch.zeros(3),  # r
                                                           torch.tensor(gt_vels[i, 0], dtype=torch.float32),  # v
                                                           torch.zeros(3),  # bw
                                                           torch.zeros(3)).to(device))  # ba
        states = [torch.stack(init_state)]
        poses = [torch.tensor([np.linalg.inv(p[0]) for p in gt_poses], dtype=torch.float32).to(device)]
        covars = [torch.zeros(timestamps.shape[0], 18, 18).to(device)]
        imu_noise = torch.eye(12, 12).to(device)
        precomp_covars = [torch.zeros(timestamps.shape[0], 18, 18).to(device)]

        states[0].requires_grad = req_grad
        covars[0].requires_grad = req_grad
        poses[0].requires_grad = req_grad

        imu_max_length = 12
        for i in range(0, timestamps.shape[1] - 1):
            imu_data = self.concat_imu_data_at_time_k(imu_max_length, imu_timestamps[:, i],
                                                      gyro_measurements[:, i],
                                                      accel_measurements[:, i])
            imu_data = torch.tensor(imu_data, dtype=torch.float32).to(device)
            pred_states, pred_covars = ekf.predict(imu_data, imu_noise, states[-1], covars[-1])

            precomp_covars.append(pred_covars[-1])

            pose, state, covar = ekf.composition(poses[-1], pred_states[-1], pred_covars[-1])

            states.append(state)
            covars.append(covar)
            poses.append(pose)

        states = torch.stack(states, 1)
        covars = torch.stack(covars, 1)
        poses = torch.stack(poses, 1)
        precomp_covars = torch.stack(precomp_covars)

        return timestamps, gt_poses, gt_vels, poses, states, covars, precomp_covars

    def test_meas_jacobian_numerically(self):
        torch.set_default_tensor_type('torch.DoubleTensor')
        device = "cpu"
        T_imu_cam = torch.eye(4, 4)
        T_imu_cam[0:3, 0:3] = torch_se3.exp_SO3(torch.tensor([1., -2., 3.]))
        T_imu_cam[0:3, 3] = torch.tensor([-3., 2., 1.])
        T_imu_cam = T_imu_cam.to(device).view(1, 4, 4)
        ekf = IMUKalmanFilter()

        C_pred = torch_se3.exp_SO3(torch.tensor([0.1, 3, -0.3])).to(device).view(1, 3, 3)
        r_pred = torch.tensor([-1.05, 20, -1.]).view(3, 1).to(device).view(1, 3, 1)
        vis_meas = torch.tensor([4., -6., 8., -7., 5., -9.]).to(device).view(1, 6, 1)
        residual, H = ekf.meas_residual_and_jacobi(C_pred, r_pred, vis_meas, T_imu_cam)

        e = 1e-6
        p = torch.eye(3, 3).view(1, 3, 3) * e
        H_C_numerical = torch.zeros(1, 6, 3)
        H_r_numerical = torch.zeros(1, 6, 3)

        for i in range(0, 3):
            pb = p[:, :, i:i + 1]
            residual_minus_pb, _ = ekf.meas_residual_and_jacobi(torch.matmul(C_pred, torch_se3.exp_SO3_b(-pb)),
                                                                r_pred, vis_meas, T_imu_cam)
            residual_plus_pb, _ = ekf.meas_residual_and_jacobi(torch.matmul(C_pred, torch_se3.exp_SO3_b(pb)),
                                                               r_pred, vis_meas, T_imu_cam)
            H_C_numerical[:, :, i] = (residual_plus_pb - residual_minus_pb).view(6) / (2 * e)

        for i in range(0, 3):
            pb = p[:, :, i:i + 1]
            residual_minus_pb, _ = ekf.meas_residual_and_jacobi(C_pred, r_pred - pb, vis_meas, T_imu_cam)
            residual_plus_pb, _ = ekf.meas_residual_and_jacobi(C_pred, r_pred + pb, vis_meas, T_imu_cam)
            H_r_numerical[:, :, i] = (residual_plus_pb - residual_minus_pb).view(6) / (2 * e)

        self.assertTrue(torch.allclose(H_C_numerical, H[:, :, 3:6], atol=1e-7))
        self.assertTrue(torch.allclose(H_r_numerical, H[:, :, 6:9], atol=1e-7))

        torch.set_default_tensor_type('torch.FloatTensor')

    def test_meas_jacobian_numerically_small_error(self):
        torch.set_default_tensor_type('torch.DoubleTensor')
        device = "cpu"
        T_imu_cam = torch.eye(4, 4)
        # T_imu_cam[0:3, 0:3] = torch_se3.exp_SO3(torch.tensor([1., -2., 3.]))
        # T_imu_cam[0:3, 3] = torch.tensor([-3., 2., 1.])
        T_imu_cam = T_imu_cam.to(device).view(1, 4, 4)
        ekf = IMUKalmanFilter()

        C_pred = torch_se3.exp_SO3(torch.tensor([0.1, 3, -0.3])).to(device).view(1, 3, 3)
        r_pred = torch.tensor([-1.05, 20, -1.]).view(3, 1).to(device).view(1, 3, 1)
        vis_meas = torch.tensor([0.1, 3, -0.3, -1.05, 20, -1.]).to(device).view(1, 6, 1)
        residual, H = ekf.meas_residual_and_jacobi(C_pred, r_pred, vis_meas, T_imu_cam)

        e = 1e-6
        p = torch.eye(3, 3).view(1, 3, 3) * e
        H_C_numerical = torch.zeros(1, 6, 3)
        H_r_numerical = torch.zeros(1, 6, 3)

        for i in range(0, 3):
            pb = p[:, :, i:i + 1]
            residual_minus_pb, _ = ekf.meas_residual_and_jacobi(torch.matmul(C_pred, torch_se3.exp_SO3_b(-pb)),
                                                                r_pred, vis_meas, T_imu_cam)
            residual_plus_pb, _ = ekf.meas_residual_and_jacobi(torch.matmul(C_pred, torch_se3.exp_SO3_b(pb)),
                                                               r_pred, vis_meas, T_imu_cam)
            H_C_numerical[:, :, i] = (residual_plus_pb - residual_minus_pb).view(6) / (2 * e)

        for i in range(0, 3):
            pb = p[:, :, i:i + 1]
            residual_minus_pb, _ = ekf.meas_residual_and_jacobi(C_pred, r_pred - pb, vis_meas, T_imu_cam)
            residual_plus_pb, _ = ekf.meas_residual_and_jacobi(C_pred, r_pred + pb, vis_meas, T_imu_cam)
            H_r_numerical[:, :, i] = (residual_plus_pb - residual_minus_pb).view(6) / (2 * e)

        self.assertTrue(torch.allclose(H_C_numerical, H[:, :, 3:6], atol=1e-7))
        self.assertTrue(torch.allclose(H_r_numerical, H[:, :, 6:9], atol=1e-7))

        torch.set_default_tensor_type('torch.FloatTensor')

    def test_ekf_cuda_graph(self):
        seqs = ["K04", "K06", "K08", "K01", "K07"]

        imu_covar = torch.diag(torch.tensor([1e-5, 1e-5, 1e-5,
                                             1e-8, 1e-8, 1e-8,
                                             1e-1, 1e-1, 1e-1,
                                             1e-3, 1e-3, 1e-3])).to("cuda")
        vis_meas_covar = torch.diag(torch.tensor([1e-2, 1e-2, 1e-2,
                                                  1e0, 1e0, 1e0])).to("cuda")
        init_covar = np.tile(np.eye(18, 18), [len(seqs), 1, 1])

        _, _, _, poses, states, covars = \
            self.ekf_test_case(seqs, [0, 150, ], init_covar, imu_covar, vis_meas_covar, "cuda", req_grad=True)

        self.assertTrue(states.requires_grad)
        self.assertTrue(states.is_cuda)
        self.assertTrue(covars.requires_grad)
        self.assertTrue(covars.is_cuda)
        self.assertTrue(poses.requires_grad)
        self.assertTrue(poses.is_cuda)

        states[0, -1, 0].backward(retain_graph=True)
        covars[0, -1, 0, 0].backward(retain_graph=True)
        poses[0, -1, 0, 0].backward(retain_graph=True)

    def test_ekf_all_plotted(self):
        output_dir = os.path.join(par.results_coll_dir, "test_ekf_all_plotted")
        logger.initialize(output_dir, use_tensorboard=False)

        seqs = ["K01", "K06", "K07", "K10"]

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
        init_covar = np.tile(init_covar, (len(seqs), 1, 1,))

        timestamps, gt_poses, gt_vels, poses, states, covars = \
            self.ekf_test_case(seqs, [0, 1091], init_covar, imu_covar, vis_meas_covar, device, req_grad)

        for i in range(0, len(seqs)):
            plot_ekf_data(os.path.join(output_dir, seqs[i]),
                          timestamps[i], gt_poses[i], gt_vels[i], poses[i], states[i])

        # seqs = ["K01", "K04", "K06", "K07", "K08", "K09", "K10"]

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
            self.ekf_test_case(["K06"], [0, 1091, ], init_covar.reshape(1, 18, 18), imu_covar, vis_meas_covar, device,
                               req_grad, accel_bias_inject=np.array([0.1, -0.2, 0.3]).reshape(1, 3))
        plot_ekf_data(os.path.join(output_dir, "K06_accel_bias"),
                      timestamps[0], gt_poses[0], gt_vels[0], poses[0], states[0])

        init_covar = np.eye(18, 18)
        init_covar[0:3, 0:3] = np.eye(3, 3) * 1e-8  # g
        init_covar[3:9, 3:9] = np.zeros([6, 6])  # C,r
        init_covar[9:12, 9:12] = np.eye(3, 3) * 1e-2  # v
        init_covar[12:15, 12:15] = np.eye(3, 3) * 1e2  # bw
        init_covar[15:18, 15:18] = np.eye(3, 3) * 1e-2  # ba

        timestamps, gt_poses, gt_vels, poses, states, covars = \
            self.ekf_test_case(["K06"], [0, 1091, ], init_covar.reshape(1, 18, 18), imu_covar, vis_meas_covar, device,
                               req_grad, gyro_bias_inject=np.array([-0.1, 0.2, -0.3]).reshape(1, 3))
        plot_ekf_data(os.path.join(output_dir, "K06_gyro_bias"),
                      timestamps[0], gt_poses[0], gt_vels[0], poses[0], states[0])

        init_covar = np.eye(18, 18)
        init_covar[0:3, 0:3] = np.eye(3, 3) * 1e-8  # g
        init_covar[3:9, 3:9] = np.zeros([6, 6])  # C,r
        init_covar[9:12, 9:12] = np.eye(3, 3) * 1e-2  # v
        init_covar[12:15, 12:15] = np.eye(3, 3) * 1e2  # bw
        init_covar[15:18, 15:18] = np.eye(3, 3) * 1e2  # ba

        timestamps, gt_poses, gt_vels, poses, states, covars = \
            self.ekf_test_case(["K06"], [0, 1091, ], init_covar.reshape(1, 18, 18), imu_covar, vis_meas_covar, device,
                               req_grad,
                               gyro_bias_inject=np.array([-0.05, 0.06, -0.07]).reshape(1, 3),
                               accel_bias_inject=np.array([-0.1, 0.3, -0.2]).reshape(1, 3))
        plot_ekf_data(os.path.join(output_dir, "K06_both_gyro_accel_bias"),
                      timestamps[0], gt_poses[0], gt_vels[0], poses[0], states[0])

    def concat_imu_data_at_time_k(self, max_length, imu_timestamps, gyro_measurements, accel_measurements,
                                  gyro_bias_inject=None, accel_bias_inject=None):
        n_batches = imu_timestamps.shape[0]
        imu_timestamps = list(imu_timestamps)
        gyro_measurements = list(gyro_measurements)
        accel_measurements = list(accel_measurements)

        if gyro_bias_inject is None:
            gyro_bias_inject = np.zeros([n_batches, 3])

        if accel_bias_inject is None:
            accel_bias_inject = np.zeros([n_batches, 3])

        for i in range(0, n_batches):
            n_to_pad = max_length - len(imu_timestamps[i])
            assert (n_to_pad >= 0)
            if len(imu_timestamps[i]) > 0:
                time_pad = imu_timestamps[i][-1]
            else:
                time_pad = 0
            imu_timestamps[i] = np.concatenate([imu_timestamps[i], np.full(n_to_pad, time_pad)])
            gyro_measurements[i] = np.concatenate([gyro_measurements[i], np.full([n_to_pad, 3], 0)]) + \
                                   gyro_bias_inject[i:i + 1]
            accel_measurements[i] = np.concatenate([accel_measurements[i], np.full([n_to_pad, 3], 0)]) + \
                                    accel_bias_inject[i:i + 1]

        imu_timestamps = np.array(imu_timestamps)
        gyro_measurements = np.array(gyro_measurements)
        accel_measurements = np.array(accel_measurements)

        return np.concatenate([np.expand_dims(imu_timestamps, -1), gyro_measurements, accel_measurements], -1)

    def ekf_test_case(self, seqs, seqs_range, init_covar, imu_covar, vis_meas_covar, device, req_grad,
                      accel_bias_inject=None, gyro_bias_inject=None):
        seqs_data = [SequenceData(seq) for seq in seqs]
        data_frames = [d.df[seqs_range[0]:seqs_range[1]] for d in seqs_data]
        data_frames_lengths = [len(d) for d in data_frames]
        assert (all(data_frames_lengths[0] == l for l in data_frames_lengths))

        timestamps = np.array([list(df.loc[:, "timestamp"].values) for df in data_frames])
        gt_poses = np.array([list(df.loc[:, "T_i_vk"].values) for df in data_frames])
        gt_vels = np.array([list(df.loc[:, "v_vk_i_vk"].values) for df in data_frames])
        T_imu_cam = np.array([d.T_cam_imu for d in seqs_data])

        if accel_bias_inject is None:
            accel_bias_inject = np.zeros([len(seqs_data), 3])
        if gyro_bias_inject is None:
            gyro_bias_inject = np.zeros([len(seqs_data), 3])

        imu_timestamps = np.array([df.loc[:, "imu_timestamps"].values for df in data_frames])
        gyro_measurements = np.array([df.loc[:, "gyro_measurements"].values for df in data_frames])
        accel_measurements = np.array([df.loc[:, "accel_measurements"].values for df in data_frames])

        ekf = IMUKalmanFilter()

        self.assertEqual(timestamps.shape[0:2], gt_poses.shape[0:2])
        self.assertEqual(timestamps.shape[0:2], gt_vels.shape[0:2])
        self.assertEqual(timestamps.shape[0], T_imu_cam.shape[0])

        g = np.array([0, 0, 9.808679801065017])
        init_state = []
        for i in range(0, len(gt_poses)):
            init_state.append(IMUKalmanFilter.encode_state(torch.tensor(gt_poses[i, 0, 0:3, 0:3].transpose().dot(g),
                                                                        dtype=torch.float32),  # g
                                                           torch.eye(3, 3),  # C
                                                           torch.zeros(3),  # r
                                                           torch.tensor(gt_vels[i, 0], dtype=torch.float32),  # v
                                                           torch.zeros(3),  # bw
                                                           torch.zeros(3)).to(device))  # ba
        init_state = torch.stack(init_state)
        init_pose = torch.tensor([np.linalg.inv(p[0]) for p in gt_poses], dtype=torch.float32).to(device)
        init_covar = torch.tensor(init_covar, dtype=torch.float32).to(device)
        imu_covar = torch.tensor(imu_covar, dtype=torch.float32).to(device)

        # collect the data
        imu_data = []
        imu_max_length = 12
        for i in range(0, timestamps.shape[1]):
            imu_data_time_k = self.concat_imu_data_at_time_k(imu_max_length, imu_timestamps[:, i],
                                                             gyro_measurements[:, i],
                                                             accel_measurements[:, i],
                                                             gyro_bias_inject, accel_bias_inject)
            imu_data.append(imu_data_time_k)
        imu_data = torch.tensor(np.stack(imu_data, 1), dtype=torch.float32).to(device)

        vis_meas = []
        for i in range(0, timestamps.shape[1] - 1):
            T_rel = np.matmul(np.linalg.inv(gt_poses[:, i]), gt_poses[:, i + 1])
            T_rel_vis = np.matmul(np.matmul(np.linalg.inv(T_imu_cam), T_rel), T_imu_cam)

            vis_meas.append(np.concatenate([np.array([log_SO3(T[:3, :3]) for T in T_rel_vis]),
                                            T_rel_vis[:, 0:3, 3]], -1))
        vis_meas = np.expand_dims(np.stack(vis_meas, 1), -1)
        vis_meas = torch.tensor(vis_meas, dtype=torch.float32).to(device)
        vis_meas_covars = vis_meas_covar.repeat(vis_meas.shape[0], vis_meas.shape[1], 1, 1).to(device)

        vis_meas_covars.requires_grad = req_grad
        imu_covar.requires_grad = req_grad
        init_covar.requires_grad = req_grad
        init_pose.requires_grad = req_grad
        init_state.requires_grad = req_grad
        init_pose.requires_grad = req_grad

        start_time = time.time()
        poses, states, covars = ekf.forward(imu_data,
                                            imu_covar,
                                            init_pose,
                                            init_state,
                                            init_covar,
                                            vis_meas,
                                            vis_meas_covars,
                                            torch.tensor(T_imu_cam, dtype=torch.float32).to(device))
        print("ekf.forward elapsed %.5f" % (time.time() - start_time))
        return timestamps, gt_poses, gt_vels, poses, states, covars


if __name__ == '__main__':
    # Test_EKF().test_predict_plotted()
    # Test_EKF().test_process_model_F_G_Q_covar()
    # Test_EKF().test_meas_jacobian_numerically_small_error()
    # Test_EKF().test_meas_jacobian_numerically()
    # Test_EKF().test_ekf_all_plotted()
    # Test_EKF().test_ekf_predict_cuda_graph()
    # Test_EKF().test_ekf_cuda_graph()
    # Test_EKF().test_ekf_K06_with_artificial_biases_plotted()
    Test_EKF().imu_predict_kitti()
    # Test_EKF().imu_predict_euroc()
    # unittest.main(verbosity=10)
