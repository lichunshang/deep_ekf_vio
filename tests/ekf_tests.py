import unittest
from data_loader import SequenceData
from model import IMUKalmanFilter
import torch
import numpy as np
from params import par
from log import logger
from se3_math import log_SO3
import os
from utils import Plotter

np.set_printoptions(linewidth=1024)


class Test_EKF(unittest.TestCase):

    def __init__(self):
        super(Test_EKF, self).__init__()
        self.output_dir = os.path.join(par.results_coll_dir, "ekf_tests")
        logger.initialize(self.output_dir, use_tensorboard=False)

    def test_predict_K06_plotted(self):
        timestamps, gt_poses, gt_vels, poses, states, covars, precomp_covars = self.predict_test_case("K06", "cpu",
                                                                                                      False)

        imu_int_positions = []
        imu_int_vels = []
        imu_int_rots = []
        for i in range(0, len(poses)):
            pose = np.linalg.inv(np.array(poses[i], dtype=np.float64))
            _, _, _, v, _, _ = IMUKalmanFilter.decode_state(states[i])
            imu_int_positions.append(np.array(pose[0:3, 3]))
            imu_int_rots.append(log_SO3(np.array(pose[0:3, 0:3])))
            imu_int_vels.append(np.array(v))

        imu_int_positions = np.squeeze(np.array(imu_int_positions))
        imu_int_vels = np.squeeze(np.array(imu_int_vels))
        imu_int_rots = np.squeeze(np.array(imu_int_rots))
        gt_rots = np.array([log_SO3(p[:3, :3]) for p in gt_poses])

        imu_int_rots[:, 0] = np.unwrap(imu_int_rots[:, 0])
        imu_int_rots[:, 1] = np.unwrap(imu_int_rots[:, 1])
        imu_int_rots[:, 2] = np.unwrap(imu_int_rots[:, 2])
        gt_rots[:, 0] = np.unwrap(gt_rots[:, 0])
        gt_rots[:, 1] = np.unwrap(gt_rots[:, 1])
        gt_rots[:, 2] = np.unwrap(gt_rots[:, 2])

        plotter = Plotter(self.output_dir)
        plotter.plot(([gt_poses[:, 0, 3], gt_poses[:, 1, 3]],
                      [imu_int_positions[:, 0], imu_int_positions[:, 1]],),
                     "x [m]", "y [m]", "XY Plot", labels=["gt_poses", "imu_int_pose"], equal_axes=True)
        plotter.plot(([gt_poses[:, 0, 3], gt_poses[:, 2, 3]],
                      [imu_int_positions[:, 0], imu_int_positions[:, 2]],),
                     "x [m]", "z [m]", "XZ Plot", labels=["gt_poses", "imu_int_pose"], equal_axes=True)
        plotter.plot(([gt_poses[:, 1, 3], gt_poses[:, 2, 3]],
                      [imu_int_positions[:, 1], imu_int_positions[:, 2]],),
                     "y [m]", "z [m]", "YZ Plot", labels=["gt_poses", "imu_int_pose"], equal_axes=True)

        plotter.plot(([timestamps, gt_vels[:, 0]], [timestamps, imu_int_vels[:, 0]],),
                     "t [s]", "v [m/s]", "Vel X", labels=["gt_vel", "imu_int_vel"])
        plotter.plot(([timestamps, gt_vels[:, 1]], [timestamps, imu_int_vels[:, 1]],),
                     "t [s]", "v [m/s]", "Vel Y", labels=["gt_vel", "imu_int_vel"])
        plotter.plot(([timestamps, gt_vels[:, 2]], [timestamps, imu_int_vels[:, 2]],),
                     "t [s]", "v [m/s]", "Vel Z", labels=["gt_vel", "imu_int_vel"])

        plotter.plot(([timestamps, gt_rots[:, 0]], [timestamps, imu_int_rots[:, 0]],),
                     "t [s]", "rot [rad]", "Rot X", labels=["gt_rot", "imu_int_rot"])
        plotter.plot(([timestamps, gt_rots[:, 1]], [timestamps, imu_int_rots[:, 1]],),
                     "t [s]", "rot [rad]", "Rot Y", labels=["gt_rot", "imu_int_rot"])
        plotter.plot(([timestamps, gt_rots[:, 2]], [timestamps, imu_int_rots[:, 2]],),
                     "t [s]", "rot [rad]", "Rot Z", labels=["gt_rot", "imu_int_rot"])

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

        T_cal = torch.eye(4, 4).to(device)
        imu_noise = torch.eye(12, 12).to(device)
        ekf = IMUKalmanFilter(imu_noise, T_cal)

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
        states = [IMUKalmanFilter.encode_state(torch.tensor(gt_poses[0, 0:3, 0:3].dot(g), dtype=torch.float32),  # g
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
            state, covar = ekf.predict(imu_data, states[-1], covars[-1])

            precomp_covars.append(covar)

            pose, state, covar = ekf.composition(poses[-1], state, covar)

            states.append(state)
            covars.append(covar)
            poses.append(pose)

        return timestamps, gt_poses, gt_vels, poses, states, covars, precomp_covars


if __name__ == '__main__':
    Test_EKF().test_pred_K04_cuda_graph()
    # Test_EKF().test_predict_K06_plotted()
    # unittest.main()
