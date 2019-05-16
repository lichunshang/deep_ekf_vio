import unittest
from eval import plot_ekf_data
import numpy as np
import torch
import os
from model import IMUKalmanFilter
from data_loader import get_subseqs, SubseqDataset, SequenceData
from params import par
from log import logger
from eval.euroc_eval import EurocErrorCalc


class Test_EKF_EUROC(unittest.TestCase):
    def test_ekf_euroc(self):
        output_dir = os.path.join(par.results_coll_dir, "test_ekf_euroc")
        logger.initialize(output_dir, use_tensorboard=False)

        # seqs = ["MH_01"]
        seqs = ["MH_01", "MH_02", "MH_03", "MH_04", "MH_05", "V1_01", "V1_02", "V1_03", "V2_01", "V2_02", "V2_03"]
        # seqs2 = ["MH_01_eval", "MH_02_eval", "MH_03_eval", "MH_04_eval", "MH_05_eval",
        #          "V1_01_eval", "V1_02_eval", "V1_03_eval", "V2_01_eval", "V2_02_eval", "V2_03_eval"]
        # seqs = seqs + seqs2

        error_calc = EurocErrorCalc(seqs)

        imu_covar = torch.diag(torch.tensor([1e-3, 1e-3, 1e-3,
                                             1e-5, 1e-5, 1e-5,
                                             1e-1, 1e-1, 1e-1,
                                             1e-2, 1e-2, 1e-2]))
        vis_meas_covar = torch.diag(torch.tensor([1e-3, 1e-3, 1e-3,
                                                  1e-3, 1e-3, 1e-3])).view(1, 1, 6, 6)
        init_covar = np.eye(18, 18)
        init_covar[0:3, 0:3] = np.eye(3, 3) * 1e-2  # g
        init_covar[3:9, 3:9] = np.zeros([6, 6])  # C,r
        init_covar[9:12, 9:12] = np.eye(3, 3) * 1e-2  # v
        init_covar[12:15, 12:15] = np.eye(3, 3) * 1e-2  # bw
        init_covar[15:18, 15:18] = np.eye(3, 3) * 1e2  # ba
        init_covar = torch.tensor(init_covar, dtype=torch.float32).view(1, 18, 18)
        ekf = IMUKalmanFilter()

        for seq in seqs:
            subseqs = get_subseqs([seq], 2, overlap=1, sample_times=1, training=False)
            dataset = SubseqDataset(subseqs, (par.img_h, par.img_w), 0, 1, True, training=False, no_image=True)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            seq_data = SequenceData(seq)

            est_poses = []
            est_ekf_states = []
            est_ekf_covars = []

            for i, data in enumerate(dataloader):
                # print('%d/%d (%.2f%%)' % (i, len(dataloader), i * 100 / len(dataloader)), end="\n")

                _, _, imu_data, init_state, _, gt_poses_inv, gt_rel_poses = data

                gt_rel_poses = gt_rel_poses.view(1, 1, 6, 1)

                if i == 0:
                    pose, ekf_state, ekf_covar = ekf.forward(imu_data, imu_covar, gt_poses_inv[:, 0], init_state,
                                                             init_covar,
                                                             gt_rel_poses, vis_meas_covar,
                                                             torch.eye(4, 4).view(1, 4, 4))
                    est_poses.append(pose[:, 0])
                    est_ekf_states.append(ekf_state[:, 0])
                    est_ekf_covars.append(ekf_covar[:, 0])
                    est_poses.append(pose[:, -1])
                    est_ekf_states.append(ekf_state[:, -1])
                    est_ekf_covars.append(ekf_covar[:, -1])
                else:
                    pose, ekf_state, ekf_covar = ekf.forward(imu_data, imu_covar, est_poses[-1], est_ekf_states[-1],
                                                             est_ekf_covars[-1],
                                                             gt_rel_poses, vis_meas_covar,
                                                             torch.eye(4, 4).view(1, 4, 4))

                    est_poses.append(pose[:, -1])
                    est_ekf_states.append(ekf_state[:, -1])
                    est_ekf_covars.append(ekf_covar[:, -1])

            est_poses_np = torch.stack(est_poses, 1).squeeze().detach().cpu().numpy().astype("float64")
            err = error_calc.accumulate_error(seq, np.linalg.inv(est_poses_np))

            logger.print("Error: %.5f" % err)

            logger.print("Plotting figures...")
            plot_ekf_data(os.path.join(output_dir, seq), seq_data.get_timestamps(),
                          seq_data.get_poses(), seq_data.get_velocities(),
                          torch.stack(est_poses, 1).squeeze(), torch.stack(est_ekf_states, 1).squeeze())

            logger.print("Finished Sequence %s" % seq)

        logger.print("Done! Ave Error: %.5f" % error_calc.get_average_error())


if __name__ == '__main__':
    Test_EKF_EUROC().test_ekf_euroc()
    # unittest.main(verbosity=10)?
