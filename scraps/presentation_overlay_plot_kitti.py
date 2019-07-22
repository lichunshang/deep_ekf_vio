from utils import Plotter
import os
import numpy as np

seq = "K10"
gt_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation Results/KITTI/"
                     "0vanillagloss_train_20190509-12-31-54_esg_vistanh_1e4beta_uncorrcovar/"
                     "saved_model.eval.traj/gt_poses", seq + ".npy"))
vanilla_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation Results/KITTI/"
                     "0vanillagloss_train_20190509-12-31-54_esg_vistanh_1e4beta_uncorrcovar/"
                     "saved_model.eval.traj/est_poses", seq + ".npy"))
hybrid_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation Results/KITTI/"
                     "2hybridgloss_train_20190529-00-14-06_esg_fiximu_hybrid_btsz48_kitti/"
                     "saved_model.eval.traj/est_poses", seq + ".npy"))
vision_only_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation Results/KITTI/"
                     "3vision_train_20190420-01-16-51_allaug_GaussCovarLoss_lr1e-4_eps1e-4r1e-2t_100k4/"
                     "saved_model.eval.traj/est_poses", seq + ".npy"))
imu_only_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation Results/KITTI/imu", seq, "est.npy"))

plotter = Plotter("/home/cs4li/Dev/deep_ekf_vio/results/Presentation Results/KITTI")
plotter.plot(([gt_poses[:, 0, 3], gt_poses[:, 1, 3]],
              [vanilla_poses[:, 0, 3], vanilla_poses[:, 1, 3]],
              [hybrid_poses[:, 0, 3], hybrid_poses[:, 1, 3]],
              [vision_only_poses[:, 0, 3], vision_only_poses[:, 1, 3]],
              [imu_only_poses[:, 0, 3], imu_only_poses[:, 1, 3]],
              ),
             "x [m]", "y [m]", "KITTI Sequence %s" % seq[1:],
             labels=["gt", "vanilla", "hybrid", "vision", "imu"],
             equal_axes=True, filename=seq)

plotter.plot(([gt_poses[:, 0, 3], gt_poses[:, 1, 3]],
              [hybrid_poses[:, 0, 3], hybrid_poses[:, 1, 3]],
              ),
             "x [m]", "y [m]", "KITTI Sequence %s" % seq[1:],
             labels=["gt", "hybrid"],
             equal_axes=True, filename=seq+"_one")
