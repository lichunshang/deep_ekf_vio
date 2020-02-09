from utils import Plotter
import os
import numpy as np
import matplotlib.pyplot as plt
import se3
import glob


def pfind(*path):
    p = glob.glob(os.path.join(*path) + "*")
    assert len(p) == 1
    return p[0]


seq = "K06"
base_dir = "/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results"
# gt_poses = np.load(os.path.join(pfind(base_dir, "KITTI", seq + "_train"), "saved_model.eval.traj/gt_poses", seq + ".npy"))
# vanilla_poses = np.load(os.path.join(pfind(base_dir, "KITTI", seq + "_train"), "saved_model.eval.traj/est_poses", seq + ".npy"))
timestmaps = np.load("/home/cs4li/Dev/deep_ekf_vio/results/train_20200130-22-45-48_epoch16_eval/saved_model.eval.traj/timestamps/K07.npy")
gt_velocities = np.load(
    os.path.join(pfind(base_dir, "KITTI", seq + "_train"), "saved_model.eval.traj/ekf_states/gt_velocities", seq + ".npy"))
est_velocities = np.load(
    os.path.join(pfind(base_dir, "KITTI", seq + "_train"), "saved_model.eval.traj/ekf_states/states", seq + ".npy"))[:,
                 15:18]
# vis_meas = np.load(os.path.join(pfind(base_dir, "KITTI", seq + "_train"), "saved_model.eval.traj/ekf_states/vis_meas", seq + ".npy"))
# vis_meas = np.load("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/Presentation_Results/KITTI/1vanillanogloss_train_20190430-14-11-52_ekf_scratch_nogloss_0.75k3_1k4_eps1e-5/saved_model.eval.traj/vis_meas/meas/K07.npy")
# vis_meas = np.load("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/Presentation_Results/KITTI/3vision_train_20190420-01-16-51_allaug_GaussCovarLoss_lr1e-4_eps1e-4r1e-2t_100k4/saved_model.eval.traj/vis_meas/meas/K07.npy")
# vis_meas = np.load("/home/cs4li/Dev/deep_ekf_vio/results/train_20200130-22-45-48/saved_model.eval.traj/vis_meas/meas/K07.npy")
# vis_meas = np.load("/home/cs4li/Dev/deep_ekf_vio/results/train_20200130-22-45-48/saved_model.eval.traj/vis_meas/meas/K07.npy")
vis_meas2 = np.load("/home/cs4li/Dev/deep_ekf_vio/results/train_20200130-22-45-48_epoch16_eval/saved_model.eval.traj/vis_meas/meas/K07.npy")

# covars = np.load(
#     os.path.join(pfind(base_dir, "KITTI", seq + "_train"), "saved_model.eval.traj/ekf_states/vis_meas_covar", seq + ".npy"))

dt = np.ediff1d(timestmaps)
plt.plot(est_velocities[:, 0])
plt.plot(gt_velocities[:, 0])
# plt.plot(vis_meas[:, 3] / dt)
plt.plot(vis_meas2[:, 3] / dt)
# plt.plot(vis_meas[:, 3] / 0.1)
# plt.plot(est_velocities[:, 1])
# plt.plot(est_velocities[:, 2])
plt.show()