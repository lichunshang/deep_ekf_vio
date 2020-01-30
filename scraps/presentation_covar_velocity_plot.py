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


seq = "K07"
base_dir = "/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results"
# gt_poses = np.load(os.path.join(pfind(base_dir, "KITTI", seq + "_train"), "saved_model.eval.traj/gt_poses", seq + ".npy"))
# vanilla_poses = np.load(os.path.join(pfind(base_dir, "KITTI", seq + "_train"), "saved_model.eval.traj/est_poses", seq + ".npy"))
gt_velocities = np.load(os.path.join(pfind(base_dir, "KITTI", seq + "_train"), "saved_model.eval.traj/ekf_states/gt_velocities", seq + ".npy"))
est_velocities = np.load(os.path.join(pfind(base_dir, "KITTI", seq + "_train"), "saved_model.eval.traj/ekf_states/states", seq + ".npy"))[:,15:18]
# vis_meas = np.load("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_vision_only_aug/K07_train_20200123-23-39-58/saved_model.eval.traj/vis_meas/meas/K07.npy")
vis_meas = np.load("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_vision_only_aug/K07_train_20200123-23-39-58/saved_model.eval.traj/vis_meas/meas/K07.npy")
covars = np.load(os.path.join(pfind(base_dir, "KITTI", seq + "_train"), "saved_model.eval.traj/ekf_states/vis_meas_covar", seq + ".npy"))

fig, axs = plt.subplots(2)
axs[0].plot(gt_velocities[:, 0])
axs[0].plot(est_velocities[:, 0])
axs[0].plot(vis_meas[:, 3] / 0.1)

# plt.plot(est_velocities[:, 1])
# plt.plot(est_velocities[:, 2])
plt.show()