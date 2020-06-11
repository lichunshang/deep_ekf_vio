from utils import Plotter
import os
import numpy as np
import matplotlib.pyplot as plt
import se3
import glob


def pfind(*path):
    print(path)
    p = glob.glob(os.path.join(*path) + "*")
    assert len(p) == 1
    return p[0]


seq = "K01"
base_dir = "/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results"
gt_poses = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss_scaled_est", seq + "_train"), "saved_model.eval.traj/gt_poses", seq + ".npy"))
vanilla_poses = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss", seq + "_train"), "saved_model.eval.traj/est_poses", seq + ".npy"))
vanilla_scaled_poses = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss_scaled_est", seq + "_train"), "saved_model.eval.traj/est_poses", seq + ".npy"))

timestmaps = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss_scaled_est", seq + "_train"), "saved_model.eval.traj/timestamps", seq+".npy"))
gt_velocities = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss_scaled_est", seq + "_train"), "saved_model.eval.traj/ekf_states/gt_velocities", seq + ".npy"))
est_velocities = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss_scaled_est", seq + "_train"), "saved_model.eval.traj/ekf_states/states", seq + ".npy"))[:,15:18]
est_scale = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss_scaled_est", seq + "_train"), "saved_model.eval.traj/ekf_states/states", seq + ".npy"))[:, 24]
vis_meas = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss_scaled_est", seq + "_train"), "saved_model.eval.traj/vis_meas/meas", seq + ".npy"))

dt = np.ediff1d(timestmaps)

plotter = Plotter(os.path.join(base_dir, "KITTI_figures"))

plotter.plot(([gt_poses[:, 0, 3], gt_poses[:, 1, 3]],
              [vanilla_poses[:, 0, 3], vanilla_poses[:, 1, 3]],
              [vanilla_scaled_poses[:, 0, 3], vanilla_scaled_poses[:, 1, 3]],
              ),
             "x [m]", "y [m]", None,
             labels=["ground truth", "proposed", "proposed-scaled"],
             colors=["blue", "red", "green"],
             equal_axes=True, filename=seq+"_scaled.pdf")

# plt.figure(2)
# plt.plot(timestmaps, gt_velocities[:, 0], color="b", label="gt vel", linewidth=1)
# plt.plot(timestmaps, est_velocities[:, 0], color="r", label="est vel", linestyle='--', linewidth=1)
# plt.plot(timestmaps[1:], vis_meas[:, 3] / dt, color="g", label="est vel")
# plt.ylabel("velocity [m/s]")
# plt.xlabel("time [s]")
#
#
# plt.legend(["gt $v_x$", "est $v_x$", "$\\tilde{r}_x / \Delta t$"], loc='upper left')
# sz = plt.gcf().get_size_inches()
# sz[1] /= 1.5
# plt.grid()
# plt.gcf().set_size_inches(sz)
# plt.savefig("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_figures/vel_x_vis_meas_scaled.pdf",  format='pdf', bbox_inches='tight', pad_inches=0)
#

fig, ax1 = plt.subplots(1)
dt = np.ediff1d(timestmaps)

l1 = ax1.plot(timestmaps, gt_velocities[:, 0], color="b", label="gt vel", linewidth=1)
l2 = ax1.plot(timestmaps, est_velocities[:, 0], color="r", label="est vel", linestyle='--', linewidth=1)
l3 = ax1.plot(timestmaps[1:], vis_meas[:, 3] / dt, color="g", label="est vel")
ax1.set_ylabel("velocity [m/s]")
ax1.set_xlabel("time [s]")

ax2 = ax1.twinx()
l4 = ax2.plot(timestmaps, est_scale, color="gold", label="scale", linewidth=1)
ax2.set_ylabel("scale []")
ax1.legend(l1 + l2 + l3 + l4, ["gt $v_x$", "est $v_x$", "$\\tilde{r}_x / \Delta t$", "$\lambda$"], loc='upper right')
sz = fig.get_size_inches()
sz[1] /= 1.5
ax1.grid()
fig.set_size_inches(sz)

plt.savefig("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_figures/vel_x_vis_meas_scaled.pdf",  format='pdf', bbox_inches='tight', pad_inches=0)