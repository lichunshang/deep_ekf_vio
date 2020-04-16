from utils import Plotter
import os
import numpy as np
import matplotlib.pyplot as plt
import se3
import glob
from data_loader import SequenceData


def pfind(*path):
    p = glob.glob(os.path.join(*path) + "*")
    assert len(p) == 1
    return p[0]


seq = "K01"
seq_data = SequenceData(seq)
base_dir = "/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results"
gt_poses = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss", seq + "_train"), "saved_model.eval.traj/gt_poses", seq + ".npy"))
vanilla_poses = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss", seq + "_train"), "saved_model.eval.traj/est_poses", seq + ".npy"))
timestmaps = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss", seq + "_train"), "saved_model.eval.traj/timestamps", seq+".npy"))
gt_velocities = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss", seq + "_train"), "saved_model.eval.traj/ekf_states/gt_velocities", seq + ".npy"))
est_velocities = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss", seq + "_train"), "saved_model.eval.traj/ekf_states/states", seq + ".npy"))[:,15:18]
vis_meas = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss", seq + "_train"), "saved_model.eval.traj/vis_meas/meas", seq + ".npy"))
covars = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss", seq + "_train"), "saved_model.eval.traj/vis_meas/covar", seq + ".npy"))
# covars = np.load("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_nogloss/K07_train_20200130-22-45-48/saved_model.eval.traj/vis_meas/covar/K07.npy")

fig, ax1 = plt.subplots(1)
dt = np.ediff1d(timestmaps)

l1 = ax1.plot(timestmaps, gt_velocities[:, 0], color="b", label="gt vel", linewidth=1)
# ax1.plot(timestmaps, vis_meas[:, 3] / dt, color="g", label="est vel")
l2 = ax1.plot(timestmaps, est_velocities[:, 0], color="r", label="est vel", linestyle='--', linewidth=1)
# ax1.plot(np.sqrt(covars[:, 3, 3], color="k")
# ax1[0].set_ylim(ymin=0)
ax1.set_ylabel("velocity [m/s]")
ax1.set_xlabel("time [s]")

ax2 = ax1.twinx()
l3 = ax2.plot(timestmaps[1:], covars[:, 3, 3], color="g", label="cov x", linewidth=1)
ax2.set_ylabel("variance [m^2]")
ax1.legend(l1 + l2 + l3, ["gt $v_x$", "est $v_x$", "var($\\tilde{r}_x$)"], loc='upper left')
sz = fig.get_size_inches()
# sz[1] /= 2
ax1.grid()
fig.set_size_inches(sz)
# ax2.set_ylim(ymin=0)

# ax1[1].plot(timestmaps, gt_poses[:,0, 3], label="")
# ax1[1].plot(timestmaps, vanilla_poses[:,0, 3])
# ax1[1].plot(timestmaps, gt_poses[:,1, 3])
# ax1[1].plot(timestmaps, vanilla_poses[:,1, 3])


# plt.plot(est_velocities[:, 1])
# plt.plot(est_velocities[:, 2])
# plt.show()
plt.savefig("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_figures/vel_x.svg",  format='svg', bbox_inches='tight', pad_inches=0)

# Figure 2
fig, ax1 = plt.subplots(1)
dt = np.ediff1d(timestmaps)

# l1 = ax1.plot(timestmaps, gt_velocities[:, 1], color="b", label="gt vel", linewidth=1)
z_rot_vel = np.array([seq_data.df.loc[i, "gyro_measurements"][0, 2] for i in range(0, len(seq_data.df) - 1)])
l2 = ax1.plot(timestmaps[:-1], np.abs(z_rot_vel), color="r", label="est vel", linestyle='--', linewidth=1)
ax1.set_ylabel("angular velocity abs [rad/s]")
ax1.set_xlabel("time [s]")

ax2 = ax1.twinx()
l3 = ax2.plot(timestmaps[1:], covars[:, 4, 4], color="g", label="cov y", linewidth=1)
ax2.set_ylabel("variance [m^2]")
ax1.legend(l2 + l3, ["gt $w_z$", "var($\\tilde{r}_y$)"], loc='upper left')
sz = fig.get_size_inches()
# sz[1] /= 2
ax1.grid()
fig.set_size_inches(sz)
# ax2.set_ylim(ymin=0)

# plt.show()
plt.savefig("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_figures/vel_y.svg",  format='svg', bbox_inches='tight', pad_inches=0)