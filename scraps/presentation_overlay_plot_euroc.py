import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

seq = "V1_03"
gt_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation_Results/EUROC/"
                     "0vanilla_train_20190520-16-55-11_esge_100k1_0.5k3",
                     "saved_model.eval.traj/gt_poses", seq + ".npy"))
vanilla_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation_Results/EUROC/"
                     "0vanilla_train_20190520-16-55-11_esge_100k1_0.5k3",
                     "saved_model.eval.traj/est_poses", seq + ".npy"))
hybrid_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation_Results/EUROC/"
                     "1hybrid_train_20190529-00-06-56_esg_fiximu_hybrid_euroc",
                     "saved_model.eval.traj/est_poses", seq + ".npy"))
vision_only_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation_Results/EUROC/"
                     "2visiononly_train_20190520-16-17-00_euroc_noekf_100k1_235x150sz",
                     "saved_model.eval.traj/est_poses", seq + ".npy"))
imu_only_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation_Results/EUROC/imu", seq, "est.npy"))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(gt_poses[:, 0, 3], gt_poses[:, 1, 3], gt_poses[:, 2, 3], label='gt')
# ax.plot(vanilla_poses[:, 0, 3], vanilla_poses[:, 1, 3], vanilla_poses[:, 2, 3], label='vanilla')
ax.plot(hybrid_poses[:, 0, 3], hybrid_poses[:, 1, 3], hybrid_poses[:, 2, 3], label='hybrid')
# ax.plot(vision_only_poses[:, 0, 3], vision_only_poses[:, 1, 3], vision_only_poses[:, 2, 3], label='vision')
# ax.plot(imu_only_poses[:, 0, 3], imu_only_poses[:, 1, 3], imu_only_poses[:, 2, 3], label='imu')
ax.legend()
ax.set_xlabel("x[m]")
ax.set_ylabel("y[m]")
ax.set_zlabel("z[m]")
plt.title("EUROC Sequence %s" % seq)

plt.show()
