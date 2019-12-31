from utils import Plotter
import os
import numpy as np
import matplotlib.pyplot as plt
import se3

seq = "K07"
gt_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation_Results/KITTI/"
                     "0vanillagloss_train_20190509-12-31-54_esg_vistanh_1e4beta_uncorrcovar/"
                     "saved_model.eval.traj/gt_poses", seq + ".npy"))
vanilla_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation_Results/KITTI/"
                     "train_20190722-00-01-20_vanilla/"
                     "saved_model.eval.traj/est_poses", seq + ".npy"))
hybrid_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation_Results/KITTI/"
                     "train_20190722-00-01-58_hybrid/"
                     "saved_model.eval.traj/est_poses", seq + ".npy"))
vision_only_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation_Results/KITTI/"
                     "3vision_train_20190420-01-16-51_allaug_GaussCovarLoss_lr1e-4_eps1e-4r1e-2t_100k4/"
                     "saved_model.eval.traj/est_poses", seq + ".npy"))
imu_only_poses = np.load(
        os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/Presentation_Results/KITTI/imu", seq, "est.npy"))

plotter = Plotter("/home/cs4li/Dev/deep_ekf_vio/results/Presentation_Results/KITTI")


def plot_callback(fig, ax):
    ax.plot(gt_poses[0, 0, 3], gt_poses[0, 1, 3], 'x', color='black', markersize=10, markeredgewidth=2, label="start")
    ax.plot()
    ax.legend(numpoints=1, prop={'size': 10})

    # ax.arrow(75, 195, -20, -20, head_width=5, head_length=5, fc='r', ec='r')
    circle = plt.Circle((45, 170), 20, color='r', fill=False, linestyle="--")
    ax.add_artist(circle)

    # plot_margin = 25
    # x0, x1, y0, y1 = ax.axis()
    # ax.axis((x0 - plot_margin,
    #          x1 + plot_margin,
    #          y0 - plot_margin,
    #          y1 + plot_margin))


plotter.plot(([gt_poses[:, 0, 3], gt_poses[:, 1, 3]],
              [vanilla_poses[:, 0, 3], vanilla_poses[:, 1, 3]],
              [hybrid_poses[:, 0, 3], hybrid_poses[:, 1, 3]],
              [vision_only_poses[:, 0, 3], vision_only_poses[:, 1, 3]],
              [imu_only_poses[:, 0, 3], imu_only_poses[:, 1, 3]],
              ),
             "x [m]", "y [m]", "KITTI Sequence %s" % seq[1:],
             labels=["gt", "vanilla", "hybrid", "vision", "imu"],
             equal_axes=True, filename=seq, callback=plot_callback)

# plotter.plot(([gt_poses[:, 0, 3], gt_poses[:, 1, 3]],
#               [hybrid_poses[:, 0, 3], hybrid_poses[:, 1, 3]],
#               ),
#              "x [m]", "y [m]", "KITTI Sequence %s" % seq[1:],
#              labels=["gt", "hybrid"],
#              equal_axes=True, filename=seq+"_one")

# IMU only errors
plt.clf()
toplot_error = np.matmul(np.linalg.inv(imu_only_poses), gt_poses)
toplot_position_error = np.linalg.norm(toplot_error[:, 0:3, 3], axis=1)
toplot_orientation_error = np.linalg.norm(np.array([se3.log_SO3(e[0:3,0:3]) for e in toplot_error]), axis=1)
time = np.linspace(0, len(gt_poses) / 10, len(gt_poses))

ax1 = plt.axes()
ax2 = ax1.twinx()
ax1.set_xlabel("time [s]")
ax1.set_ylabel("position error norm [m]", color="red")
ax1.plot(time, toplot_position_error, color="red")
ax1.tick_params(axis='y', labelcolor="red")

ax2.set_ylabel("orientation error norm [deg]", color="blue")
ax2.plot(time, toplot_orientation_error * 180 / np.pi, color="blue")
ax2.tick_params(axis='y', labelcolor="blue")

plt.title("KITTI Sequence 07 IMU Only Errors")
plt.show()