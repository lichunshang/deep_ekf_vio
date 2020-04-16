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
gt_poses = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss", seq + "_train"), "saved_model.eval.traj/gt_poses", seq + ".npy"))
vanilla_poses = np.load(os.path.join(pfind(base_dir, "KITTI_nogloss", seq + "_train"), "saved_model.eval.traj/est_poses", seq + ".npy"))
vision_only_poses = np.load(
        os.path.join(pfind(base_dir, "KITTI_vision_only_aug", seq), "saved_model.eval.traj/est_poses", seq + ".npy"))
msf_fusion_poses = np.load(os.path.join(base_dir, "KITTI_msf", seq, "est_shifted.npy"))
imu_only_poses = np.load(os.path.join(base_dir, "KITTI_imu_only", seq, "est.npy"))

plotter = Plotter(os.path.join(base_dir, "KITTI_figures"))


def plot_callback(fig, ax):
    ax.plot(gt_poses[0, 0, 3], gt_poses[0, 1, 3], 'x', color='black', markersize=10, markeredgewidth=2, label="start")
    ax.plot()
    ax.legend(numpoints=1, prop={'size': 8})

    #K07
    if seq == "K07":
        ax.arrow(75, 195, -20, -20, head_width=5, head_length=5, fc='r', ec='r')
        circle = plt.Circle((45, 170), 20, color='r', fill=False, linestyle="--")
        ax.add_artist(circle)

        plot_margin = 25
        x0, x1, y0, y1 = ax.axis()
        ax.axis((x0 - plot_margin,
                 x1 + plot_margin,
                 y0 - plot_margin,
                 y1 + plot_margin))

    #K08
    if seq == "K08":
        ax.axis((0,1000,-600,400))


plotter.plot(([imu_only_poses[:, 0, 3], imu_only_poses[:, 1, 3]],
              [vision_only_poses[:, 0, 3], vision_only_poses[:, 1, 3]],
              [msf_fusion_poses[:, 0, 3], msf_fusion_poses[:, 1, 3]],
              [gt_poses[:, 0, 3], gt_poses[:, 1, 3]],
              [vanilla_poses[:, 0, 3], vanilla_poses[:, 1, 3]],
              ),
             "x [m]", "y [m]", None,
             labels=["IMU",  "vision", "ORB+MSF", "ground truth", "proposed"],
             colors=["turquoise", "gold", "green", "blue", "red"],
             equal_axes=True, filename=seq+".svg", callback=plot_callback)

# plotter.plot(([gt_poses[:, 0, 3], gt_poses[:, 1, 3]],
#               [hybrid_poses[:, 0, 3], hybrid_poses[:, 1, 3]],
#               ),
#              "x [m]", "y [m]", "KITTI Sequence %s" % seq[1:],
#              labels=["gt", "hybrid"],
#              equal_axes=True, filename=seq+"_one")

# IMU only errors
# plt.clf()
# toplot_error = np.matmul(np.linalg.inv(imu_only_poses), gt_poses)
# toplot_position_error = np.linalg.norm(toplot_error[:, 0:3, 3], axis=1)
# toplot_orientation_error = np.linalg.norm(np.array([se3.log_SO3(e[0:3, 0:3]) for e in toplot_error]), axis=1)
# time = np.linspace(0, len(gt_poses) / 10, len(gt_poses))
#
# ax1 = plt.axes()
# ax2 = ax1.twinx()
# ax1.set_xlabel("time [s]")
# ax1.set_ylabel("position error norm [m]", color="red")
# ax1.plot(time, toplot_position_error, color="red")
# ax1.tick_params(axis='y', labelcolor="red")
#
# ax2.set_ylabel("orientation error norm [deg]", color="blue")
# ax2.plot(time, toplot_orientation_error * 180 / np.pi, color="blue")
# ax2.tick_params(axis='y', labelcolor="blue")
#
# plt.title("KITTI Sequence 07 IMU Only Errors")
# plt.show()
