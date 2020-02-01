from utils import Plotter
import os
import numpy as np
from data_loader import SequenceData
import matplotlib.pyplot as plt
import se3
import glob
from evo.tools import file_interface
from evo.core import trajectory, sync, metrics

def pfind(*path):
    p = glob.glob(os.path.join(*path) + "*")
    assert len(p) == 1
    return p[0]


def read_tum(vins_mono_poses):
    wxyz = np.zeros_like(vins_mono_poses[:, 4:])
    wxyz[:, 0] = vins_mono_poses[:, 7]
    wxyz[:, 1:4] = vins_mono_poses[:, 4:7]

    xyz = vins_mono_poses[:,1:4]
    ts = vins_mono_poses[:,0]

    return trajectory.PoseTrajectory3D(positions_xyz=xyz, orientations_quat_wxyz=wxyz, timestamps=ts)

seq = "MH_05"
base_dir = "/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results"
timestamps_rel = np.load(os.path.join(pfind(base_dir, "EUROC", seq + "_train"), "saved_model.eval.traj/timestamps", seq + ".npy"))
# gt_poses = np.load(os.path.join(pfind(base_dir, "EUROC", seq + "_train"), "saved_model.eval.traj/gt_poses", seq + ".npy"))
gt_poses = np.loadtxt(os.path.join(base_dir, "EUROC_vins_mono", seq, "gt.tum"))
vanilla_poses = np.load(os.path.join(pfind(base_dir, "EUROC", seq + "_train"), "saved_model.checkpoint.traj/est_poses", seq + ".npy"))
vision_only_poses = np.load(
        os.path.join(pfind(base_dir, "EUROC_vision_only", seq), "saved_model.eval.traj/est_poses", seq + ".npy"))
vins_mono_poses = np.loadtxt(os.path.join(base_dir, "EUROC_vins_mono", seq, "vinsmono_output.tum"))
# imu_only_poses = np.load(os.path.join(base_dir, "KITTI_imu_only", seq, "est.npy"))
seq_data = SequenceData(seq)
raw_timestamps = np.array(seq_data.df.loc[:, "timestamp_raw"])
timestamps = timestamps_rel + raw_timestamps[0] / 1e9

gt_traj = read_tum(gt_poses)
vanilla_traj = trajectory.PoseTrajectory3D(poses_se3=vanilla_poses, timestamps=timestamps)
vision_only_traj = trajectory.PoseTrajectory3D(poses_se3=vision_only_poses, timestamps=timestamps)

vins_mono_traj = read_tum(vins_mono_poses)

gt_traj_synced_vanilla, vanilla_traj_synced = sync.associate_trajectories(gt_traj, vanilla_traj, max_diff=0.01)
gt_traj_synced_vision_only, vision_only_traj_synced = sync.associate_trajectories(gt_traj, vision_only_traj, max_diff=0.01)
gt_traj_synced_vins_mono, vins_mono_traj_synced = sync.associate_trajectories(gt_traj, vins_mono_traj, max_diff=0.01)

vanilla_traj_aligned = trajectory.align_trajectory(vanilla_traj_synced, gt_traj_synced_vanilla,correct_scale=False, correct_only_scale=False)
vision_only_traj_aligned = trajectory.align_trajectory(vision_only_traj_synced, gt_traj_synced_vision_only,correct_scale=False, correct_only_scale=False)
vins_mono_traj_aligned = trajectory.align_trajectory(vins_mono_traj_synced, gt_traj_synced_vins_mono,correct_scale=False, correct_only_scale=False)

plotter = Plotter(os.path.join(base_dir, "KITTI_figures"))

def plot_callback(fig, ax):
    pass
    # ax.plot(gt_poses[0, 0, 3], gt_poses[0, 1, 3], 'x', color='black', markersize=10, markeredgewidth=2, label="start")
    # ax.plot()
    # ax.legend(numpoints=1, prop={'size': 8})

    #K07
    # ax.arrow(75, 195, -20, -20, head_width=5, head_length=5, fc='r', ec='r')
    # circle = plt.Circle((45, 170), 20, color='r', fill=False, linestyle="--")
    # ax.add_artist(circle)
    #
    # plot_margin = 25
    # x0, x1, y0, y1 = ax.axis()
    # ax.axis((x0 - plot_margin,
    #          x1 + plot_margin,
    #          y0 - plot_margin,
    #          y1 + plot_margin))

    #K08
    # ax.axis((0,1000,-600,400))


plotter.plot(([vision_only_traj_aligned.positions_xyz[:, 0], vision_only_traj_aligned.positions_xyz[:, 1]],
              [vins_mono_traj_aligned.positions_xyz[:, 0], vins_mono_traj_aligned.positions_xyz[:, 1]],
              [gt_traj_synced_vanilla.positions_xyz[:, 0], gt_traj_synced_vanilla.positions_xyz[:, 1]],
              [vanilla_traj_aligned.positions_xyz[:, 0], vanilla_traj_aligned.positions_xyz[:, 1]],
              ),
             "x [m]", "y [m]", None,
             labels=["vision", "vins_mono", "gt", "proposed"],
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
