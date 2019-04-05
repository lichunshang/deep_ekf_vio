import numpy as np
import matplotlib.pyplot as plt
import se3
from log import logger, Logger
import os

if "DISPLAY" not in os.environ:
    plt.switch_backend("Agg")


def plot_trajectory(working_dir):
    output_dir = os.path.join(working_dir, "figures")
    pose_est_dir = os.path.join(working_dir, "est_poses")
    pose_gt_dir = os.path.join(working_dir, "gt_poses")
    assert sorted(os.listdir(pose_est_dir)) == sorted(os.listdir(pose_gt_dir))
    pose_est_files = sorted(os.listdir(pose_est_dir))

    Logger.make_dir_if_not_exist(output_dir)
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ PLOT TRAJECTORY ================")
    logger.print("Working on directory:", working_dir)
    logger.print("Found pose estimate files: \n" + "\n".join(pose_est_files))

    for i, pose_est_file in enumerate(pose_est_files):
        sequence = os.path.splitext(pose_est_file)[0]

        trajectory = np.load(os.path.join(pose_est_dir, "%s.npy" % sequence))
        trajectory_gt = np.load(os.path.join(pose_gt_dir, "%s.npy" % sequence))

        trans_xyz = np.array([se3.r_from_T(T) for T in trajectory])
        rot_xyz = np.array([se3.log_SO3(se3.C_from_T(T)) for T in trajectory])
        trans_xyz_gt = np.array([se3.r_from_T(T) for T in trajectory_gt])
        rot_xyz_gt = np.array([se3.log_SO3(se3.C_from_T(T)) for T in trajectory_gt])

        trans_x = trans_xyz[:, 0]
        trans_y = trans_xyz[:, 1]
        trans_z = trans_xyz[:, 2]
        rot_x = rot_xyz[:, 0]
        rot_y = rot_xyz[:, 1]
        rot_z = rot_xyz[:, 2]

        trans_x_gt = trans_xyz_gt[:, 0]
        trans_y_gt = trans_xyz_gt[:, 1]
        trans_z_gt = trans_xyz_gt[:, 2]
        rot_x_gt = rot_xyz_gt[:, 0]
        rot_y_gt = rot_xyz_gt[:, 1]
        rot_z_gt = rot_xyz_gt[:, 2]

        # ########## XYZ Planes Plots ##########
        # plt.figure()
        plt.clf()
        plt.plot(trans_x, trans_y, linewidth=1.0, color="r", label="Estimate")
        plt.plot(trans_x_gt, trans_y_gt, linewidth=1.0, color="b", label="Ground Truth")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Seq. %s Trajectory XY Plane" % sequence)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "seq_%s_00_xy_traj.png" % sequence))

        plt.clf()
        plt.plot(trans_x, trans_z, linewidth=1.0, color="r", label="Estimate")
        plt.plot(trans_x_gt, trans_z_gt, linewidth=1.0, color="b", label="Ground Truth")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("z [m]")
        plt.title("Seq. %s Trajectory XZ Plane" % sequence)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "seq_%s_01_xz_traj.png" % sequence))

        plt.clf()
        plt.plot(trans_y, trans_z, linewidth=1.0, color="r", label="Estimate")
        plt.plot(trans_y_gt, trans_z_gt, linewidth=1.0, color="b", label="Ground Truth")
        plt.axis("equal")
        plt.xlabel("y [m]")
        plt.ylabel("z [m]")
        plt.title("Seq. %s Trajectory YZ Plane" % sequence)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "seq_%s_02_yz_traj.png" % sequence))

        # ########## Progression Throughout Planes Plot ##########
        labels = ["Trans X", "Trans Y", "Trans Z", "Rot X", "Rot Y", "Rot Z"]
        data = [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
        data_gt = [trans_x_gt, trans_y_gt, trans_z_gt, rot_x_gt, rot_y_gt, rot_z_gt]

        for j in range(0, len(labels)):
            plt.clf()
            plt.plot(data[j], linewidth=1.0, color="r", label="Estimate")
            plt.plot(data_gt[j], linewidth=1.0, color="b", label="Ground Truth")
            plt.xlabel("frame # []")
            plt.ylabel(labels[j].lower())
            plt.title("Seq. %s %s" % (sequence, labels[j]))
            plt.legend()
            plt.savefig(os.path.join(output_dir, "seq_%s_%02d_%s_plt.png" %
                                     (sequence, j + 3, "_".join(labels[j].lower().split()))))

        logger.print("Plot saved for sequence %s" % sequence)
