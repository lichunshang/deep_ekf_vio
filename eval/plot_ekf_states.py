import numpy as np
from model import IMUKalmanFilter
from se3 import log_SO3
from utils import Plotter
import matplotlib.pyplot as plt
import torch
import os
from log import Logger, logger

if "DISPLAY" not in os.environ:
    plt.switch_backend("Agg")


def plot_ekf_data(output_dir, timestamps, gt_poses, gt_vels, est_poses, est_states, g_const=9.80665):
    # convert all to np
    timestamps = np.array([np.array(item) for item in timestamps], dtype=np.float64)
    gt_poses = np.array([np.array(item) for item in gt_poses], dtype=np.float64)
    gt_vels = np.array([np.array(item) for item in gt_vels], dtype=np.float64)
    est_poses = np.array([np.array(item) for item in est_poses], dtype=np.float64)
    est_states = np.array([np.array(item) for item in est_states], dtype=np.float64)

    est_positions = []
    est_vels = []
    est_rots = []
    est_gravities = []
    est_ba = []
    est_bw = []

    for i in range(0, len(est_poses)):
        pose = np.linalg.inv(est_poses[i])
        g, C, r, v, bw, ba = IMUKalmanFilter.decode_state(torch.tensor(est_states[i]))
        est_positions.append(pose[0:3, 3])
        est_rots.append(log_SO3(pose[0:3, 0:3]))
        est_vels.append(np.array(v))
        est_gravities.append(np.array(g))
        est_bw.append(np.array(bw))
        est_ba.append(np.array(ba))

    est_positions = np.squeeze(est_positions)
    est_vels = np.squeeze(est_vels)
    est_rots = np.squeeze(est_rots)
    est_gravities = np.squeeze(est_gravities)
    est_bw = np.squeeze(est_bw)
    est_ba = np.squeeze(est_ba)

    gt_rots = np.array([log_SO3(p[:3, :3]) for p in gt_poses])
    gt_gravities = np.array([gt_poses[i, 0:3, 0:3].transpose().dot([0, 0, g_const])
                             for i in range(0, len(gt_poses))])

    est_rots[:, 0] = np.unwrap(est_rots[:, 0])
    est_rots[:, 1] = np.unwrap(est_rots[:, 1])
    est_rots[:, 2] = np.unwrap(est_rots[:, 2])
    gt_rots[:, 0] = np.unwrap(gt_rots[:, 0])
    gt_rots[:, 1] = np.unwrap(gt_rots[:, 1])
    gt_rots[:, 2] = np.unwrap(gt_rots[:, 2])

    plotter = Plotter(output_dir)
    plotter.plot(([gt_poses[:, 0, 3], gt_poses[:, 1, 3]],
                  [est_positions[:, 0], est_positions[:, 1]],),
                 "x [m]", "y [m]", "XY Plot", labels=["gt_poses", "est_pose"], equal_axes=True)
    plotter.plot(([gt_poses[:, 0, 3], gt_poses[:, 2, 3]],
                  [est_positions[:, 0], est_positions[:, 2]],),
                 "x [m]", "z [m]", "XZ Plot", labels=["gt_poses", "est_pose"], equal_axes=True)
    plotter.plot(([gt_poses[:, 1, 3], gt_poses[:, 2, 3]],
                  [est_positions[:, 1], est_positions[:, 2]],),
                 "y [m]", "z [m]", "YZ Plot", labels=["gt_poses", "est_pose"], equal_axes=True)

    plotter.plot(([timestamps, gt_poses[:, 0, 3]], [timestamps, est_positions[:, 0]],),
                 "t [s]", "p [m]", "Pos X", labels=["gt", "est"])
    plotter.plot(([timestamps, gt_poses[:, 1, 3]], [timestamps, est_positions[:, 1]],),
                 "t [s]", "p [m]", "Pos Y", labels=["gt", "est"])
    plotter.plot(([timestamps, gt_poses[:, 2, 3]], [timestamps, est_positions[:, 2]],),
                 "t [s]", "p [m]", "Pos Z", labels=["gt", "est"])

    plotter.plot(([timestamps, gt_vels[:, 0]], [timestamps, est_vels[:, 0]],),
                 "t [s]", "v [m/s]", "Vel X", labels=["gt", "est"])
    plotter.plot(([timestamps, gt_vels[:, 1]], [timestamps, est_vels[:, 1]],),
                 "t [s]", "v [m/s]", "Vel Y", labels=["gt", "est"])
    plotter.plot(([timestamps, gt_vels[:, 2]], [timestamps, est_vels[:, 2]],),
                 "t [s]", "v [m/s]", "Vel Z", labels=["gt", "est"])

    plotter.plot(([timestamps, gt_rots[:, 0]], [timestamps, est_rots[:, 0]],),
                 "t [s]", "rot [rad]", "Rot X", labels=["gt", "est"])
    plotter.plot(([timestamps, gt_rots[:, 1]], [timestamps, est_rots[:, 1]],),
                 "t [s]", "rot [rad]", "Rot Y", labels=["gt", "est"])
    plotter.plot(([timestamps, gt_rots[:, 2]], [timestamps, est_rots[:, 2]],),
                 "t [s]", "rot [rad]", "Rot Z", labels=["gt", "set"])

    plotter.plot(([timestamps, gt_gravities[:, 0]], [timestamps, est_gravities[:, 0]],),
                 "t [s]", "accel [m/s^2]", "Gravity X", labels=["gt", "est"])
    plotter.plot(([timestamps, gt_gravities[:, 1]], [timestamps, est_gravities[:, 1]],),
                 "t [s]", "accel [m/s^2]", "Gravity Y", labels=["gt", "est"])
    plotter.plot(([timestamps, gt_gravities[:, 2]], [timestamps, est_gravities[:, 2]],),
                 "t [s]", "accel [m/s^2]", "Gravity Z", labels=["gt", "est"])

    plotter.plot(([timestamps, est_bw[:, 0]],), "t [s]", "w [rad/s]", "Gyro Bias X")
    plotter.plot(([timestamps, est_bw[:, 1]],), "t [s]", "w [rad/s]", "Gyro Bias Y")
    plotter.plot(([timestamps, est_bw[:, 2]],), "t [s]", "w [rad/s]", "Gyro Bias Z")

    plotter.plot(([timestamps, est_ba[:, 0]],), "t [s]", "a [m/s^2]", "Accel Bias X")
    plotter.plot(([timestamps, est_ba[:, 1]],), "t [s]", "a [m/s^2]", "Accel Bias Y")
    plotter.plot(([timestamps, est_ba[:, 2]],), "t [s]", "a [m/s^2]", "Accel Bias Z")


def plot_ekf_states(working_dir):
    output_dir = os.path.join(working_dir, "figures")

    timestamps_dir = os.path.join(working_dir, "timestamps")
    poses_dir = os.path.join(working_dir, "ekf_states", "poses")
    states_dir = os.path.join(working_dir, "ekf_states", "states")
    gt_velocities_dir = os.path.join(working_dir, "ekf_states", "gt_velocities")
    gt_poses_dir = os.path.join(working_dir, "gt_poses")

    pose_files = sorted(os.listdir(poses_dir))
    assert sorted(os.listdir(poses_dir)) == sorted(os.listdir(states_dir)) == sorted(os.listdir(gt_velocities_dir)) == \
           sorted(os.listdir(timestamps_dir))

    Logger.make_dir_if_not_exist(output_dir)
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ PLOT EKF States ================")
    logger.print("Working on directory:", working_dir)
    logger.print("Found pose estimate files: \n" + "\n".join(pose_files))

    for i, pose_est_file in enumerate(pose_files):
        seq = os.path.splitext(pose_est_file)[0]
        plot_ekf_data(os.path.join(working_dir, "ekf_states", "figures", seq),
                      np.load(os.path.join(timestamps_dir, seq + ".npy")),
                      np.load(os.path.join(gt_poses_dir, seq + ".npy")),
                      np.load(os.path.join(gt_velocities_dir, seq + ".npy")),
                      np.load(os.path.join(poses_dir, seq + ".npy")),
                      np.load(os.path.join(states_dir, seq + ".npy")))
        logger.print("Plot saved for sequence %s" % seq)
