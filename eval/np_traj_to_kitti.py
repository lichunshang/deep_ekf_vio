import os
import numpy as np
from log import Logger, logger


def write_trj(file_handle, pose):
    pose = np.concatenate(pose[0:3])
    file_handle.write(" ".join(["%f" % val for val in list(pose)]) + "\n")


def np_traj_to_kitti(working_dir):
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ CONVERT TO KITTI ================")
    logger.print("Working on directory:", working_dir)

    pose_est_dir = os.path.join(working_dir, "est_poses")
    pose_gt_dir = os.path.join(working_dir, "gt_poses")
    kitti_traj_output = os.path.join(working_dir, "kitti")
    Logger.make_dir_if_not_exist(kitti_traj_output)
    pose_est_files = sorted(os.listdir(pose_est_dir))

    logger.print("Found pose estimate files: \n" + "\n".join(pose_est_files))

    for i, pose_est_file in enumerate(pose_est_files):
        sequence = os.path.splitext(pose_est_file)[0]

        traj_est = np.load(os.path.join(pose_est_dir, "%s.npy" % sequence))
        traj_gt = np.load(os.path.join(pose_gt_dir, "%s.npy" % sequence))
        kitti_est_file = open(os.path.join(kitti_traj_output, "%s_est.txt" % sequence), "w")
        kitti_gt_file = open(os.path.join(kitti_traj_output, "%s_gt.txt" % sequence), "w")

        assert (traj_est.shape[0] == traj_gt.shape[0])

        for j in range(0, traj_est.shape[0]):
            write_trj(kitti_est_file, traj_est[j, :, :])
            write_trj(kitti_gt_file, traj_gt[j, :, :])

        kitti_est_file.close()
        kitti_gt_file.close()

    logger.print("All Done.")
