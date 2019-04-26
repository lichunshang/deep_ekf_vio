import os
import numpy as np
from log import Logger, logger
from se3 import log_SO3, r_from_T, C_from_T


def calc_error(working_dir):
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ CALCULATE ERRORS ================")
    logger.print("Working on directory:", working_dir)

    pose_est_dir = os.path.join(working_dir, "est_poses")
    pose_gt_dir = os.path.join(working_dir, "gt_poses")
    vis_meas_dir = os.path.join(working_dir, "vis_meas")
    errors_output_dir = os.path.join(working_dir, "errors")
    Logger.make_dir_if_not_exist(errors_output_dir)
    pose_est_files = sorted(os.listdir(pose_est_dir))

    logger.print("Found pose estimate files: \n" + "\n".join(pose_est_files))

    for i, pose_est_file in enumerate(pose_est_files):
        sequence = os.path.splitext(pose_est_file)[0]

        traj_est = np.load(os.path.join(pose_est_dir, "%s.npy" % sequence))
        traj_gt = np.load(os.path.join(pose_gt_dir, "%s.npy" % sequence))
        vis_meas = np.load(os.path.join(vis_meas_dir, "meas", "%s.npy" % sequence))
        assert (traj_est.shape[0] == traj_gt.shape[0])

        abs_traj_error = []
        rel_traj_error = [np.zeros(6)]
        vis_meas_error = [np.zeros(6)]

        for j in range(0, traj_est.shape[0]):
            pose_est = traj_est[j]
            pose_gt = traj_gt[j]

            abs_pose_err = np.linalg.inv(pose_est).dot(pose_gt)
            abs_traj_error.append(np.concatenate([log_SO3(C_from_T(abs_pose_err)), r_from_T(abs_pose_err)]))

        for j in range(1, traj_est.shape[0]):
            rel_pose_gt = np.linalg.inv(traj_gt[j - 1]).dot(traj_gt[j])
            rel_pose_est = np.linalg.inv(traj_est[j - 1]).dot(traj_est[j])
            rel_pose_err = np.linalg.inv(rel_pose_est).dot(rel_pose_gt)
            rel_traj_error.append(np.concatenate([log_SO3(C_from_T(rel_pose_err)), r_from_T(rel_pose_err)]))
            vis_meas_error.append(np.concatenate([log_SO3(C_from_T(rel_pose_gt)), r_from_T(rel_pose_gt)]) -
                                  vis_meas[j - 1])

        np.save(logger.ensure_file_dir_exists(os.path.join(errors_output_dir, "abs", sequence + ".npy")),
                np.array(abs_traj_error))
        np.save(logger.ensure_file_dir_exists(os.path.join(errors_output_dir, "rel", sequence + ".npy")),
                np.array(rel_traj_error))
        np.save(logger.ensure_file_dir_exists(os.path.join(errors_output_dir, "vis_meas", sequence + ".npy")),
                np.array(vis_meas_error))
        logger.print("Error saved for sequence %s" % sequence)

    logger.print("All Done.")
