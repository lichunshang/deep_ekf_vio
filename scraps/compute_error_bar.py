import numpy as np
from eval.kitti_eval_pyimpl import *


def calc_kitti_f2f_errors(gt_poses, est_poses):
    assert (len(gt_poses) == len(est_poses))
    gt_poses = gt_poses.astype(np.float64)
    est_poses = est_poses.astype(np.float64)
    step_size = 10
    distances = calc_trajectory_dist(gt_poses)
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    errors_by_length = {k: [] for k in lengths}

    est_rel = np.array([np.linalg.inv(est_poses[i]).dot(est_poses[i + 1]) for i in range(0, len(est_poses) - 1)])
    gt_rel = np.array([np.linalg.inv(gt_poses[i]).dot(gt_poses[i + 1]) for i in range(0, len(gt_poses) - 1)])
    err_rel = np.matmul(np.linalg.inv(est_rel), gt_rel)

    for i in range(0, len(gt_poses) - 1, step_size):
        for length in lengths:
            j = last_frame_from_segment_length(distances, i, length)
            if j < 0:
                continue

            trans_err = np.linalg.norm(err_rel[:, 0:3, 3], axis=-1)

            rot_diag = np.diagonal(err_rel[:, 0:3, 0:3], axis1=-2, axis2=-1)
            rot_err = np.arccos(np.clip((np.sum(rot_diag, axis=-1) - 1.0) * 0.5, a_min=- 1.0, a_max=1.0))
            e = np.stack([trans_err, rot_err], axis=-1)
            errors_by_length[length] += list(e)

    return errors_by_length


gt = np.load(
    "/home/cs4li/Dev/deep_ekf_vio/results/train_20190422-19-21-05_ekf_ekf_gloss_0.5k3_1e-5lr_imucovfix/saved_model.eval.traj/gt_poses/K10.npy")
est = np.load(
    "/home/cs4li/Dev/deep_ekf_vio/results/train_20190422-19-21-05_ekf_ekf_gloss_0.5k3_1e-5lr_imucovfix/saved_model.eval.traj/est_poses/K10.npy")

err = calc_kitti_f2f_errors(gt, est)

print("Done")
