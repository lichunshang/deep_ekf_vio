import numpy as np
from data_loader import SequenceData


def calc_trajectory_dist(poses):
    distances = [0, ]
    for k in range(1, len(poses)):
        pose_km1 = poses[k - 1]
        pose_k = poses[k]
        dist_rel = np.linalg.norm(pose_km1[0:3, 3] - pose_k[0:3, 3])
        distances.append(distances[-1] + dist_rel)
    return distances


def calc_error(gt_pose, est_pose):
    diff = np.linalg.inv(est_pose).dot(gt_pose)
    rot_diag = np.diag(diff[0:3, 0:3])
    rot_err = np.arccos(max(min((np.sum(rot_diag) - 1.0) * 0.5, 1.0), -1.0))

    trans_err = np.linalg.norm(diff[0:3, 3])

    return trans_err, rot_err


def last_frame_from_segment_length(distances, first_frame, length):
    for i in range(first_frame, len(distances)):
        if distances[i] > distances[first_frame] + length:
            return i
    return -1


# written according to KITTI
def calc_kitti_seq_errors(gt_poses, est_poses):
    assert (len(gt_poses) == len(est_poses))
    gt_poses = gt_poses.astype(np.float64)
    est_poses = est_poses.astype(np.float64)
    errors_div_length = []
    step_size = 10
    distances = calc_trajectory_dist(gt_poses)

    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    errors_by_length = {k: [] for k in lengths}
    errors_by_length_div_nframes = {k: [] for k in lengths}

    for i in range(0, len(gt_poses), step_size):
        for length in lengths:
            j = last_frame_from_segment_length(distances, i, length)
            if j < 0:
                continue
            gt_t_ij = np.linalg.inv(gt_poses[i]).dot(gt_poses[j])
            est_t_ij = np.linalg.inv(est_poses[i]).dot(est_poses[j])

            trans_err, rot_err = calc_error(gt_t_ij, est_t_ij)
            errors_div_length.append([trans_err / length, rot_err / length, ])
            errors_by_length[length].append([trans_err, rot_err])
            errors_by_length_div_nframes[length].append([trans_err / (j - i), rot_err / (j - i)])

    return errors_div_length, errors_by_length, errors_by_length_div_nframes


class KittiErrorCalc(object):
    def __init__(self, sequences):
        self.errors = []
        self.gt_poses = {}

        for seq in sequences:
            gt_abs_poses = SequenceData(seq).get_poses()
            self.gt_poses[seq] = gt_abs_poses

    def accumulate_error(self, seq, est):
        assert (seq in self.gt_poses)
        errors = calc_kitti_seq_errors(self.gt_poses[seq][:len(est)], est)[0]
        self.errors += errors
        return np.average(np.array(errors)[:, 0])

    def get_average_error(self):
        # only use translation as average error
        return np.average(np.array(self.errors)[:, 0])

    def clear(self):
        self.errors = []
