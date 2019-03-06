import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from params import par
from log import Logger
from data_loader import SequenceData

working_dir = os.path.abspath(sys.argv[1])
pose_est_dir = os.path.join(working_dir, "est_poses")
pose_gt_dir = os.path.join(working_dir, "gt_poses")
pose_est_files = sorted(os.listdir(pose_est_dir))

for i, pose_est_file in enumerate(pose_est_files):
    sequence = os.path.splitext(pose_est_file)[0]

    traj_est = np.load(os.path.join(pose_est_dir, "%s.npy" % sequence))
    length = traj_est.shape[0]

    traj_gt = SequenceData(sequence).get_poses()
    traj_gt = traj_gt[0:length, :, :]
    np.save(Logger.ensure_file_dir_exists(os.path.join(pose_gt_dir, "%s.npy" % sequence)), traj_gt)

print("All Done")
