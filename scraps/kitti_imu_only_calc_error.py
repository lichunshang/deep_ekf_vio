import sys
import pandas as pd
import os
import numpy as np
import time
import transformations
from data_loader import SequenceData
from eval.kitti_eval_pyimpl import calc_kitti_seq_errors

results_dir = "/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_imu_only"
sequences = ["K01", "K04", "K06", "K07", "K08", "K09", "K10"]

for seq in sequences:
    seq_data = SequenceData(seq)
    est_traj = np.load(os.path.join(results_dir, seq, "est.npy"))
    gt_traj = np.array([T for T in  seq_data.df.loc[:, "T_i_vk"]])

    d = np.sum([np.linalg.norm((np.linalg.inv(gt_traj[i]).dot(gt_traj[i + 1]))[:3,3]) for i in range(0, len(gt_traj) - 1)])
    # timestamps
    err = np.array(calc_kitti_seq_errors(gt_traj, est_traj)[0])
    print("Seq Error seq %s (t,r): %.15f, %.15f, %.15f dist: %.15f" % (seq, np.average(err[:, 0]), np.average(err[:, 1]) * 180 / np.pi * 100,
          np.average(err[:, 1]) * 180 / np.pi, d / 1e3))
    # print("%.15f, %.15f" % (np.average(err[:, 0]), np.average(err[:, 1]) * 180 / np.pi))


