import sys
import pandas as pd
import os
import numpy as np
import time
import transformations
from data_loader import SequenceData
from eval.kitti_eval_pyimpl import calc_kitti_seq_errors

results_dir = "/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_msf"
sequences = ["K01", "K04", "K06", "K07", "K08", "K09", "K10"]

REF_PATH = {
    "K01": "2011_10_03/2011_10_03_drive_0042",
    "K04": "2011_09_30/2011_09_30_drive_0016",
    "K06": "2011_09_30/2011_09_30_drive_0020",
    "K07": "2011_09_30/2011_09_30_drive_0027",
    "K08": "2011_09_30/2011_09_30_drive_0028",
    "K09": "2011_09_30/2011_09_30_drive_0033",
    "K10": "2011_09_30/2011_09_30_drive_0034",
}

for seq in sequences:
    seq_data = SequenceData(seq)

    imu_ref_time = np.datetime64(open(os.path.join("/home/cs4li/Dev/KITTI/dataset",
                                                   REF_PATH[seq] + "_extract", "oxts",
                                                   "timestamps.txt"), "r").readline().strip())
    imu_ref_time_flt = (imu_ref_time - np.datetime64(0, "s")) / np.timedelta64(1, "s")
    df_timestamps = np.array(list(seq_data.df.loc[:, "timestamp"]))

    est_traj = np.load(os.path.join(results_dir, seq, "est.npy"))
    gt_traj = np.load(os.path.join(results_dir, seq, "gt.npy"))
    timestamps = np.load(os.path.join(results_dir, seq, "timestamps.npy"))
    timestamps_adj = np.array(timestamps) - imu_ref_time_flt

    nearest_idx = np.abs((df_timestamps - timestamps_adj[0])).argmin()
    assert(np.abs(df_timestamps[nearest_idx] - timestamps_adj[0]) < 1e-5)
    T_shift = seq_data.df.loc[nearest_idx, "T_i_vk"]
    est_traj_shifted = np.array([T_shift.dot(T) for T in est_traj])
    np.save(os.path.join(results_dir, seq, "est_shifted.npy"), est_traj_shifted)

    # timestamps
    err = np.array(calc_kitti_seq_errors(gt_traj, est_traj)[0])
    print("Seq Error seq %s (t,r): %.15f, %.15f, idx %d" % (seq, np.average(err[:, 0]), np.average(err[:, 1]) * 180 / np.pi, nearest_idx))
    # print("%.15f, %.15f" % (np.average(err[:, 0]), np.average(err[:, 1]) * 180 / np.pi))


