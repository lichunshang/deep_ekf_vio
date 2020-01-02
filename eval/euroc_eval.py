import numpy as np
import os
import prettytable
from evo.tools import file_interface
from evo.core import trajectory, sync, metrics
from params import par
from data_loader import SequenceData
from log import logger


# from evo.tools import log
# log.configure_logging(verbose=True, debug=True, silent=False)
#
# import numpy as np
#
# from evo.tools import plot
# import matplotlib.pyplot as plt
#
# # temporarily override some package settings
# from evo.tools.settings import SETTINGS
# SETTINGS.plot_usetex = False

def calc_euroc_seq_errors(est_traj, gt_traj):
    gt_traj_synced, est_traj_synced = sync.associate_trajectories(gt_traj, est_traj, max_diff=0.01)
    est_traj_aligned = trajectory.align_trajectory(est_traj_synced, gt_traj_synced,
                                                   correct_scale=False, correct_only_scale=False)
    pose_relation = metrics.PoseRelation.translation_part
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data((gt_traj_synced, est_traj_aligned,))
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)

    # ape_metric = metrics.RPE(pose_relation)
    # ape_metric.process_data((gt_traj_synced, est_traj_aligned,))
    # ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)

    # fig = plt.figure()
    # traj_by_label = {
    #     "estimate (not aligned)": est_traj,
    #     "estimate (aligned)": est_traj_aligned,
    #     "reference": gt_traj
    # }
    # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    # plt.show()

    return ape_metric, ape_stat


class EurocErrorCalc(object):
    def __init__(self, sequences):
        self.errors = []
        self.gt_traj = {}
        self.raw_timestamps = {}

        for seq in sequences:
            gt_traj = file_interface.read_euroc_csv_trajectory(os.path.join(par.data_dir, seq, "groundtruth.csv"))
            self.gt_traj[seq] = gt_traj
            self.raw_timestamps[seq] = np.array(SequenceData(seq).get_timestamps_raw()) / 10 ** 9

    def accumulate_error(self, seq, est):
        assert (seq in self.gt_traj)
        est_traj = trajectory.PoseTrajectory3D(poses_se3=est, timestamps=self.raw_timestamps[seq][:len(est)])
        ape_metric, ape_stat = calc_euroc_seq_errors(est_traj, self.gt_traj[seq])
        self.errors.append(ape_stat)

        return ape_stat

    def get_average_error(self):
        return np.average(np.array(self.errors))

    def clear(self):
        self.errors = []


def euroc_eval(working_dir, seqs):
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ Evaluate EUROC ================")
    logger.print("Working on directory:", working_dir)

    pose_est_dir = os.path.join(working_dir, "est_poses")
    pose_est_files = sorted(os.listdir(pose_est_dir))

    logger.print("Evaluating seqs: %s" % ", ".join(seqs))
    available_seqs = [seq.replace(".npy", "") for seq in pose_est_files]

    table = prettytable.PrettyTable()
    table.field_names = ["Seq.", "RMSE APE Err"]
    table.align["Seq."] = "l"
    table.align["RMSE APE Err"] = "r"

    assert set(seqs).issubset(available_seqs), "est file is not available, have seqs: %s" % \
                                               ", ".join(list(available_seqs))
    error_calc = EurocErrorCalc(seqs)
    stats = []

    for seq in seqs:
        if seq not in available_seqs:
            raise RuntimeError("File for seq %s not available" % seq)

        poses_est = np.load(os.path.join(pose_est_dir, "%s.npy" % seq))
        stat = error_calc.accumulate_error(seq, poses_est)
        table.add_row([seq, "%.6f" % stat])
        stats.append(stat)

    table.add_row(["Ave.", "%.6f" % error_calc.get_average_error()])
    logger.print(table)

    logger.print("Copy to Google Sheets:", ",".join([str(stat) for stat in stats]))
