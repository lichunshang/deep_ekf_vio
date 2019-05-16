import numpy as np
import os
from evo.tools import file_interface
from evo.core import trajectory, sync, metrics
from params import par
from data_loader import SequenceData


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

class EurocErrorCalc(object):
    def __init__(self, sequences):
        self.errors = []
        self.gt_traj = {}
        self.raw_timestamps = {}

        for seq in sequences:
            if seq.endswith("eval"):
                seq_name_eval = seq
            else:
                seq_name_eval = seq + "_eval"
            gt_traj = file_interface.read_euroc_csv_trajectory(
                    os.path.join(par.data_dir, seq_name_eval, "groundtruth.csv"))
            self.gt_traj[seq] = gt_traj
            self.raw_timestamps[seq] = np.array(SequenceData(seq).get_timestamps_raw()) / 10 ** 9

    def accumulate_error(self, seq, est):
        assert (seq in self.gt_traj)
        est_traj = trajectory.PoseTrajectory3D(poses_se3=est, timestamps=self.raw_timestamps[seq][:len(est)])
        gt_traj, est_traj = sync.associate_trajectories(self.gt_traj[seq], est_traj, max_diff=0.01)
        est_traj_aligned = trajectory.align_trajectory(est_traj, gt_traj,
                                                       correct_scale=False, correct_only_scale=False)
        pose_relation = metrics.PoseRelation.translation_part
        ape_metric = metrics.APE(pose_relation)
        ape_metric.process_data((gt_traj, est_traj_aligned,))
        ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)

        self.errors.append(ape_stat)

        # fig = plt.figure()
        # traj_by_label = {
        #     "estimate (not aligned)": est_traj,
        #     "estimate (aligned)": est_traj_aligned,
        #     "reference": gt_traj
        # }
        # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
        # plt.show()

        return ape_stat

    def get_average_error(self):
        return np.average(np.array(self.errors))

    def clear(self):
        self.errors = []
