import numpy as np
from eval.kitti_eval_pyimpl import *
import matplotlib.pyplot as plt

# def calc_kitti_f2f_errors(gt_poses, est_poses):
#     assert (len(gt_poses) == len(est_poses))
#     gt_poses = gt_poses.astype(np.float64)
#     est_poses = est_poses.astype(np.float64)
#     step_size = 10
#     distances = calc_trajectory_dist(gt_poses)
#     lengths = [100, 200, 300, 400, 500, 600, 700, 800]
#     errors_by_length = {k: [] for k in lengths}
#
#     est_rel = np.array([np.linalg.inv(est_poses[i]).dot(est_poses[i + 1]) for i in range(0, len(est_poses) - 1)])
#     gt_rel = np.array([np.linalg.inv(gt_poses[i]).dot(gt_poses[i + 1]) for i in range(0, len(gt_poses) - 1)])
#     err_rel = np.matmul(np.linalg.inv(est_rel), gt_rel)
#
#     for i in range(0, len(gt_poses) - 1, step_size):
#         for length in lengths:
#             j = last_frame_from_segment_length(distances, i, length)
#             if j < 0:
#                 continue
#
#             trans_err = np.linalg.norm(err_rel[:, 0:3, 3], axis=-1)
#
#             rot_diag = np.diagonal(err_rel[:, 0:3, 0:3], axis1=-2, axis2=-1)
#             rot_err = np.arccos(np.clip((np.sum(rot_diag, axis=-1) - 1.0) * 0.5, a_min=- 1.0, a_max=1.0))
#             e = np.stack([trans_err, rot_err], axis=-1)
#             errors_by_length[length] += list(e)
#
#     return errors_by_length


gt = np.load(
        "/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_nogloss/K10_train_20200131-15-31-04/saved_model.eval.traj/gt_poses/K10.npy")
est = np.load(
        "/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_nogloss/K10_train_20200131-15-31-04/saved_model.eval.traj/est_poses/K10.npy")

_, error_by_length, _ = calc_kitti_seq_errors(gt, est)

# fig1, ax1 = plt.subplots()
# plt.title('Basic Plot')

# data100 = [np.array(error_by_length[100])[:, 0], [0, 0, 0, 2.18, 3]]
# data200 = [np.array(error_by_length[200])[:, 0], [2.5, 1.01, 5.43, 9.5]]
# data300 = [np.array(error_by_length[300])[:, 0], [6, 3.26, 17.9, 33]]
# data400 = [np.array(error_by_length[400])[:, 0], [0.5, 10.3, 5.43, 39.6, 76]]
# data500 = [np.array(error_by_length[500])[:, 0], [1.5, 16.8, 8.6, 70.1, 131]]

data_a = [np.array(error_by_length[100])[:, 0], np.array(error_by_length[200])[:, 0],
          np.array(error_by_length[300])[:, 0], np.array(error_by_length[400])[:, 0],
          np.array(error_by_length[500])[:, 0]]
data_b = [[0, 0, 0, 2.18, 3], [2.5, 1.01, 5.43, 9.5],
          [6, 3.26, 17.9, 33], [0.5, 10.3, 5.43, 39.6, 76],
          [1.5, 16.8, 8.6, 70.1, 131]]

ticks = ['100', '200', '300', '400', '500']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

tick_space = 4
bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*tick_space-0.4, sym='', widths=0.6)
bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*tick_space+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='proposed')
plt.plot([], c='#2C7BB6', label='VINet')
plt.xlabel("distance [m]")
plt.ylabel("error [m]")
plt.legend()
plt.grid(axis='y', which="both")

plt.yticks(np.arange(0, 45, 10.0))
plt.xticks(range(0, len(ticks) * tick_space, tick_space), ticks)
plt.xlim(-tick_space, len(ticks)*tick_space)
plt.ylim(0, 45)
sz = plt.gcf().get_size_inches()
sz[1] /= 2
plt.gcf().set_size_inches(sz)

plt.savefig("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_figures/vinet.svg",  format='svg', bbox_inches='tight', pad_inches=0)

for i in range(100, 801, 100):
    e = np.array(error_by_length[i])
    t = e[:, 0]
    # print("%d trans: %f, rot %f" % (i, np.median(e[:,0]), np.median(e[:, 1]) * 180 / np.pi))
    print("%d trans: %f 1st: %f 3rd: %f max: %f" % (
    i, np.median(t), np.quantile(t, 0.25), np.quantile(t, 0.75), np.max(t)))
# plt.show()
print("Done")
