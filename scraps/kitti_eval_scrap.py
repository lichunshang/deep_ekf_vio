from eval import kitti_eval_pyimpl
import numpy as np


def to_matrix(m):
    m = np.concatenate([m, np.tile(np.array([0, 0, 0, 1]), [len(m), 1])], axis=1)
    m = m.reshape(len(m), 4, 4)
    return m


error_calc = kitti_eval_pyimpl.KittiErrorCalc(["K04", "K06", "K07", "K10"])

est_04 = to_matrix(np.loadtxt("/home/cs4li/Dev/ORB_SLAM2/eval/orbslam_stereo_results/CameraTrajectory_StereoK04.txt"))
est_06 = to_matrix(np.loadtxt("/home/cs4li/Dev/ORB_SLAM2/eval/orbslam_stereo_results/CameraTrajectory_StereoK06.txt"))
est_07 = to_matrix(np.loadtxt("/home/cs4li/Dev/ORB_SLAM2/eval/orbslam_stereo_results/CameraTrajectory_StereoK07.txt"))
est_10 = to_matrix(np.loadtxt("/home/cs4li/Dev/ORB_SLAM2/eval/orbslam_stereo_results/CameraTrajectory_StereoK10.txt"))

gt_04 = to_matrix(np.loadtxt("/home/cs4li/Dev/KITTI/dataset/poses/04.txt"))
gt_06 = to_matrix(np.loadtxt("/home/cs4li/Dev/KITTI/dataset/poses/06.txt"))
gt_07 = to_matrix(np.loadtxt("/home/cs4li/Dev/KITTI/dataset/poses/07.txt"))
gt_10 = to_matrix(np.loadtxt("/home/cs4li/Dev/KITTI/dataset/poses/10.txt"))

print("04")
print(np.average(np.array(kitti_eval_pyimpl.calc_kitti_seq_errors(gt_04, est_04)[0])[:, 0]))
print(np.average(np.array(kitti_eval_pyimpl.calc_kitti_seq_errors(gt_04, est_04)[0])[:, 1]))

print("06")
print(np.average(np.array(kitti_eval_pyimpl.calc_kitti_seq_errors(gt_06, est_06)[0])[:, 0]))
print(np.average(np.array(kitti_eval_pyimpl.calc_kitti_seq_errors(gt_06, est_06)[0])[:, 1]))

print("07")
print(np.average(np.array(kitti_eval_pyimpl.calc_kitti_seq_errors(gt_07, est_07)[0])[:, 0]))
print(np.average(np.array(kitti_eval_pyimpl.calc_kitti_seq_errors(gt_07, est_07)[0])[:, 1]))


print("10")
print(np.average(np.array(kitti_eval_pyimpl.calc_kitti_seq_errors(gt_10, est_10)[0])[:, 0]))
print(np.average(np.array(kitti_eval_pyimpl.calc_kitti_seq_errors(gt_10, est_10)[0])[:, 1]))