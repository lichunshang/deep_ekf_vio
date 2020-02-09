import eval.kitti_eval_pyimpl as e
import os
import glob
import numpy as np

def pfind(*path):
    p = glob.glob(os.path.join(*path) + "*")
    assert len(p) == 1
    return p[0]

base_dir = "/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results"
seqs = ["K04", "K06", "K07", "K08", "K09", "K10"]
error_calc_vanilla = e.KittiErrorCalc(seqs)
error_calc_imu = e.KittiErrorCalc(seqs)
error_calc_vision = e.KittiErrorCalc(seqs)
error_calc_msf_errors = []

for seq in seqs:
    gt_poses = np.load(
        os.path.join(pfind(base_dir, "KITTI_nogloss", seq + "_train"), "saved_model.eval.traj/gt_poses", seq + ".npy"))
    vanilla_poses = np.load(
        os.path.join(pfind(base_dir, "KITTI_nogloss", seq + "_train"), "saved_model.eval.traj/est_poses", seq + ".npy"))
    vision_only_poses = np.load(
        os.path.join(pfind(base_dir, "KITTI_vision_only_aug", seq), "saved_model.eval.traj/est_poses", seq + ".npy"))
    imu_only_poses = np.load(os.path.join(base_dir, "KITTI_imu_only", seq, "est.npy"))

    msf_fusion_poses = np.load(os.path.join(base_dir, "KITTI_msf", seq, "est_shifted.npy"))
    msf_fusion_gt_poses = np.load(os.path.join(base_dir, "KITTI_msf", seq, "gt.npy"))
    error_calc_msf_errors += e.calc_kitti_seq_errors(msf_fusion_gt_poses, msf_fusion_poses)[0]

    error_calc_vanilla.accumulate_error(seq, vanilla_poses)
    error_calc_imu.accumulate_error(seq, imu_only_poses)
    error_calc_vision.accumulate_error(seq, vision_only_poses)
    # error_calc_msf.accumulate_error(seq, msf_fusion_poses, "saved_model.eval.traj/est_poses", s + ".npy")))

print("vanilla trans/rot: %.2f %.3f" % (np.mean(np.array(error_calc_vanilla.errors)[:, 0]) * 100, np.mean(np.array(error_calc_vanilla.errors)[:, 1]) * 180 / np.pi * 100))
print("imu trans/rot: %.2f %.3f" % (np.mean(np.array(error_calc_imu.errors)[:, 0]) * 100, np.mean(np.array(error_calc_imu.errors)[:, 1]) * 180 / np.pi * 100))
print("vision trans/rot: %.2f %.3f" % (np.mean(np.array(error_calc_vision.errors)[:, 0]) * 100, np.mean(np.array(error_calc_vision.errors)[:, 1]) * 180 / np.pi * 100))
print("ORB+MSF trans/rot: %.2f %.3f" % (np.average(np.array(error_calc_msf_errors)[:, 0]) * 100, np.average(np.array(error_calc_msf_errors)[:, 1]) * 180 / np.pi * 100))
