from log import logger
import numpy as np
import os
import transformations
from se3 import log_SO3, exp_SO3, interpolate_SE3, interpolate_SO3
from data_loader import SequenceData
from utils import Plotter
import matplotlib.pyplot as plt
import time
import csv

px = 0  # p_RS_R_x [m]
py = 1  # p_RS_R_y [m]
pz = 2  # p_RS_R_z [m]
qw = 3  # q_RS_w []
qx = 4  # q_RS_x []
qy = 5  # q_RS_y []
qz = 6  # q_RS_z []
vx = 7  # v_RS_R_x [m s^-1]
vy = 8  # v_RS_R_y [m s^-1]
vz = 9  # v_RS_R_z [m s^-1]
bwx = 10  # b_w_RS_S_x [rad s^-1]
bwy = 11  # b_w_RS_S_y [rad s^-1]
bwz = 12  # b_w_RS_S_z [rad s^-1]
bax = 13  # b_a_RS_S_x [m s^-2]
bay = 14  # b_a_RS_S_y [m s^-2]
baz = 15  # b_a_RS_S_z [m s^-2]

wx = 0  # w_RS_S_x [rad s^-1]
wy = 1  # w_RS_S_y [rad s^-1]
wz = 2  # w_RS_S_z [rad s^-1]
ax = 3  # a_RS_S_x [m s^-2]
ay = 4  # a_RS_S_y [m s^-2]
az = 5  # a_RS_S_z [m s^-2]


def package_euroc_data(camera_timestamps, imu_timestamps, imu_data, gt_timestamps=None, gt_data=None):
    data_frames = []
    first_image_found = False
    ref_time = np.datetime64(min([camera_timestamps[0], imu_timestamps[0], ]), "ns")
    for i in range(0, len(camera_timestamps) - 1):
        t_k = camera_timestamps[i]
        t_kp1 = camera_timestamps[i + 1]

        # raises value error is not found\
        try:
            imu_start_idx = imu_timestamps.index(t_k)
        except ValueError as e:
            if not first_image_found:
                continue
            else:
                raise e

        try:
            imu_end_idx = imu_timestamps.index(t_kp1)
        except ValueError:
            break

        if not first_image_found:
            logger.print("First image idx %d ts %d" % (i, t_k))
            first_image_found = True

        if not imu_end_idx - imu_start_idx == 10:
            logger.print("WARN imu_end_idx - imu_start_idx != 10, image: [%d -> %d] imu: [%d -> %d, %d -> %d]" %
                         (i, i + 1, imu_start_idx, imu_end_idx, t_k, t_kp1))


# all the input data must be aligned
def find_initial_gravity(imu_timestamps, imu_data, gt_timestamps, gt_data, desired_gnorm):
    every_N_frames = 10

    length = len(imu_timestamps)
    assert length == len(imu_data)
    assert length == len(gt_timestamps)
    assert length == len(gt_data)

    H_blocks = []
    z_blocks = []

    R_b0_w = transformations.quaternion_matrix(gt_data[0, qw:qz + 1])[:3, :3]
    for i in range(0, len(imu_data) - 1, every_N_frames):
        R_bk_b0 = R_b0_w.dot(transformations.quaternion_matrix(gt_data[i, qw:qz + 1])[:3, :3].transpose())
        Dt = (imu_timestamps[i + every_N_frames] - imu_timestamps[i]) / 10 ** 9
        H_blocks.append(np.concatenate([-0.5 * R_bk_b0 * Dt ** 2, - R_bk_b0 * Dt], 0))

        alpha = np.zeros([3])
        beta = np.zeros([3])
        # compute the IMU pre-integrations
        for j in range(i, i + every_N_frames):
            R_bt_b0 = R_b0_w.dot(transformations.quaternion_matrix(gt_data[j, qw:qz + 1])[:3, :3].transpose())
            R_bk_bt = R_bk_b0.dot(R_bt_b0.transpose())
            dt = (imu_timestamps[j + 1] - imu_timestamps[j]) / 10 ** 9
            alpha = alpha + beta * dt + 0.5 * R_bk_bt.dot(imu_data[j, ax:az + 1]) * dt ** 2
            beta = beta + R_bk_bt.dot(imu_data[j, ax:az + 1]) * dt

        z_blocks.append(np.concatenate([alpha, beta]))

    H_blocks = np.concatenate(H_blocks)
    b = H_blocks.transpose().dot(np.concatenate(z_blocks)) * 1000.0
    A = H_blocks.transpose().dot(H_blocks) * 1000
    g_est = np.linalg.inv(A).dot(b)

    print("estimated g:", g_est)
    print("estimated g norm: ", np.sqrt(g_est.dot(g_est)))


def preprocess_euroc(seq_dir, output_dir, cam_still_range):
    logger.initialize(working_dir=output_dir, use_tensorboard=False)
    logger.print("================ PREPROCESS KITTI RAW ================")
    logger.print("Preprocessing %s" % seq_dir)
    logger.print("Output to: %s" % output_dir)

    left_cam_csv = open(os.path.join(seq_dir, 'cam0', 'data.csv'), 'r')
    imu_csv = open(os.path.join(seq_dir, 'imu0', "data.csv"), 'r')
    gt_csv = open(os.path.join(seq_dir, "state_groundtruth_estimate0", "data.csv"), "r")
    camera_timestamps = []
    imu_timestamps = []
    imu_data = []
    gt_timestamp = []
    gt_data = []

    left_cam_csv.readline()  # skip header
    for line in left_cam_csv:
        line = line.split(",")
        timestamp_str = line[0]
        assert str(timestamp_str + ".png" == line[1])
        camera_timestamps.append(int(timestamp_str))

    imu_csv.readline()  # skip header
    for line in imu_csv:
        line = line.split(",")
        timestamp = int(line[0])
        data = [float(line[i + 1]) for i in range(0, 6)]
        imu_timestamps.append(timestamp)
        imu_data.append(data)

    gt_csv.readline()  # skip header
    for line in gt_csv:
        line = line.split(",")
        timestamp = int(line[0])
        data = [float(line[i + 1]) for i in range(0, 16)]
        gt_timestamp.append(timestamp)
        gt_data.append(data)

    camera_timestamps = np.array(camera_timestamps)
    imu_timestamps = np.array(imu_timestamps)
    imu_data = np.array(imu_data)
    gt_timestamp = np.array(gt_timestamp)
    gt_data = np.array(gt_data)

    assert np.all(np.diff(camera_timestamps) > 0), "nonmonotonic timestamp"
    assert np.all(np.diff(imu_timestamps) > 0), "nonmonotonic timestamp"
    assert np.all(np.diff(gt_timestamp) > 0), "nonmonotonic timestamp"

    # align imu and ground truth timestamps, imu and camera timestamps are always aligned
    imu_gt_start = (np.abs(imu_timestamps - gt_timestamp[0])).argmin()
    min_length = min(len(gt_timestamp), len(imu_timestamps[imu_gt_start:]))
    assert np.all(np.abs(gt_timestamp[0:min_length] - imu_timestamps[imu_gt_start:imu_gt_start + min_length])
                  < 1000), "timestamp out of sync by > 1 us"

    find_initial_gravity(imu_timestamps[imu_gt_start:imu_gt_start + 5001],
                         imu_data[imu_gt_start:imu_gt_start + 5001],
                         gt_timestamp[0:5001],
                         gt_data[0:5001], 0)

    # frame_k = SequenceData.Frame(camera_timestamps[i],
    #                              (np.datetime64(t_k, "ns") - ref_time) / np.timedelta64(1, 's'),
    #                              T_i_vk, v_vk,
    #                              imu_poses, imu_timestamps_k_kp1, accel_measurements_k_kp1, gyro_measurements_k_kp1)
    # data_frames.append(frame_k)

    print("Done")


preprocess_euroc("/home/cs4li/Dev/EUROC/MH_03_medium", "/home/cs4li/Dev/deep_ekf_vio/results/euroc_proprocess_test",
                 [0, 40])
# preprocess_euroc("/home/cs4li/Dev/EUROC/V2_03_difficult", "/home/cs4li/Dev/deep_ekf_vio/results/euroc_proprocess_test",
#                  0, 0)
# preprocess_euroc("/home/cs4li/Dev/EUROC/V2_03_difficult", "/home/cs4li/Dev/deep_ekf_vio/results/euroc_proprocess_test",
#                  0, 0)
