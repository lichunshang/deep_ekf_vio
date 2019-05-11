from log import logger
import numpy as np
import os
import transformations
from data_loader import SequenceData
import yaml
from shutil import copyfile

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


def package_euroc_data(cam_timestamps, imu_timestamps, imu_data, gt_timestamps, gt_data):
    assert len(gt_timestamps) == len(imu_timestamps)
    assert len(gt_timestamps) == len(imu_data)
    assert np.max(np.abs(np.array(imu_timestamps) - np.array(gt_timestamps))) < 1000

    data_frames = []
    first_image_found = False
    ref_time = np.datetime64(int(min([cam_timestamps[0], imu_timestamps[0], ])), "ns")
    for i in range(0, len(cam_timestamps) - 1):
        t_k = cam_timestamps[i]
        t_kp1 = cam_timestamps[i + 1]

        # raises value error is not found
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

        imu_poses = []
        imu_timestamps_k_kp1 = []
        accel_measurements_k_kp1 = []
        gyro_measurements_k_kp1 = []
        for j in range(imu_start_idx, imu_end_idx + 1, 1):
            imu_pose = transformations.quaternion_matrix(gt_data[j, [qw, qx, qy, qz]])
            imu_pose[0:3, 3] = gt_data[j, [px, py, pz]]
            imu_poses.append(imu_pose)
            imu_timestamps_k_kp1.append(np.datetime64(imu_timestamps[j], "ns") - ref_time)
            accel_measurements_k_kp1.append(imu_data[j, [ax, ay, az]])
            gyro_measurements_k_kp1.append(imu_data[j, [wx, wy, wz]])

        v_vk = gt_data[imu_start_idx, [vx, vy, vz]]
        T_i_vk = imu_poses[0]
        frame_k = SequenceData.Frame("%09d.png" % t_k,
                                     np.datetime64(t_k, "ns") - ref_time,
                                     T_i_vk,
                                     v_vk,
                                     imu_poses,
                                     imu_timestamps_k_kp1,
                                     accel_measurements_k_kp1,
                                     gyro_measurements_k_kp1,
                                     timestamp_raw=t_k)

        data_frames.append(frame_k)

    T_i_vkp1 = imu_poses[-1]
    data_frames.append(SequenceData.Frame("%09d.png" % t_kp1,
                                          np.datetime64(t_kp1, "ns") - ref_time,
                                          T_i_vkp1,
                                          gt_data[imu_end_idx, [vx, vy, vz]],
                                          np.zeros([0, 4, 4]), np.zeros([0]), np.zeros([0, 3]), np.zeros([0, 3]),
                                          timestamp_raw=t_kp1))

    return data_frames


# all the input data must be aligned
def find_initial_gravity(imu_timestamps, imu_data, gt_timestamps, gt_data, every_N_frames):
    length = len(imu_timestamps)
    assert length == len(imu_data)
    assert length == len(gt_timestamps)
    assert length == len(gt_data)

    H_blocks = []
    z_blocks = []

    R_b0_w = transformations.quaternion_matrix(gt_data[0, [qw, qx, qy, qz]])[:3, :3].transpose()
    for i in range(0, len(imu_data) - every_N_frames, every_N_frames):
        R_bk_w = transformations.quaternion_matrix(gt_data[i, [qw, qx, qy, qz]])[:3, :3].transpose()
        Dt = (imu_timestamps[i + every_N_frames] - imu_timestamps[i]) / 10 ** 9
        H_blocks.append(np.concatenate([0.5 * R_bk_w * Dt ** 2,
                                        R_bk_w * Dt], 0))

        alpha = np.zeros([3])
        beta = np.zeros([3])
        # compute the IMU pre-integrations
        for j in range(i, i + every_N_frames):
            R_bt_w = transformations.quaternion_matrix(gt_data[j, [qw, qx, qy, qz]])[:3, :3].transpose()
            R_bk_bt = R_bk_w.dot(R_bt_w.transpose())
            dt = (imu_timestamps[j + 1] - imu_timestamps[j]) / 10 ** 9
            alpha = alpha + beta * dt + 0.5 * R_bk_bt.dot(imu_data[j, [ax, ay, az]]) * dt ** 2
            beta = beta + R_bk_bt.dot(imu_data[j, [ax, ay, az]]) * dt

        z_blocks.append(np.concatenate([alpha, beta]))

    H_blocks = np.concatenate(H_blocks)
    b = H_blocks.transpose().dot(np.concatenate(z_blocks))
    A = H_blocks.transpose().dot(H_blocks)
    g_est = np.linalg.inv(A).dot(b)

    print("initial estimated g_w:", g_est)
    print("initial estimated g_b0:", R_b0_w.dot(g_est))
    print("initial estimated g norm: ", np.sqrt(g_est.dot(g_est)))

    return g_est


def preprocess_euroc(seq_dir, output_dir, cam_still_range):
    logger.initialize(working_dir=output_dir, use_tensorboard=False)
    logger.print("================ PREPROCESS EUROC ================")
    logger.print("Preprocessing %s" % seq_dir)
    logger.print("Output to: %s" % output_dir)

    left_cam_csv = open(os.path.join(seq_dir, 'cam0', 'data.csv'), 'r')
    imu_csv = open(os.path.join(seq_dir, 'imu0', "data.csv"), 'r')
    gt_csv = open(os.path.join(seq_dir, "state_groundtruth_estimate0", "data.csv"), "r")
    cam_sensor_yaml_config = yaml.load(open(os.path.join(seq_dir, "cam0", "sensor.yaml")))
    T_cam_imu = np.linalg.inv(np.array(cam_sensor_yaml_config["T_BS"]["data"]).reshape(4, 4))
    cam_timestamps = []
    imu_timestamps = []
    imu_data = []
    gt_timestamps = []
    gt_data = []

    left_cam_csv.readline()  # skip header
    for line in left_cam_csv:
        line = line.split(",")
        timestamp_str = line[0]
        assert str(timestamp_str + ".png" == line[1])
        cam_timestamps.append(int(timestamp_str))

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
        gt_timestamps.append(timestamp)
        gt_data.append(data)

    cam_timestamps = cam_timestamps
    imu_timestamps = imu_timestamps
    imu_data = np.array(imu_data)
    gt_timestamps = gt_timestamps
    gt_data = np.array(gt_data)

    assert np.all(np.diff(cam_timestamps) > 0), "nonmonotonic timestamp"
    assert np.all(np.diff(imu_timestamps) > 0), "nonmonotonic timestamp"
    assert np.all(np.diff(gt_timestamps) > 0), "nonmonotonic timestamp"

    # align imu and ground truth timestamps, and the time difference should not be more than 1 us
    assert (gt_timestamps[0] > imu_timestamps[0] and gt_timestamps[-1] < imu_timestamps[-1])
    imu_gt_aligned_start_idx = (np.abs(np.array(imu_timestamps) - gt_timestamps[0])).argmin()  # gta = gt aligned
    imu_timestamps_gt_aligned = imu_timestamps[imu_gt_aligned_start_idx:imu_gt_aligned_start_idx + len(gt_timestamps)]
    imu_data_gt_aligned = imu_data[imu_gt_aligned_start_idx:imu_gt_aligned_start_idx + len(gt_timestamps)]

    gt_align_time_sync_diff = np.array(gt_timestamps) - np.array(imu_timestamps_gt_aligned)
    assert np.all(np.abs(gt_align_time_sync_diff) < 1000), "timestamp out of sync by > 1 us"

    # first_cam_timestamp
    cam_imu_aligned_start_idx = -1
    for i in range(0, len(cam_timestamps)):
        if cam_timestamps[i] in imu_timestamps:
            cam_imu_aligned_start_idx = i
            break
    assert cam_imu_aligned_start_idx >= 0

    cam_imu_aligned_end_idx = -1
    for i in range(len(cam_timestamps) - 1, -1, -1):
        if cam_timestamps[i] in imu_timestamps:
            cam_imu_aligned_end_idx = i
            break
    assert cam_imu_aligned_end_idx >= 0

    assert cam_imu_aligned_start_idx < cam_still_range[1], "sanity check that the alignment is at the start"

    imu_still_idx_start = imu_timestamps.index(cam_timestamps[max(cam_still_range[0], cam_imu_aligned_start_idx)])
    imu_still_idx_end = imu_timestamps.index(cam_timestamps[cam_still_range[1]])

    gyro_bias_from_still = np.mean(imu_data[imu_still_idx_start:imu_still_idx_end, [wx, wy, wz]], axis=0)
    gravity_from_still = np.mean(imu_data[imu_still_idx_start:imu_still_idx_end, [ax, ay, az]], axis=0)

    print("still estimated g_b0:", gravity_from_still)
    print("still estimated g_b0 norm: ", np.linalg.norm(gravity_from_still))

    # for training /validation
    every_N_frames = 10
    rounded_length = len(imu_timestamps_gt_aligned) // every_N_frames * every_N_frames
    gravity_from_gt = find_initial_gravity(imu_timestamps_gt_aligned[0:rounded_length],
                                           imu_data_gt_aligned[0:rounded_length],
                                           gt_timestamps[0:rounded_length], gt_data[0:rounded_length], 10)

    cam_gt_aligned_start_idx = -1
    for i in range(0, len(cam_timestamps)):
        if cam_timestamps[i] in imu_timestamps_gt_aligned:
            cam_gt_aligned_start_idx = i
            break
    assert cam_gt_aligned_start_idx >= 0

    cam_gt_aligned_end_idx = -1
    for i in range(len(cam_timestamps) - 1, -1, -1):
        if cam_timestamps[i] in imu_timestamps_gt_aligned:
            cam_gt_aligned_end_idx = i
            break
    assert cam_gt_aligned_end_idx >= 0

    gt_cam_aligned_start_idx = imu_timestamps_gt_aligned.index(cam_timestamps[cam_gt_aligned_start_idx])
    gt_cam_aligned_end_idx = imu_timestamps_gt_aligned.index(cam_timestamps[cam_gt_aligned_end_idx])

    data_frames = package_euroc_data(cam_timestamps[cam_gt_aligned_start_idx:cam_gt_aligned_end_idx + 1],
                                     imu_timestamps_gt_aligned[gt_cam_aligned_start_idx:gt_cam_aligned_end_idx],
                                     imu_data_gt_aligned[gt_cam_aligned_start_idx:gt_cam_aligned_end_idx],
                                     gt_timestamps[gt_cam_aligned_start_idx:gt_cam_aligned_end_idx],
                                     gt_data[gt_cam_aligned_start_idx:gt_cam_aligned_end_idx])

    SequenceData.save_as_pd(data_frames, gravity_from_gt, gyro_bias_from_still, T_cam_imu, output_dir)

    # for evaluation
    imu_cam_aligned_start_idx = imu_timestamps.index(cam_timestamps[cam_imu_aligned_start_idx])
    imu_cam_aligned_end_idx = imu_timestamps.index(cam_timestamps[cam_imu_aligned_end_idx])
    imu_timestamps_cam_aligned = imu_timestamps[imu_cam_aligned_start_idx:imu_cam_aligned_end_idx]
    imu_data_cam_aligned = imu_data[imu_cam_aligned_start_idx:imu_cam_aligned_end_idx]
    zeros_gt = np.zeros([len(imu_timestamps_cam_aligned), 16])
    zeros_gt[: qw] = 1

    data_frames = package_euroc_data(cam_timestamps[cam_imu_aligned_start_idx:cam_imu_aligned_end_idx + 1],
                                     imu_timestamps_cam_aligned,
                                     imu_data_cam_aligned,
                                     imu_timestamps_cam_aligned,
                                     zeros_gt)
    eval_output_dir = output_dir + "_eval"
    logger.make_dir_if_not_exist(eval_output_dir)
    SequenceData.save_as_pd(data_frames, gravity_from_still, gyro_bias_from_still, T_cam_imu, eval_output_dir)
    copyfile(os.path.join(seq_dir, "state_groundtruth_estimate0", "data.csv"),
             os.path.join(eval_output_dir, "groundtruth.csv"))


# preprocess_euroc("/home/cs4li/Dev/EUROC/V2_03_difficult", "/home/cs4li/Dev/deep_ekf_vio/results/euroc_proprocess_test",
#                  [0, 70])
preprocess_euroc("/home/cs4li/Dev/EUROC/MH_01_easy", "/home/cs4li/Dev/deep_ekf_vio/results/euroc_proprocess_test",
                 [421, 621])
# preprocess_euroc("/home/cs4li/Dev/EUROC/MH_03_medium", "/home/cs4li/Dev/deep_ekf_vio/results/euroc_proprocess_test",
#                  [0, 40])
# preprocess_euroc("/home/cs4li/Dev/EUROC/MH_04_difficult", "/home/cs4li/Dev/deep_ekf_vio/results/euroc_proprocess_test",
#                  [0, 20])
# preprocess_euroc("/home/cs4li/Dev/EUROC/V2_03_difficult", "/home/cs4li/Dev/deep_ekf_vio/results/euroc_proprocess_test",
#                  0, 0)
# preprocess_euroc("/home/cs4li/Dev/EUROC/V2_03_difficult", "/home/cs4li/Dev/deep_ekf_vio/results/euroc_proprocess_test",
#                  0, 0)
