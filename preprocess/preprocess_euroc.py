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


def package_euroc_data(seq_dir, cam_timestamps, imu_timestamps, imu_data, gt_timestamps, gt_data):
    assert len(gt_timestamps) == len(imu_timestamps)
    assert len(gt_timestamps) == len(imu_data)
    assert np.max(np.abs(np.array(imu_timestamps) - np.array(gt_timestamps))) < 1000
    assert cam_timestamps[0] == imu_timestamps[0]
    assert cam_timestamps[-1] == imu_timestamps[-1]

    data_frames = []
    ref_time = np.datetime64(int(min([cam_timestamps[0], imu_timestamps[0], ])), "ns")

    cam_period = 100 * 10 ** 6  # nanoseconds
    imu_skip = 2
    i_start = 0
    # for i in range(0, len(cam_timestamps) - 1):
    while i_start < len(cam_timestamps) - 1:
        t_k = cam_timestamps[i_start]
        i_end = (np.abs(np.array(cam_timestamps) - (cam_timestamps[i_start] + cam_period))).argmin()
        t_kp1 = cam_timestamps[i_end]

        imu_start_idx = imu_timestamps.index(t_k)
        imu_end_idx = imu_timestamps.index(t_kp1)

        if not t_kp1 - t_k == cam_period:
            # ignore the last frame if it is does not at the desired rate
            if i_end == len(cam_timestamps) - 1:
                break

            logger.print("WARN imu_end_idx - imu_start_idx != %.5s, "
                         "image: [%d -> %d] imu: [%d -> %d] time: [%d -> %d] diff %.5f"
                         % (cam_period / 10 ** 9, i_start, i_end, imu_start_idx, imu_end_idx, t_k, t_kp1,
                            (t_kp1 - t_k) / 10 ** 9))

        imu_poses = []
        imu_timestamps_k_kp1 = []
        accel_measurements_k_kp1 = []
        gyro_measurements_k_kp1 = []
        for j in range(imu_start_idx, imu_end_idx + 1, imu_skip):
            imu_pose = transformations.quaternion_matrix(gt_data[j, [qw, qx, qy, qz]])
            imu_pose[0:3, 3] = gt_data[j, [px, py, pz]]
            imu_poses.append(imu_pose)
            imu_timestamps_k_kp1.append((np.datetime64(imu_timestamps[j], "ns") - ref_time) / np.timedelta64(1, "s"))
            accel_measurements_k_kp1.append(imu_data[j, [ax, ay, az]])
            gyro_measurements_k_kp1.append(imu_data[j, [wx, wy, wz]])

        T_i_vk = imu_poses[0]
        frame_k = SequenceData.Frame(os.path.join(seq_dir, "cam0", "data", "%09d.png" % t_k),
                                     (np.datetime64(t_k, "ns") - ref_time) / np.timedelta64(1, "s"),
                                     T_i_vk,
                                     T_i_vk[0:3, 0:3].transpose().dot(gt_data[imu_start_idx, [vx, vy, vz]]),  # v_vk
                                     imu_poses,
                                     imu_timestamps_k_kp1,
                                     accel_measurements_k_kp1,
                                     gyro_measurements_k_kp1,
                                     timestamp_raw=t_k)

        data_frames.append(frame_k)
        i_start = i_end

    T_i_vkp1 = imu_poses[-1]
    data_frames.append(SequenceData.Frame(os.path.join(seq_dir, "cam0", "data", "%09d.png" % t_kp1),
                                          (np.datetime64(t_kp1, "ns") - ref_time) / np.timedelta64(1, "s"),
                                          T_i_vkp1,
                                          T_i_vkp1[0:3, 0:3].transpose().dot(gt_data[imu_end_idx, [vx, vy, vz]]),
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
    logger.print("Camera still range [%d -> %d]" % (cam_still_range[0], cam_still_range[1]))

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
    logger.print("Processing data for training...")
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

    logger.print("Camera index [%d -> %d]" % (cam_gt_aligned_start_idx, cam_gt_aligned_end_idx))
    data_frames = package_euroc_data(seq_dir, cam_timestamps[cam_gt_aligned_start_idx:cam_gt_aligned_end_idx + 1],
                                     imu_timestamps_gt_aligned[gt_cam_aligned_start_idx:gt_cam_aligned_end_idx + 1],
                                     imu_data_gt_aligned[gt_cam_aligned_start_idx:gt_cam_aligned_end_idx + 1],
                                     gt_timestamps[gt_cam_aligned_start_idx:gt_cam_aligned_end_idx + 1],
                                     gt_data[gt_cam_aligned_start_idx:gt_cam_aligned_end_idx + 1])

    SequenceData.save_as_pd(data_frames, gravity_from_gt, gyro_bias_from_still, T_cam_imu, output_dir)
    copyfile(os.path.join(seq_dir, "state_groundtruth_estimate0", "data.csv"),
             os.path.join(output_dir, "groundtruth.csv"))

    # for evaluation
    logger.print("Processing data for evaluation...")
    imu_cam_aligned_start_idx = imu_timestamps.index(cam_timestamps[cam_still_range[1]])
    imu_cam_aligned_end_idx = imu_timestamps.index(cam_timestamps[cam_imu_aligned_end_idx])
    imu_timestamps_cam_aligned = imu_timestamps[imu_cam_aligned_start_idx:imu_cam_aligned_end_idx + 1]
    imu_data_cam_aligned = imu_data[imu_cam_aligned_start_idx:imu_cam_aligned_end_idx + 1]
    zeros_gt = np.zeros([len(imu_timestamps_cam_aligned), 16])
    zeros_gt[: qw] = 1

    logger.print("Camera index [%d -> %d]" % (cam_still_range[1], cam_imu_aligned_end_idx))
    eval_output_dir = output_dir + "_eval"
    data_frames = package_euroc_data(seq_dir, cam_timestamps[cam_still_range[1]:cam_imu_aligned_end_idx + 1],
                                     imu_timestamps_cam_aligned,
                                     imu_data_cam_aligned,
                                     imu_timestamps_cam_aligned,
                                     zeros_gt)
    logger.make_dir_if_not_exist(eval_output_dir)
    SequenceData.save_as_pd(data_frames, gravity_from_still, gyro_bias_from_still, T_cam_imu, eval_output_dir)
    copyfile(os.path.join(seq_dir, "state_groundtruth_estimate0", "data.csv"),
             os.path.join(eval_output_dir, "groundtruth.csv"))