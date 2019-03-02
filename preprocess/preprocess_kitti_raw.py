from log import logger
import numpy as np
from numpy.linalg import inv
import os
import transformations
from se3_math import log_SO3, exp_SO3, interpolate_SE3, interpolate_SO3
import matplotlib.pyplot as plt
import time
import pandas

if "DISPLAY" not in os.environ:
    plt.switch_backend("Agg")

lat, lon, alt, roll, pitch, yaw, vn, ve, vf, vl, vu, ax, ay, az, af, al, au, wx, wy, wz, wf, wl, wu, \
posacc, velacc, navstat, numsats, posmode, velmode, orimode = list(range(0, 30))


class Frame(object):
    def __init__(self, image_path, timestamp, T_i_vk, T_cam_imu, v_vk_i_vk,
                 imu_poses, imu_timestamps, accel_measurements, gyro_measurements):
        self.image_path = image_path
        self.timestamp = timestamp
        self.T_i_vk = T_i_vk  # inertial to vehicle frame pose
        self.T_cam_imu = T_cam_imu  # calibration from imu to camera
        self.v_vk_i_vk = v_vk_i_vk  # velocity expressed in vehicle frame
        self.imu_timestamps = imu_timestamps
        self.imu_poses = imu_poses
        self.accel_measurements = accel_measurements
        self.gyro_measurements = gyro_measurements

        assert (len(imu_timestamps) == len(accel_measurements))
        assert (len(imu_timestamps) == len(gyro_measurements))
        assert (len(imu_timestamps) == len(imu_poses))


def read_timestamps(ts_file):
    f = open(ts_file, "r")
    timestamps = []
    for line in f:
        line = line.strip()
        if line:
            timestamps.append(np.datetime64(line))
    # put the time in local frame of reference, and convert to seconds
    return np.array(timestamps)


def interpolate(imu_data_i, imu_data_j, pose_i, pose_j, alpha):
    # rotation
    C_i_vi = transformations.euler_matrix(imu_data_i[yaw], imu_data_i[pitch], imu_data_i[roll], 'rzyx')[0:3, 0:3]
    C_i_vj = transformations.euler_matrix(imu_data_j[yaw], imu_data_j[pitch], imu_data_j[roll], 'rzyx')[0:3, 0:3]
    C_i_vk = interpolate_SO3(C_i_vi, C_i_vj, alpha)
    C_i_hi = transformations.euler_matrix(imu_data_i[yaw], 0, 0, 'rzyx')[0:3, 0:3]
    C_i_hj = transformations.euler_matrix(imu_data_j[yaw], 0, 0, 'rzyx')[0:3, 0:3]
    C_i_hk = interpolate_SO3(C_i_hi, C_i_hj, alpha)

    # pose
    T_i_vk = interpolate_SE3(pose_i, pose_j, alpha)

    C_vk_vj = C_i_vk.transpose().dot(C_i_vj)
    C_vk_vi = C_i_vk.transpose().dot(C_i_vi)

    # acceleration
    a_vi = np.array([imu_data_i[ax], imu_data_i[ay], imu_data_i[az]])
    a_vj = np.array([imu_data_j[ax], imu_data_j[ay], imu_data_j[az]])
    a_vk = alpha * (C_vk_vj.dot(a_vj) - C_vk_vi.dot(a_vi)) + C_vk_vi.dot(a_vi)

    # angular velocity
    w_vi = np.array([imu_data_i[wx], imu_data_i[wy], imu_data_i[wz]])
    w_vj = np.array([imu_data_j[wx], imu_data_j[wy], imu_data_j[wz]])
    w_vk = alpha * (C_vk_vj.dot(w_vj) - C_vk_vi.dot(w_vi)) + C_vk_vi.dot(w_vi)

    # velocity
    v_hi = np.array([imu_data_i[vf], imu_data_i[vl], imu_data_i[vu]])
    v_hj = np.array([imu_data_j[vf], imu_data_j[vl], imu_data_j[vu]])
    v_hk = alpha * (v_hj - v_hi) + v_hi
    v_vk = C_i_vk.transpose().dot(C_i_hk).dot(v_hk)
    # v_vk[1] = -v_vk[1]

    return T_i_vk, v_vk, w_vk, a_vk


def preprocess_kitti_raw(raw_seq_dir, output_dir, cam_subset_range):
    logger.initialize(working_dir=output_dir, use_tensorboard=False)
    logger.print("================ PREPROCESS KITTI RAW ================")
    logger.print("Preprocessing %s" % raw_seq_dir)
    logger.print("Output to: %s" % output_dir)
    logger.print("Camera images: %d => %d" % (cam_subset_range[0], cam_subset_range[1]))

    oxts_dir = os.path.join(raw_seq_dir, "oxts")
    image_dir = os.path.join(raw_seq_dir, "image_02")
    gps_poses = np.loadtxt(os.path.join(oxts_dir, "poses.txt"))
    gps_poses = np.array(
            [np.vstack([np.reshape(p, [3, 4]), [0, 0, 0, 1]]) for p in gps_poses])  # convert to 4x4 matrices
    T_velo_imu = np.loadtxt(os.path.join(raw_seq_dir, "../T_velo_imu.txt"))
    T_cam_velo = np.loadtxt(os.path.join(raw_seq_dir, '../T_cam_velo.txt'))
    T_cam_imu = T_cam_velo.dot(T_velo_imu)

    # load IMU data
    imu_data = []
    imu_data_files = sorted(os.listdir(os.path.join(oxts_dir, "data")))
    assert (len(imu_data_files) == len(gps_poses))
    start_time = time.time()
    for i in range(0, len(imu_data_files)):
        print("Loading IMU data files %d/%d (%.2f%%)" %
              (i + 1, len(imu_data_files), 100 * (i + 1) / len(imu_data_files)), end='\r')
        imu_data.append(np.loadtxt(os.path.join(oxts_dir, "data", imu_data_files[i])))
    imu_data = np.array(imu_data)
    logger.print("Loading IMU data took %.2fs" % (time.time() - start_time))

    # imu timestamps
    imu_timestamps = read_timestamps(os.path.join(oxts_dir, "timestamps.txt"))
    assert (len(imu_timestamps) == len(gps_poses))

    # load image data
    cam_timestamps = read_timestamps(os.path.join(image_dir, "timestamps.txt"))
    image_paths = sorted(os.listdir(os.path.join(image_dir, "data")))  # image data exists are part of paths
    assert (len(cam_timestamps) == len(image_paths))
    assert (cam_subset_range[0] >= 0 and cam_subset_range[1] < len(image_paths))

    # the first camera timestamps must be between IMU timestamps
    assert (cam_timestamps[cam_subset_range[0]] >= imu_timestamps[0])
    assert (cam_timestamps[cam_subset_range[1]] <= imu_timestamps[-1])
    # convert to local time reference in seconds
    cam_timestamps = (cam_timestamps - imu_timestamps[0]) / np.timedelta64(1, 's')
    imu_timestamps = (imu_timestamps - imu_timestamps[0]) / np.timedelta64(1, 's')

    idx_imu_slice_start = 0
    idx_imu_slice_end = 0
    data_frames = []
    start_time = time.time()
    for k in range(cam_subset_range[0], cam_subset_range[1]):
        print("Processing IMU data files %d/%d (%.2f%%)" %
              (k + 1 - cam_subset_range[0], cam_subset_range[1] - cam_subset_range[0] + 1,
               100 * (k + 1 - cam_subset_range[0]) / (cam_subset_range[1] - cam_subset_range[0]) + 1), end='\r')

        t_k = cam_timestamps[k]
        t_kp1 = cam_timestamps[k + 1]

        # the start value does not need to be recomputed, since you can get that from the previous time step, but
        # i am a lazy person, this will work
        while imu_timestamps[idx_imu_slice_start] < t_k:
            idx_imu_slice_start += 1
        assert (imu_timestamps[idx_imu_slice_start - 1] <= t_k <= imu_timestamps[idx_imu_slice_start])
        # interpolate
        tk_i = imu_timestamps[idx_imu_slice_start - 1]
        tk_j = imu_timestamps[idx_imu_slice_start]
        alpha_k = (t_k - tk_i) / (tk_j - tk_i)
        T_i_vk, v_vk, w_vk, a_vk = \
            interpolate(imu_data[idx_imu_slice_start - 1], imu_data[idx_imu_slice_start],
                        gps_poses[idx_imu_slice_start - 1], gps_poses[idx_imu_slice_start], alpha_k)

        while imu_timestamps[idx_imu_slice_end] < t_kp1:
            idx_imu_slice_end += 1
        assert (imu_timestamps[idx_imu_slice_end - 1] <= t_kp1 <= imu_timestamps[idx_imu_slice_end])
        # interpolate
        tkp1_i = imu_timestamps[idx_imu_slice_end - 1]
        tkp1_j = imu_timestamps[idx_imu_slice_end]
        alpha_kp1 = (t_kp1 - tkp1_i) / (tkp1_j - tkp1_i)
        T_i_vkp1, v_vkp1, w_vkp1, a_vkp1 = \
            interpolate(imu_data[idx_imu_slice_end - 1], imu_data[idx_imu_slice_end],
                        gps_poses[idx_imu_slice_end - 1], gps_poses[idx_imu_slice_end], alpha_kp1)

        imu_timestamps_k_kp1 = np.concatenate(
                [[t_k], imu_timestamps[idx_imu_slice_start:idx_imu_slice_end - 1], [t_kp1]])
        imu_poses = np.concatenate([[T_i_vk], gps_poses[idx_imu_slice_start:idx_imu_slice_end - 1], [T_i_vkp1]])
        accel_measurements_k_kp1 = np.concatenate([[a_vk],
                                                   imu_data[idx_imu_slice_start: idx_imu_slice_end - 1, ax:az + 1],
                                                   [a_vkp1]])
        gyro_measurements_k_kp1 = np.concatenate([[w_vk],
                                                  imu_data[idx_imu_slice_start: idx_imu_slice_end - 1, wx:wz + 1],
                                                  [w_vkp1]])
        frame_k = Frame(image_paths[k], t_k, T_i_vk, T_cam_imu, v_vk,
                        imu_poses, imu_timestamps_k_kp1, accel_measurements_k_kp1, gyro_measurements_k_kp1)
        data_frames.append(frame_k)

        # assertions for sanity check
        assert (np.allclose(data_frames[-1].timestamp, data_frames[-1].imu_timestamps[0], atol=1e-13))
        assert (np.allclose(data_frames[-1].T_i_vk, data_frames[-1].imu_poses[0], atol=1e-13))
        if len(data_frames) > 1:
            # self.image_path = image_path
            # self.timestamp = timestamp
            # self.T_i_vk = T_i_vk  # inertial to vehicle frame pose
            # self.T_cam_imu = T_cam_imu  # calibration from imu to camera
            # self.v_vk_i_vk = v_vk_i_vk  # velocity expressed in vehicle frame
            # self.imu_timestamps = imu_timestamps
            # self.imu_poses = imu_poses
            # self.accel_measurements = accel_measurements
            # self.gyro_measurements = gyro_measurements
            assert (np.allclose(data_frames[-1].timestamp, data_frames[-2].imu_timestamps[-1], atol=1e-13))
            assert (np.allclose(data_frames[-1].T_i_vk, data_frames[-2].imu_poses[-1], atol=1e-13))
            assert (
                np.allclose(data_frames[-1].accel_measurements[0], data_frames[-2].accel_measurements[-1], atol=1e-13))
            assert (
                np.allclose(data_frames[-1].accel_measurements[0], data_frames[-2].accel_measurements[-1], atol=1e-13))

    # add the last frame without any IMU data
    data_frames.append(Frame(image_paths[-1], cam_timestamps[-1], T_i_vkp1, T_cam_imu, v_vkp1,
                             np.zeros([0, 4, 4]), np.zeros([0]), np.zeros([0, 3]), np.zeros([0, 3])))

    logger.print("Processing data took %.2fs" % (time.time() - start_time))

    start_time = time.time()
    data = {"image_path": [f.image_path for f in data_frames],
            "timestamp": [f.timestamp for f in data_frames],
            "T_i_vk": [f.T_i_vk for f in data_frames],
            "T_cam_imu": [f.T_cam_imu for f in data_frames],
            "v_vk_i_vk": [f.v_vk_i_vk for f in data_frames],
            "imu_timestamps": [f.imu_timestamps for f in data_frames],
            "imu_poses": [f.imu_poses for f in data_frames],
            "accel_measurements": [f.accel_measurements for f in data_frames],
            "gyro_measurements": [f.gyro_measurements for f in data_frames]}
    pandas_df = pandas.DataFrame(data, columns=data.keys())
    pandas_df.to_pickle(os.path.join(output_dir, "data.pickle"))
    logger.print("Saving pandas took %.2fs" % (time.time() - start_time))

    # saving some figures for sanity test
    # plot trajectory
    start_time = time.time()
    poses = np.array(data["T_i_vk"])
    plt.clf()
    plt.plot(poses[:, 0, 3], poses[:, 1, 3])  # XY
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("XY Plot")
    plt.axis('equal')
    plt.grid()
    plt.savefig(os.path.join(output_dir, "00_xy_plot.png"))

    plt.clf()
    plt.plot(poses[:, 0, 3], poses[:, 2, 3])  # XZ
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title("XZ Plot")
    plt.axis('equal')
    plt.grid()
    plt.savefig(os.path.join(output_dir, "01_xz_plot.png"))

    plt.clf()
    plt.plot(poses[:, 1, 3], poses[:, 2, 3])  # YZ
    plt.xlabel("Y [m]")
    plt.ylabel("Z [m]")
    plt.title("YZ Plot")
    plt.axis('equal')
    plt.grid()
    plt.savefig(os.path.join(output_dir, "02_yz_plot.png"))

    # three velocities in one plot
    velocities = np.array(data["v_vk_i_vk"])
    poses_timestamps = np.array(data["timestamp"])
    plt.clf()
    plt.plot(poses_timestamps, velocities[:, 0], label="vx")
    plt.plot(poses_timestamps, velocities[:, 1], label="vy")
    plt.plot(poses_timestamps, velocities[:, 2], label="vz")
    plt.xlabel("t [s]")
    plt.ylabel("v [m/s]")
    plt.title("Velocities Plot")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "03_velocities_plot.png"))

    # integrate gyroscope to compare against rotation
    data_imu_timestamps = np.concatenate([d[:-1] for d in data['imu_timestamps']])
    data_gyro_measurements = np.concatenate([d[:-1] for d in data['gyro_measurements']])
    assert (len(data_imu_timestamps) == len(data_gyro_measurements))
    orientations_data_int = [data["T_i_vk"][0][:3, :3]]
    for i in range(0, len(data_imu_timestamps) - 1):
        dt = data_imu_timestamps[i + 1] - data_imu_timestamps[i]
        orientations_data_int.append(orientations_data_int[-1].dot(exp_SO3(dt * data_gyro_measurements[i])))

    # integration of original gyro data
    # orientations_imu_int = [np.eye(3,3)]
    # for i in range(0, len(imu_timestamps) - 1):
    #     dt = imu_timestamps[i + 1] - imu_timestamps[i]
    #     orientations_imu_int.append(orientations_imu_int[-1].dot(exp_SO3(dt * imu_data[i, wx:wz + 1])))

    orientations_data_int = np.array([log_SO3(o) for o in orientations_data_int])
    # orientations_imu_int = np.array([log_SO3(o) for o in orientations_imu_int])
    # orientation_gps = np.array([log_SO3(data["T_i_vk"][0][:3, :3].transpose().dot(p[:3, :3])) for p in gps_poses])
    orientation_data_cam = np.array([log_SO3(p[:3, :3]) for p in data["T_i_vk"]])

    plt.clf()
    plt.plot(data_imu_timestamps, orientations_data_int[:, 0], label="data_int")
    plt.plot(data["timestamp"], orientation_data_cam[:, 0], label="data_poses")
    # plt.plot(imu_timestamps, orientations_imu_int[:, 0], label="imu_int")
    # plt.plot(imu_timestamps, orientation_gps[:, 0], label="gps")
    plt.xlabel("t [s]")
    plt.ylabel("theta [rad]")
    plt.title("Theta X Plot")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "04_theta_x_cmp_plot.png"))

    plt.clf()
    plt.plot(data_imu_timestamps, orientations_data_int[:, 1], label="data_int")
    plt.plot(data["timestamp"], orientation_data_cam[:, 1], label="data_pose")
    # plt.plot(imu_timestamps, orientations_imu_int[:, 1], label="imu_int")
    # plt.plot(imu_timestamps, orientation_gps[:, 1], label="gps")
    plt.xlabel("t [s]")
    plt.ylabel("theta [rad]")
    plt.title("Theta Y Plot")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "05_theta_y_cmp_plot.png"))

    plt.clf()
    plt.plot(data_imu_timestamps, np.unwrap(orientations_data_int[:, 2]), label="data_int")
    plt.plot(data["timestamp"], np.unwrap(orientation_data_cam[:, 2]), label="data_pose")
    # plt.plot(imu_timestamps, orientations_imu_int[:, 2], label="imu_int")
    # plt.plot(imu_timestamps, orientation_gps[:, 2], label="gps")
    plt.xlabel("t [s]")
    plt.ylabel("theta [rad]")
    plt.title("Theta Z Plot")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "06_theta_z_cmp_plot.png"))

    # integrate accel to compare against velocity
    data_accel_measurements = np.concatenate([d[:-1] for d in data["accel_measurements"]])
    data_imu_poses = np.concatenate([d[:-1, :, :] for d in data["imu_poses"]])

    velocities_data_int = [velocities[0, :]]
    # g = np.array([0, 0, 9.80665])
    g = np.array([0, 0, 9.808679801065017])
    for i in range(0, len(data_imu_timestamps) - 1):
        dt = data_imu_timestamps[i + 1] - data_imu_timestamps[i]
        C_i_vk = data_imu_poses[i][:3, :3]
        C_vkp1_vk = data_imu_poses[i + 1][:3, :3].transpose().dot(data_imu_poses[i][:3, :3])

        v_vk_i_vk = velocities_data_int[-1]
        v_vkp1_vk_vk = dt * (data_accel_measurements[i] - C_i_vk.transpose().dot(g))
        v_vkp1_i_vk = v_vk_i_vk + v_vkp1_vk_vk
        velocities_data_int.append(C_vkp1_vk.dot(v_vkp1_i_vk))
    velocities_data_int = np.array(velocities_data_int)

    velocities_from_rel_poses = []
    for i in range(0, len(gps_poses) - 1):
        dt = imu_timestamps[i + 1] - imu_timestamps[i]
        velocities_from_rel_poses.append(np.linalg.inv(gps_poses[i]).dot(gps_poses[i + 1])[0:3, 3] / dt)
    velocities_from_rel_poses = np.array(velocities_from_rel_poses)

    # velocities_from_rel_poses = []
    # for i in range(0, len(data["T_i_vk"]) - 1):
    #     dt = data["timestamp"][i + 1] - data["timestamp"][i]
    #     velocities_from_rel_poses.append(np.linalg.inv(data["T_i_vk"][i]).dot(data["T_i_vk"][i + 1])[0:3, 3] / dt)
    # velocities_from_rel_poses = np.array(velocities_from_rel_poses)

    plt.clf()
    plt.plot(imu_timestamps[1:], velocities_from_rel_poses[:, 0], label="gps_poses_diff")
    plt.plot(data_imu_timestamps, velocities_data_int[:, 0], label="data_int")
    plt.plot(poses_timestamps, velocities[:, 0], label="data_vel")
    # plt.plot(imu_timestamps, imu_data[:, vf], label="gps_vel")
    plt.xlabel("t [s]")
    plt.ylabel("v [m/s]")
    plt.title("Velocity X Plot")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "07_velocity_x_cmp_plot.png"))

    plt.clf()
    plt.plot(imu_timestamps[1:], velocities_from_rel_poses[:, 1], label="gps_poses_diff")
    plt.plot(data_imu_timestamps, velocities_data_int[:, 1], label="data_int")
    plt.plot(poses_timestamps, velocities[:, 1], label="data_vel")
    # plt.plot(imu_timestamps, imu_data[:, vl], label="gps_vel")
    plt.xlabel("t [s]")
    plt.ylabel("v [m/s]")
    plt.title("Velocity Y Plot")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "08_velocity_y_cmp_plot.png"))

    plt.clf()
    plt.plot(imu_timestamps[1:], velocities_from_rel_poses[:, 2], label="gps_poses_diff")
    plt.plot(data_imu_timestamps, velocities_data_int[:, 2], label="data_int")
    plt.plot(poses_timestamps, velocities[:, 2], label="data_vel")
    # plt.plot(imu_timestamps, imu_data[:, vu], label="gps_vel")
    plt.xlabel("t [s]")
    plt.ylabel("v [m/s]")
    plt.title("Velocity Z Plot")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "09_velocity_z_cmp_plot.png"))

    logger.print("Generating figures took %.2fs" % (time.time() - start_time))
    logger.print("All done!")
