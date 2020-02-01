import sys
import pandas as pd
import os
import numpy as np
import time
import transformations

ROS = True
if ROS:
    import rospy
    from sensor_msgs.msg import Imu
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import TransformStamped, PoseStamped
    from nav_msgs.msg import Odometry, Path
    from cv_bridge import CvBridge, CvBridgeError
    import cv2


class SequenceData(object):
    def __init__(self, seq):
        data_dir = "/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_ORB_SLAM2_Mono/data_py2_pickle"
        self.seq = seq
        self.seq_dir = os.path.join(data_dir, seq)
        self.pd_path = os.path.join(data_dir, seq, "data.pickle")
        self.df = pd.read_pickle(self.pd_path)

        self.constants_path = os.path.join(data_dir, seq, "constants.npy")
        self.constants = np.load(self.constants_path, allow_pickle=True).item()

        self.g_i = self.constants["g_i"]
        self.T_cam_imu = self.constants["T_cam_imu"]
        self.bw_0 = self.constants["bw_0"]
        if "ba_0" in self.constants:
            self.ba_0 = self.constants["ba_0"]
        else:
            self.ba_0 = np.zeros_like(self.bw_0)


def quat_matrix(quat):
    quat2 = np.array([quat[3], quat[0], quat[1], quat[2]])
    return transformations.quaternion_matrix(quat2)


def main():
    # ----------- ros stuff -----------
    if ROS:
        rospy.init_node('kitti_msf_data_node')
        imu_pub = rospy.Publisher("/imu", Imu, queue_size=1000)
        transform_pub = rospy.Publisher("/orb_slam_mono_vo", TransformStamped, queue_size=1000)
        odom_pub = rospy.Publisher("/orb_slam_mono_vo_odom", Odometry, queue_size=1000)
        image_pub = rospy.Publisher("/image", Image, queue_size=1)
        gt_pub = rospy.Publisher("/gt", PoseStamped, queue_size=1)
        gt_path_pub = rospy.Publisher("/gt_path", Path, queue_size=1000)
        gt_path = Path()
        cv_bridge = CvBridge()

    # ----------- datset stuff -----------
    # seq = "K06"
    seq = sys.argv[1]
    valid_seqs = ["K01", "K04", "K06", "K07", "K08", "K09", "K10"]

    REF_FRAME = {
        "K01": 0,
        "K04": 0,
        "K06": 0,
        "K07": 0,
        "K08": 1100,
        "K09": 0,
        "K10": 0,
    }

    REF_PATH = {
        "K01": "2011_10_03/2011_10_03_drive_0042",
        "K04": "2011_09_30/2011_09_30_drive_0016",
        "K06": "2011_09_30/2011_09_30_drive_0020",
        "K07": "2011_09_30/2011_09_30_drive_0027",
        "K08": "2011_09_30/2011_09_30_drive_0028",
        "K09": "2011_09_30/2011_09_30_drive_0033",
        "K10": "2011_09_30/2011_09_30_drive_0034",
    }

    assert seq in valid_seqs
    seq_data = SequenceData(seq)
    orb_slam_outputs = np.loadtxt(
            os.path.join("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_ORB_SLAM2_Mono",
                         "KeyFrameTrajectory_%s_stereo.txt" % seq))
    orb_slam_timestamps = orb_slam_outputs[:, 0]
    orb_slam_position = orb_slam_outputs[:, 1:4]
    orb_slam_quat = orb_slam_outputs[:, 4:8]  # qw is the last column

    orb_slam_ref_time = np.datetime64(open(os.path.join("/home/cs4li/Dev/KITTI/dataset",
                                                        REF_PATH[seq] + "_sync", "image_02",
                                                        "timestamps.txt"), "r").readlines()[REF_FRAME[seq]].strip())
    imu_ref_time = np.datetime64(open(os.path.join("/home/cs4li/Dev/KITTI/dataset",
                                                   REF_PATH[seq] + "_extract", "oxts",
                                                   "timestamps.txt"), "r").readline().strip())

    df_timestamps = list(seq_data.df.loc[:, "timestamp"])

    orb_slam_ref_time_flt = (orb_slam_ref_time - np.datetime64(0, "s")) / np.timedelta64(1, "s")
    orb_slam_timestamps_adj = orb_slam_timestamps + orb_slam_ref_time_flt
    imu_ref_time_flt = (imu_ref_time - np.datetime64(0, "s")) / np.timedelta64(1, "s")
    df_timestamps_adj = imu_ref_time_flt + df_timestamps

    # start_frame = 0
    # while imu_ref_time_flt + df_timestamps[start_frame] < orb_slam_timestamps_adj[0]:
    #     start_frame += 1
    #
    # print("start_frame %d, diff %.2f" % (start_frame,
    #     imu_ref_time_flt + df_timestamps[start_frame] - orb_slam_ref_time_flt + orb_slam_timestamps[0]))

    prev_orb_slam_meas_idx = -1
    warn_count = 0
    orb_slam_meas_count = 0
    first_meas_T = None
    first_gt_T = None
    scale = -1
    for i in range(0, len(seq_data.df) - 1):
        gyro_measurements = seq_data.df.loc[i, "gyro_measurements"]
        accel_measurements = seq_data.df.loc[i, "accel_measurements"]
        imu_timestamps = seq_data.df.loc[i, "imu_timestamps"] + imu_ref_time_flt
        assert (len(gyro_measurements) == len(accel_measurements) == len(imu_timestamps))

        if ROS:
            if rospy.is_shutdown():
                break

        orb_slam_meas_idx = np.abs(orb_slam_timestamps_adj - imu_timestamps[0]).argmin()
        if np.abs(orb_slam_timestamps_adj[orb_slam_meas_idx] - imu_timestamps[0]) < 5e-3:

            if prev_orb_slam_meas_idx > 0 and not orb_slam_meas_idx - prev_orb_slam_meas_idx == 1:
                warn_count += 1
                print("WARN!!!!")

            if prev_orb_slam_meas_idx < 0:
                first_meas_T = quat_matrix(orb_slam_quat[orb_slam_meas_idx])
                first_meas_T[0:3, 3] = orb_slam_position[orb_slam_meas_idx]

                first_gt_T = seq_data.df.loc[i, "T_i_vk"]

                print("g_0: [{:.8f}, {:.8f}, {:.8f}]".format(*list(seq_data.df.loc[i, "T_i_vk"][:3,:3].transpose().dot(seq_data.g_i))))
                print("v_0: [{:.8f}, {:.8f}, {:.8f}]".format(*list(seq_data.df.loc[i, "v_vk_i_vk"])))

            if orb_slam_meas_count == 1:
                dist_orb = np.linalg.norm(orb_slam_position[orb_slam_meas_idx] -
                                          orb_slam_position[prev_orb_slam_meas_idx])

                gt1_idx = np.abs(orb_slam_timestamps_adj[orb_slam_meas_idx] - df_timestamps_adj).argmin()
                gt2_idx = np.abs(orb_slam_timestamps_adj[prev_orb_slam_meas_idx] - df_timestamps_adj).argmin()

                print("gt1_time_diff: %.5f ms" % (orb_slam_timestamps_adj[orb_slam_meas_idx] - df_timestamps_adj[gt1_idx]))
                print("gt2_time_diff: %.5f ms" % (orb_slam_timestamps_adj[orb_slam_meas_idx] - df_timestamps_adj[gt2_idx]))

                gt1 = seq_data.df.loc[gt1_idx, "T_i_vk"]
                gt2 = seq_data.df.loc[gt2_idx, "T_i_vk"]

                dist_gt = np.linalg.norm(gt2[:3, 3] - gt1[:3, 3])
                scale = dist_gt / dist_orb
                print("Scale: %f" % scale)

            prev_orb_slam_meas_idx = orb_slam_meas_idx

            if orb_slam_meas_count > 0:
                meas_T = quat_matrix(orb_slam_quat[orb_slam_meas_idx])
                meas_T[0:3, 3] = orb_slam_position[orb_slam_meas_idx]
                meas_T_rel = np.linalg.inv(first_meas_T).dot(meas_T)
                meas_T_rel_quat = transformations.quaternion_from_matrix(meas_T_rel)
                # meas_T_rel_pos = meas_T_rel[0:3, 3] * scale
                meas_T_rel_pos = meas_T_rel[0:3, 3]

                x = [orb_slam_timestamps_adj[orb_slam_meas_idx]] + \
                    list(orb_slam_position[orb_slam_meas_idx]) + list(orb_slam_quat[orb_slam_meas_idx])
                x2 = [orb_slam_timestamps_adj[orb_slam_meas_idx]] + list(meas_T_rel_pos) + list(meas_T_rel_quat)

                print("orb_slam_ts: {:20} pos: [{:9.2e} {:9.2e} {:9.2e}]  "
                      "quat: [{:9.2e} {:9.2e} {:9.2e} {:9.2e}]".format(*x))
                print("orb_slam_ts_rel: {:20} pos: [{:9.2e} {:9.2e} {:9.2e}]  "
                      "quat: [{:9.2e} {:9.2e} {:9.2e} {:9.2e}]".format(*x2))

                if ROS:
                    t_msg = TransformStamped()
                    # t_msg.header.stamp = rospy.Time(orb_slam_timestamps_adj[orb_slam_meas_idx])
                    t_msg.header.stamp = rospy.Time(seq_data.df.loc[i, "timestamp"] + imu_ref_time_flt)
                    t_msg.header.frame_id = "vision"
                    t_msg.child_frame_id = "camera"
                    t_msg.transform.translation.x = meas_T_rel_pos[0]
                    t_msg.transform.translation.y = meas_T_rel_pos[1]
                    t_msg.transform.translation.z = meas_T_rel_pos[2]
                    t_msg.transform.rotation.x = meas_T_rel_quat[1]
                    t_msg.transform.rotation.y = meas_T_rel_quat[2]
                    t_msg.transform.rotation.z = meas_T_rel_quat[3]
                    t_msg.transform.rotation.w = meas_T_rel_quat[0]

                    o_msg = Odometry()
                    o_msg.header = t_msg.header
                    o_msg.child_frame_id = t_msg.child_frame_id
                    o_msg.header.frame_id = "world"
                    o_msg.pose.pose.position.x = t_msg.transform.translation.x
                    o_msg.pose.pose.position.y = t_msg.transform.translation.y
                    o_msg.pose.pose.position.z = t_msg.transform.translation.z
                    o_msg.pose.pose.orientation.x = t_msg.transform.rotation.x
                    o_msg.pose.pose.orientation.y = t_msg.transform.rotation.y
                    o_msg.pose.pose.orientation.z = t_msg.transform.rotation.z
                    o_msg.pose.pose.orientation.w = t_msg.transform.rotation.w

                    transform_pub.publish(t_msg)
                    odom_pub.publish(o_msg)

            orb_slam_meas_count += 1

        if prev_orb_slam_meas_idx < 0:
            continue

        if ROS:
            # publish image
            image_pub.publish(cv_bridge.cv2_to_imgmsg(cv2.imread(seq_data.df.loc[i, "image_path"]), encoding="bgr8"))
            gt_T_rel = np.linalg.inv(first_gt_T).dot(seq_data.df.loc[i, "T_i_vk"])
  
            gt_p_msg = PoseStamped()
            gt_p_msg.header.stamp = rospy.Time(seq_data.df.loc[i, "timestamp"] + imu_ref_time_flt)
            gt_p_msg.header.frame_id = "world"
            gt_p_msg.pose.position.x = gt_T_rel[0, 3]
            gt_p_msg.pose.position.y = gt_T_rel[1, 3]
            gt_p_msg.pose.position.z = gt_T_rel[2, 3]

            q = transformations.quaternion_from_matrix(gt_T_rel)
            gt_p_msg.pose.orientation.x = q[1]
            gt_p_msg.pose.orientation.y = q[2]
            gt_p_msg.pose.orientation.z = q[3]
            gt_p_msg.pose.orientation.w = q[0]

            
            gt_path.header = gt_p_msg.header
            gt_path.poses.append(gt_p_msg)
            gt_path_pub.publish(gt_path)
            gt_pub.publish(gt_p_msg)

        for j in range(0, len(gyro_measurements) - 1):
            x = [imu_timestamps[j]] + list(gyro_measurements[j]) + list(accel_measurements[j])
            print("imu_ts: {:20} gyro: [{:9.2e} {:9.2e} {:9.2e}]  accel: [{:9.2e} {:9.2e} {:9.2e}]".format(*x))

            if ROS:
                imu_msg = Imu()
                imu_msg.header.stamp = rospy.Time(imu_timestamps[j])
                imu_msg.header.frame_id = "imu"
                imu_msg.angular_velocity.x = gyro_measurements[j, 0]
                imu_msg.angular_velocity.y = gyro_measurements[j, 1]
                imu_msg.angular_velocity.z = gyro_measurements[j, 2]
                imu_msg.linear_acceleration.x = accel_measurements[j, 0]
                imu_msg.linear_acceleration.y = accel_measurements[j, 1]
                imu_msg.linear_acceleration.z = accel_measurements[j, 2]
                imu_pub.publish(imu_msg)

                d = rospy.Duration((imu_timestamps[j + 1] - imu_timestamps[j]) / 10.0, 0)
                rospy.sleep(d)
            else:
                time.sleep(imu_timestamps[j + 1] - imu_timestamps[j])

    print("Done! %d" % warn_count)


if __name__ == "__main__":
    main()
