import sys
import os
import rospy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import transformations

poses = []
gt_poses = []
path = Path()
last_ts = None
def est_callback(data):
    global last_ts
    global poses
    global path
    global gt_poses

    if last_ts and abs((data.header.stamp - last_ts).to_sec()) > 0.1:
        print("Reset!")
        poses = []
        gt_poses = []
    
    pose_stamped = PoseStamped()
    pose_stamped.header = data.header
    pose_stamped.pose = data.pose.pose
    poses.append(pose_stamped)
    path.header = data.header
    path.poses = poses
    last_ts = data.header.stamp

def gt_callback(data):
    global gt_poses

    gt_poses.append(data)
    

def main():
    global gt_poses
    global poses

    rospy.init_node('kitti_msf_ros2file')

    rospy.Subscriber("/msf_core/pose", PoseWithCovarianceStamped, est_callback)
    rospy.Subscriber("/gt", PoseStamped, gt_callback)
    path_pub = rospy.Publisher("/path", Path, queue_size=1)

    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        if len(path.poses) > 0:
            path_pub.publish(path)
        rate.sleep()


    # align the ground truth and estimate
    est_poses_ts = np.array([p.header.stamp.to_sec() for p in poses])
    gt_poses_ts = np.array([p.header.stamp.to_sec() for p in gt_poses])
    
    timestamps = []
    est_poses_mat = []
    gt_poses_mat = []
    num_skipped = 0
    for i in range(0, len(est_poses_ts)):
        j = np.abs(gt_poses_ts - est_poses_ts[i]).argmin()
        # print("diff %f" % np.abs(gt_poses_ts[j] - est_poses_ts[i]))
        # assert (np.abs(gt_poses_ts[j] - est_poses_ts[i]) < 1e-4)
        if np.abs(gt_poses_ts[j] - est_poses_ts[i]) > 1e-4:
            # print("skip id %d" % i)
            num_skipped += 1
            continue
            

        timestamps.append(est_poses_ts[i])

        est_T = transformations.quaternion_matrix(np.array([poses[i].pose.orientation.w, 
                            poses[i].pose.orientation.x, 
                            poses[i].pose.orientation.y, 
                            poses[i].pose.orientation.z ]))
        est_T[:3, 3] = np.array([poses[i].pose.position.x, poses[i].pose.position.y, poses[i].pose.position.z])
        est_poses_mat.append(est_T)

        gt_T = transformations.quaternion_matrix(np.array([gt_poses[j].pose.orientation.w, 
                            gt_poses[j].pose.orientation.x, 
                            gt_poses[j].pose.orientation.y, 
                            gt_poses[j].pose.orientation.z ]))
        gt_T[:3, 3] = np.array([gt_poses[j].pose.position.x, gt_poses[j].pose.position.y, gt_poses[j].pose.position.z])
        gt_poses_mat.append(gt_T)

    print("num_skipped: %d" % num_skipped)
    print("num timestamps %d" % len(timestamps))
    print("num gt %d" % len(gt_poses_mat))
    print("num est %d" % len(est_poses_mat))
    np.save("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_msf/timestamps.npy", np.array(timestamps))
    np.save("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_msf/gt.npy", np.array(gt_poses_mat))
    np.save("/home/cs4li/Dev/deep_ekf_vio/results/final_thesis_results/KITTI_msf/est.npy", np.array(est_poses_mat))


if __name__ == "__main__":
    main()
