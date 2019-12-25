import numpy as np
import os
import matplotlib.pyplot as plt

# path = "/home/cs4li/Dev/deep_ekf_vio/results/Presentation_Results/KITTI/" \
#        "1vanillanogloss_train_20190430-14-11-52_ekf_scratch_nogloss_0.75k3_1k4_eps1e-5/saved_model.eval.traj"
path = "/home/cs4li/Dev/deep_ekf_vio/results/Presentation_Results/KITTI/" \
       "train_20190722-00-01-20_vanilla/saved_model.train.traj"
vis_meas_covars = np.sqrt(np.diagonal(np.load(os.path.join(path, "vis_meas", "covar", "K08.npy")), axis1=1, axis2=2)) * 3
vis_meas_errors = np.abs(np.load(os.path.join(path, "errors", "vis_meas", "K08.npy"))[1:])

plt.figure(1)
plt.scatter(vis_meas_covars[:,0], vis_meas_errors[:,0])
plt.figure(2)
plt.scatter(vis_meas_covars[:,1], vis_meas_errors[:,1])
plt.figure(3)
plt.scatter(vis_meas_covars[:,2], vis_meas_errors[:,2])
plt.figure(4)
plt.scatter(vis_meas_covars[:,3], vis_meas_errors[:,3])
plt.figure(5)
plt.scatter(vis_meas_covars[:,4], vis_meas_errors[:,4])
plt.figure(6)
plt.scatter(vis_meas_covars[:,5], vis_meas_errors[:,5])
plt.show()