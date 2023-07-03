from data_loader import get_subseqs, SubseqDataset
from torch.utils.data import DataLoader
from params import par
from trainer import _TrainAssistant, E2EVIO
from log import logger
from eval import plot_ekf_data
import torch
import se3
import numpy as np
from new_loss import *
train_subseqs = get_subseqs(["K04"], 4, overlap=1, sample_times=par.sample_times, training=True)
train_dataset = SubseqDataset(train_subseqs, (par.img_h, par.img_w), par.img_means,
                              par.img_stds, par.minus_point_5)
train_dl = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
train_dl_iter = iter(train_dl)

e2e_vio_model = E2EVIO()
resume_path = "/mnt/data/teamAI/duy/deep_ekf_vio/results/train_20230630-10-51-30/saved_model.eval"
save_path = "/mnt/data/teamAI/duy/deep_ekf_vio/results/all_test"
e2e_vio_model.load_state_dict(logger.clean_state_dict_key(torch.load(resume_path)))
e2e_vio_model = e2e_vio_model.cuda()
e2e_vio_ta = _TrainAssistant(e2e_vio_model)

logger.initialize(save_path, use_tensorboard=True)

for i in range(0, 1):
    data = next(train_dl_iter)
    meta_data, images, imu_data, prev_state, T_imu_cam, gt_poses, gt_rel_poses = data

    gt_rel_poses = np.array(gt_rel_poses.squeeze())
    gt_poses = np.array(gt_poses.squeeze())
    gt_rel_mat = []
    predicted_abs_poses = [gt_poses[0]]
    for i, rel_pose in enumerate(gt_rel_poses):
        gt_rel_mat.append(euler_to_matrix_np(rel_pose))
        T_vkm1_vk = se3.T_from_Ct(se3.exp_SO3(rel_pose[0:3]/np.linalg.norm(rel_pose[0:3])), rel_pose[3:6])
        T_i_vk = predicted_abs_poses[i].dot(T_vkm1_vk)
        se3.log_SO3(T_i_vk[0:3, 0:3])  # just here to check for warnings
        predicted_abs_poses.append(T_i_vk)

    # pose02 = np.matmul(gt_rel_mat[0],gt_rel_mat[1])
    # pose02_euler = matrix_to_euler(pose02)
    # T_vkm1_vk = se3.T_from_Ct(se3.exp_SO3(pose02_euler[0:3]), pose02_euler[3:6])
    # T_i_vk = predicted_abs_poses[0].dot(T_vkm1_vk)
    print(gt_poses[1] - predicted_abs_poses[1])

    print('===')



    vis_meas, vis_meas_covar, lstm_states, poses, ekf_states, ekf_covars = \
        e2e_vio_ta.model.forward(images.cuda(),
                                 imu_data.cuda(),
                                 None,
                                 gt_poses[:, 0].cuda(),
                                 prev_state.cuda(), T_imu_cam.cuda())

    plot_ekf_data(save_path,
                  imu_data[0, :, 0, 0].detach().cpu(),
                  torch.inverse(gt_poses[0]),
                  torch.zeros(poses.shape[1], 3),
                  poses[0].detach().cpu(),
                  ekf_states[0].detach().cpu(),
                  g_const=9.80665)

    loss = e2e_vio_ta.ekf_loss(poses, gt_poses.cuda())
