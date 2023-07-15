from data_loader import get_subseqs, SubseqDataset
from torch.utils.data import DataLoader
from params import par
from trainer import _TrainAssistant, E2EVIO
from log import logger
from eval import plot_ekf_data
import torch
import se3
import torch_se3
import numpy as np
from new_loss import *
train_subseqs = get_subseqs(["K04"], 5, overlap=1, sample_times=par.sample_times, training=True)
train_dataset = SubseqDataset(train_subseqs, (par.img_h, par.img_w), par.img_means,
                              par.img_stds, par.minus_point_5)
train_dl = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
train_dl_iter = iter(train_dl)

e2e_vio_model = E2EVIO()
# resume_path = "//mnt/data/teamAI/duy/deep_ekf_vio/results/train_20230713-16-57-37/saved_model.eval"
save_path = "/mnt/data/teamAI/duy/deep_ekf_vio/results/all_test"
# e2e_vio_model.load_state_dict(logger.clean_state_dict_key(torch.load(resume_path)))
e2e_vio_model = e2e_vio_model.cuda()
e2e_vio_ta = _TrainAssistant(e2e_vio_model)

logger.initialize(save_path, use_tensorboard=True)

data = next(train_dl_iter)
meta_data, images, imu_data, prev_state, T_imu_cam, gt_poses, gt_rel_poses = data

print(gt_poses.shape)
print(gt_rel_poses.shape)
calc_pose = []
for k in range(gt_rel_poses.size(1)):
    rel_pose = gt_rel_poses[:,k]
    T_vkm1_vk = torch_se3.T_from_Ct_b(torch_se3.rpy_to_so3(rel_pose[:,0:3]), rel_pose[:,3:6])
    T_i_vk = torch.matmul(gt_poses[:,:,0],(T_vkm1_vk))
    calc_pose.append(T_i_vk)
    print(gt_poses[:,:,1]-calc_pose[k])
    quit()
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
