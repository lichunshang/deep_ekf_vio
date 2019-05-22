from model import IMUKalmanFilter
from data_loader import *
from params import par
import torch

ekf_model = IMUKalmanFilter()

batch_sz = 8
seq_len = 6

imu_noise_covar_weights = torch.nn.Linear(1, 4, bias=False)
imu_noise_covar_weights.weight.data /= 10

covar = 10 ** (par.imu_noise_covar_beta * torch.tanh(par.imu_noise_covar_gamma * imu_noise_covar_weights(
        torch.ones(1, device=imu_noise_covar_weights.weight.device))))

imu_noise_covar_diag = torch.tensor([1, 1, 1, 1], dtype=torch.float32,
                                    device=imu_noise_covar_weights.weight.device).repeat_interleave(3) * \
                       torch.stack([covar[0], covar[0], covar[0],
                                    covar[1], covar[1], covar[1],
                                    covar[2], covar[2], covar[2],
                                    covar[3], covar[3], covar[3]])

# vis_meas = np.load("/home/cs4li/Dev/deep_ekf_vio/results/"
#                    "train_20190515-11-39-21_esg_0.5k3_vis3b1g_imu4b1g_fix_imu_ref_noinv_256rnn/"
#                    "saved_model.eval.traj/vis_meas/meas/K06.npy")
subseqs = get_subseqs(["K06"], seq_len, overlap=1, sample_times=1, training=False)
dataset = SubseqDataset(subseqs, (par.img_h, par.img_w), 0, 1, True, training=False, no_image=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_sz, shuffle=False, num_workers=0)

data = next(iter(dataloader))
meta_data, _, imu_data, init_state, T_imu_cam, gt_poses, gt_rel_poses = data

# (subseq.length, subseq.seq, subseq.type, subseq.id, subseq.id_next),
# images, imu_data, init_state, T_imu_cam, gt_poses, gt_rel_poses

vis_meas = (gt_rel_poses + torch.randn_like(gt_rel_poses) / 1).unsqueeze(-1)
# vis_meas.requires_grad = True
init_covar = torch.eye(18, 18).repeat(batch_sz, 1, 1)
init_covar.requires_grad = True

# imu_data, imu_noise_covar, prev_pose, prev_state, prev_covar, vis_meas, vis_meas_covar, T_imu_cam
poses, ekf_states, ekf_covars = ekf_model.forward(imu_data, torch.diag(imu_noise_covar_diag),
                                                  gt_poses[:, 0].inverse(), init_state,
                                                  init_covar,
                                                  vis_meas,
                                                  torch.diag_embed(torch.ones(batch_sz, seq_len - 1, 6)),
                                                  T_imu_cam)


abs_errors = torch.matmul(poses[:, 1:], gt_poses[:, 1:])
length_div = torch.arange(start=1, end=abs_errors.size(1) + 1, device=abs_errors.device,
                          dtype=torch.float32).view(1, -1, 1)
abs_trans_errors_sq = torch.sum((abs_errors[:, :, 0:3, 3] / length_div) ** 2, dim=-1)
abs_trans_loss = torch.mean(abs_trans_errors_sq[4])


print("Hello")
