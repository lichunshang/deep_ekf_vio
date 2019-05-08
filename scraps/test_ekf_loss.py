import torch
import numpy as np
from params import par
import se3


def ekf_loss(est_poses, gt_poses, ekf_states, gt_rel_poses, vis_meas, vis_meas_covar):
    abs_errors = torch.matmul(torch.inverse(est_poses[:, 1:]), gt_poses[:, 1:])
    # length_div = torch.arange(start=1, end=abs_errors.size(1) + 1, device=abs_errors.device,
    #                           dtype=torch.float32).view(1, -1, 1)
    # calculate the F norm squared from identity
    I_minus_angle_errors = (torch.eye(3, 3, device=abs_errors.device) -
                            abs_errors[:, :, 0:3, 0:3])
    I_minus_angle_errors_sq = torch.matmul(I_minus_angle_errors, I_minus_angle_errors.transpose(-2, -1))
    abs_angle_errors_sq = torch.sum(torch.diagonal(I_minus_angle_errors_sq, dim1=-2, dim2=-1), dim=-1)
    # abs_angle_errors = torch.squeeze(torch_se3.log_SO3_b(abs_errors[:, :, 0:3, 0:3]), -1) / length_div
    # abs_angle_errors_sq = torch.sum(abs_angle_errors ** 2, dim=-1)  # norm squared

    abs_trans_errors_sq = torch.sum((abs_errors[:, :, 0:3, 3]) ** 2, dim=-1)

    abs_angle_loss = torch.mean(abs_angle_errors_sq)
    abs_trans_loss = torch.mean(abs_trans_errors_sq)

    # _, C_rel, r_rel, _, _, _ = IMUKalmanFilter.decode_state_b(ekf_states)
    # rel_angle_errors = (gt_rel_poses[:, :, 0:3] - torch.squeeze(torch_se3.log_SO3_b(C_rel[:, 1:]), -1)) ** 2
    # rel_angle_errors_sq = torch.sum(rel_angle_errors ** 2, dim=-1)
    # rel_trans_error_sq = torch.sum((gt_rel_poses[:, :, 3:6] - torch.squeeze(r_rel[:, 1:], -1)) ** 2, dim=-1)
    # rel_angle_loss = torch.mean(rel_angle_errors_sq)
    # rel_trans_loss = torch.mean(rel_trans_error_sq)

    # k3 = self.schedule(par.k3)

    loss_abs = (par.k2 * abs_angle_loss + abs_trans_loss) * par.k4 ** 2
    # loss_rel = (par.k1 * rel_angle_loss + rel_trans_loss)
    # loss = k3 * loss_rel + (1 - k3) * loss_abs
    # loss_vis_meas = self.vis_meas_loss(vis_meas, vis_meas_covar, gt_rel_poses)
    # loss = k3 * loss_vis_meas + (1 - k3) * loss_abs

    # assert not torch.any(torch.isnan(loss))

    # add to tensorboard
    trans_errors = abs_errors[:, :, 0:3, 3].detach().cpu().numpy()
    angle_errors_np = []
    errors_np = abs_errors.detach().cpu().numpy()
    for i in range(0, abs_errors.size(0)):
        angle_errors_over_ts = []
        for j in range(0, abs_errors.size(1)):
            angle_errors_over_ts.append(se3.log_SO3(errors_np[i, j, 0:3, 0:3]))
        angle_errors_np.append(np.stack(angle_errors_over_ts))
    angle_errors_np = np.stack(angle_errors_np)

    last_rot_x_loss = np.mean(np.abs(angle_errors_np[:, -1, 0]))
    last_rot_y_loss = np.mean(np.abs(angle_errors_np[:, -1, 1]))
    last_rot_z_loss = np.mean(np.abs(angle_errors_np[:, -1, 2]))
    last_trans_x_loss = np.mean(np.abs(trans_errors[:, -1, 0]))
    last_trans_y_loss = np.mean(np.abs(trans_errors[:, -1, 1]))
    last_trans_z_loss = np.mean(np.abs(trans_errors[:, -1, 2]))

    print("abs_angle_loss", abs_angle_loss)
    print("abs_trans_loss", abs_trans_loss)
    print("loss_abs", loss_abs)

    print("last_rot_x_loss", last_rot_x_loss)
    print("last_rot_y_loss", last_rot_y_loss)
    print("last_rot_z_loss", last_rot_z_loss)
    print("last_trans_x_loss", last_trans_x_loss)
    print("last_trans_y_loss", last_trans_y_loss)
    print("last_trans_z_loss", last_trans_z_loss)

int0 = 100
int1 = int0 + 32
torch.set_default_tensor_type('torch.DoubleTensor')
# 1
est1 = np.load(
    "/home/cs4li/Dev/deep_ekf_vio/results/train_20190505-03-03-36_ekf_scratch_gloss_0.999k3_uncorrcovar/saved_model.eval.traj/est_poses/K01.npy")
gt1 = np.load(
    "/home/cs4li/Dev/deep_ekf_vio/results/train_20190505-03-03-36_ekf_scratch_gloss_0.999k3_uncorrcovar/saved_model.eval.traj/gt_poses/K01.npy")

est1_abs = np.array([np.dot(gt1[int0], np.linalg.inv(est1[int0]).dot(p)) for p in est1[int0:int1]])
ekf_loss(torch.unsqueeze(torch.tensor(est1_abs, dtype=torch.float64), 0), torch.unsqueeze(torch.tensor(gt1[int0:int1], dtype=torch.float64), 0), None, None, None, None)
print("\n")

# est1_rel = np.array([np.linalg.inv(est1[int0]).dot(p) for p in est1[int0:int1]])
# gt1_rel = np.array([np.linalg.inv(gt1[int0]).dot(p) for p in gt1[int0:int1]])
# ekf_loss(torch.unsqueeze(torch.tensor(est1_rel, dtype=torch.float64), 0), torch.unsqueeze(torch.tensor(gt1_rel, dtype=torch.float64), 0), None, None, None, None)

#
# print("\n")
#
est2 = np.load(
    "/home/cs4li/Dev/deep_ekf_vio/results/train_20190505-03-03-36_ekf_scratch_gloss_0.999k3_uncorrcovar/saved_model.checkpoint.traj/est_poses/K01.npy")
gt2 = np.load(
    "/home/cs4li/Dev/deep_ekf_vio/results/train_20190505-03-03-36_ekf_scratch_gloss_0.999k3_uncorrcovar/saved_model.checkpoint.traj/gt_poses/K01.npy")
est2_abs = np.array([np.dot(gt2[int0], np.linalg.inv(est2[int0]).dot(p)) for p in est2[int0:int1]])
ekf_loss(torch.unsqueeze(torch.tensor(est2_abs, dtype=torch.float64), 0), torch.unsqueeze(torch.tensor(gt2[int0:int1], dtype=torch.float64), 0), None, None, None, None)