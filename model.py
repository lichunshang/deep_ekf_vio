import torch
import torch.nn as nn
import numpy as np
import data_loader
import torch_se3
from params import par
from backmodel.newnet import   PoseRegressor, Res
from backmodel.ra import RAFT
from new_loss import euler_to_matrix, matrix_to_euler

class IMUKalmanFilter(nn.Module):
    STATE_VECTOR_DIM = 19

    def __init__(self):
        super(IMUKalmanFilter, self).__init__()

    def force_symmetrical(self, M):
        M_upper = torch.triu(M)
        return M_upper + M_upper.transpose(-2, -1) * \
               (1 - torch.eye(M_upper.size(-2), M_upper.size(-1), device=M.device).repeat(M_upper.size(0), 1, 1))

    def predict_one_step(self, t_accum, C_accum, r_accum, v_accum, dt, g_k, v_k, bw_k, ba_k, covar,
                         gyro_meas, accel_meas, imu_noise_covar):
        mm = torch.matmul
        batch_size = dt.size(0)

        dt2 = dt * dt
        w = gyro_meas - bw_k
        w_skewed = torch_se3.skew3_b(w)
        C_accum_transpose = C_accum.transpose(-2, -1)
        a = accel_meas - ba_k
        v = mm(C_accum_transpose, v_k - g_k * t_accum + v_accum)
        v_skewed = torch_se3.skew3_b(v)
        I3 = torch.eye(3, 3, device=covar.device).repeat(batch_size, 1, 1)
        exp_int_w = torch_se3.exp_SO3_b(dt * w)
        exp_int_w_transpose = exp_int_w.transpose(-2, -1)

        # propagate uncertainty, 2nd order
        F = torch.zeros(batch_size, 19, 19, device=covar.device)
        F[:, 3:6, 3:6] = -w_skewed
        F[:, 3:6, 12:15] = -I3
        F[:, 6:9, 3:6] = -mm(C_accum, v_skewed)
        F[:, 6:9, 9:12] = C_accum
        F[:, 9:12, 0:3] = -C_accum_transpose
        F[:, 9:12, 3:6] = -torch_se3.skew3_b(mm(C_accum_transpose, g_k))
        F[:, 9:12, 9:12] = -w_skewed
        F[:, 9:12, 12:15] = -v_skewed
        F[:, 9:12, 15:18] = -I3

        G = torch.zeros(batch_size, 19, 12, device=covar.device)
        G[:, 3:6, 0:3] = -I3
        G[:, 9:12, 0:3] = -v_skewed
        G[:, 9:12, 6:9] = -I3
        G[:, 12:15, 3:6] = I3
        G[:, 15:18, 9:12] = I3

        Phi = torch.eye(19, 19, device=covar.device).repeat(batch_size, 1, 1) + \
              F * dt + 0.5 * mm(F, F) * dt2
        Phi[:, 6:9, 12:15] = torch.zeros(3, 3, device=covar.device)  # this blocks is exactly zero in 2nd order approx
        Phi[:, 3:6, 3:6] = exp_int_w_transpose
        Phi[:, 9:12, 9:12] = exp_int_w_transpose

        Q = mm(mm(mm(mm(Phi, G), imu_noise_covar.repeat(batch_size, 1, 1)),
                  G.transpose(-2, -1)), Phi.transpose(-2, -1)) * dt
        covar = mm(mm(Phi, covar), Phi.transpose(-2, -1)) + Q
        covar = self.force_symmetrical(covar)

        # propagate nominal states
        r_accum = r_accum + v_accum * dt + 0.5 * mm(C_accum, (dt2 * a))
        v_accum = v_accum + mm(C_accum, (dt * a))
        C_accum = mm(C_accum, exp_int_w)
        t_accum = t_accum + dt

        return t_accum, C_accum, r_accum, v_accum, covar, F, G, Phi, Q

    def predict(self, imu_meas, imu_noise_covar, prev_state, prev_covar):
        num_batches = imu_meas.size(0)
        C_accum = torch.eye(3, 3, device=imu_meas.device).repeat(num_batches, 1, 1)
        r_accum = torch.zeros(num_batches, 3, 1, device=imu_meas.device)
        v_accum = torch.zeros(num_batches, 3, 1, device=imu_meas.device)
        t_accum = torch.zeros(num_batches, 1, 1, device=imu_meas.device)

        # set C, r covariances to zero
        U = torch.diag(torch.tensor([1.] * 3 + [0.] * 6 + [1.] * 10, device=imu_meas.device)).repeat(num_batches, 1, 1)
        pred_covar = torch.matmul(torch.matmul(U, prev_covar), U.transpose(-2, -1))

        pred_states = []
        pred_covars = []

        # Note C and r always gonna be identity and at each time k
        g_k, _, _, v_k, bw_k, ba_k, lambd = IMUKalmanFilter.decode_state_b(prev_state)
        for tau in range(0, imu_meas.size(1) - 1):
            t, gyro_meas, accel_meas = data_loader.SubseqDataset.decode_imu_data_b(imu_meas[:, tau, :])
            tp1, _, _ = data_loader.SubseqDataset.decode_imu_data_b(imu_meas[:, tau + 1, :])
            dt = tp1 - t
        
            t_accum, C_accum, r_accum, v_accum, pred_covar, _, _, _, _ = \
                self.predict_one_step(t_accum, C_accum, r_accum, v_accum, dt, g_k,
                                      v_k, bw_k, ba_k, pred_covar,
                                      gyro_meas, accel_meas, imu_noise_covar)

            pred_covars.append(pred_covar)
            pred_states.append(IMUKalmanFilter.encode_state_b(g_k,
                                                              C_accum,
                                                              v_k * t_accum - 0.5 * g_k * t_accum * t_accum + r_accum,
                                                              torch.matmul(C_accum.transpose(-2, -1),
                                                                           v_k - g_k * t_accum + v_accum),
                                                              bw_k, ba_k, lambd))
        return pred_states, pred_covars

    def meas_residual_and_jacobi(self, C_pred, r_pred, lambd_pred, vis_meas, T_imu_cam):
        # C_cal = T_imu_cam[:, 0:3, 0:3]
        # C_cal_transpose = C_cal.transpose(-2, -1)
        # r_cal = T_imu_cam[:, 0:3, 3:4]

        mm = torch.matmul
        vis_meas_rot = vis_meas[:, 0:3, :]
        vis_meas_trans = vis_meas[:, 3:6, :]
        # residual_rot = torch_se3.log_SO3_b(mm(mm(mm(torch_se3.exp_SO3_b(vis_meas_rot), C_cal_transpose),
        #                                          C_pred.transpose(-2, -1)), C_cal))
        phi_pred = phi_pred = torch_se3.log_SO3_b(C_pred)
        residual_rot = vis_meas_rot - phi_pred
        residual_trans = vis_meas_trans - lambd_pred * r_pred
        residual = torch.cat([residual_rot, residual_trans], dim=1)

        H = torch.zeros(vis_meas.shape[0], 6, 19, device=vis_meas.device)
        # H[:, 0:3, 3:6] = -mm(mm(torch_se3.J_left_SO3_inv_b(-residual_rot), C_cal_transpose), C_pred)
        H[:, 0:3, 3:6] = -torch_se3.J_left_SO3_inv_b(-phi_pred)
        H[:, 3:6, 6:9] = -torch.eye(3, device=vis_meas.device) * lambd_pred
        H[:, 3:6, 18:19] = -r_pred

        return residual, H

    def update(self, pred_state, pred_covar, vis_meas, vis_meas_covar, T_imu_cam):
        mm = torch.matmul
        g_pred, C_pred, r_pred, v_pred, bw_pred, ba_pred, lambd_pred = IMUKalmanFilter.decode_state_b(pred_state)
        residual, H = self.meas_residual_and_jacobi(C_pred, r_pred, lambd_pred, vis_meas, T_imu_cam)

        H = -H  # this is required for EKF, since the way we derived the Jacobian are for batch methods
        H_transpose = H.transpose(-2, -1)
        S = mm(mm(H, pred_covar), H_transpose) + vis_meas_covar
        K = mm(mm(pred_covar, H_transpose), S.inverse())

        est_error = mm(K, residual)

        I19 = torch.eye(19, 19, device=pred_state.device).repeat(vis_meas.size(0), 1, 1)
        est_covar = mm(I19 - mm(K, H), pred_covar)


        g_err = est_error[:, 0:3]
        C_err = est_error[:, 3:6]
        r_err = est_error[:, 6:9]
        v_err = est_error[:, 9:12]
        bw_err = est_error[:, 12:15]
        ba_err = est_error[:, 15:18]
        lambd_err = est_error[:, 18:19]

        est_state = IMUKalmanFilter.encode_state_b(g_pred + g_err,
                                                   mm(C_pred, torch_se3.exp_SO3_b(C_err)),
                                                   r_pred + r_err,
                                                   v_pred + v_err,
                                                   bw_pred + bw_err,
                                                   ba_pred + ba_err,
                                                   lambd_pred + lambd_err)
        return est_state, est_covar

    def composition(self, prev_pose, est_state, est_covar):
        batch_size = est_state.size(0)
        g, C, r, v, bw, ba, lambd = IMUKalmanFilter.decode_state_b(est_state)
        C_transpose = C.transpose(-2, -1)

        new_pose = torch.eye(4, 4, device=prev_pose.device).repeat(batch_size, 1, 1)
        new_pose[:, 0:3, 0:3] = torch.matmul(C_transpose, prev_pose[:, 0:3, 0:3])
        new_pose[:, 0:3, 3:4] = torch.matmul(C_transpose, prev_pose[:, 0:3, 3:4] - r)
        new_g = torch.matmul(C_transpose, g)

        new_state = IMUKalmanFilter.encode_state_b(new_g, C, r, v, bw, ba, lambd)
        U = torch.eye(19, 19, device=prev_pose.device).repeat(batch_size, 1, 1)
        U[:, 0:3, 0:3] = C_transpose
        U[:, 0:3, 3:6] = torch_se3.skew3_b(new_g)
        new_covar = torch.matmul(torch.matmul(U, est_covar), U.transpose(-2, -1))
        new_covar = self.force_symmetrical(new_covar)

        return new_pose, new_state, new_covar

    def forward(self, imu_data, imu_noise_covar,
                prev_pose, prev_state, prev_covar,
                vis_meas, vis_meas_covar, T_imu_cam):

        num_timesteps = vis_meas.size(1)  # equals to imu_data.size(1) - 1

        poses_over_timesteps = [prev_pose]
        states_over_timesteps = [prev_state]
        covars_over_timesteps = [prev_covar]
        temp_state = []
        for k in range(0, num_timesteps):
            pred_states, pred_covars = self.predict(imu_data[:, k], imu_noise_covar,
                                                    states_over_timesteps[-1], covars_over_timesteps[-1])
            temp_state.append(pred_states[-1])
            for _ in range(12):
                est_state, est_covar = self.update(temp_state[-1], pred_covars[-1],
                                               vis_meas[:, k], vis_meas_covar[:, k], T_imu_cam)
                temp_state.append(est_state)
            new_pose, new_state, new_covar = self.composition(poses_over_timesteps[-1], temp_state[-1], est_covar)

            poses_over_timesteps.append(new_pose)
            states_over_timesteps.append(new_state)
            covars_over_timesteps.append(new_covar)

        return torch.stack(poses_over_timesteps, 1), \
               torch.stack(states_over_timesteps, 1), \
               torch.stack(covars_over_timesteps, 1)

    @staticmethod
    def decode_state_b(state_vector):
        sz = list(state_vector.shape[:-1])
        g = state_vector[..., 0:3].view(sz + [3, 1])
        C = state_vector[..., 3:12].view(sz + [3, 3])
        r = state_vector[..., 12:15].view(sz + [3, 1])
        v = state_vector[..., 15:18].view(sz + [3, 1])
        bw = state_vector[..., 18:21].view(sz + [3, 1])
        ba = state_vector[..., 21:24].view(sz + [3, 1])
        lambd = state_vector[..., 24].view(sz + [1, 1])

        return g, C, r, v, bw, ba, lambd

    @staticmethod
    def encode_state_b(g, C, r, v, bw, ba, lambd):
        return torch.cat((g.view(-1, 3),
                          C.view(-1, 9), r.view(-1, 3),
                          v.view(-1, 3),
                          bw.view(-1, 3), ba.view(-1, 3), lambd.view(-1, 1)), -1)

    @staticmethod
    def encode_state(g, C, r, v, bw, ba, lambd):
        return torch.squeeze(IMUKalmanFilter.encode_state_b(g, C, r, v, bw, ba, lambd))

    @staticmethod
    def decode_state(state_vector):
        g, C, r, v, bw, ba, lambd = IMUKalmanFilter.decode_state_b(state_vector)
        return g.view(3, 1), C.view(3, 3), r.view(3, 1), v.view(3, 1), bw.view(3, 1), ba.view(3, 1), lambd.view(1)

    @staticmethod
    def state_to_so3(state_vector):
        g, C, r, v, bw, ba, lambd = IMUKalmanFilter.decode_state_b(state_vector)
        phi = torch_se3.log_SO3_b(C)
        return torch.cat((g.view(-1, 3),
                          phi.view(-1, 3), r.view(-1, 3),
                          v.view(-1, 3),
                          bw.view(-1, 3), ba.view(-1, 3), lambd.view(-1, 1)), -1)

class E2EVIO(nn.Module):
    def __init__(self, iters = par.iters):
        super(E2EVIO, self).__init__()
        self.iters = iters
        self.module = RAFT()
        self.regressor = PoseRegressor()
        # self.gru_layer = ConvGRU(input_size=256, hidden_size=256, kernel_size=1, num_layers=1)
        self.imu_noise_covar_weights = torch.nn.Linear(1, 4, bias=False)
        self.vis_scale = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad= True)
        if not par.train_imu_noise_covar:
            for p in self.imu_noise_covar_weights.parameters():
                p.requires_grad = False
            self.imu_noise_covar_weights.weight.data.zero_()
        else:
            self.imu_noise_covar_weights.weight.data /= 10

        self.init_covar_diag_sqrt = nn.Parameter(torch.tensor(np.zeros(18), dtype=torch.float32))
        if not par.train_init_covar:
            self.init_covar_diag_sqrt.requires_grad = False

        if par.fix_vo_weights:
            for param in self.module.parameters():
                param.requires_grad = False

        self.ekf_module = IMUKalmanFilter()

    def get_imu_noise_covar(self):
        covar = 10 ** (par.imu_noise_covar_beta * torch.tanh(par.imu_noise_covar_gamma * self.imu_noise_covar_weights(
                torch.ones(1, device=self.imu_noise_covar_weights.weight.device))))

        imu_noise_covar_diag = torch.tensor(par.imu_noise_covar_diag, dtype=torch.float32,
                                            device=self.imu_noise_covar_weights.weight.device).repeat_interleave(3) * \
                               torch.stack([covar[0], covar[0], covar[0],
                                            covar[1], covar[1], covar[1],
                                            covar[2], covar[2], covar[2],
                                            covar[3], covar[3], covar[3]])
        return torch.diag(imu_noise_covar_diag)
    
    def forward(self, images, imu_data, prev_pose, prev_state, prev_covar, T_imu_cam):
        vis_meas_covar_scale = torch.ones(6, device=images.device)
        vis_meas_covar_scale[0:3] = vis_meas_covar_scale[0:3] * par.k4
        imu_noise_covar = self.get_imu_noise_covar()

        if prev_covar is None:
            init_covar_diag_sqrt = torch.tensor(par.init_covar_diag_sqrt, dtype=torch.float32, device=images.device)
            prev_covar = torch.diag(init_covar_diag_sqrt * init_covar_diag_sqrt +
                                    par.init_covar_diag_eps).repeat(images.shape[0], 1, 1)
        # fmap1, fmap2, cnet = self.module.encode_image(images)
        images_seq = torch.cat((images[:, :-1], images[:, 1:]), dim=2)
        coords0, coords1 = self.module.initialize_flow(images[:,0,:])
        num_timesteps = images.size(1) - 1  # equals to imu_data.size(1) - 1

        poses_over_timesteps = [prev_pose]
        states_over_timesteps = [prev_state]
        covars_over_timesteps = [prev_covar]
        vis_meas_over_timesteps = []
        vis_meas_covar_over_timesteps = []

        # abs_pose_from_rel_over_timesteps = [prev_pose]
        for k in range(0, num_timesteps):

            pred_states, pred_covars = self.ekf_module.predict(imu_data[:, k], imu_noise_covar,
                                                            states_over_timesteps[-1], covars_over_timesteps[-1])

            image_pair = images_seq[:,k]
            corr_fn, net, inp = self.module.prepare(image1=image_pair[:,:3,:], image2=image_pair[:,3:,:])

            temp_state_dict = [pred_states[-1]]
            temp_vis_dict = []
            for itr in range(self.iters):
                coords1, net = self.module.update(coords0, coords1, corr_fn, net, inp)
                # F(t+1) = F(t) + \Delta(t)

                vis_meas_and_covar = self.regressor((coords1 - coords0).view(coords1.size(0),-1))
                vis_meas = vis_meas_and_covar[:,:6]
                temp_vis_dict.append(vis_meas)
            # vis_meas_mat = euler_to_matrix(vis_meas)
            # abs_pose_rel = torch.matmul(abs_pose_from_rel_over_timesteps[-1],vis_meas_mat)

                vis_meas_covar_diag = par.vis_meas_covar_init_guess * \
                                    10 ** (par.vis_meas_covar_beta *
                                            torch.tanh(par.vis_meas_covar_gamma * vis_meas_and_covar[:, 6:]))
                vis_meas_covar_scaled = torch.diag_embed(vis_meas_covar_diag / vis_meas_covar_scale.view(1, 6))
                # vis_meas_covar = torch.diag_embed(vis_meas_covar_diag)

        # ekf correct
                est_state, est_covar = self.ekf_module.update(temp_state_dict[-1], pred_covars[-1],
                                                            vis_meas.unsqueeze(-1),
                                                            vis_meas_covar_scaled,
                                                            T_imu_cam)
                temp_state_dict.append(est_state)
                # temp_covar_dict.append(est_covar)
            vis_meas_one_timestep = torch.stack(temp_vis_dict, dim=1) # B x Iter x 6
            new_pose, new_state, new_covar = self.ekf_module.composition(poses_over_timesteps[-1],
                                                                        temp_state_dict[-1], est_covar)

            poses_over_timesteps.append(new_pose)
            states_over_timesteps.append(new_state)
            covars_over_timesteps.append(new_covar)
            # vis_meas_over_timesteps.append(vis_meas)
            vis_meas_over_timesteps.append(vis_meas_one_timestep)  # B x Seq X Iter x 6
            vis_meas_covar_over_timesteps.append(vis_meas_covar_diag)
            # abs_pose_from_rel_over_timesteps.append(abs_pose_rel)

        return torch.stack(vis_meas_over_timesteps, 1), \
            torch.stack(vis_meas_covar_over_timesteps, 1), \
            torch.stack(poses_over_timesteps, 1), \
            torch.stack(states_over_timesteps, 1), \
            torch.stack(covars_over_timesteps, 1),\
            # torch.stack(abs_pose_from_rel_over_timesteps, 1)