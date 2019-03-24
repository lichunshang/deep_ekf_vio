import torch
import torch.nn as nn
import numpy as np
import data_loader
import time
from params import par
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)  # , inplace=True)
        )


class TorchSE3(object):
    @staticmethod
    def exp_SO3(phi):
        phi_norm = torch.norm(phi)

        if phi_norm > 1e-8:
            unit_phi = phi / phi_norm
            unit_phi_skewed = TorchSE3.skew3(unit_phi)
            C = torch.eye(3, 3, device=phi.device) + torch.sin(phi_norm) * unit_phi_skewed + \
                (1 - torch.cos(phi_norm)) * torch.mm(unit_phi_skewed, unit_phi_skewed)
        else:
            phi_skewed = TorchSE3.skew3(phi)
            C = torch.eye(3, 3, device=phi.device) + phi_skewed + 0.5 * torch.mm(phi_skewed, phi_skewed)

        return C

    # assumes small rotations
    @staticmethod
    def log_SO3(C):
        phi_norm = torch.acos(torch.clamp((torch.trace(C) - 1) / 2, -1.0, 1.0))
        if torch.sin(phi_norm) > 1e-6:
            phi = phi_norm * TorchSE3.unskew3(C - C.transpose(0, 1)) / (2 * torch.sin(phi_norm))
        else:
            phi = 0.5 * TorchSE3.unskew3(C - C.transpose(0, 1))

        return phi

    @staticmethod
    def log_SO3_eigen(C):  # no autodiff
        phi_norm = torch.acos(torch.clamp((torch.trace(C) - 1) / 2, -1.0, 1.0))

        # eig is not very food for C close to identity, will only keep around 3 decimals places
        w, v = torch.eig(C, eigenvectors=True)
        a = torch.tensor([0., 0., 0.], device=C.device)
        for i in range(0, w.size(0)):
            if torch.abs(w[i, 0] - 1.0) < 1e-6 and torch.abs(w[i, 1] - 0.0) < 1e-6:
                a = v[:, i]

        assert (torch.abs(torch.norm(a) - 1.0) < 1e-6)

        if torch.allclose(TorchSE3.exp_SO3(phi_norm * a), C, atol=1e-3):
            return phi_norm * a
        elif torch.allclose(TorchSE3.exp_SO3(-phi_norm * a), C, atol=1e-3):
            return -phi_norm * a
        else:
            raise ValueError("Invalid logarithmic mapping")

    @staticmethod
    def skew3(v):
        m = torch.zeros(3, 3, device=v.device)
        m[0, 1] = -v[2]
        m[0, 2] = v[1]
        m[1, 0] = v[2]

        m[1, 2] = -v[0]
        m[2, 0] = -v[1]
        m[2, 1] = v[0]

        return m

    @staticmethod
    def unskew3(m):
        return torch.stack([m[2, 1], m[0, 2], m[1, 0]])

    @staticmethod
    def J_left_SO3_inv(phi):
        phi = phi.view(3, 1)
        phi_norm = torch.norm(phi)
        if torch.abs(phi_norm) > 1e-6:
            a = phi / phi_norm
            cot_half_phi_norm = 1.0 / torch.tan(phi_norm / 2)
            J_inv = (phi_norm / 2) * cot_half_phi_norm * torch.eye(3, 3, device=phi.device) + \
                    (1 - (phi_norm / 2) * cot_half_phi_norm) * \
                    torch.mm(a, a.transpose(0, 1)) - (phi_norm / 2) * TorchSE3.skew3(a)
        else:
            J_inv = torch.eye(3, 3, device=phi.device) - 0.5 * TorchSE3.skew3(phi)
        return J_inv

    @staticmethod
    def J_left_SO3(phi):
        phi = phi.view(3, 1)
        phi_norm = torch.norm(phi)
        if torch.abs(phi_norm) > 1e-6:
            a = phi / phi_norm
            J = (torch.sin(phi_norm) / phi_norm) * torch.eye(3, 3, device=phi.device) + \
                (1 - (torch.sin(phi_norm) / phi_norm)) * torch.mm(a, a.transpose(0, 1)) + \
                ((1 - torch.cos(phi_norm)) / phi_norm) * TorchSE3.skew3(a)
        else:
            J = torch.eye(3, 3, device=phi.device) + 0.5 * TorchSE3.skew3(phi)
        return J


class IMUKalmanFilter(nn.Module):
    def __init__(self):
        super(IMUKalmanFilter, self).__init__()

    def force_symmetrical(self, M):
        M_upper = torch.triu(M)
        return M_upper + M_upper.transpose(0, 1) * (1 - torch.eye(*M_upper.size(), device=M.device))

    def predict_one_step(self, t_accum, C_accum, r_accum, v_accum, dt, g_k, v_k, bw_k, ba_k, covar,
                         gyro_meas, accel_meas, imu_noise_covar):
        dt2 = dt * dt
        w = gyro_meas - bw_k
        w_skewed = TorchSE3.skew3(w)
        C_accum_transpose = C_accum.transpose(0, 1)
        a = accel_meas - ba_k
        v = torch.mm(C_accum_transpose, v_k - g_k * t_accum + v_accum)
        v_skewed = TorchSE3.skew3(v)
        I3 = torch.eye(3, 3, device=covar.device)
        exp_int_w = TorchSE3.exp_SO3(dt * w)

        # propagate uncertainty, 2nd order
        F = torch.zeros(18, 18, device=covar.device)
        F[3:6, 3:6] = -w_skewed
        F[3:6, 12:15] = -I3
        F[6:9, 3:6] = -torch.mm(C_accum, v_skewed)
        F[6:9, 9:12] = C_accum
        F[9:12, 0:3] = -C_accum_transpose
        F[9:12, 3:6] = -TorchSE3.skew3(torch.mm(C_accum_transpose, g_k))
        F[9:12, 9:12] = -w_skewed
        F[9:12, 12:15] = -v_skewed
        F[9:12, 15:18] = -I3

        G = torch.zeros(18, 12, device=covar.device)
        G[3:6, 0:3] = -I3
        G[9:12, 0:3] = -v_skewed
        G[9:12, 6:9] = -I3
        G[12:15, 3:6] = I3
        G[15:18, 9:12] = I3

        mm = torch.mm
        Phi = torch.eye(18, 18, device=covar.device) + F * dt + 0.5 * mm(F, F) * dt2
        Phi[6:9, 12:15] = torch.zeros(3, 3, device=covar.device)  # this blocks is exactly zero in 2nd order approx
        Phi[3:6, 3:6] = exp_int_w.transpose(0, 1)
        Phi[9:12, 9:12] = exp_int_w.transpose(0, 1)

        Q = mm(mm(mm(mm(Phi, G), imu_noise_covar), G.transpose(0, 1)), Phi.transpose(0, 1)) * dt
        covar = mm(mm(Phi, covar), Phi.transpose(0, 1)) + Q
        covar = self.force_symmetrical(covar)

        # propagate nominal states
        r_accum = r_accum + v_accum * dt + 0.5 * torch.mm(C_accum, (dt2 * a))
        v_accum = v_accum + torch.mm(C_accum, (dt * a))
        C_accum = torch.mm(C_accum, exp_int_w)
        t_accum = t_accum + dt

        return t_accum, C_accum, r_accum, v_accum, covar, F, G, Phi, Q

    def predict(self, imu_meas, imu_noise_covar, prev_state, prev_covar):
        C_accum = torch.eye(3, 3, device=imu_meas.device)
        r_accum = torch.zeros(3, 1, device=imu_meas.device)
        v_accum = torch.zeros(3, 1, device=imu_meas.device)
        t_accum = torch.tensor(0.0, device=imu_meas.device)
        pred_covar = prev_covar

        g_k, C_k, r_k, v_k, bw_k, ba_k = IMUKalmanFilter.decode_state(prev_state)
        for tau in range(0, len(imu_meas) - 1):
            t, gyro_meas, accel_meas = data_loader.SubseqDataset.decode_imu_data(imu_meas[tau, :])
            tp1, _, _ = data_loader.SubseqDataset.decode_imu_data(imu_meas[tau + 1, :])
            dt = tp1 - t
            t_accum, C_accum, r_accum, v_accum, pred_covar, _, _, _, _ = \
                self.predict_one_step(t_accum, C_accum, r_accum, v_accum, dt, g_k, v_k, bw_k, ba_k, pred_covar,
                                      gyro_meas, accel_meas, imu_noise_covar)

        pred_state = IMUKalmanFilter.encode_state(g_k,
                                                  torch.mm(C_k, C_accum),
                                                  r_k + v_k * t_accum - 0.5 * g_k * t_accum * t_accum + r_accum,
                                                  torch.mm(C_accum.transpose(0, 1), v_k - g_k * t_accum + v_accum),
                                                  bw_k, ba_k)

        return pred_state, pred_covar

    def meas_residual_and_jacobi(self, C_pred, r_pred, vis_meas, T_imu_cam):
        C_cal = T_imu_cam[0:3, 0:3]
        C_cal_transpose = C_cal.transpose(0, 1)
        r_cal = T_imu_cam[0:3, 3].view(3, 1)

        mm = torch.mm
        se3 = TorchSE3
        vis_meas_rot = vis_meas[0:3, :]
        vis_meas_trans = vis_meas[3:6, :]
        residual_rot = se3.log_SO3(mm(mm(mm(se3.exp_SO3(vis_meas_rot), C_cal_transpose),
                                         C_pred.transpose(0, 1)), C_cal))
        residual_trans = vis_meas_trans - mm(mm(C_cal_transpose, C_pred), r_cal) - \
                         mm(C_cal_transpose, r_pred - r_cal)
        residual = torch.cat([residual_rot.view(3, 1), residual_trans], dim=0)

        H = torch.zeros(6, 18, device=vis_meas.device)
        H[0:3, 3:6] = -mm(mm(se3.J_left_SO3_inv(-residual_rot), C_cal_transpose), C_pred)
        H[3:6, 3:6] = mm(mm(C_cal_transpose, C_pred), se3.skew3(r_cal))
        H[3:6, 6:9] = -C_cal_transpose

        return residual, H

    def update(self, pred_state, pred_covar, vis_meas, vis_meas_covar, T_imu_cam):
        mm = torch.mm
        g_pred, C_pred, r_pred, v_pred, bw_pred, ba_pred = IMUKalmanFilter.decode_state(pred_state)
        residual, H = self.meas_residual_and_jacobi(C_pred, r_pred, vis_meas, T_imu_cam)

        H = -H  # this is required for EKF, since the way we derived the Jacobian are for batch methods
        H_transpose = H.transpose(0, 1)

        S = mm(mm(H, pred_covar), H_transpose) + vis_meas_covar
        K = mm(mm(pred_covar, H_transpose), S.inverse())

        est_error = mm(K, residual)

        I18 = torch.eye(18, 18, device=pred_state.device)
        est_covar = mm(I18 - mm(K, H), pred_covar)

        g_err = est_error[0:3]
        C_err = est_error[3:6]
        r_err = est_error[6:9]
        v_err = est_error[9:12]
        bw_err = est_error[12:15]
        ba_err = est_error[15:18]

        est_state = IMUKalmanFilter.encode_state(g_pred + g_err,
                                                 mm(C_pred, TorchSE3.exp_SO3(C_err)),
                                                 r_pred + r_err,
                                                 v_pred + v_err,
                                                 bw_pred + bw_err,
                                                 ba_pred + ba_err)
        return est_state, est_covar

    def composition(self, prev_pose, est_state, est_covar):
        g, C, r, v, bw, ba = IMUKalmanFilter.decode_state(est_state)
        C_transpose = C.transpose(0, 1)

        new_pose = torch.eye(4, 4, device=prev_pose.device)
        new_pose[0:3, 0:3] = torch.mm(C_transpose, prev_pose[0:3, 0:3])
        new_pose[0:3, 3] = torch.mm(C_transpose, prev_pose[0:3, 3].view(3, 1) - r).view(3)
        new_g = torch.mm(C_transpose, g)

        new_state = IMUKalmanFilter.encode_state(new_g,
                                                 torch.eye(3, 3, device=prev_pose.device),
                                                 torch.zeros(3, device=prev_pose.device),
                                                 v, bw, ba)
        U = torch.zeros(18, 18, device=prev_pose.device)
        U[0:3, 0:3] = C_transpose
        U[0:3, 3:6] = TorchSE3.skew3(new_g)
        U[9:18, 9:18] = torch.eye(9, 9, device=prev_pose.device)
        new_covar = torch.mm(torch.mm(U, est_covar), U.transpose(0, 1))
        new_covar = self.force_symmetrical(new_covar)

        return new_pose, new_state, new_covar

    def forward(self, imu_data_idxs, imu_data, imu_noise_covar,
                prev_pose, prev_state, prev_covar,
                vis_meas, vis_meas_covar, T_imu_cam):

        num_batches = vis_meas.size(0)
        num_timesteps = vis_meas.size(1)

        poses_over_batches = []
        states_over_batches = []
        covars_over_batches = []
        for i in range(0, num_batches):
            poses_over_timesteps = [prev_pose[i]]
            states_over_timesteps = [prev_state[i]]
            covars_over_timesteps = [prev_covar[i]]
            for j in range(0, num_timesteps):
                imu_meas = imu_data[i, imu_data_idxs[i, j]:imu_data_idxs[i, j + 1]]
                pred_state, pred_covar = self.predict(imu_meas, imu_noise_covar,
                                                      states_over_timesteps[-1], covars_over_timesteps[-1])
                est_state, est_covar = self.update(pred_state, pred_covar,
                                                   vis_meas[i, j].view(6, 1), vis_meas_covar[i, j], T_imu_cam[i])
                new_pose, new_state, new_covar = self.composition(poses_over_timesteps[-1], est_state, est_covar)

                poses_over_timesteps.append(new_pose)
                states_over_timesteps.append(new_state)
                covars_over_timesteps.append(new_covar)

            poses_over_batches.append(torch.stack(poses_over_timesteps))
            states_over_batches.append(torch.stack(states_over_timesteps))
            covars_over_batches.append(torch.stack(covars_over_timesteps))

        return torch.stack(poses_over_batches), \
               torch.stack(states_over_batches), \
               torch.stack(covars_over_batches)

    @staticmethod
    def decode_error_state(state_vector):
        g = state_vector[0:3]
        C = state_vector[3:6]
        r = state_vector[6:9]
        v = state_vector[9:12]
        bw = state_vector[12:15]
        ba = state_vector[15:18]

        return g, C, r, v, bw, ba

    @staticmethod
    def decode_state(state_vector):
        g = state_vector[0:3].view(3, 1)
        C = state_vector[3:12].view(3, 3)
        r = state_vector[12:15].view(3, 1)
        v = state_vector[15:18].view(3, 1)
        bw = state_vector[18:21].view(3, 1)
        ba = state_vector[21:24].view(3, 1)

        return g, C, r, v, bw, ba

    @staticmethod
    def encode_state(g, C, r, v, bw, ba):
        return torch.cat((g.view(3), C.view(9), r.view(3), v.view(3), bw.view(3), ba.view(3),))


class DeepVO(nn.Module):
    def __init__(self, imsize1, imsize2, batchNorm):
        super(DeepVO, self).__init__()
        # CNN
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
        self.conv3_1 = conv(self.batchNorm, 256, 256, kernel_size=3, stride=1, dropout=par.conv_dropout[3])
        self.conv4 = conv(self.batchNorm, 256, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        self.conv4_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[5])
        self.conv5 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[6])
        self.conv5_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[7])
        self.conv6 = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=par.conv_dropout[8])
        # Compute the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self.encode_image(__tmp)

        # RNN
        self.rnn = nn.LSTM(
                input_size=int(np.prod(__tmp.size())),
                hidden_size=par.rnn_hidden_size,
                num_layers=par.rnn_num_layers,
                dropout=par.rnn_dropout_between,
                batch_first=True)
        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)
        self.linear = nn.Linear(in_features=par.rnn_hidden_size, out_features=6)

        # Initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)  # orthogonal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(m.weight_ih_l1)  # orthogonal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l1.data[start:end].fill_(1.)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, lstm_init_state=None):
        # x: (batch, seq_len, channel, width, height)
        # stack_image
        x = torch.cat((x[:, :-1], x[:, 1:]), dim=2)
        batch_size = x.size(0)
        seq_len = x.size(1)
        # CNN
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)

        # lstm_init_state has the dimension of (# batch, 2 (hidden/cell), lstm layers, lstm hidden size)
        if lstm_init_state is not None:
            hidden_state = lstm_init_state[:, 0, :, :].permute(1, 0, 2).contiguous()
            cell_state = lstm_init_state[:, 1, :, :].permute(1, 0, 2).contiguous()
            lstm_init_state = (hidden_state, cell_state,)

        # RNN
        # lstm_state is (hidden state, cell state,)
        # each hidden/cell state has the shape (lstm layers, batch size, lstm hidden size)
        out, lstm_state = self.rnn(x, lstm_init_state)
        out = self.rnn_drop_out(out)
        out = self.linear(out)

        # rearrange the shape back to (# batch, 2 (hidden/cell), lstm layers, lstm hidden size)
        lstm_state = torch.stack(lstm_state, dim=0)
        lstm_state = lstm_state.permute(2, 0, 1, 3)

        return out, lstm_state

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class E2EVIO(nn.Module):
    def __init__(self):
        super(E2EVIO, self).__init__()

        self.vo_module = DeepVO(par.img_h, par.img_w, par.batch_norm)

        self.imu_noise_covar_diag_sqrt = nn.Parameter(torch.tensor(par.imu_noise_covar_diag_sqrt, dtype=torch.float32))
        if not par.train_imu_noise_covar:
            self.imu_noise_covar_diag_sqrt.require_grad = False

        self.init_covar_diag_sqrt = nn.Parameter(torch.tensor(par.init_covar_diag_sqrt, dtype=torch.float32))
        if not par.train_init_covar:
            self.init_covar_diag_sqrt.require_grad = False

        self.ekf_module = IMUKalmanFilter()

    def forward(self, images, imu_data_idxs, imu_data, prev_lstm_states, prev_pose, prev_state, T_imu_cam):
        vis_meas, lstm_states = self.vo_module.forward(images, lstm_init_state=prev_lstm_states)

        if par.vis_meas_covar_use_fixed:
            vis_meas_covar = torch.diag(torch.tensor(par.vis_meas_fixed_covar, dtype=torch.float32)). \
                repeat(vis_meas.shape[0],
                       vis_meas.shape[1], 1, 1).cuda()

        if not par.enable_ekf:
            return vis_meas, vis_meas_covar, lstm_states, None, None, None

        imu_noise_covar = torch.diag(self.imu_noise_covar_diag_sqrt * self.imu_noise_covar_diag_sqrt +
                                     par.imu_noise_covar_diag_eps)
        init_covar = torch.diag(self.init_covar_diag_sqrt * self.init_covar_diag_sqrt +
                                par.init_covar_diag_eps).repeat(vis_meas.shape[0], 1, 1)

        poses, ekf_states, ekf_covars = self.ekf_module.forward(imu_data_idxs, imu_data, imu_noise_covar,
                                                                prev_pose, prev_state, init_covar,
                                                                vis_meas, vis_meas_covar, T_imu_cam)

        return vis_meas, vis_meas_covar, lstm_states, poses, ekf_states, ekf_covars
