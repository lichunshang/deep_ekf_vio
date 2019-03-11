import torch
import torch.nn as nn
import numpy as np
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

    def predict_one_step(self, dt, accel_meas, gyro_meas, prev_state, prev_covar):
        pass

    def predict(self, imu_meas, prev_state, prev_covar):
        # phi_matrix = []
        for tau in range(0, len(imu_meas)):
            pass

    def update(self):
        pass

    def forward(self, imu_meas, prev_state, prev_covar, vis_meas, vis_meas_covar):
        pass


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
