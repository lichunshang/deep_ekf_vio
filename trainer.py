from log import logger
from params import par
import torch
import torch.nn.functional
from data_helper import ImageSequenceDataset


class Trainer(object):

    def __init__(self, model):
        self.model = model
        self.num_train_iterations = 0
        self.num_val_iterations = 0
        self.clip = par.clip
        self.lstm_state_cache = {}

    def update_lstm_state(self, t_x_meta, lstm_states):
        _, seq_list, type_list, id_list = ImageSequenceDataset.decode_batch_meta_info(t_x_meta)
        # lstm_states = (hidden state, cell state)

        assert (len(seq_list) == len(type_list) and
                len(seq_list) == len(id_list) and
                len(seq_list) == lstm_states[0].size(1) and
                len(seq_list) == lstm_states[1].size(1))
        num_batches = len(seq_list)

        for i in range(0, num_batches):
            key = "%s_%s_%d" % (seq_list[i], type_list[i], id_list[i][-1])
            self.lstm_state_cache[key] = (lstm_states[0][:, i, :], lstm_states[1][:, i, :],)

    def retrieve_lstm_state(self, t_x_meta):
        _, seq_list, type_list, id_list = ImageSequenceDataset.decode_batch_meta_info(t_x_meta)
        assert (len(seq_list) == len(type_list) and len(seq_list) == len(id_list))
        num_batches = len(seq_list)

        lstm_hidden_states = []
        lstm_cell_states = []

        for i in range(0, num_batches):
            key = "%s_%s_%d" % (seq_list[i], type_list[i], id_list[i][0])
            if key in self.lstm_state_cache:
                tmp = self.lstm_state_cache[key]
            else:
                num_layers = self.model.rnn.num_layers
                hidden_size = self.model.rnn.hidden_size
                tmp = [torch.zeros([num_layers, hidden_size]), torch.zeros([num_layers, hidden_size])]
            lstm_hidden_states.append(tmp[0])
            lstm_cell_states.append(tmp[1])

        return [torch.stack(lstm_hidden_states, dim=1), torch.stack(lstm_cell_states, dim=1), ]

    def get_loss(self, t_x_meta, x, y):
        prev_lstm_states = None
        if par.stateful_training:
            prev_lstm_states = self.retrieve_lstm_state(t_x_meta)
            prev_lstm_states[0] = prev_lstm_states[0].cuda()
            prev_lstm_states[1] = prev_lstm_states[1].cuda()

        predicted, lstm_states = self.model.forward(x, prev_lstm_states)

        if par.stateful_training:
            lstm_states = list(lstm_states)
            lstm_states[0] = lstm_states[0].detach().cpu()
            lstm_states[1] = lstm_states[1].detach().cpu()
            self.update_lstm_state(t_x_meta, lstm_states)

        # Weighted MSE Loss
        angle_loss = torch.nn.functional.mse_loss(predicted[:, :, 3:6], y[:, :, 3:6])
        trans_loss = torch.nn.functional.mse_loss(predicted[:, :, 0:3], y[:, :, 0:3])
        loss = (100 * angle_loss + trans_loss)

        # log the loss
        loss_name = "train_loss" if self.model.training else "val_loss"
        iterations = self.num_train_iterations if self.model.training else self.num_val_iterations
        trans_x_loss = torch.nn.functional.mse_loss(predicted[:, :, 0], y[:, :, 0])
        trans_y_loss = torch.nn.functional.mse_loss(predicted[:, :, 1], y[:, :, 1])
        trans_z_loss = torch.nn.functional.mse_loss(predicted[:, :, 2], y[:, :, 2])
        rot_x_loss = torch.nn.functional.mse_loss(predicted[:, :, 3], y[:, :, 3])
        rot_y_loss = torch.nn.functional.mse_loss(predicted[:, :, 4], y[:, :, 4])
        rot_z_loss = torch.nn.functional.mse_loss(predicted[:, :, 5], y[:, :, 5])
        logger.tensorboard.add_scalar(loss_name + "/total_loss", loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/rot_loss", angle_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/rot_loss/x", rot_x_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/rot_loss/y", rot_y_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/rot_loss/z", rot_z_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/trans_loss", trans_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/trans_loss/x", trans_x_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/trans_loss/y", trans_y_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/trans_loss/z", trans_z_loss, iterations)

        if self.model.training:
            self.num_train_iterations += 1
        else:
            self.num_val_iterations += 1

        return loss

    def step(self, t_x_meta, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(t_x_meta, x, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm(self.model.rnn.parameters(), self.clip)
        optimizer.step()
        return loss
