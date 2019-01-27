from log import logger
from params import par
import torch
import torch.nn.functional


class Trainer(object):

    def __init__(self, model):
        self.model = model
        self.num_train_iterations = 0
        self.num_val_iterations = 0
        self.clip = par.clip

    def get_loss(self, x, y):
        predicted = self.model.forward(x)

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

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm(self.model.rnn.parameters(), self.clip)
        optimizer.step()
        return loss