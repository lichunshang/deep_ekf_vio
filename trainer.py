import torch
import torch.nn.functional
import numpy as np
import os
import time
import se3
import torch_se3
from params import par
from model import E2EVIO, IMUKalmanFilter
from data_loader import get_subseqs, SubseqDataset, convert_subseqs_list_to_panda
from log import logger
from torch.utils.data import DataLoader
from eval.kitti_eval_pyimpl import KittiErrorCalc
from eval.gen_trajectory import gen_trajectory_rel_iter, gen_trajectory_abs_iter


class _OnlineDatasetEvaluator(object):
    def __init__(self, model, sequences, eval_length):
        self.model = model  # this is a reference
        self.dataloaders = {}
        self.error_calc = KittiErrorCalc(sequences)
        logger.print("Loading data for the online dataset evaluator...")
        for seq in sequences:
            subseqs = get_subseqs([seq], eval_length, overlap=1, sample_times=1, training=False)
            dataset = SubseqDataset(subseqs, (par.img_h, par.img_w), par.img_means, par.img_stds, par.minus_point_5,
                                    training=False)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
            self.dataloaders[seq] = dataloader

    def evaluate(self):
        if par.enable_ekf:
            return self.evaluate_abs()
        else:
            return self.evaluate_rel()

    def evaluate_rel(self):
        start_time = time.time()
        seqs = sorted(list(self.dataloaders.keys()))
        for seq in seqs:
            predicted_abs_poses = gen_trajectory_rel_iter(self.model, self.dataloaders[seq], True)
            self.error_calc.accumulate_error(seq, predicted_abs_poses)
        ave_err = self.error_calc.get_average_error()
        self.error_calc.clear()
        logger.print("Online evaluation took %.2fs, err %.6f" % (time.time() - start_time, ave_err))

        return ave_err

    def evaluate_abs(self):
        start_time = time.time()
        _, _, est_poses_dict, _, _ = gen_trajectory_abs_iter(self.model, self.dataloaders)
        for k, v in est_poses_dict.items():
            self.error_calc.accumulate_error(k, np.linalg.inv(np.array(v, dtype=np.float64)))
        ave_err = self.error_calc.get_average_error()
        self.error_calc.clear()
        logger.print("Online evaluation abs took %.2fs, err %.6f" % (time.time() - start_time, ave_err))

        return ave_err


class _TrainAssistant(object):
    def __init__(self, model):
        self.model = model
        self.num_train_iterations = 0
        self.num_val_iterations = 0
        self.clip = par.clip
        self.lstm_state_cache = {}
        self.epoch = 0

    def update_lstm_state(self, t_x_meta, lstm_states):
        # lstm_states has the dimension of (# batch, 2 (hidden/cell), lstm layers, lstm hidden size)
        _, seq_list, type_list, _, id_next_list = SubseqDataset.decode_batch_meta_info(t_x_meta)
        assert (len(seq_list) == lstm_states.size(0) and len(seq_list) == lstm_states.size(0))
        num_batches = len(seq_list)

        for i in range(0, num_batches):
            key = "%s_%s_%d" % (seq_list[i], type_list[i], id_next_list[i])
            self.lstm_state_cache[key] = lstm_states[i, :, :, :]

    def retrieve_lstm_state(self, t_x_meta):
        _, seq_list, type_list, id_list, id_next_list = SubseqDataset.decode_batch_meta_info(t_x_meta)
        num_batches = len(seq_list)

        lstm_states = []

        for i in range(0, num_batches):
            key = "%s_%s_%d" % (seq_list[i], type_list[i], id_list[i])
            if key in self.lstm_state_cache:
                tmp = self.lstm_state_cache[key]
            else:
                # This assert only checks "vanilla" sequences for now
                assert (not (self.epoch > 0 and id_list[i] >= par.seq_len - 1 and id_next_list[i] > id_list[i]))
                num_layers = par.rnn_num_layers
                hidden_size = par.rnn_hidden_size
                tmp = torch.zeros(2, num_layers, hidden_size)
            lstm_states.append(tmp)

        return torch.stack(lstm_states, dim=0)

    def get_loss(self, data):
        meta_data, images, imu_data, prev_state, T_imu_cam, gt_poses, gt_rel_poses = data

        prev_lstm_states = None
        if par.stateful_training:
            prev_lstm_states = self.retrieve_lstm_state(meta_data)
            prev_lstm_states = prev_lstm_states.cuda()

        vis_meas, vis_meas_covar, lstm_states, poses, ekf_states, ekf_covars = \
            self.model.forward(images.cuda(),
                               imu_data.cuda(),
                               prev_lstm_states,
                               gt_poses[:, 0].cuda(),
                               prev_state.cuda(), None,
                               T_imu_cam.cuda())

        if par.enable_ekf:
            loss = self.ekf_loss(poses, gt_poses.cuda(), ekf_states, gt_rel_poses.cuda())
        else:
            loss = self.vis_meas_loss(vis_meas, gt_rel_poses.cuda())

        if par.stateful_training:
            lstm_states = lstm_states.detach().cpu()
            self.update_lstm_state(meta_data, lstm_states)

        if self.model.training:
            self.num_train_iterations += 1
        else:
            self.num_val_iterations += 1

        return loss

    def vis_meas_loss(self, predicted_rel_poses, gt_rel_poses):
        # Weighted MSE Loss
        angle_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 0:3], gt_rel_poses[:, :, 0:3])
        trans_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 3:6], gt_rel_poses[:, :, 3:6])
        loss = (100 * angle_loss + trans_loss)

        # log the loss
        loss_name = "train_loss" if self.model.training else "val_loss"
        iterations = self.num_train_iterations if self.model.training else self.num_val_iterations
        rot_x_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 0], gt_rel_poses[:, :, 0])
        rot_y_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 1], gt_rel_poses[:, :, 1])
        rot_z_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 2], gt_rel_poses[:, :, 2])
        trans_x_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 3], gt_rel_poses[:, :, 3])
        trans_y_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 4], gt_rel_poses[:, :, 4])
        trans_z_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 5], gt_rel_poses[:, :, 5])
        logger.tensorboard.add_scalar(loss_name + "/total_loss", loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/rot_loss", angle_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/rot_loss/x", rot_x_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/rot_loss/y", rot_y_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/rot_loss/z", rot_z_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/trans_loss", trans_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/trans_loss/x", trans_x_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/trans_loss/y", trans_y_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/trans_loss/z", trans_z_loss, iterations)

        return loss

    def ekf_loss(self, est_poses, gt_poses, ekf_states, gt_rel_poses):
        abs_errors = torch.matmul(torch.inverse(est_poses), gt_poses)
        # calculate the F norm squared from identity
        # I_minus_angle_errors = torch.eye(3, 3, device=errors.device) - errors[:, :, 0:3, 0:3]
        # I_minus_angle_errors_sq = torch.matmul(I_minus_angle_errors, I_minus_angle_errors.transpose(-2, -1))
        # angle_errors_F_norm_sq = torch.sum(torch.diagonal(I_minus_angle_errors_sq, dim1=-2, dim2=-1), dim=-1)
        abs_angle_errors = torch.squeeze(torch_se3.log_SO3_b(abs_errors[:, :, 0:3, 0:3]), -1)
        abs_angle_errors_sq = torch.sum(abs_angle_errors ** 2, dim=-1)
        abs_trans_errors_sq = torch.sum(abs_errors[:, :, 0:3, 3] ** 2, dim=-1)
        abs_angle_loss = torch.mean(abs_angle_errors_sq)
        abs_trans_loss = torch.mean(abs_trans_errors_sq)

        _, C_rel, r_rel, _, _, _ = IMUKalmanFilter.decode_state_b(ekf_states)
        rel_angle_errors = (gt_rel_poses[:, :, 0:3] - torch.squeeze(torch_se3.log_SO3_b(C_rel[:, 1:]), -1)) ** 2
        rel_angle_errors_sq = torch.sum(rel_angle_errors ** 2, dim=-1)
        rel_trans_error_sq = torch.sum((gt_rel_poses[:, :, 3:6] - torch.squeeze(r_rel[:, 1:], -1)) ** 2, dim=-1)
        rel_angle_loss = torch.mean(rel_angle_errors_sq)
        rel_trans_loss = torch.mean(rel_trans_error_sq)

        loss_abs = (par.k2 * abs_angle_loss + abs_trans_loss)
        loss_rel = (par.k1 * rel_angle_loss + rel_trans_loss)
        loss = par.k3 * loss_rel + (1 - par.k3) * loss_abs

        assert not torch.any(torch.isnan(loss))

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

        loss_name = "train_loss" if self.model.training else "val_loss"
        iterations = self.num_train_iterations if self.model.training else self.num_val_iterations
        logger.tensorboard.add_scalar(loss_name + "/abs_total_loss", loss_abs, iterations)
        logger.tensorboard.add_scalar(loss_name + "/abs_rot_loss", abs_angle_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/last_abs_rot_loss/x", last_rot_x_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/last_abs_rot_loss/y", last_rot_y_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/last_abs_rot_loss/z", last_rot_z_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/abs_trans_loss", abs_trans_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/last_abs_trans_loss/x", last_trans_x_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/last_abs_trans_loss/y", last_trans_y_loss, iterations)
        logger.tensorboard.add_scalar(loss_name + "/last_abs_trans_loss/z", last_trans_z_loss, iterations)

        if self.model.training:
            model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            imu_noise_covar_diag = model.imu_noise_covar_diag_sqrt.detach().cpu() ** 2 + par.imu_noise_covar_diag_eps
            logger.tensorboard.add_scalar("imu_noise_diag/w_x", imu_noise_covar_diag[0], iterations)
            logger.tensorboard.add_scalar("imu_noise_diag/w_y", imu_noise_covar_diag[1], iterations)
            logger.tensorboard.add_scalar("imu_noise_diag/w_z", imu_noise_covar_diag[2], iterations)
            logger.tensorboard.add_scalar("imu_noise_diag/bw_x", imu_noise_covar_diag[3], iterations)
            logger.tensorboard.add_scalar("imu_noise_diag/bw_y", imu_noise_covar_diag[4], iterations)
            logger.tensorboard.add_scalar("imu_noise_diag/bw_z", imu_noise_covar_diag[5], iterations)
            logger.tensorboard.add_scalar("imu_noise_diag/a_x", imu_noise_covar_diag[6], iterations)
            logger.tensorboard.add_scalar("imu_noise_diag/a_y", imu_noise_covar_diag[7], iterations)
            logger.tensorboard.add_scalar("imu_noise_diag/a_z", imu_noise_covar_diag[8], iterations)
            logger.tensorboard.add_scalar("imu_noise_diag/ba_x", imu_noise_covar_diag[9], iterations)
            logger.tensorboard.add_scalar("imu_noise_diag/ba_y", imu_noise_covar_diag[10], iterations)
            logger.tensorboard.add_scalar("imu_noise_diag/ba_z", imu_noise_covar_diag[11], iterations)

        return loss

    def step(self, data, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(data)
        loss.backward()
        if self.clip is not None:
            if isinstance(self.model, torch.nn.DataParallel):
                torch.nn.utils.clip_grad_norm(self.model.module.rnn.parameters(), self.clip)
            else:
                torch.nn.utils.clip_grad_norm(self.model.rnn.parameters(), self.clip)
        optimizer.step()
        return loss


def train(resume_model_path, resume_optimizer_path):
    logger.initialize(working_dir=par.results_dir, use_tensorboard=True)
    logger.print("================ TRAIN ================")

    train_description = input("Enter a description of this training run: ")
    logger.print("Train description: ", train_description)
    logger.tensorboard.add_text("description", train_description)

    logger.log_parameters()
    logger.log_source_files()

    # Prepare Data
    train_subseqs = get_subseqs(par.train_seqs, par.seq_len, overlap=1, sample_times=par.sample_times, training=True)
    convert_subseqs_list_to_panda(train_subseqs).to_pickle(os.path.join(par.results_dir, "train_df.pickle"))
    train_dataset = SubseqDataset(train_subseqs, (par.img_h, par.img_w), par.img_means,
                                  par.img_stds, par.minus_point_5)
    train_dl = DataLoader(train_dataset, batch_size=par.batch_size, shuffle=True, num_workers=par.n_processors,
                          pin_memory=par.pin_mem, drop_last=False)
    logger.print('Number of samples in training dataset: %d' % len(train_subseqs))

    valid_subseqs = get_subseqs(par.valid_seqs, par.seq_len, overlap=1, sample_times=1, training=False)
    convert_subseqs_list_to_panda(valid_subseqs).to_pickle(os.path.join(par.results_dir, "valid_df.pickle"))
    valid_dataset = SubseqDataset(valid_subseqs, (par.img_h, par.img_w), par.img_means,
                                  par.img_stds, par.minus_point_5, training=False)
    valid_dl = DataLoader(valid_dataset, batch_size=par.batch_size, shuffle=False, num_workers=par.n_processors,
                          pin_memory=par.pin_mem, drop_last=False)
    logger.print('Number of samples in validation dataset: %d' % len(valid_subseqs))

    # Model
    e2e_vio_model = E2EVIO()
    e2e_vio_model = e2e_vio_model.cuda()
    online_evaluator = _OnlineDatasetEvaluator(e2e_vio_model, par.valid_seqs, 50)

    # Load FlowNet weights pretrained with FlyingChairs
    # NOTE: the pretrained model assumes image rgb values in range [-0.5, 0.5]
    if par.pretrained_flownet and not resume_model_path:
        pretrained_w = torch.load(par.pretrained_flownet)
        logger.print('Load FlowNet pretrained model')
        # Use only conv-layer-part of FlowNet as CNN for DeepVO
        vo_model_dict = e2e_vio_model.vo_module.state_dict()
        update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in vo_model_dict}
        assert (len(update_dict) > 0)
        vo_model_dict.update(update_dict)
        e2e_vio_model.vo_module.load_state_dict(vo_model_dict)

    # Create optimizer
    optimizer = par.optimizer(e2e_vio_model.parameters(), **par.optimizer_args)

    # Load trained DeepVO model and optimizer
    if resume_model_path:
        state_dict_update = logger.clean_state_dict_key(torch.load(resume_model_path))
        state_dict_update = {key: state_dict_update[key] for key in state_dict_update
                             if key not in par.exclude_resume_weights}
        state_dict = e2e_vio_model.state_dict()
        state_dict.update(state_dict_update)
        e2e_vio_model.load_state_dict(state_dict)
        logger.print('Load model from: %s' % resume_model_path)
        if resume_optimizer_path:
            optimizer.load_state_dict(torch.load(resume_optimizer_path))
            logger.print('Load optimizer from: %s' % resume_optimizer_path)

    # if to use more than one GPU
    if par.n_gpu > 1:
        assert (torch.cuda.device_count() == par.n_gpu)
        e2e_vio_model = torch.nn.DataParallel(e2e_vio_model)

    e2e_vio_ta = _TrainAssistant(e2e_vio_model)

    # Train
    min_loss_t = 1e10
    min_loss_v = 1e10
    min_err_eval = 1e10
    for epoch in range(par.epochs):
        e2e_vio_ta.epoch = epoch
        st_t = time.time()
        logger.print('=' * 50)
        # Train
        e2e_vio_model.train()
        loss_mean = 0
        t_loss_list = []
        count = 0
        for data in train_dl:
            print("%d/%d (%.2f%%)" % (count, len(train_dl), 100 * count / len(train_dl)), end='\r')
            ls = e2e_vio_ta.step(data, optimizer).data.cpu().numpy()
            t_loss_list.append(float(ls))
            loss_mean += float(ls)
            count += 1
        logger.print('Train take {:.1f} sec'.format(time.time() - st_t))
        loss_mean /= len(train_dl)
        logger.tensorboard.add_scalar("train_loss/epochs", loss_mean, epoch)

        # Validation
        st_t = time.time()
        e2e_vio_model.eval()
        loss_mean_valid = 0
        v_loss_list = []
        for data in valid_dl:
            v_ls = e2e_vio_ta.get_loss(data).data.cpu().numpy()
            v_loss_list.append(float(v_ls))
            loss_mean_valid += float(v_ls)
        logger.print('Valid take {:.1f} sec'.format(time.time() - st_t))
        loss_mean_valid /= len(valid_dl)
        logger.tensorboard.add_scalar("val_loss/epochs", loss_mean_valid, epoch)

        logger.print('Epoch {}\ntrain loss mean: {}, std: {}\nvalid loss mean: {}, std: {}\n'.
                     format(epoch + 1, loss_mean, np.std(t_loss_list), loss_mean_valid, np.std(v_loss_list)))

        err_eval = online_evaluator.evaluate()
        logger.tensorboard.add_scalar("eval_loss/epochs", err_eval, epoch)

        # Save model
        if (epoch + 1) % 5 == 0:
            logger.log_training_state("checkpoint", epoch + 1, e2e_vio_model.state_dict(), optimizer.state_dict())
        if loss_mean_valid < min_loss_v:
            min_loss_v = loss_mean_valid
            logger.log_training_state("valid", epoch + 1, e2e_vio_model.state_dict())
        if loss_mean < min_loss_t:
            min_loss_t = loss_mean
            logger.log_training_state("train", epoch + 1, e2e_vio_model.state_dict())
        if err_eval < min_err_eval:
            min_err_eval = err_eval
            logger.log_training_state("eval", epoch + 1, e2e_vio_model.state_dict())

        logger.print("Latest saves:",
                     " ".join(["%s: %s" % (k, v) for k, v in logger.log_training_state_latest_epoch.items()]))
