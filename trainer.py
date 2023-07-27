import torch
import torch.nn.functional
import numpy as np
import os
import time
import se3
import params
import re
from params import par
from model import E2EVIO
from data_loader import get_subseqs, SubseqDataset, convert_subseqs_list_to_panda
from log import logger
from torch.utils.data import DataLoader
from eval import EurocErrorCalc, KittiErrorCalc
from eval.gen_trajectory import gen_trajectory_rel_iter, gen_trajectory_abs_iter
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from new_loss import scale_pose

class _OnlineDatasetEvaluator(object):
    def __init__(self, model, sequences, eval_length):
        self.model = model  # this is a reference
        self.dataloaders = {}

        if par.dataset() == "KITTI":
            self.error_calc = KittiErrorCalc(sequences)
        elif par.dataset() == "EUROC":
            self.error_calc = EurocErrorCalc(sequences)

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
            # gt_abs_poses = np.load(os.path.join(par.pose_dir, seq,  "constants.npy"))
            predicted_abs_poses, _, _ = gen_trajectory_rel_iter(self.model, self.dataloaders[seq])
            seq_err = self.error_calc.accumulate_error(seq, np.array(predicted_abs_poses))
            logger.print("%s: %.5f" % (seq, seq_err), end=" ")
        logger.print()
        ave_err = self.error_calc.get_average_error()
        self.error_calc.clear()
        logger.print("Online evaluation took %.2fs, err %.6f" % (time.time() - start_time, ave_err))

        return ave_err

    def evaluate_abs(self):
        start_time = time.time()
        _, _, est_poses_dict, _, _ = gen_trajectory_abs_iter(self.model, self.dataloaders)
        for k, v in est_poses_dict.items():
            seq_err = self.error_calc.accumulate_error(k, np.linalg.inv(np.array(v, dtype=np.float64)))
            logger.print("%s: %.5f" % (k, seq_err), end=" ")
        logger.print()
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
        self.epoch = 0
        # self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.L1Loss()
        self.loss_fn1 = torch.nn.L1Loss()
        self.loss_fn = torch.nn.MSELoss()
        self.loss_h = torch.nn.HuberLoss()

    def get_loss(self, data):
        meta_data, images, imu_data, prev_state, T_imu_cam, gt_poses, gt_rel_poses = data

        _, _, _, _, _, invalid_imu_list = SubseqDataset.decode_batch_meta_info(meta_data)

        vis_meas, vis_meas_covar, poses, ekf_states, ekf_covars = \
            self.model.forward(images.cuda(),
                               imu_data.cuda(),
                               gt_poses[:, 0].inverse().cuda(),
                               prev_state.cuda(), None,
                               T_imu_cam.cuda(),
                               gt_rel_poses[:,0].cuda(),
                               None)

        if par.enable_ekf and not par.gaussian_pdf_loss:
            # note that the estimated poses are already inversed
            s = np.array(invalid_imu_list)
            loss_abs = 0
            loss_vis_meas = 0
            vis_meas_loss_invalid_imu = 0
            if not np.all(s):
                _, loss_abs, loss_vis_meas = self.ekf_loss(poses[~s], gt_poses[~s].cuda(), ekf_states[~s],
                                                              gt_rel_poses[~s].cuda(), vis_meas[~s], vis_meas_covar[~s])
            if np.any(s):
                vis_meas_loss_invalid_imu = self.vis_meas_loss(vis_meas[s], vis_meas_covar[s], gt_rel_poses[s].cuda())

            # loss = vis_meas_loss_invalid_imu + loss_vis_meas +  loss_abs
            loss =  vis_meas_loss_invalid_imu + 100 * loss_vis_meas + loss_abs
        elif par.enable_ekf:
            loss, _, _ = self.ekf_loss(poses, gt_poses.cuda(), ekf_states, gt_rel_poses.cuda(), vis_meas, vis_meas_covar)
        else:
            loss = self.vis_meas_loss(vis_meas, vis_meas_covar, gt_rel_poses.cuda())


        if self.model.training:
            self.num_train_iterations += 1
        else:
            self.num_val_iterations += 1

        return loss

    def vis_meas_loss(self, predicted_rel_poses, vis_meas_covar, gt_rel_poses):
        # Weighted MSE Loss
        # angle_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 0:3], gt_rel_poses[:, :, 0:3])
        # trans_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 3:6], gt_rel_poses[:, :, 3:6])

        trans_loss = self.loss_h(predicted_rel_poses[:, :, 3:6], gt_rel_poses[:, :, 3:6])
        # trans_loss = self.sequence_loss(predicted_rel_poses[:, :, 0:3],gt_rel_poses[:, :, 0:3])
                        # self.loss_fn1(predicted_rel_poses[:, :, 3:6], gt_rel_poses[:, :, 3:6])
        # gt_trans_norm = torch.norm(gt_rel_poses[:, :, 3:6], dim=2).unsqueeze(2)
        # pred_trans_norm = torch.norm(predicted_rel_poses[:, :, 3:6], dim=2).unsqueeze(2)
        # pred_trans_scaled = predicted_rel_poses[:, :, 3:6]/ pred_trans_norm * gt_trans_norm
        # trans_loss = self.loss_h(pred_trans_scaled, gt_rel_poses[:,:,3:6])
        # angle_loss = self.loss_h(predicted_rel_poses[:,:,0:3],gt_rel_poses[:, :, 0:3])
                        # self.loss_fn1(predicted_rel_poses[:, :, 0:3], gt_rel_poses[:, :, 0:3])) \
        # trans_loss = self.loss_fn1(predicted_rel_poses[:,:,3:6],gt_rel_poses[:,:,3:6])

        angle_loss = self.loss_h(predicted_rel_poses[:, :, 0:3], gt_rel_poses[:, :, 0:3])
        # covar_loss = torch.mean(vis_meas_covar)

        if par.gaussian_pdf_loss:
            Q_det = torch.prod(torch.diagonal(vis_meas_covar, dim1=-2, dim2=-1), -1)
            log_Q_norm = torch.log(Q_det)
            err = predicted_rel_poses - gt_rel_poses
            scale = np.ones(6, dtype=np.float32)
            scale[0:3] = scale[0:3] * np.sqrt(par.k1)
            err = par.k4 * torch.unsqueeze(err * torch.tensor(scale, device=vis_meas_covar.device).view(1, 1, 6), -1)
            err_weighted_by_covar = torch.matmul(torch.matmul(err.transpose(-2, -1), vis_meas_covar.inverse()), err)
            loss = torch.mean(log_Q_norm + torch.squeeze(err_weighted_by_covar))
        else:
            loss = (par.k1 * angle_loss + trans_loss)
            # loss = ( angle_loss + par.k1 * trans_loss)
            # loss = angle_loss + trans_loss

        # log the loss
        tag_name = "train" if self.model.training else "val"
        iterations = self.num_train_iterations if self.model.training else self.num_val_iterations
        add_scalar = logger.tensorboard.add_scalar
        rot_x_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 0], gt_rel_poses[:, :, 0])
        rot_y_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 1], gt_rel_poses[:, :, 1])
        rot_z_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 2], gt_rel_poses[:, :, 2])
        trans_x_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 3], gt_rel_poses[:, :, 3])
        trans_y_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 4], gt_rel_poses[:, :, 4])
        trans_z_loss = torch.nn.functional.mse_loss(predicted_rel_poses[:, :, 5], gt_rel_poses[:, :, 5])
        add_scalar(tag_name + "_vis/total_loss", loss, iterations)
        add_scalar(tag_name + "_vis/rot_loss", angle_loss, iterations)
        add_scalar(tag_name + "_vis/rot_loss/x", rot_x_loss, iterations)
        add_scalar(tag_name + "_vis/rot_loss/y", rot_y_loss, iterations)
        add_scalar(tag_name + "_vis/rot_loss/z", rot_z_loss, iterations)
        add_scalar(tag_name + "_vis/trans_loss", trans_loss, iterations)
        add_scalar(tag_name + "_vis/trans_loss/x", trans_x_loss, iterations)
        add_scalar(tag_name + "_vis/trans_loss/y", trans_y_loss, iterations)
        add_scalar(tag_name + "_vis/trans_loss/z", trans_z_loss, iterations)

        vis_meas_covar_diag = torch.diagonal(vis_meas_covar, dim1=-2, dim2=-1)
        add_hist = logger.tensorboard.add_histogram
        add_scalar(tag_name + "_vis_covar/ave/rot_x", torch.mean(vis_meas_covar_diag[:, :, 0]), iterations)
        add_scalar(tag_name + "_vis_covar/ave/rot_y", torch.mean(vis_meas_covar_diag[:, :, 1]), iterations)
        add_scalar(tag_name + "_vis_covar/ave/rot_z", torch.mean(vis_meas_covar_diag[:, :, 2]), iterations)
        add_scalar(tag_name + "_vis_covar/ave/trans_x", torch.mean(vis_meas_covar_diag[:, :, 3]), iterations)
        add_scalar(tag_name + "_vis_covar/ave/trans_y", torch.mean(vis_meas_covar_diag[:, :, 4]), iterations)
        add_scalar(tag_name + "_vis_covar/ave/trans_z", torch.mean(vis_meas_covar_diag[:, :, 5]), iterations)
        add_hist(tag_name + "_vis_covar/hist/rot_x", vis_meas_covar_diag[:, :, 0].view(-1), iterations)
        add_hist(tag_name + "_vis_covar/hist/rot_y", vis_meas_covar_diag[:, :, 1].view(-1), iterations)
        add_hist(tag_name + "_vis_covar/hist/rot_z", vis_meas_covar_diag[:, :, 2].view(-1), iterations)
        add_hist(tag_name + "_vis_covar/hist/trans_x", vis_meas_covar_diag[:, :, 3].view(-1), iterations)
        add_hist(tag_name + "_vis_covar/hist/trans_y", vis_meas_covar_diag[:, :, 4].view(-1), iterations)
        add_hist(tag_name + "_vis_covar/hist/trans_z", vis_meas_covar_diag[:, :, 5].view(-1), iterations)

        return loss

    def ekf_loss(self, est_poses, gt_poses, ekf_states, gt_rel_poses, vis_meas, vis_meas_covar):
        est_poses = scale_pose(est_poses,gt_poses).to(device=gt_poses.device)
        # print(est_poses)
        abs_errors = torch.matmul(est_poses[:, 1:], gt_poses[:, 1:])
        length_div = torch.arange(start=1, end=abs_errors.size(1) + 1, device=abs_errors.device,
                                  dtype=torch.float32).view(1, -1, 1)
        # calculate the F norm squared from identity
        I_minus_angle_errors = (torch.eye(3, 3, device=abs_errors.device) -
                                abs_errors[:, :, 0:3, 0:3]) / length_div.view(1, -1, 1, 1)
        I_minus_angle_errors_sq = torch.matmul(I_minus_angle_errors, I_minus_angle_errors.transpose(-2, -1))
        abs_angle_errors_sq = torch.sum(torch.diagonal(I_minus_angle_errors_sq, dim1=-2, dim2=-1), dim=-1)
        # abs_angle_errors = torch.squeeze(torch_se3.log_SO3_b(abs_errors[:, :, 0:3, 0:3]), -1) / length_div
        # abs_angle_errors_sq = torch.sum(abs_angle_errors ** 2, dim=-1)  # norm squared

        abs_trans_errors_sq = torch.sum((abs_errors[:, :, 0:3, 3] / length_div) ** 2, dim=-1)

        abs_angle_loss = torch.mean(abs_angle_errors_sq)
        abs_trans_loss = torch.mean(abs_trans_errors_sq)

        # _, C_rel, r_rel, _, _, _ = IMUKalmanFilter.decode_state_b(ekf_states)
        # rel_angle_errors = (gt_rel_poses[:, :, 0:3] - torch.squeeze(torch_se3.log_SO3_b(C_rel[:, 1:]), -1)) ** 2
        # rel_angle_errors_sq = torch.sum(rel_angle_errors ** 2, dim=-1)
        # rel_trans_error_sq = torch.sum((gt_rel_poses[:, :, 3:6] - torch.squeeze(r_rel[:, 1:], -1)) ** 2, dim=-1)
        # rel_angle_loss = torch.mean(rel_angle_errors_sq)
        # rel_trans_loss = torch.mean(rel_trans_error_sq)

        k3 = self.schedule(par.k3)

        loss_abs = (par.k2 *  abs_angle_loss + abs_trans_loss) * par.k4 ** 2
        # loss_rel = (par.k1 * rel_angle_loss + rel_trans_loss)
        # loss = k3 * loss_rel + (1 - k3) * loss_abs
        loss_vis_meas = self.vis_meas_loss(vis_meas, vis_meas_covar, gt_rel_poses)
        loss = k3 * loss_vis_meas + (1 - k3) * loss_abs

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

        tag_name = "train" if self.model.training else "val"
        iterations = self.num_train_iterations if self.model.training else self.num_val_iterations
        add_scalar = logger.tensorboard.add_scalar
        add_scalar(tag_name + "_abs/abs_total_loss", loss_abs, iterations)
        add_scalar(tag_name + "_abs/abs_rot_loss", abs_angle_loss, iterations)
        add_scalar(tag_name + "_abs/last_rot_loss/x", last_rot_x_loss, iterations)
        add_scalar(tag_name + "_abs/last_rot_loss/y", last_rot_y_loss, iterations)
        add_scalar(tag_name + "_abs/last_rot_loss/z", last_rot_z_loss, iterations)
        add_scalar(tag_name + "_abs/abs_loss", abs_trans_loss, iterations)
        add_scalar(tag_name + "_abs/last_trans_loss/x", last_trans_x_loss, iterations)
        add_scalar(tag_name + "_abs/last_trans_loss/y", last_trans_y_loss, iterations)
        add_scalar(tag_name + "_abs/last_trans_loss/z", last_trans_z_loss, iterations)

        if self.model.training:
            model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            imu_noise_covar = torch.diagonal(model.get_imu_noise_covar())
            add_scalar("imu_noise_diag/w_x", imu_noise_covar[0], iterations)
            add_scalar("imu_noise_diag/w_y", imu_noise_covar[1], iterations)
            add_scalar("imu_noise_diag/w_z", imu_noise_covar[2], iterations)
            add_scalar("imu_noise_diag/bw_x", imu_noise_covar[3], iterations)
            add_scalar("imu_noise_diag/bw_y", imu_noise_covar[4], iterations)
            add_scalar("imu_noise_diag/bw_z", imu_noise_covar[5], iterations)
            add_scalar("imu_noise_diag/a_x", imu_noise_covar[6], iterations)
            add_scalar("imu_noise_diag/a_y", imu_noise_covar[7], iterations)
            add_scalar("imu_noise_diag/a_z", imu_noise_covar[8], iterations)
            add_scalar("imu_noise_diag/ba_x", imu_noise_covar[9], iterations)
            add_scalar("imu_noise_diag/ba_y", imu_noise_covar[10], iterations)
            add_scalar("imu_noise_diag/ba_z", imu_noise_covar[11], iterations)
            add_scalar("params/k3", k3, iterations)
        return loss, loss_abs, loss_vis_meas

    

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

    def schedule(self, d):
        epochs = sorted(list(d.keys()))
        for i in range(0, len(epochs)):
            if epochs[len(epochs) - i - 1] <= self.epoch:
                return d[epochs[len(epochs) - i - 1]]
        raise ValueError("Invalid Schedule")


def train(resume_model_path, resume_optimizer_path, train_description ='train'):
    logger.initialize(working_dir=par.results_dir, use_tensorboard=True)
    logger.print("================ TRAIN ================")

    if not train_description:
        train_description = "1"
        # train_description = input("Enter a description of this training run: ")
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
    # for param in e2e_vio_model.vo_module.parameters():
    #     param.requires_grad = True
    # for param in e2e_vio_model.vo_module.extractor.model.feature_encoder.parameters():
    #     param.requires_grad = False
    # for param in e2e_vio_model.vo_module.extractor.model.context_encoder.parameters():
    #     param.requires_grad = False
    
    online_evaluator = _OnlineDatasetEvaluator(e2e_vio_model, par.valid_seqs, par.seq_len)

    # Load FlowNet weights pretrained with FlyingChairs
    # NOTE: the pretrained model assumes image rgb values in range [-0.5, 0.5]
    if par.pretrained and not resume_model_path:
        pretrained_w = torch.load(par.pretrained)
        logger.print('Load pretrained model')

        vo_model_dict = e2e_vio_model.vo_module.state_dict()
        update_dict = {k: v for k, v in pretrained_w.items() if k in vo_model_dict}
        assert (len(update_dict) > 0)
        vo_model_dict.update(update_dict)
        e2e_vio_model.vo_module.load_state_dict(vo_model_dict)

    # Create optimizer
    logger.print("Optimizing on parameters:")
    optimizer_params = []
    for p_name, p in e2e_vio_model.named_parameters():
        specific_lr_found = False
        for k in par.param_specific_lr:
            regex = re.compile(k)
            if regex.match(p_name):
                optimizer_params.append({"params": p, "lr": par.param_specific_lr[k]})
                logger.print("%s, lr:%f" % (p_name, par.param_specific_lr[k]))
                specific_lr_found = True

        if not specific_lr_found:
            optimizer_params.append({"params": p})
            logger.print(p_name)
    assert(len(optimizer_params) == sum(1 for _ in e2e_vio_model.parameters()))

    optimizer = par.optimizer(optimizer_params, **par.optimizer_args)

    # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    # Load trained DeepVO model and optimizer
    if resume_model_path:
        state_dict_update = logger.clean_state_dict_key(torch.load(resume_model_path))
        state_dict_update = {key: state_dict_update[key] for key in state_dict_update
                             if key not in par.exclude_resume_weights
                             and 'vo_module.extractor' not in key
                            #  and key != 'vo_module.regressor.firstconv.0.0.weight'
                             }
        state_dict = e2e_vio_model.state_dict()
        state_dict_update = {k: v for k, v in state_dict_update.items() if k in state_dict}
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
        logger.tensorboard.add_scalar("epoch/train_loss", loss_mean, epoch)

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
        logger.tensorboard.add_scalar("epoch/val_loss", loss_mean_valid, epoch)

        logger.print('Epoch {}\ntrain loss mean: {}, std: {}\nvalid loss mean: {}, std: {}\n'.
                     format(epoch + 1, loss_mean, np.std(t_loss_list), loss_mean_valid, np.std(v_loss_list)))

        err_eval = online_evaluator.evaluate()
        logger.tensorboard.add_scalar("epoch/eval_loss", err_eval, epoch)

        # scheduler.step()

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