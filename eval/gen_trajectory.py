import numpy as np
import torch
import os
import time
import se3
import collections
from data_loader import get_subseqs, SubseqDataset, SequenceData
from params import par
from model import E2EVIO
from log import logger
from new_loss import scale_pose

def gen_trajectory_rel_iter(model, dataloader, initial_pose=np.eye(4, 4)):
    predicted_abs_poses = [np.array(initial_pose), ]
    predicted_rel_poses = []
    vis_meas_covars = []
    # lstm_states = None  # none defaults to zero
    for i, data in enumerate(dataloader):
        print('%d/%d (%.2f%%)' % (i, len(dataloader), i * 100 / len(dataloader)), end="\r")

        # images = data[1].cuda()
        meta_data, images, imu_data, prev_state, T_imu_cam, gt_poses, gt_rel_poses = data
        gt_trans_norm = torch.norm(gt_rel_poses[:, :, 3:6], dim=2).unsqueeze(2).cpu()
        # lstm_states = lstm_states if prop_lstm_states else None
        # we only care about the results from the VO front ends here
        vis_meas, vis_meas_covar,  _, _, _ = model.forward(images.cuda(),
                                                                       imu_data.cuda(),
                                                                       gt_poses[:, 0].inverse().cuda(),
                                                                       prev_state.cuda(), None,
                                                                       T_imu_cam.cuda(),
                                                                       gt_rel_poses[:,0].cuda(),
                                                                       None)
        # lstm_states = lstm_states.detach()
        vis_meas = vis_meas.cpu()
        vis_meas_rot = vis_meas[:,:,:3]
        
        vis_meas_trans_norm = torch.norm(vis_meas[:,:,3:], dim=2).unsqueeze(2).detach().cpu()
        vis_meas_trans = vis_meas[:,:,3:]/vis_meas_trans_norm * gt_trans_norm
        vis_meas = torch.cat((vis_meas_rot, vis_meas_trans), dim=2).detach().cpu().numpy()
        vis_meas_covar = vis_meas_covar.detach().cpu().numpy()

        for i, rel_pose in enumerate(vis_meas[-1]):  # select the only batch
            T_vkm1_vk = se3.T_from_Ct(se3.exp_SO3(rel_pose[0:3]), rel_pose[3:6])
            T_i_vk = predicted_abs_poses[-1].dot(T_vkm1_vk)
            se3.log_SO3(T_i_vk[0:3, 0:3])  # just here to check for warnings
            predicted_abs_poses.append(T_i_vk)
            vis_meas_covars.append(vis_meas_covar[-1][i])
            predicted_rel_poses.append(rel_pose)

    return predicted_abs_poses, predicted_rel_poses, vis_meas_covars


def gen_trajectory_abs_iter(model, dataloaders):
    for d in dataloaders:
        assert dataloaders[d].batch_size == 1
    est_poses_dict = {k: [] for k in dataloaders.keys()}
    est_states_dict = {k: [] for k in dataloaders.keys()}
    est_covars_dict = {k: [] for k in dataloaders.keys()}
    est_vis_meas_dict = {k: [] for k in dataloaders.keys()}
    vis_meas_covar_dict = {k: [] for k in dataloaders.keys()}
    # lstm_states_dict = dict.fromkeys(dataloaders.keys())

    max_length = max([len(dataloaders[k]) for k in dataloaders])
    data_loader_iter = {k: iter(v) for k, v in dataloaders.items()}

    for i in range(0, max_length):
        print('%d/%d (%.2f%%)' % (i, max_length, i * 100 / max_length), end="\r")

        # use Ordered Dict guarantee deterministic order
        data = collections.OrderedDict({k: next(data_loader_iter[k]) for k, v in dataloaders.items() if i < len(v)})
        data_list = list(data.values())
        data_keys = list(data.keys())
        # meta_data, images, imu_data, prev_state, T_imu_cam, gt_poses, gt_rel_poses
        images = torch.stack([torch.squeeze(d[1], 0) for d in data_list]).cuda()
        imu_data = torch.stack([torch.squeeze(d[2], 0) for d in data_list]).cuda()
        T_imu_cam = torch.stack([torch.squeeze(d[4], 0) for d in data_list]).cuda()
        gt_poses = torch.stack([torch.squeeze(d[5], 0) for d in data_list]).cuda()
        gt_rel_poses = torch.stack([torch.squeeze(d[6], 0) for d in data_list]).cuda()
        gt_trans_norm = torch.norm(gt_rel_poses[:, :, 3:6], dim=2).unsqueeze(2)
        # use returned states for all iterations after the first
        if i > 0:
            prev_pose = torch.stack([torch.tensor(est_poses_dict[k][-1]) for k in data_keys]).cuda()
            prev_state = torch.stack([torch.tensor(est_states_dict[k][-1]) for k in data_keys]).cuda()
            prev_covar = torch.stack([torch.tensor(est_covars_dict[k][-1]) for k in data_keys]).cuda()
            prev_vis_meas = torch.stack([torch.tensor(est_vis_meas_dict[k][-1]) for k in data_keys]).cuda()
            prev_vis_meas_covar = torch.stack([torch.tensor(vis_meas_covar_dict[k][-1]) for k in data_keys]).cuda()
        else:
            prev_pose = torch.stack([torch.squeeze(d[5], 0) for d in data_list])[:, 0].inverse().cuda()
            prev_state = torch.stack([torch.squeeze(d[3], 0) for d in data_list]).cuda()
            prev_covar = None
            prev_vis_meas = torch.stack([torch.squeeze(d[6], 0) for d in data_list])[:, 0].cuda()
            prev_vis_meas_covar = None
            # lstm_states = None

        vis_meas, vis_meas_covar, est_poses, est_ekf_states, est_ekf_covars = \
            model.forward(images, imu_data,  prev_pose, prev_state, prev_covar, T_imu_cam, prev_vis_meas, prev_vis_meas_covar)

        est_poses = scale_pose(est_poses, gt_poses)

        vis_meas_rot = vis_meas[:,:,:3]
        
        vis_meas_trans_norm = torch.norm(vis_meas[:,:,3:], dim=2).unsqueeze(2)
        vis_meas_trans = vis_meas[:,:,3:]/vis_meas_trans_norm * gt_trans_norm
        vis_meas = torch.cat((vis_meas_rot, vis_meas_trans), dim=2)

        for j, k in enumerate(data):
            # if it is the first estimate, include the initial pose as well, otherwise just 1: onwards
            slice_start = 1 if i > 0 else 0
            est_poses_dict[k] += list(est_poses[j].detach().cpu().numpy())[slice_start:]
            est_states_dict[k] += list(est_ekf_states[j].detach().cpu().numpy())[slice_start:]
            est_covars_dict[k] += list(est_ekf_covars[j].detach().cpu().numpy())[slice_start:]

            est_vis_meas_dict[k] += list(vis_meas[j].detach().cpu().numpy())
            vis_meas_covar_dict[k] += list(vis_meas_covar[j].detach().cpu().numpy())
            # lstm_states_dict[k] = lstm_states[j].detach()

    return est_vis_meas_dict, vis_meas_covar_dict, est_poses_dict, est_states_dict, est_covars_dict


def gen_trajectory(model_file_path, sequences, seq_len):
    # Path
    model_file_path = os.path.abspath(model_file_path)
    assert (os.path.exists(model_file_path))
    working_dir = os.path.join(os.path.dirname(model_file_path), os.path.basename(model_file_path) + ".traj")
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ GENERATE TRAJECTORY REL ================")

    # Load model
    logger.print("Constructing model...")
    model = E2EVIO()
    model = model.cuda()
    logger.print("Loading model from: ", model_file_path)
    model.load_state_dict(logger.clean_state_dict_key(torch.load(model_file_path)))
    model.eval()

    logger.log_parameters()
    logger.print("Using sequence length:", seq_len)
    # logger.print("Prop LSTM states:", prop_lstm_states)
    logger.print("Sequences: \n" + "\n".join(sequences))

    for seq in sequences:
        logger.print("Generating trajectory for seq...", seq)
        start_time = time.time()

        subseqs = get_subseqs([seq], seq_len, overlap=1, sample_times=1, training=False)
        dataset = SubseqDataset(subseqs, (par.img_h, par.img_w), par.img_means, par.img_stds, par.minus_point_5,
                                training=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        seq_data = SequenceData(seq)
        gt_abs_poses = seq_data.get_poses()
        timestamps = seq_data.get_timestamps()

        if par.enable_ekf:
            logger.print("With EKF enabled ...")
            est_vis_meas_dict, vis_meas_covar_dict, est_poses_dict, est_states_dict, est_covars_dict = \
                gen_trajectory_abs_iter(model, {seq: dataloader})
            est_vis_meas = est_vis_meas_dict[seq]
            vis_meas_covar = vis_meas_covar_dict[seq]
            est_states = est_states_dict[seq]
            est_covars = est_covars_dict[seq]
            est_poses = est_poses_dict[seq]
            np.save(logger.ensure_file_dir_exists(
                    os.path.join(working_dir, "ekf_states", "vis_meas", seq + ".npy")), est_vis_meas)
            np.save(logger.ensure_file_dir_exists(
                    os.path.join(working_dir, "ekf_states", "vis_meas_covar", seq + ".npy")), vis_meas_covar)
            np.save(logger.ensure_file_dir_exists(
                    os.path.join(working_dir, "ekf_states", "poses", seq + ".npy")), est_poses)
            np.save(logger.ensure_file_dir_exists(
                    os.path.join(working_dir, "ekf_states", "states", seq + ".npy")), est_states)
            np.save(logger.ensure_file_dir_exists(
                    os.path.join(working_dir, "ekf_states", "covars", seq + ".npy")), est_covars)
            np.save(logger.ensure_file_dir_exists(
                    os.path.join(working_dir, "ekf_states", "gt_velocities", seq + ".npy")), seq_data.get_velocities())

            est_poses = np.linalg.inv(np.array(est_poses_dict[seq]).astype(np.float64))
        else:
            logger.print("Without EKF enabled ...")
            est_poses, est_vis_meas, vis_meas_covar = gen_trajectory_rel_iter(
                    model, dataloader,initial_pose=gt_abs_poses[0, :, :])

        np.save(logger.ensure_file_dir_exists(os.path.join(working_dir, "vis_meas", "meas", seq + ".npy")),
                est_vis_meas)
        np.save(logger.ensure_file_dir_exists(os.path.join(working_dir, "vis_meas", "covar", seq + ".npy")),
                vis_meas_covar)
        np.save(logger.ensure_file_dir_exists(os.path.join(working_dir, "est_poses", seq + ".npy")), est_poses)
        np.save(logger.ensure_file_dir_exists(os.path.join(working_dir, "gt_poses", seq + ".npy")),
                gt_abs_poses[:len(est_poses)])  # ensure same length as est poses
        np.save(logger.ensure_file_dir_exists(os.path.join(working_dir, "timestamps", seq + ".npy")),
                timestamps[:len(est_poses)])
        logger.print("Done, took %.2f seconds" % (time.time() - start_time))

    return working_dir