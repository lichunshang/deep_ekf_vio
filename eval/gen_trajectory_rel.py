import numpy as np
import torch
import os
import time
import se3_math
from data_helper import get_data_info, ImageSequenceDataset
from params import par
from model import DeepVO
from torch.utils.data import DataLoader
from log import logger


def gen_trajectory_rel(model_file_path, sequences, seq_len, prop_lstm_states):
    # Path
    model_file_path = os.path.abspath(model_file_path)
    assert (os.path.exists(model_file_path))
    working_dir = os.path.join(os.path.dirname(model_file_path), os.path.basename(model_file_path) + ".traj")
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ GENERATE TRAJECTORY REL ================")

    # Load model
    logger.print("Constructing model...")
    M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
    M_deepvo = M_deepvo.cuda()
    logger.print("Loading model from: ", model_file_path)
    M_deepvo.load_state_dict(logger.clean_state_dict_key(torch.load(model_file_path)))
    M_deepvo.eval()

    logger.log_parameters()
    logger.print("Using sequence length:", seq_len)
    logger.print("Prop LSTM states:", prop_lstm_states)
    logger.print("Sequences: \n" + "\n".join(sequences))

    for seq in sequences:
        logger.print("Generating trajectory for seq...", seq)
        start_time = time.time()

        df = get_data_info(sequences=[seq], seq_len=seq_len, overlap=1, sample_times=1)
        dataset = ImageSequenceDataset(df, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        gt_abs_poses = np.load(os.path.join(par.pose_dir, seq + ".npy"))

        predicted_abs_poses = [np.eye(4, 4), ]
        lstm_states = None  # none defaults to zero

        for i, batch in enumerate(dataloader):
            print('%d/%d (%.2f%%)' % (i, len(dataloader), i * 100 / len(dataloader)), end="\r")
            _, x, _ = batch

            lstm_states = lstm_states if prop_lstm_states else None
            print(lstm_states)
            predicted_rel_poses, lstm_states = M_deepvo.forward(x.cuda(), lstm_states)

            lstm_states = list(lstm_states)
            lstm_states[0] = lstm_states[0].detach()
            lstm_states[1] = lstm_states[1].detach()
            predicted_rel_poses = predicted_rel_poses.detach().cpu().numpy()

            for rel_pose in predicted_rel_poses[-1]:  # select the only batch
                T_vkm1_vk = se3_math.T_from_Ct(se3_math.exp_SO3(rel_pose[3:6]), rel_pose[0:3])
                T_i_vk = predicted_abs_poses[-1].dot(T_vkm1_vk)
                se3_math.log_SO3(T_i_vk[0:3, 0:3])  # just here to check for warnings
                predicted_abs_poses.append(T_i_vk)

        np.save(logger.ensure_file_dir_exists(os.path.join(working_dir, "est_poses", seq + ".npy")),
                predicted_abs_poses)
        np.save(logger.ensure_file_dir_exists(os.path.join(working_dir, "gt_poses", seq + ".npy")),
                gt_abs_poses[:len(predicted_abs_poses)])  # ensure same length as est poses
        logger.print("Done, took %.2f seconds" % (time.time() - start_time))

    return working_dir
