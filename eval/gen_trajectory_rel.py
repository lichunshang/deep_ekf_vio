import numpy as np
import torch
import os
import time
import se3
from data_loader import get_subseqs, SubseqDataset, SequenceData
from params import par
from model import E2EVIO
from log import logger


def gen_trajectory_rel_iter(model, dataloader, prop_lstm_states, initial_pose=np.eye(4, 4)):
    predicted_abs_poses = [np.array(initial_pose), ]
    lstm_states = None  # none defaults to zero
    for i, data in enumerate(dataloader):
        print('%d/%d (%.2f%%)' % (i, len(dataloader), i * 100 / len(dataloader)), end="\r")

        images = data[1].cuda()

        lstm_states = lstm_states if prop_lstm_states else None
        predicted_rel_poses, _, lstm_states, _, _, _ = model.forward(images, None, None, lstm_states, None, None, None)

        lstm_states = lstm_states.detach()
        predicted_rel_poses = predicted_rel_poses.detach().cpu().numpy()

        for rel_pose in predicted_rel_poses[-1]:  # select the only batch
            T_vkm1_vk = se3.T_from_Ct(se3.exp_SO3(rel_pose[3:6]), rel_pose[0:3])
            T_i_vk = predicted_abs_poses[-1].dot(T_vkm1_vk)
            se3.log_SO3(T_i_vk[0:3, 0:3])  # just here to check for warnings
            predicted_abs_poses.append(T_i_vk)

    return predicted_abs_poses


def gen_trajectory_rel(model_file_path, sequences, seq_len, prop_lstm_states):
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
    logger.print("Prop LSTM states:", prop_lstm_states)
    logger.print("Sequences: \n" + "\n".join(sequences))

    for seq in sequences:
        logger.print("Generating trajectory for seq...", seq)
        start_time = time.time()

        subseqs = get_subseqs([seq], seq_len, overlap=1, sample_times=1, training=False)
        dataset = SubseqDataset(subseqs, (par.img_h, par.img_w), par.img_means, par.img_stds, par.minus_point_5,
                                training=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        gt_abs_poses = SequenceData(seq).get_poses()

        predicted_abs_poses = gen_trajectory_rel_iter(model, dataloader, prop_lstm_states,
                                                      initial_pose=gt_abs_poses[0, :, :])

        np.save(logger.ensure_file_dir_exists(os.path.join(working_dir, "est_poses", seq + ".npy")),
                predicted_abs_poses)
        np.save(logger.ensure_file_dir_exists(os.path.join(working_dir, "gt_poses", seq + ".npy")),
                gt_abs_poses[:len(predicted_abs_poses)])  # ensure same length as est poses
        logger.print("Done, took %.2f seconds" % (time.time() - start_time))

    return working_dir
