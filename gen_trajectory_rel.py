import numpy as np
import torch
import os
import time
import se3_math
import argparse
from data_helper import get_data_info, ImageSequenceDataset
from params import par
from model import DeepVO
from torch.utils.data import DataLoader
from log import logger

default_sequences = ["00", "01", "02", "04", "05", "06", "07", "08", "09", "10"]
default_seq_len = 2

arg_parser = argparse.ArgumentParser(description='Generate trajectory')
arg_parser.add_argument('model_file_path', type=str, help='path to the saved model state dict')
arg_parser.add_argument('--sequences', type=str, nargs="+", help="Select the sequences", default=default_sequences)
arg_parser.add_argument('--seq_len', type=int, help="sequence length", default=default_seq_len)
arg_parsed = arg_parser.parse_args()
sequences = arg_parsed.sequences
seq_len = arg_parsed.seq_len
seq_len_range = (seq_len, seq_len,)

# Path
model_file_path = os.path.abspath(arg_parsed.model_file_path)
assert (os.path.exists(model_file_path))
working_dir = os.path.join(os.path.dirname(model_file_path), os.path.basename(model_file_path) + ".traj")
logger.initialize(working_dir=working_dir, use_tensorboard=False)

# Load model
logger.print("Constructing model...")
M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
M_deepvo = M_deepvo.cuda()
logger.print("Loading model from: ", model_file_path)
M_deepvo.load_state_dict(torch.load(model_file_path))
M_deepvo.eval()

logger.log_parameters()
logger.print("Using sequence length:", seq_len)
logger.print("Sequences: \n" + "\n".join(sequences))

for seq in sequences:
    logger.print("Generating trajectory for seq...", seq)
    start_time = time.time()

    df = get_data_info(folder_list=[seq], seq_len_range=seq_len_range, overlap=1, sample_times=1, shuffle=False,
                       sort=False)
    # df = df.loc[df.seq_len == seq_len_range[0]]
    dataset = ImageSequenceDataset(df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds,
                                   par.minus_point_5)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    predicted_abs_poses = [np.eye(4, 4), ]
    errors = [np.zeros([6, 1, ]), ]
    lstm_states = None  # none defaults to zero

    for i, batch in enumerate(dataloader):
        print('%d/%d (%.2f%%)' % (i, len(dataloader), i * 100 / len(dataloader)), end="\r")
        _, x, _ = batch
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

    np.save(logger.ensure_file_dir_exists(os.path.join(working_dir, "est_poses", seq + ".npy")), predicted_abs_poses)
    logger.print("Done, took %.2f seconds" % (time.time() - start_time))
