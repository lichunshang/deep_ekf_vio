import eval
import argparse
import sys
import os
from params import par

script = sys.argv[1]
args = sys.argv[2:]

if script == "gen_trajectory_rel":
    default_sequences = par.train_video + par.valid_video
    default_seq_len = 2
    arg_parser = argparse.ArgumentParser(description='Generate trajectory')
    arg_parser.add_argument('model_file_path', type=str, help='path to the saved model state dict')
    arg_parser.add_argument('--sequences', type=str, nargs="+", help="Select the sequences", default=default_sequences)
    arg_parser.add_argument('--seq_len', type=int, help="sequence length", default=default_seq_len)
    arg_parsed = arg_parser.parse_args(args=args)
    sequences = arg_parsed.sequences
    seq_len = arg_parsed.seq_len
    seq_len_range = (seq_len, seq_len,)

    eval.gen_trajectory_rel(os.path.abspath(arg_parsed.model_file_path), arg_parsed.sequences, arg_parsed.seq_len)

elif script == "plot_trajectory":
    arg_parser = argparse.ArgumentParser(description='Plot trajectory')
    arg_parser.add_argument("working_dir", type=str, help="working directory of generated results")
    eval.plot_trajectory(arg_parser.parse_args(args=args).working_dir)

elif script == "np_traj_to_kitti":
    arg_parser = argparse.ArgumentParser(description='Convert np trajectory to KITTI')
    arg_parser.add_argument("working_dir", type=str, help="working directory of generated results")
    eval.np_traj_to_kitti(arg_parser.parse_args(args=args).working_dir)

elif script == "kitti_eval":
    arg_parser = argparse.ArgumentParser(description='KITTI evaluation')
    arg_parser.add_argument('working_dir', type=str, help='path to the saved model state dict')
    arg_parser.add_argument('--train_seqs', type=str, nargs="+", help="Select training sequences",
                            default=par.train_video)
    arg_parser.add_argument('--val_seqs', type=str, nargs="+", help="Select validation sequences",
                            default=par.valid_video)
    arg_parsed = arg_parser.parse_args(args=args)
    eval.kitti_eval(arg_parsed.working_dir, arg_parsed.train_seqs, arg_parsed.val_seqs)

else:
    print("Invalid selection!")
