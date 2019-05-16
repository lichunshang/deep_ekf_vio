import eval
import argparse
import os
from params import par
import preprocess

choices = ["gen_trajectory", "plot_trajectory", "plot_ekf_states", "np_traj_to_kitti", "kitti_eval", "calc_error",
           "plot_error", "preprocess_kitti_raw", "preprocess_euroc", "check_time_discontinuities",
           "calc_image_mean_std", "euroc_eval"]

top_level_arg_parser = argparse.ArgumentParser(description='Execute scripts')
top_level_arg_parser.add_argument('script', type=str, help='The program to run', choices=choices)
top_level_arg_parsed, args = top_level_arg_parser.parse_known_args()

if top_level_arg_parsed.script == "gen_trajectory":
    default_sequences = par.train_seqs + par.valid_seqs
    default_seq_len = 2
    arg_parser = argparse.ArgumentParser(description='Generate trajectory')
    arg_parser.add_argument('model_file_path', type=str, help='path to the saved model state dict')
    arg_parser.add_argument('--sequences', type=str, nargs="+", help="Select the sequences", default=default_sequences)
    arg_parser.add_argument('--seq_len', type=int, help="sequence length", default=default_seq_len)
    arg_parser.add_argument('--no_prop_lstm_states', help="Don't propagate LSTM states", default=False,
                            action='store_true')
    arg_parsed = arg_parser.parse_args(args=args)

    eval.gen_trajectory(os.path.abspath(arg_parsed.model_file_path), arg_parsed.sequences, arg_parsed.seq_len,
                        not arg_parsed.no_prop_lstm_states)

elif top_level_arg_parsed.script == "plot_trajectory":
    arg_parser = argparse.ArgumentParser(description='Plot trajectory')
    arg_parser.add_argument("working_dir", type=str, help="working directory of generated results")
    eval.plot_trajectory(arg_parser.parse_args(args=args).working_dir)

elif top_level_arg_parsed.script == "plot_ekf_states":
    arg_parser = argparse.ArgumentParser(description='Plot EKF States')
    arg_parser.add_argument("working_dir", type=str, help="working directory of generated results")
    eval.plot_ekf_states(arg_parser.parse_args(args=args).working_dir)


elif top_level_arg_parsed.script == "plot_error":
    arg_parser = argparse.ArgumentParser(description='Plot error')
    arg_parser.add_argument("working_dir", type=str, help="working directory of generated results")
    eval.plot_errors(arg_parser.parse_args(args=args).working_dir)

elif top_level_arg_parsed.script == "calc_error":
    arg_parser = argparse.ArgumentParser(description='Calculate error')
    arg_parser.add_argument("working_dir", type=str, help="working directory of generated results")
    eval.calc_error(arg_parser.parse_args(args=args).working_dir)

elif top_level_arg_parsed.script == "np_traj_to_kitti":
    arg_parser = argparse.ArgumentParser(description='Convert np trajectory to KITTI')
    arg_parser.add_argument("working_dir", type=str, help="working directory of generated results")
    eval.np_traj_to_kitti(arg_parser.parse_args(args=args).working_dir)

elif top_level_arg_parsed.script == "kitti_eval":
    arg_parser = argparse.ArgumentParser(description='KITTI evaluation')
    arg_parser.add_argument('working_dir', type=str, help='path to the saved model state dict')
    arg_parser.add_argument('--train_seqs', type=str, nargs="+", help="Select training sequences",
                            default=par.train_seqs)
    arg_parser.add_argument('--val_seqs', type=str, nargs="+", help="Select validation sequences",
                            default=par.valid_seqs)
    arg_parser.add_argument('--simple', default=False, action='store_true',
                            help="Compute with kitti python implementation, only computes error, looks at --seqs only")
    arg_parser.add_argument('--seqs', type=str, nargs="*", help="select sequences, only with --simple")
    arg_parsed = arg_parser.parse_args(args=args)

    if arg_parsed.simple:
        eval.kitti_eval_simple(arg_parsed.working_dir, arg_parsed.seqs)
    else:
        eval.kitti_eval(arg_parsed.working_dir, arg_parsed.train_seqs, arg_parsed.val_seqs)


elif top_level_arg_parsed.script == "preprocess_kitti_raw":
    arg_parser = argparse.ArgumentParser(description='Preprocess RAW Kitti Data')
    arg_parser.add_argument('kitti_raw_dir', type=str, help='KITTI raw data path (extract)')
    arg_parser.add_argument('output_dir', type=str,
                            help='Output directory, the name of the output dir is also the sequence name')
    arg_parser.add_argument('cam_subset_range', type=int, nargs=2,
                            help="The subset the sequence to process, "
                                 "indicated by camera frame start and end index, inclusive ")
    arg_parsed = arg_parser.parse_args(args=args)
    preprocess.preprocess_kitti_raw(arg_parsed.kitti_raw_dir, arg_parsed.output_dir, arg_parsed.cam_subset_range)
elif top_level_arg_parsed.script == "check_time_discontinuities":
    arg_parser = argparse.ArgumentParser(description='Check time discontinuities')
    arg_parser.add_argument('kitti_raw_dir', type=str, help='KITTI raw data path (extract)')
    arg_parsed = arg_parser.parse_args(args=args)
    preprocess.check_time_discontinuities(arg_parsed.kitti_raw_dir)

elif top_level_arg_parsed.script == "preprocess_euroc":
    arg_parser = argparse.ArgumentParser(description='Preprocess EUROC Data')
    arg_parser.add_argument('euroc_dir', type=str, help='EUROC_dir')
    arg_parser.add_argument('output_dir', type=str,
                            help='Output directory, the name of the output dir is also the sequence name')
    arg_parser.add_argument('cam_still_range', type=int, nargs=2,
                            help="The portion of the sequence when the drone is not moving, "
                                 "used for inference time evaluation")
    arg_parsed = arg_parser.parse_args(args=args)
    preprocess.preprocess_euroc(arg_parsed.euroc_dir, arg_parsed.output_dir, arg_parsed.cam_still_range)

elif top_level_arg_parsed.script == "euroc_eval":
    arg_parser = argparse.ArgumentParser(description='Evaluate EUROC')
    arg_parser.add_argument('working_dir', type=str, help='path to the saved model state dict')
    arg_parser.add_argument('seqs', type=str, nargs="+", help="select sequences")
    arg_parsed = arg_parser.parse_args(args=args)
    eval.euroc_eval(arg_parsed.working_dir, arg_parsed.seqs)

elif top_level_arg_parsed.script == "calc_image_mean_std":
    arg_parser = argparse.ArgumentParser(description='Calculate the image channel mean and std')
    arg_parser.add_argument('--seqs', type=str, nargs="+", help="Select sequences",
                            default=par.train_seqs)
    arg_parsed = arg_parser.parse_args(args=args)
    preprocess.calc_image_mean_std(arg_parsed.seqs)
else:
    print("Invalid selection!")
