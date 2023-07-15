import os
import argparse
import trainer
import time
from params import par
from log import logger
from eval import gen_trajectory, plot_trajectory, kitti_eval, np_traj_to_kitti, calc_error, plot_errors, euroc_eval

arg_parser = argparse.ArgumentParser(description='Train E2E VIO')
arg_parser.add_argument('--description', type=str, help="Description of this training run")
arg_parser.add_argument('--gpu_id', type=int, nargs="+", help="select the GPU to perform training on")
arg_parser.add_argument('--run_eval_only', default=False, action='store_true',
                        help="Only run evaluation in current working directory")
arg_parser.add_argument('--resume_model_from', type=str, help="path of model state to resume from")
arg_parser.add_argument('--resume_optimizer_from', type=str, help="path of optimizer state to resume from")
arg_parsed = arg_parser.parse_args()
gpu_ids = arg_parsed.gpu_id
resume_model_path = os.path.abspath(arg_parsed.resume_model_from) if arg_parsed.resume_model_from else None
resume_optimizer_path = os.path.abspath(arg_parsed.resume_optimizer_from) if arg_parsed.resume_optimizer_from else None

results_dir = os.path.abspath(os.path.dirname(__file__)) if arg_parsed.run_eval_only else par.results_dir
logger.initialize(working_dir=results_dir, use_tensorboard=True)

# set the visible GPUs
if gpu_ids:
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(i) for i in gpu_ids])
    logger.print("CUDA_VISIVLE_DEVICES: %s" % os.environ["CUDA_VISIBLE_DEVICES"])

if not arg_parsed.run_eval_only:
    start_t = time.time()
    trainer.train(resume_model_path, resume_optimizer_path, arg_parsed.description)
    logger.print("Training took %.2fs" % (time.time() - start_t))

for tag in ["eval", "valid", "train", "checkpoint" ]:
    seq_results_dir = gen_trajectory(os.path.join(results_dir, "saved_model.%s" % tag),
                                     par.valid_seqs + par.train_seqs, par.seq_len)
    plot_trajectory(seq_results_dir)
    calc_error(seq_results_dir)
    plot_errors(seq_results_dir)

    if par.dataset() == "KITTI":
        np_traj_to_kitti(seq_results_dir)
        kitti_eval(seq_results_dir, par.train_seqs, par.valid_seqs)
    elif par.dataset() == "EUROC":
        euroc_eval(seq_results_dir, par.train_seqs)
        euroc_eval(seq_results_dir, par.valid_seqs)