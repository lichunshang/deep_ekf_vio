import numpy as np
import os
import argparse
import trainer
import time
from params import par
from log import logger
from eval import gen_trajectory_rel, plot_trajectory, kitti_eval, np_traj_to_kitti

np.set_printoptions(linewidth=1024)
logger.initialize(working_dir=par.results_dir, use_tensorboard=True)

arg_parser = argparse.ArgumentParser(description='Train E2E VIO')
arg_parser.add_argument('--gpu_id', type=int, nargs="+", help="select the GPU to perform training on")
arg_parser.add_argument('--resume_model_from', type=str, help="path of model state to resume from")
arg_parser.add_argument('--resume_optimizer_from', type=str, help="path of optimizer state to resume from")
arg_parsed = arg_parser.parse_args()
gpu_ids = arg_parsed.gpu_id
resume_model_path = os.path.abspath(arg_parsed.resume_model_from) if arg_parsed.resume_model_from else None
resume_optimizer_path = os.path.abspath(arg_parsed.resume_optimizer_from) if arg_parsed.resume_optimizer_from else None

# set the visible GPUs
if gpu_ids:
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(i) for i in gpu_ids])
    logger.print("CUDA_VISIVLE_DEVICES: %s" % os.environ["CUDA_VISIBLE_DEVICES"])

start_t = time.time()
trainer.train(resume_model_path, resume_optimizer_path)
logger.print("Training took %.2fs" % (time.time() - start_t))

for tag in ["valid", "train"]:
    seq_results_dir = gen_trajectory_rel(os.path.join(par.results_dir, "saved_model.%s" % tag),
                                         par.valid_seqs + par.train_seqs, 2, True)
    plot_trajectory(seq_results_dir)
    np_traj_to_kitti(seq_results_dir)
    kitti_eval(seq_results_dir, par.train_seqs, par.valid_seqs)
