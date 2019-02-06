import os
import subprocess
from log import logger, Logger
import glob
import numpy as np
import prettytable


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        if "page written on" not in stdout_line and "Heiko Oberdiek" not in stdout_line:
            yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def compute_error_for_each_seq(kitti_eval_out_dir):
    # print the errors
    seq_error_files = os.listdir(os.path.join(kitti_eval_out_dir, "errors"))
    seq_errors = {}
    for seq_error_file in seq_error_files:
        seq = os.path.splitext(seq_error_file)[0]
        errors = np.loadtxt(os.path.join(kitti_eval_out_dir, "errors", seq_error_file))
        seq_errors[seq] = (np.average(errors[:, 2]), np.average(errors[:, 1]),)  # translation and rotation

    ave_errors = np.loadtxt(os.path.join(kitti_eval_out_dir, "stats.txt"))

    return seq_errors, (ave_errors[0], ave_errors[1],)


def print_error_table(errors, ave_errors):
    table = prettytable.PrettyTable()
    table.field_names = ["Seq.", "Trans. Err", "Rot. Error"]
    table.align["Seq."] = "l"
    table.align["Trans. Err"] = "r"
    table.align["Rot. Error"] = "r"
    keys = sorted(list(errors.keys()))
    for key in keys:
        table.add_row([key, "%.6f" % errors[key][0], "%.6f" % (errors[key][1] * 180 / np.pi)])
    table.add_row(["Ave.", "%.6f" % ave_errors[0], "%.6f" % (ave_errors[1] * 180 / np.pi)])
    logger.print(table)


def kitti_eval(working_dir, train_sequences, val_sequences):
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ Evaluate KITTI ================")
    logger.print("Working on directory:", working_dir)

    executable = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "kitti_eval", "cpp", "evaluate_odometry")
    kitti_dir = os.path.join(working_dir, "kitti")

    available_seqs = []
    for f in sorted(os.listdir(kitti_dir)):
        if f.endswith(".txt"):
            available_seqs.append(f[0:2])

    assert (len(available_seqs) // 2 == len(set(available_seqs)))
    available_seqs = list(set(available_seqs))

    print("Poses not available for these sequences:",
          ", ".join(list(set(available_seqs) ^ set(train_sequences + val_sequences))))

    if train_sequences:
        logger.print("Evaluating training sequences...")
        logger.print("Training sequences: ", ", ".join(list(set(available_seqs) & set(train_sequences))))
        cmd = [executable, kitti_dir, Logger.make_dir_if_not_exist(os.path.join(kitti_dir, "train"))] + list(
                set(available_seqs) & set(train_sequences))
        for line in execute(cmd):
            logger.print(line.strip())

    if val_sequences:
        logger.print("Evaluating validation sequences...")
        logger.print("Validation sequences:", " ,".join(list(set(available_seqs) & set(val_sequences))))
        cmd = [executable, kitti_dir, Logger.make_dir_if_not_exist(os.path.join(kitti_dir, "valid"))] + list(
                set(available_seqs) & set(val_sequences))
        execute(cmd)
        for line in execute(cmd):
            logger.print(line.strip())

    logger.print("Deleting useless files...")
    selected_files = list(glob.iglob(os.path.join(kitti_dir, "train", "**"), recursive=True)) + \
                     list(glob.iglob(os.path.join(kitti_dir, "valid", "**"), recursive=True))
    for filename in selected_files:
        if filename.endswith(".tex") or filename.endswith(".eps") or \
                filename.endswith(".pdf") or filename.endswith(".gp"):
            os.remove(filename)

    logger.print("Finished running KITTI evaluation!")

    # print the errors
    train_errors, ave_train_errors = compute_error_for_each_seq(os.path.join(kitti_dir, "train"))
    val_errors, ave_val_errors = compute_error_for_each_seq(os.path.join(kitti_dir, "valid"))
    logger.print("Training errors are:")
    print_error_table(train_errors, ave_train_errors)
    logger.print("Validation errors are:")
    print_error_table(val_errors, ave_val_errors)
