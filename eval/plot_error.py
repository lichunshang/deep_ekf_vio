import os
import numpy as np
import matplotlib.pyplot as plt
from params import par
from log import logger, Logger
from scipy import stats

if "DISPLAY" not in os.environ:
    plt.switch_backend("Agg")


def plot_errors(working_dir):
    errors_dir = os.path.join(working_dir, "errors")
    rel_errors_dir = os.path.join(errors_dir, "rel")
    abs_errors_dir = os.path.join(errors_dir, "abs")
    vis_meas_errors_dir = os.path.join(errors_dir, "vis_meas")
    vis_meas_covars_dir = os.path.join(working_dir, "vis_meas", "covar")
    abs_errors_files = sorted(os.listdir(abs_errors_dir))
    rel_errors_files = sorted(os.listdir(rel_errors_dir))
    vis_meas_errors_files = sorted(os.listdir(vis_meas_errors_dir))
    vis_meas_covars_files = sorted(os.listdir(vis_meas_covars_dir))

    assert (abs_errors_files == rel_errors_files == vis_meas_errors_files == vis_meas_covars_files)
    sequences = [f[:-4] for f in abs_errors_files]

    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ PLOT ERROR ================")
    logger.print("Working on directory:", working_dir)
    logger.print("Found sequences: [%s]" % ", ".join(sequences))

    for i, sequence in enumerate(sequences):
        error_rel = np.load(os.path.join(rel_errors_dir, "%s.npy" % sequence))
        error_abs = np.load(os.path.join(abs_errors_dir, "%s.npy" % sequence))
        error_vis_meas = np.load(os.path.join(vis_meas_errors_dir, "%s.npy" % sequence))
        covar_vis_meas = np.load(os.path.join(vis_meas_covars_dir, "%s.npy" % sequence))

        labels = ["Rot X", "Rot Y", "Rot Z", "Trans X", "Trans Y", "Trans Z"]

        for j in range(0, 6):
            err = error_rel[:, j]
            plt.clf()
            plt.plot(err, color="r")
            plt.xlabel("frame # []")
            plt.ylabel(labels[j].lower())
            plt.title("Seq. %s %s Rel Error" % (sequence, labels[j]))
            plt.savefig(Logger.ensure_file_dir_exists(
                    os.path.join(errors_dir, "rel_error_figures",
                                 "seq_%s_%02d_%s_rel_err_plt.png" % (
                                     sequence, j, "_".join(labels[j].lower().split())))))
            plt.clf()
            plt.hist(err, bins=np.linspace(start=np.min(err), stop=np.max(err), num=100), normed=True)

            ticks = plt.xticks()[0]
            lnspc = np.linspace(min(ticks), max(ticks), len(err))
            mean, std_dev = stats.norm.fit(err)
            pdf_g = stats.norm.pdf(lnspc, mean, std_dev)
            plt.plot(lnspc, pdf_g, color="r")
            plt.text(0.05, 0.9, r"$\mu$=%.4g" % mean + "\n" + r"$\sigma$=%.4g" % std_dev,
                     transform=plt.gca().transAxes)
            plt.title("Seq. %s %s Rel Error Hist." % (sequence, labels[j]))
            plt.savefig(Logger.ensure_file_dir_exists(
                    os.path.join(errors_dir, "rel_error_histograms",
                                 "fig_%s_%02d_%s_rel_err_hist.png" % (
                                     sequence, j, "_".join(labels[j].lower().split())))))

        for j in range(0, 6):
            err = np.abs(error_vis_meas[:, j])
            covar = covar_vis_meas[:, j, j]
            if j in [0, 1, 2]: covar = covar / par.k1

            covar_sig = np.sqrt(covar)
            plt.clf()
            plt.plot(err, color="r", label="Err")
            plt.plot(-err, color="r", label="Err")
            plt.plot(covar_sig, color="b", label="3sig")
            plt.plot(-covar_sig, color="b", label="3sig")
            plt.plot()
            plt.xlabel("frame # []")
            plt.ylabel(labels[j].lower())
            plt.title("Seq. %s %s Vis Meas Error & Covar" % (sequence, labels[j]))
            plt.legend()
            plt.savefig(Logger.ensure_file_dir_exists(
                    os.path.join(errors_dir, "vis_meas_error_covar_figures",
                                 "fig_%s_%02d_%s_vis_meas_error_covar.png" % (
                                     sequence, j, "_".join(labels[j].lower().split())))))

        for j in range(0, 6):
            err = error_abs[:, j]
            plt.clf()
            plt.plot(err, color="r")
            plt.xlabel("frame # []")
            plt.ylabel(labels[j].lower())
            plt.title("Seq. %s %s Abs Error" % (sequence, labels[j]))
            plt.savefig(Logger.ensure_file_dir_exists(
                    os.path.join(errors_dir, "abs_error_figures",
                                 "seq_%s_%02d_%s_abs_err_plt.png" % (
                                     sequence, j, "_".join(labels[j].lower().split())))))

        logger.print("Plots saved for sequence %s" % sequence)
