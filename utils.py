import matplotlib.pyplot as plt
import os
import log

class Plotter(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.counter = 0

    def plot(self, plots, xlabel, ylabel, title, labels=None, equal_axes=False, filename=None):
        if not labels:
            labels_txt = [None] * len(plots)
        else:
            labels_txt = labels
        assert (len(plots) == len(labels_txt))

        plt.clf()
        for i in range(0, len(plots)):
            plt.plot(*plots[i], label=labels_txt[i])

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        if equal_axes:
            plt.axis("equal")

        if labels:
            plt.legend()

        plt.grid()
        if filename is None:
            filename = "%02d_%s.png" % (self.counter, "_".join(title.lower().split()))
        plt.savefig(log.Logger.ensure_file_dir_exists(os.path.join(self.output_dir, filename)))
        self.counter += 1
