import matplotlib.pyplot as plt
import os
import log


class Plotter(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.counter = 0

    def plot(self, plots, xlabel, ylabel, title, labels=None, equal_axes=False, filename=None, callback=None, colors=None):
        if not labels:
            labels_txt = [None] * len(plots)
        else:
            labels_txt = labels
        assert (len(plots) == len(labels_txt))

        plt.clf()
        for i in range(0, len(plots)):
            args = {
                "label": labels_txt[i]
            }
            if colors:
                args["color"] = colors[i]
            plt.plot(*plots[i], linewidth=1, **args)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)

        if equal_axes:
            plt.axis("equal")

        if labels:
            plt.legend()

        plt.grid()
        if filename is None:
            filename = "%02d_%s.svg" % (self.counter, "_".join(title.lower().split()))

        if callback is not None:
            callback(plt.gcf(), plt.gca())

        plt.savefig(log.Logger.ensure_file_dir_exists(os.path.join(self.output_dir, filename)),  format='svg', bbox_inches='tight', pad_inches=0)
        self.counter += 1
