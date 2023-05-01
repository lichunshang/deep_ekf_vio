import matplotlib.pyplot as plt
import os
import log
from scipy.spatial.transform import Rotation as R
import numpy as np


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

def line2mat(line_data):
    '''
    12 -> 4 x 4
    '''
    mat = np.eye(4)
    mat[0:3,:] = line_data.reshape(3,4)
    return np.matrix(mat)
def pose2motion(data, skip=0):
    '''
    data: N x 12
    all_motion (N-1-skip) x 12
    '''
    data_size = data.shape[0]
    all_motion = np.zeros((data_size-1-skip,12))
    for i in range(0,data_size-1-skip):
        pose_curr = line2mat(data[i,:])
        pose_next = line2mat(data[i+1+skip,:])
        motion = pose_curr.I*pose_next
        motion_line = np.array(motion[0:3,:]).reshape(1,12)
        all_motion[i,:] = motion_line
    return all_motion