from log import logger
import numpy as np
import os
import transformations
from se3 import log_SO3, exp_SO3, interpolate_SE3, interpolate_SO3
from data_loader import SequenceData
from utils import Plotter
import matplotlib.pyplot as plt
import time

if "DISPLAY" not in os.environ:
    plt.switch_backend("Agg")

def preprocess_kitti_raw(seq_dir, output_dir, cam_subset_range):
    pass