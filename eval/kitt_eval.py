import os
import numpy as np
from log import Logger, logger


def ktti_eval(working_dir, sequences):
    logger.initialize(working_dir=working_dir, use_tensorboard=False)
    logger.print("================ Evaluate KITTI ================")
    logger.print("Working on directory:", working_dir)

