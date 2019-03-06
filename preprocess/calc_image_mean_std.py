import os
import pandas
import torch
import torchvision
from PIL import Image
from params import par
from log import logger


def calc_image_mean_std(sequences):
    logger.initialize(working_dir=par.data_dir, use_tensorboard=False)
    logger.print("================ PREPROCESS KITTI RAW ================")
    to_tensor = torchvision.transforms.ToTensor()

    for seq in sequences:
        seq_data_path = os.path.join(par.data_dir, seq, "data.pickle")
        dataframe = pandas.read_pickle(seq_data_path)
        image_paths = dataframe.loc[:, "image_path"].values
        cnt_pixels = 0

        for path in image_paths:
            img = to_tensor(Image.open(path))
            torch.mean(img)
            torch.std(img)
            # if par.minus_point_5:
            #     img_as_tensor = img_as_tensor - 0.5
            # img_as_np = np.array(img_as_img)
            # img_as_np = np.rollaxis(img_as_np, 2, 0)
            # cnt_pixels += img_as_np.shape[1] * img_as_np.shape[2]