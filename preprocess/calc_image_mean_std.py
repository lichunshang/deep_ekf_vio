import os
import pandas
import torchvision
import numpy as np
from PIL import Image
from params import par
from log import logger


def calc_image_mean_std(sequences):
    logger.initialize(working_dir=par.data_dir, use_tensorboard=False)
    logger.print("================ CALC IMAGE MEAN STD ================")
    logger.print("Sequences: [%s]" % ",".join(sequences))
    to_tensor = torchvision.transforms.ToTensor()

    image_count = 0
    mean_sum = np.array([0.0, 0.0, 0.0])
    var_sum = np.array([0.0, 0.0, 0.0])
    image_paths = []

    for i, seq in enumerate(sequences):
        print("Collecting image paths %d/%d (%.2f%%)" %
              (i, len(sequences), i * 100 / len(sequences)), end="\r")
        seq_data_path = os.path.join(par.data_dir, seq, "data.pickle")
        dataframe = pandas.read_pickle(seq_data_path)
        image_paths += list(dataframe.loc[:, "image_path"].values)
    print()

    for i, path in enumerate(image_paths):
        print("Computing mean %d/%d (%.2f%%)" % (i, len(image_paths), i * 100 / len(image_paths)), end="\r")
        img = np.array(to_tensor(Image.open(path)))

        if par.minus_point_5:
            img = img - 0.5

        mean_sum += np.mean(img, (1, 2,))
        image_count += 1
    print()

    mean = mean_sum / image_count

    num_pixels = 0
    for i, path in enumerate(image_paths):
        print("Computing standard deviation %d/%d (%.2f%%)" %
              (i, len(image_paths), i * 100 / len(image_paths)), end="\r")
        img = np.array(to_tensor(Image.open(path)))

        if par.minus_point_5:
            img = img - 0.5

        img[0, :, :] = img[0, :, :] - mean[0]
        img[1, :, :] = img[1, :, :] - mean[1]
        img[2, :, :] = img[2, :, :] - mean[2]
        var_sum += np.sum(np.square(img), (1, 2,))
        num_pixels += img.shape[1] * img.shape[2]

    print()
    
    std = np.sqrt(var_sum / (num_pixels - 1))

    logger.print("Mean: [%f, %f, %f]" % (mean[0], mean[1], mean[2]))
    logger.print("Std: [%f, %f, %f]" % (std[0], std[1], std[2]))
