import torchvision
import numpy as np
import time
from PIL import Image
from params import par
from log import logger
from data_loader import SequenceData


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
              (i + 1, len(sequences), (i + 1) * 100 / len(sequences)), end="\r")
        image_paths += list(SequenceData(seq).get_images_paths())
    print()

    start_time = time.time()
    for i, path in enumerate(image_paths):
        print("Computing mean %d/%d (%.2f%%)" % (i + 1, len(image_paths), (i + 1) * 100 / len(image_paths)), end="\r")
        img = np.array(to_tensor(Image.open(path)))

        if par.minus_point_5:
            img = img - 0.5

        mean_sum += np.mean(img, (1, 2,))
        image_count += 1
    print("\nTook %.2fs" % (time.time() - start_time))

    mean = mean_sum / image_count

    num_pixels = 0
    start_time = time.time()
    for i, path in enumerate(image_paths):
        print("Computing standard deviation %d/%d (%.2f%%)" %
              (i + 1, len(image_paths), (i + 1) * 100 / len(image_paths)), end="\r")
        img = np.array(to_tensor(Image.open(path)))

        if par.minus_point_5:
            img = img - 0.5

        img[0, :, :] = img[0, :, :] - mean[0]
        img[1, :, :] = img[1, :, :] - mean[1]
        img[2, :, :] = img[2, :, :] - mean[2]
        var_sum += np.sum(np.square(img), (1, 2,))
        num_pixels += img.shape[1] * img.shape[2]
    print("\nTook %.2fs" % (time.time() - start_time))

    std = np.sqrt(var_sum / (num_pixels - 1))

    logger.print("Mean: [%f, %f, %f]" % (mean[0], mean[1], mean[2]))
    logger.print("Std: [%f, %f, %f]" % (std[0], std[1], std[2]))
