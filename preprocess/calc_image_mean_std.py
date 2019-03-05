import os
from params import par


def calc_image_mean_std(sequences):



    for seq in sequences:

        seq_data_path = os.path.join(par.data_dir, seq)
        pass
