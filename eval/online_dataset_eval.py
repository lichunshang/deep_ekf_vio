from torch.utils.data import DataLoader
from eval.kitti_eval_pyimpl import KittiErrorCalc
from eval.gen_trajectory_rel import gen_trajectory_rel_iter
from data_helper import get_subseqs, SubseqDataset
from params import par
import time


class OnlineDatasetEvaluator(object):
    def __init__(self, model, sequences, eval_length):
        self.model = model  # this is a reference
        self.dataloaders = {}
        self.error_calc = KittiErrorCalc(sequences)
        for seq in sequences:
            subseqs = get_subseqs([seq], eval_length, overlap=1, sample_times=1, training=False)
            dataset = SubseqDataset(subseqs, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5,
                                    training=False)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
            self.dataloaders[seq] = dataloader

    def evaluate_rel(self):
        start_time = time.time()
        seqs = sorted(list(self.dataloaders.keys()))

        for seq in seqs:
            predicted_abs_poses = gen_trajectory_rel_iter(self.model, self.dataloaders[seq], True)
            self.error_calc.accumulate_error(seq, predicted_abs_poses)
        ave_err = self.error_calc.get_average_error()
        self.error_calc.clear()
        print("Online evaluation took %.2fs, err %.6f" % (time.time() - start_time, ave_err))

        return
