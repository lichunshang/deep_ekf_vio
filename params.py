import os
import datetime
import torch
import fnmatch
import re


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


class Parameters(object):
    __instance = None

    def __init__(self):
        self.timestamp = datetime.datetime.today()

        self.n_processors = 8
        self.n_gpu = 2

        # Path
        self.project_dir = "/home/cs4li/Dev/deep_ekf_vio/"
        self.data_dir = os.path.join(self.project_dir, "data")
        self.results_coll_dir = os.path.join(self.project_dir, "results")
        self.pose_dir = os.path.join(self.data_dir, 'pose_GT')
        self.results_dir = os.path.join(self.results_coll_dir,
                                        "train" + "_%s" % self.timestamp.strftime('%Y%m%d-%H-%M-%S'))

        self.train_seqs = self.wc(['K00_*', 'K01', 'K02_*', 'K05_*', 'K08', 'K09'])
        self.valid_seqs = ['K04', 'K06', 'K07', 'K10']

        self.img_w = 320 * 2
        self.img_h = 96 * 2
        self.img_means = (-0.138843, -0.119405, -0.123209)
        self.img_stds = (1, 1, 1)
        self.minus_point_5 = True

        self.seq_len = 112
        self.sample_times = 3

        # Model
        self.rnn_hidden_size = 1000
        self.rnn_num_layers = 2
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0  # 0: no dropout
        self.clip = None
        self.batch_norm = True

        # Training
        self.epochs = 200
        self.batch_size = 2
        self.pin_mem = True
        self.cache_image = True
        self.optimizer = torch.optim.Adam
        self.optimizer_args = {'lr': 0.0001}

        self.stateful_training = False

        # data augmentation
        self.data_aug_rand_color = AttrDict({
            "enable": True,
            "params": {
                "brightness": 0.1,
                "contrast": 0.1,
                "saturation": 0.1,
                "hue": 0.05
            }
        })
        self.data_aug_transforms = AttrDict({
            "enable": True,
            "lr_flip": True,
            "reverse": True,
        })

        # Pretrain, Resume training
        self.pretrained_flownet = os.path.join(self.project_dir, './pretrained/flownets_bn_EPE2.459.pth.tar')
        # Choice:
        # None
        # './pretrained/flownets_bn_EPE2.459.pth.tar'
        # './pretrained/flownets_EPE1.951.pth.tar'

        # validation
        assert (len(list(set(self.train_seqs) & set(self.valid_seqs))) == 0)

    @staticmethod
    def get_instance():
        if not Parameters.__instance:
            Parameters.__instance = Parameters()
        return Parameters.__instance

    def wc(self, seqs):
        available_seqs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        ret_seqs = []
        for seq in seqs:
            regex = re.compile(fnmatch.translate(seq))
            start_cnt = len(ret_seqs)
            for available_seq in sorted(available_seqs):
                if regex.match(available_seq):
                    ret_seqs.append(available_seq)
            assert (len(ret_seqs) > start_cnt)
        return ret_seqs


par = Parameters.get_instance()
