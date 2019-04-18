import os
import datetime
import torch
import fnmatch
import re
import numpy as np

torch.set_printoptions(linewidth=1024, precision=10)
np.set_printoptions(linewidth=1024)

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


class Parameters(object):
    __instance = None

    def __init__(self):
        self.timestamp = datetime.datetime.today()

        self.n_processors = 8
        self.n_gpu = 1

        # Path Parameters
        self.project_dir = "/home/cs4li/Dev/deep_ekf_vio/"
        self.data_dir = os.path.join(self.project_dir, "data")
        self.results_coll_dir = os.path.join(self.project_dir, "results")
        self.pose_dir = os.path.join(self.data_dir, 'pose_GT')
        self.results_dir = os.path.join(self.results_coll_dir,
                                        "train" + "_%s" % self.timestamp.strftime('%Y%m%d-%H-%M-%S'))

        self.train_seqs = self.wc(['K00_*', 'K01', 'K02_*', 'K05_*', 'K08', 'K09'])
        self.valid_seqs = ['K04', 'K06', 'K07', 'K10']
        # self.train_seqs = ['K08']
        # self.valid_seqs = ['K07']

        self.seq_len = 32
        self.sample_times = 3

        self.exclude_resume_weights = ["imu_noise_covar_diag_sqrt", "init_covar_diag_sqrt"]

        self.k1 = 100.  # rel loss angle multiplier
        self.k2 = 500.  # abs loss angle multiplier
        self.k3 = 0.5  # (1-k3)*abs + k3*rel weighting

        # VO Model parameters
        self.fix_vo_weights = False
        self.img_w = 320
        self.img_h = 96
        self.img_means = (-0.138843, -0.119405, -0.123209)
        self.img_stds = (1, 1, 1)
        self.minus_point_5 = True

        self.rnn_hidden_size = 1000
        self.rnn_num_layers = 2
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0  # 0: no dropout
        self.clip = None
        self.batch_norm = True
        self.stateful_training = True
        self.gaussian_pdf_loss = True

        # EKF parameters
        self.enable_ekf = True
        self.T_imu_cam_override = np.eye(4, 4)
        self.cal_override_enable = True
        #
        self.init_covar_diag_sqrt = np.array([1e-4, 1e-4, 1e-4,  # g
                                              0, 0, 0, 0, 0, 0,  # C, r
                                              1e-2, 1e-2, 1e-2,  # v
                                              1e-8, 1e-8, 1e-8,  # bw
                                              1e-1, 1e-1, 1e-1])  # ba
        self.train_init_covar = False
        self.init_covar_diag_eps = 1e-12
        #
        self.imu_noise_covar_diag_sqrt = np.sqrt(np.array([1e-7, 1e-7, 1e-7,
                                                           1e-7, 1e-7, 1e-7,
                                                           1e-2, 1e-2, 1e-2,
                                                           1e-3, 1e-3, 1e-3]))
        self.train_imu_noise_covar = False
        self.imu_noise_covar_diag_eps = 1e-12
        #
        self.vis_meas_fixed_covar = np.array([1e0, 1e0, 1e0,
                                              1e0, 1e0, 1e0])
        self.vis_meas_covar_use_fixed = False
        self.vis_meas_covar_diag_eps = np.array([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6])

        # Training parameters
        self.epochs = 200
        self.batch_size = 16
        self.pin_mem = True
        self.cache_image = False
        self.optimizer = torch.optim.Adam
        self.optimizer_args = {'lr': 1e-4}

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
            "enable": False,
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
