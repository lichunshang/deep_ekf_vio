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
torch.backends.cudnn.benchmark = True


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


class Parameters(object):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(Parameters, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        self.timestamp = datetime.datetime.today()

        self.n_processors = 4
        self.n_gpu = 1

        # Path Parameters
        self.project_dir = "/mnt/data/teamAI/duy/deep_ekf_vio"
        self.data_dir = os.path.join(self.project_dir, "data")
        self.results_coll_dir = os.path.join(self.project_dir, "results")
        self.pose_dir = self.data_dir
        self.results_dir = os.path.join(self.results_coll_dir,
                                        "train" + "_%s" % self.timestamp.strftime('%Y%m%d-%H-%M-%S'))

        
        self.sample_times = 1

        self.exclude_resume_weights = ["imu_noise_covar_weights", "init_covar_diag_sqrt"]

        # VO Model parameters
        self.fix_vo_weights = False

        self.hybrid_recurrency = False
        self.rnn_hidden_size = 1000
        self.rnn_num_layers = 2
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0  # 0: no dropout
        self.clip = None
        self.batch_norm = True
        self.stateful_training = True

        # EKF parameters
        self.enable_ekf = False
        self.T_imu_cam_override = np.eye(4, 4)
        self.cal_override_enable = True

        self.train_init_covar = False
        self.train_imu_noise_covar = False
        self.vis_meas_covar_use_fixed = False

        # Training parameters
        self.epochs = 50
        self.batch_size = 64
        self.seq_len = 4
        self.iters = 1
        self.pin_mem = True
        self.cache_image = True
        self.optimizer = torch.optim.Adam
        self.optimizer_args = {'lr': 1e-4}
        self.param_specific_lr = {
            "init_covar_diag_sqrt": 2*1e-1,
            "imu_noise_covar_weights.*": 2*1e-1
        }

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
        # Pretrain, Resume training
        # self.pretrained_flownet = os.path.join(self.project_dir, './pretrained/ems_transposenet_7scenes_pretrained.pth')
        self.pretrained = None
        # self.pretrained = '/mnt/data/teamAI/duy/deep_ekf_vio/results/train_20230715-01-47-04/saved_model.eval'
        # Choice:
        # None
        # './pretrained/flownets_bn_EPE2.459.pth.tar'
        # './pretrained/flownets_EPE1.951.pth.tar'

    def wc(self, seqs):
        available_seqs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        ret_seqs = []
        for seq in seqs:
            regex = re.compile(fnmatch.translate(seq))
            start_cnt = len(ret_seqs)
            for available_seq in sorted(available_seqs):
                if regex.match(available_seq):
                    ret_seqs.append(available_seq)
            if not (len(ret_seqs) > start_cnt):
                print("WARN!!! len(ret_seqs) > start_cnt")
        return ret_seqs
    

    def dataset(self):
        raise NotImplementedError("Dataset no specified")


class KITTIParams(Parameters):
    def __init__(self):
        Parameters.__init__(self)

        self.all_seqs = self.wc(['K00_*', 'K01', 'K02_*', 'K04', 'K05_*', 'K06', 'K07', 'K08', 'K09', 'K10'])
        self.eval_seq = 'K10'


        # self.valid_seqs = ['K10']
        # self.train_seqs = [x for x in self.all_seqs if not x == self.eval_seq and x not in self.valid_seqs]

        self.train_seqs = ['K07']
        self.valid_seqs = ['K07']

        self.img_w = 320
        self.img_h = 96
        self.img_means = (-0.138843, -0.119405, -0.123209)
        self.img_stds = (1, 1, 1)
        self.minus_point_5 = True

        #
        self.init_covar_diag_sqrt = np.array([1e-4, 1e-4, 1e-4,  # g
                                              0, 0, 0, 0, 0, 0,  # C, r
                                              1e-2, 1e-2, 1e-2,  # v
                                              1e-8, 1e-8, 1e-8,  # bw
                                              1e-1, 1e-1, 1e-1,
                                              5e0])  # ba
        self.init_covar_diag_eps = 1e-12
        #
        self.imu_noise_covar_diag = np.array([1e-7,  # w
                                              1e-7,  # bw
                                              1e-2,  # a
                                              1e-3])  # ba
        self.imu_noise_covar_beta = 4
        self.imu_noise_covar_gamma = 1

        self.vis_meas_fixed_covar = np.array([1e0, 1e0, 1e0,
                                              1e0, 1e0, 1e0])
        self.vis_meas_covar_init_guess = 1e1
        self.vis_meas_covar_beta = 3
        self.vis_meas_covar_gamma = 1

        # -----------------------------------------
        self.k1 = 1000  # rel loss angle multiplier
        self.k2 = 5000.  # abs loss angle multiplier
        self.k3 = {  # (1-k3)*abs + k3*rel weighting, not actually used
            0: 0.9,
        }
        # error scale for covar loss, not really used,
        # but must be 1.0 for self.gaussian_pdf_loss = False
        self.k4 = 1.0

        self.gaussian_pdf_loss = False

        self.data_aug_transforms = AttrDict({
            "enable": True,
            "lr_flip": True,
            "ud_flip": False,
            "lrud_flip": False,
            "reverse": True,
        })

    def dataset(self):
        return "KITTI"


class EUROCParams(Parameters):

    def __init__(self):
        Parameters.__init__(self)

        self.all_seqs = ['MH_01', 'MH_02', 'MH_03', 'MH_04', 'MH_05', "V1_01", "V1_02", 'V1_03', "V2_01", "V2_02", "V2_03"]
        self.eval_seq = "V1_03"

        # self.train_seqs = [x for x in self.all_seqs if not x == self.eval_seq]
        # self.train_seqs = ['MH_01']
        # self.valid_seqs = [self.eval_seq]

        self.train_seqs = ['MH_01', 'MH_02', 'MH_03', 'MH_04', "V1_01", "V1_02", "V2_01", "V2_02",'MH_05']
        self.valid_seqs = [ "V1_03"]

        self.img_w = 256
        self.img_h = 160
        # self.img_w = 320
        # self.img_h = 96
        self.img_means = (0,)
        self.img_stds = (1,)
        self.minus_point_5 = True

        #
        self.init_covar_diag_sqrt = np.array([1e-1, 1e-1, 1e-1,  # g
                                              0, 0, 0, 0, 0, 0,  # C, r
                                              1e-2, 1e-2, 1e-2,  # v
                                              1e-1, 1e-1, 1e-1,  # bw
                                              1e1, 1e1, 1e1,
                                              5e0])  # ba
        self.init_covar_diag_eps = 1e-12
        #
        self.imu_noise_covar_diag = np.array([1e-3,  # w
                                              1e-5,  # bw
                                              1e-1,  # a
                                              1e-2])  # ba
        self.imu_noise_covar_beta = 4
        self.imu_noise_covar_gamma = 1

        self.vis_meas_fixed_covar = np.array([1e0, 1e0, 1e0,
                                              1e0, 1e0, 1e0])
        self.vis_meas_covar_init_guess = 1e1
        self.vis_meas_covar_beta = 3
        self.vis_meas_covar_gamma = 1

        self.k1 = 1000  # rel loss angle multiplier
        self.k2 = 5000.  # abs loss angle multiplier
        self.k3 = {  # (1-k3)*abs + k3*rel weighting
            0: 0.1,
        }
        self.k4 = 1.  # error scale for covar loss

        self.gaussian_pdf_loss = False

        self.data_aug_transforms = AttrDict({
            "enable": True,
            "lr_flip": True,
            "ud_flip": False,
            "lrud_flip": False,
            "reverse": True,
        })

    def dataset(self):
        return "EUROC"


# par = KITTIParams()
par = EUROCParams()