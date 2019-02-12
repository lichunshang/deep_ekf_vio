import os
import datetime
import torch


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
        self.data_dir = os.path.join(self.project_dir, "KITTI")
        self.results_dir = os.path.join(self.project_dir, "results")
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.pose_dir = os.path.join(self.data_dir, 'pose_GT')
        self.results_dir = os.path.join(self.results_dir, "train" + "_%s" % self.timestamp.strftime('%Y%m%d-%H-%M-%S'))

        self.train_seqs = ['00', '01', '02', '05', '08', '09']
        self.valid_seqs = ['04', '06', '07', '10']
        # self.train_seqs = ['04']
        # self.valid_seqs = ['06']

        self.img_w = 320
        self.img_h = 96
        self.img_means = (-0.151812640483464, -0.13357509111350818, -0.14181910364786987)
        self.img_stds = (0.3174070577943728, 0.31982824445835345, 0.32372934976798146)
        self.minus_point_5 = True

        self.seq_len = 128
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
        self.batch_size = 8
        self.pin_mem = True
        self.optimizer = torch.optim.Adagrad
        self.optimizer_args = {'lr': 0.001}

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
            "lr_flip": False,
            "td_flip": False,
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


par = Parameters.get_instance()

# elapsed time = 0.766002893447876
# Numbers of frames in training dataset: 17437
# mean_tensor =  [-0.151812640483464, -0.13357509111350818, -0.14181910364786987]
# mean_np =  [88.78708011161852, 93.43778497818349, 91.33551888646076]
# std_tensor =  [0.3174070577943728, 0.31982824445835345, 0.32372934976798146]
# std_np =  [80.93941240862273, 81.557427180421, 82.55097977909139]

# self.img_stds = (1, 1, 1)  # (0.309122, 0.315710, 0.3226514)
