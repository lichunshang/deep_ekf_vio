import os
import datetime
import torch


class Parameters(object):
    __instance = None

    def __init__(self):
        self.timestamp = datetime.datetime.today()

        self.n_processors = 8
        self.n_gpu = 1

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

        # Data Preprocessing
        self.img_w = 320  # original size is about 1226
        self.img_h = 96  # original size is about 370
        self.img_means = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
        self.img_stds = (1, 1, 1)  # (0.309122, 0.315710, 0.3226514)
        self.minus_point_5 = True

        self.seq_len = 32
        self.sample_times = 3

        # Model
        self.rnn_hidden_size = 1000
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0  # 0: no dropout
        self.clip = None
        self.batch_norm = True

        # Training
        self.epochs = 200
        self.batch_size = 16
        self.pin_mem = True
        self.optimizer = torch.optim.Adam
        self.optimizer_args = {'lr': 0.001}

        self.stateful_training = True
        self.data_augmentation = {
            "reverse": True,
            "mirror": True,
            "reverse_mirror": True,
        }

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
