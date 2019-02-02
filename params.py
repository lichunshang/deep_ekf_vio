import os
import datetime


class Parameters(object):
    __instance = None

    @staticmethod
    def get_instance():
        if not Parameters.__instance:
            Parameters.__instance = Parameters()
        return Parameters.__instance

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

        self.train_video = ['00', '01', '02', '05', '08', '09']
        self.valid_video = ['04', '06', '07', '10']
        self.partition = None  # partition videos in 'train_video' to train / valid dataset  #0.8

        # Data Preprocessing
        self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
        self.img_w = 320  # original size is about 1226
        self.img_h = 96  # original size is about 370
        self.img_means = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
        self.img_stds = (1, 1, 1)  # (0.309122, 0.315710, 0.3226514)
        self.minus_point_5 = True

        self.seq_len = (32, 32)
        self.sample_times = 3

        # Data info path
        self.train_data_info_path = 'datainfo/train_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(
                ''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1],
                self.sample_times)
        self.valid_data_info_path = 'datainfo/valid_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(
                ''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1],
                self.sample_times)

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
        self.optim = {'opt': 'Adagrad', 'lr': 0.0005}
        # Choice:
        # {'opt': 'Adagrad', 'lr': 0.001}
        # {'opt': 'Adam'}
        # {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}

        # Pretrain, Resume training
        self.pretrained_flownet = './pretrained/flownets_bn_EPE2.459.pth.tar'
        # Choice:
        # None
        # './pretrained/flownets_bn_EPE2.459.pth.tar'
        # './pretrained/flownets_EPE1.951.pth.tar'
        self.resume = False  # resume training
        self.resume_t_or_v = '.train'
        self.load_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(''.join(self.train_video),
                                                                                           ''.join(self.valid_video),
                                                                                           self.img_h, self.img_w,
                                                                                           self.seq_len[0],
                                                                                           self.seq_len[1],
                                                                                           self.batch_size,
                                                                                           self.rnn_hidden_size,
                                                                                           '_'.join([k + str(v) for k, v
                                                                                                     in
                                                                                                     self.optim.items()]),
                                                                                           self.resume_t_or_v)
        self.load_optimizer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(
                ''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0],
                self.seq_len[1], self.batch_size, self.rnn_hidden_size,
                '_'.join([k + str(v) for k, v in self.optim.items()]), self.resume_t_or_v)

        if not os.path.isdir(os.path.dirname(self.train_data_info_path)):
            os.makedirs(os.path.dirname(self.train_data_info_path))


par = Parameters.get_instance()

# elapsed time = 0.766002893447876
# Numbers of frames in training dataset: 17437
# mean_tensor =  [-0.151812640483464, -0.13357509111350818, -0.14181910364786987]
# mean_np =  [88.78708011161852, 93.43778497818349, 91.33551888646076]
# std_tensor =  [0.3174070577943728, 0.31982824445835345, 0.32372934976798146]
# std_np =  [80.93941240862273, 81.557427180421, 82.55097977909139]
