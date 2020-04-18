import os
import pandas as pd
import numpy as np
import torch
import se3
from PIL import Image
from torch.utils.data import Dataset
from log import logger
from torchvision import transforms
import model
import copy
import time
from params import par


class Subsequence(object):

    def __init__(self, frames, g_i, bw_0, ba_0, T_cam_imu, length, seq, type, idx, idx_next):
        assert (length == len(frames))
        assert (length == len(frames))
        self.gt_poses = np.array([f.T_i_vk for f in frames])
        self.gt_velocities = np.array([f.v_vk_i_vk for f in frames])
        self.image_paths = [f.image_path for f in frames]
        self.imu_timestamps = [f.imu_timestamps for f in frames]
        self.accel_measurements = [f.accel_measurements for f in frames]
        self.gyro_measurements = [f.gyro_measurements for f in frames]

        self.g_i = g_i
        self.bw_0 = bw_0
        self.ba_0 = ba_0
        self.T_cam_imu = T_cam_imu
        self.length = length
        self.type = type
        self.id = idx
        self.id_next = idx_next  # id to the next subsequence
        self.seq = seq


class SequenceData(object):
    class Frame(object):
        def __init__(self, image_path, timestamp, T_i_vk, v_vk_i_vk,
                     imu_poses, imu_timestamps, accel_measurements, gyro_measurements, timestamp_raw=0):
            self.image_path = image_path
            self.timestamp = timestamp
            self.T_i_vk = T_i_vk  # inertial to vehicle frame pose
            self.v_vk_i_vk = v_vk_i_vk  # velocity expressed in vehicle frame
            self.imu_timestamps = imu_timestamps
            self.imu_poses = imu_poses
            self.accel_measurements = accel_measurements
            self.gyro_measurements = gyro_measurements
            self.timestamp_raw = timestamp_raw

            assert (len(imu_timestamps) == len(accel_measurements))
            assert (len(imu_timestamps) == len(gyro_measurements))
            assert (len(imu_timestamps) == len(imu_poses))

    def __init__(self, seq):
        self.seq = seq
        self.seq_dir = os.path.join(par.data_dir, seq)
        self.pd_path = os.path.join(par.data_dir, seq, "data.pickle")
        self.df = pd.read_pickle(self.pd_path)

        self.constants_path = os.path.join(par.data_dir, seq, "constants.npy")
        self.constants = np.load(self.constants_path, allow_pickle=True).item()

        self.g_i = self.constants["g_i"]
        self.T_cam_imu = self.constants["T_cam_imu"]
        self.bw_0 = self.constants["bw_0"]
        if "ba_0" in self.constants:
            self.ba_0 = self.constants["ba_0"]
        else:
            self.ba_0 = np.zeros_like(self.bw_0)

    def get_poses(self):
        return np.array(list(self.df.loc[:, "T_i_vk"].values))

    def get_velocities(self):
        return np.array(list(self.df.loc[:, "v_vk_i_vk"]))

    def get_timestamps(self):
        return np.array(list(self.df.loc[:, "timestamp"]))

    def get_timestamps_raw(self):
        return np.array(list(self.df.loc[:, "timestamp_raw"]))

    def get_images_paths(self):
        return list(self.df.loc[:, "image_path"].values)

    def get(self, i):
        image_path = self.df.loc[i, "image_path"]
        timestamp = self.df.loc[i, "timestamp"]
        T_i_vk = self.df.loc[i, "T_i_vk"]
        v_vk_i_vk = self.df.loc[i, "v_vk_i_vk"]
        imu_poses = self.df.loc[i, "imu_poses"]
        imu_timestamps = self.df.loc[i, "imu_timestamps"]
        accel_measurements = self.df.loc[i, "accel_measurements"]
        gyro_measurements = self.df.loc[i, "gyro_measurements"]
        timestamp_raw = self.df.loc[i, "timestamp_raw"]
        return SequenceData.Frame(image_path, timestamp, T_i_vk, v_vk_i_vk,
                                  imu_poses, imu_timestamps, accel_measurements, gyro_measurements, timestamp_raw)

    def as_frames(self):
        frames = []
        for i in range(len(self.df)):
            frames.append(self.get(i))
        return frames

    @staticmethod
    def save_as_pd(data_frames, g_i, bw_0, T_cam_imu, output_dir, ba_0=np.zeros(3)):
        start_time = time.time()
        data = {"image_path": [f.image_path for f in data_frames],
                "timestamp": [f.timestamp for f in data_frames],
                "T_i_vk": [f.T_i_vk for f in data_frames],
                "v_vk_i_vk": [f.v_vk_i_vk for f in data_frames],
                "imu_timestamps": [f.imu_timestamps for f in data_frames],
                "imu_poses": [f.imu_poses for f in data_frames],
                "accel_measurements": [f.accel_measurements for f in data_frames],
                "gyro_measurements": [f.gyro_measurements for f in data_frames],
                "timestamp_raw": [f.timestamp_raw for f in data_frames]}
        df = pd.DataFrame(data, columns=data.keys())
        df.to_pickle(os.path.join(output_dir, "data.pickle"))

        constants = {
            "g_i": g_i,
            "T_cam_imu": T_cam_imu,
            "bw_0": bw_0,
            "ba_0": ba_0
        }

        np.save(os.path.join(output_dir, "constants.npy"), constants)

        logger.print("Saving pandas took %.2fs" % (time.time() - start_time))

        return df


def convert_subseqs_list_to_panda(subseqs):
    # Convert to pandas data frames
    data = {'seq_len': [subseq.length for subseq in subseqs],
            'image_path': [subseq.image_paths for subseq in subseqs],
            "seq": [subseq.seq for subseq in subseqs],
            "type": [subseq.type for subseq in subseqs],
            "id": [subseq.id for subseq in subseqs],
            "id_next": [subseq.id_next for subseq in subseqs],
            'gt_poses': [subseq.gt_poses for subseq in subseqs]}
    return pd.DataFrame(data, columns=data.keys())


def get_subseqs(sequences, seq_len, overlap, sample_times, training):
    subseq_list = []

    for seq in sequences:
        start_t = time.time()
        seq_data = SequenceData(seq)
        frames = seq_data.as_frames()

        if sample_times > 1:
            sample_interval = int(np.ceil(seq_len / sample_times))
            start_frames = list(range(0, seq_len, sample_interval))
            logger.print('Sample start from frame {}'.format(start_frames))
        else:
            start_frames = [0]

        for st in start_frames:
            jump = seq_len - overlap
            subseqs_buffer = []
            # The original image and data
            sub_seqs_vanilla = []
            for i in range(st, len(frames), jump):
                if i + seq_len <= len(frames):  # this will discard a few frames at the end
                    subseq = Subsequence(frames[i:i + seq_len], seq_data.g_i, seq_data.bw_0, seq_data.ba_0,
                                         seq_data.T_cam_imu,
                                         length=seq_len, seq=seq, type="vanilla", idx=i, idx_next=i + jump)
                    sub_seqs_vanilla.append(subseq)
            subseqs_buffer += sub_seqs_vanilla

            if training and par.data_aug_transforms.enable:
                # assert not par.enable_ekf, "Data aug transforms not compatible with EKF"
                if par.data_aug_transforms.lr_flip:
                    subseq_flipped_buffer = []
                    H = np.diag([1, -1, 1])  # reflection matrix, flip y, across the xz plane
                    for subseq in sub_seqs_vanilla:
                        subseq_flipped = copy.deepcopy(subseq)
                        subseq_flipped.gt_poses = \
                            np.array([se3.T_from_Ct(H.dot(T[0:3, 0:3].dot(H.transpose())), H.dot(T[0:3, 3]))
                                      for T in subseq.gt_poses])
                        subseq_flipped.type = subseq.type + "_flippedlr"
                        subseq_flipped_buffer.append(subseq_flipped)
                    subseqs_buffer += subseq_flipped_buffer

                if par.data_aug_transforms.ud_flip:
                    assert par.dataset() == "EUROC", "up down flips only supported for EUROC"
                    subseq_flipped_buffer = []
                    H = np.diag([-1, 1, 1])  # reflection matrix, flip x, across the yz plane, only for EUROC
                    for subseq in sub_seqs_vanilla:
                        subseq_flipped = copy.deepcopy(subseq)
                        subseq_flipped.gt_poses = \
                            np.array([se3.T_from_Ct(H.dot(T[0:3, 0:3].dot(H.transpose())), H.dot(T[0:3, 3]))
                                      for T in subseq.gt_poses])
                        subseq_flipped.type = subseq.type + "_flippedud"
                        subseq_flipped_buffer.append(subseq_flipped)
                    subseqs_buffer += subseq_flipped_buffer

                if par.data_aug_transforms.lrud_flip:
                    assert par.dataset() == "EUROC", "left right up down flips only supported for EUROC"
                    subseq_flipped_buffer = []
                    H = np.diag([-1, -1, 1])  # reflection matrix, flip x, across the yz plane, only for EUROC
                    for subseq in sub_seqs_vanilla:
                        subseq_flipped = copy.deepcopy(subseq)
                        subseq_flipped.gt_poses = \
                            np.array([se3.T_from_Ct(H.dot(T[0:3, 0:3].dot(H.transpose())), H.dot(T[0:3, 3]))
                                      for T in subseq.gt_poses])
                        subseq_flipped.type = subseq.type + "_flippedlrud"
                        subseq_flipped_buffer.append(subseq_flipped)
                    subseqs_buffer += subseq_flipped_buffer

                # Reverse, effectively doubles the number of examples
                if par.data_aug_transforms.reverse:
                    subseqs_rev_buffer = []
                    for subseq in subseqs_buffer:
                        subseq_rev = copy.deepcopy(subseq)
                        subseq_rev.image_paths = list(reversed(subseq.image_paths))
                        subseq_rev.gt_poses = np.flip(subseq.gt_poses, axis=0).copy()
                        subseq_rev.type = subseq.type + "_reversed"
                        subseq_rev.id = subseq.id_next
                        subseq_rev.id_next = subseq.id
                        subseqs_rev_buffer.append(subseq_rev)
                    subseqs_buffer += subseqs_rev_buffer

            # collect the sub-sequences
            subseq_list += subseqs_buffer

        logger.print('Folder %s finish in %.2g sec' % (seq, time.time() - start_t))

    return subseq_list


class SubseqDataset(Dataset):
    __cache = {}  # cache using across multiple SubseqDataset objects

    def __init__(self, subseqs, img_size=None, img_mean=None, img_std=(1, 1, 1),
                 minus_point_5=False, training=True, no_image=False):

        # Transforms
        self.pre_runtime_transformer = transforms.Compose([
            transforms.Resize((img_size[0], img_size[1]))
        ])

        if training:
            transform_ops = []
            if par.data_aug_rand_color.enable:
                transform_ops.append(transforms.ColorJitter(**par.data_aug_rand_color.params))
            transform_ops.append(transforms.ToTensor())
            self.runtime_transformer = transforms.Compose(transform_ops)
        else:
            self.runtime_transformer = transforms.ToTensor()

        # Normalization
        self.minus_point_5 = minus_point_5
        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std)
        self.no_image = no_image

        # log
        # logger.print("Transform parameters: ")
        # logger.print("pre_runtime_transformer:", self.pre_runtime_transformer)
        # logger.print("runtime_transformer:", self.runtime_transformer)
        # logger.print("minus_point_5:", self.minus_point_5)
        # logger.print("normalizer:", self.normalizer)

        self.subseqs = subseqs
        self.load_image_func = lambda p: self.pre_runtime_transformer(Image.open(p))

        if par.cache_image:
            total_images = self.subseqs[0].length * len(subseqs)
            counter = 0
            start_t = time.time()
            for subseq in self.subseqs:
                for path in subseq.image_paths:
                    if path not in SubseqDataset.__cache:
                        SubseqDataset.__cache[path] = self.load_image_func(path)
                    counter += 1
                    print("Processed %d/%d (%.2f%%)" % (counter, total_images, counter / total_images * 100), end="\r")
            logger.print("Image preprocessing took %.2fs" % (time.time() - start_t))

        # since IMU data might have different length, find the longest length of IMU data so we can pad the
        # the rest with zeros
        imu_data_lengths = []
        for s in self.subseqs:
            imu_data_lengths += [len(t) for t in s.imu_timestamps]
        self.max_imu_data_length = max(imu_data_lengths)

    def __getitem__(self, index):
        subseq = self.subseqs[index]

        gt_rel_poses = []
        imu_data = []
        for i in range(1, len(subseq.gt_poses)):
            # get relative poses
            T_i_vkm1 = subseq.gt_poses[i - 1]
            T_i_vk = subseq.gt_poses[i]
            T_vkm1_vk = se3.reorthogonalize_SE3(np.linalg.inv(T_i_vkm1).dot(T_i_vk))
            r_vk_vkm1_vkm1 = T_vkm1_vk[0:3, 3]  # get the translation from T
            phi_vkm1_vk = se3.log_SO3(T_vkm1_vk[0:3, 0:3])
            gt_rel_poses.append(np.concatenate([phi_vkm1_vk, r_vk_vkm1_vkm1, ]))

        for i in range(0, len(subseq.gt_poses)):
            imu_dat_concat = np.concatenate([np.expand_dims(subseq.imu_timestamps[i], 1),
                                             subseq.gyro_measurements[i], subseq.accel_measurements[i]], axis=1)
            imu_dat_padded = np.full([self.max_imu_data_length, 7], 0.0)
            imu_dat_padded[0:len(imu_dat_concat), :] = imu_dat_concat
            imu_dat_padded[len(imu_dat_concat):, 0] = imu_dat_concat[-1, 0] if len(imu_dat_concat) > 0 else 0

            imu_data.append(imu_dat_padded)

        gt_rel_poses = torch.tensor(gt_rel_poses, dtype=torch.float32)
        gt_poses = torch.tensor(subseq.gt_poses, dtype=torch.float32)
        gt_velocities = torch.tensor(subseq.gt_velocities, dtype=torch.float32)
        imu_data = torch.tensor(imu_data, dtype=torch.float32)

        init_g = torch.tensor(subseq.gt_poses[0, 0:3, 0:3].transpose().dot(subseq.g_i), dtype=torch.float32)
        bw_0 = torch.tensor(subseq.bw_0, dtype=torch.float32)
        ba_0 = torch.tensor(subseq.ba_0, dtype=torch.float32)

        if par.cal_override_enable:
            T_imu_cam = torch.tensor(par.T_imu_cam_override, dtype=torch.float32)
        else:
            T_imu_cam = torch.tensor(np.linalg.inv(subseq.T_cam_imu), dtype=torch.float32)  # EKF takes T_imu_cam

        init_state = model.IMUKalmanFilter.encode_state(init_g,
                                                        torch.eye(3, 3),  # C
                                                        torch.zeros(3),  # r
                                                        gt_velocities[0],  # v
                                                        bw_0,  # bw
                                                        ba_0,  # ba
                                                        torch.tensor(1, dtype=torch.float32))  # lambda

        if self.no_image:
            return (subseq.length, subseq.seq, subseq.type, subseq.id, subseq.id_next), \
                   torch.zeros(1), imu_data, init_state, T_imu_cam, gt_poses, gt_rel_poses

        # process images
        images = []
        for img_path in subseq.image_paths:

            if par.cache_image:
                image = SubseqDataset.__cache[img_path]
            else:
                image = self.load_image_func(img_path)

            if "flippedlr" in subseq.type:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif "flippedud" in subseq.type:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            elif "flippedlrud" in subseq.type:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image = image.transpose(Image.FLIP_TOP_BOTTOM)

            image = self.runtime_transformer(image)
            if self.minus_point_5:
                image = image - 0.5  # from [0, 1] -> [-0.5, 0.5]
            image = self.normalizer(image)

            # if monochrome, repeat channel 3 times
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)

            images.append(image)
        images = torch.stack(images, 0)

        if "flipped" in subseq.type or "reversed" in subseq.type:
            invalid_imu = True
        else:
            invalid_imu = False

        return (subseq.length, subseq.seq, subseq.type, subseq.id, subseq.id_next, invalid_imu), \
               images, imu_data, init_state, T_imu_cam, gt_poses, gt_rel_poses

    @staticmethod
    def decode_batch_meta_info(batch_meta_info):
        seq_len_list = batch_meta_info[0]
        seq_list = batch_meta_info[1]
        type_list = batch_meta_info[2]
        id_list = batch_meta_info[3]
        id_next_list = batch_meta_info[4]
        invalid_imu_list = batch_meta_info[5]

        # check batch dimension is consistent
        assert (len(seq_list) == len(seq_len_list))
        assert (len(seq_list) == len(type_list))
        assert (len(seq_list) == len(id_list))
        assert (len(seq_list) == len(id_next_list))
        assert (len(seq_list) == len(invalid_imu_list))

        return seq_len_list, seq_list, type_list, id_list, id_next_list, invalid_imu_list

    @staticmethod
    def decode_imu_data_b(imu_data):
        t = imu_data[..., 0].view(-1, 1, 1)
        gyro = imu_data[..., 1:4].view(-1, 3, 1)
        accel = imu_data[..., 4:7].view(-1, 3, 1)

        return t, gyro, accel

    def __len__(self):
        return len(self.subseqs)
