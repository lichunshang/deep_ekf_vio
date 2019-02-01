import os
import glob
import pandas as pd
import numpy as np
import torch
import se3_math
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import time
from params import par


def get_data_info(sequences, seq_len_range, overlap, sample_times=1, pad_y=False, shuffle=False, sort=True):
    subseq_image_path_list = []
    subseq_len_list = []
    subseq_type_list = []
    subseq_seq_list = []
    subseq_id_list = []
    subseq_gt_pose_list = []
    assert (seq_len_range[0] == seq_len_range[1])
    for seq in sequences:
        start_t = time.time()
        gt_poses = np.load(os.path.join(par.pose_dir, seq + ".npy"))
        fpaths = sorted(glob.glob(os.path.join(par.image_dir, seq, "*.png")))
        assert (len(gt_poses) == len(fpaths))  # make sure the number of images corresponds to number of poses

        if sample_times > 1:
            sample_interval = int(np.ceil(seq_len_range[0] / sample_times))
            start_frames = list(range(0, seq_len_range[0], sample_interval))
            print('Sample start from frame {}'.format(start_frames))
        else:
            start_frames = [0]

        for st in start_frames:
            seq_len = seq_len_range[0]
            jump = seq_len - overlap

            # The original image and data
            subseq_image_path, subseq_gt_pose, subseq_ids = [], [], []
            for i in range(st, len(fpaths), jump):
                if i + seq_len <= len(fpaths):  # this will discard a few frames at the end
                    subseq_image_path.append(fpaths[i:i + seq_len])
                    subseq_gt_pose.append(gt_poses[i:i + seq_len])
                    subseq_ids.append(np.array([i, i + jump]))

            subseq_type = ["normal"] * len(subseq_image_path)
            subseq_seq = [seq] * len(subseq_image_path)

            # TODO Mirrors and going in reverse

            subseq_gt_pose_list += subseq_gt_pose
            subseq_image_path_list += subseq_image_path
            subseq_len_list += [len(xs) for xs in subseq_image_path]
            subseq_seq_list += subseq_seq
            subseq_type_list += subseq_type
            subseq_id_list += subseq_ids

            # ensure all sequence length are the same
            assert (subseq_len_list.count(seq_len) == len(subseq_len_list))
        print('Folder {} finish in {} sec'.format(seq, time.time() - start_t))

    # Convert to pandas dataframes
    data = {'seq_len': subseq_len_list, 'image_path': subseq_image_path_list, "seq": subseq_seq_list,
            "type": subseq_type_list, "id": subseq_id_list, 'pose': subseq_gt_pose_list}
    return pd.DataFrame(data, columns=data.keys())


class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)
        # Calculate len (num of batches, not num of samples)
        self.len = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            self.len += n_batch

    def __iter__(self):

        # Calculate number of sameples in each group (grouped by seq_len)
        list_batch_indexes = []
        start_idx = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            rand_idxs = (start_idx + torch.randperm(n_sample)).tolist()
            tmp = [rand_idxs[s * self.batch_size: s * self.batch_size + self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len


class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_sizeize=None, img_mean=None, img_std=(1, 1, 1),
                 minus_point_5=False):
        # Transforms
        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((new_sizeize[0], new_sizeize[1])))
        elif resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((new_sizeize[0], new_sizeize[1])))
        transform_ops.append(transforms.ToTensor())
        # transform_ops.append(transforms.Normalize(mean=img_mean, std=img_std))
        self.transformer = transforms.Compose(transform_ops)
        self.minus_point_5 = minus_point_5
        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std)

        self.data_info = info_dataframe
        self.subseq_len_list = list(self.data_info.seq_len)
        self.subseq_image_path_list = np.asarray(self.data_info.image_path)  # image paths
        self.subseq_gt_pose_list = np.asarray(self.data_info.pose)

        self.subseq_type_list = np.asarray(self.data_info.type)
        self.subseq_seq_list = np.asarray(self.data_info.seq)
        self.subseq_id_list = np.asarray(self.data_info.id)

    def __getitem__(self, index):
        gt_poses = self.subseq_gt_pose_list[index]
        type = self.subseq_type_list[index]
        seq = self.subseq_seq_list[index]
        id = self.subseq_id_list[index]
        # transform
        gt_rel_poses = []
        for i in range(1, len(gt_poses)):
            T_i_vkm1 = gt_poses[i - 1]
            T_i_vk = gt_poses[i]
            T_vkm1_k = se3_math.reorthogonalize_SE3(np.linalg.inv(T_i_vkm1).dot(T_i_vk))
            r_vk_vkm1_vkm1 = T_vkm1_k[0:3, 3]  # get the translation from T
            phi_vkm1_vk = se3_math.log_SO3(T_vkm1_k[0:3, 0:3])
            gt_rel_poses.append(np.concatenate([r_vk_vkm1_vkm1, phi_vkm1_vk, ]))

        gt_rel_poses = torch.FloatTensor(gt_rel_poses)

        image_paths = self.subseq_image_path_list[index]
        assert (self.subseq_len_list[index] == len(image_paths))
        seq_len = self.subseq_len_list[index]

        image_sequence = []
        for img_path in image_paths:
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transformer(img_as_img)
            if self.minus_point_5:
                img_as_tensor = img_as_tensor - 0.5  # from [0, 1] -> [-0.5, 0.5]
            img_as_tensor = self.normalizer(img_as_tensor)
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)

        return (seq_len, seq, type, id), image_sequence, gt_rel_poses

    def __len__(self):
        return len(self.data_info.index)
