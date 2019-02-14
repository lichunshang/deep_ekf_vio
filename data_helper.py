import os
import glob
import pandas as pd
import numpy as np
import torch
import se3_math
from PIL import Image
from torch.utils.data import Dataset
from log import logger
from torchvision import transforms
import time
from params import par
from cache import image_cache


class Subsequence(object):
    def __init__(self, image_paths, length, seq, type, idx, idx_next, gt_poses):
        self.image_paths = image_paths[:]
        self.length = length
        self.type = type
        self.id = idx
        self.id_next = idx_next  # id to the next subsequence
        self.gt_poses = gt_poses
        self.seq = seq

        assert (self.length == len(image_paths))
        assert (self.length == len(gt_poses))


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
        gt_poses = np.load(os.path.join(par.pose_dir, seq + ".npy"))
        image_paths = sorted(glob.glob(os.path.join(par.image_dir, seq, "*.png")))
        assert (len(gt_poses) == len(image_paths))  # make sure the number of images corresponds to number of poses

        if sample_times > 1:
            sample_interval = int(np.ceil(seq_len / sample_times))
            start_frames = list(range(0, seq_len, sample_interval))
            print('Sample start from frame {}'.format(start_frames))
        else:
            start_frames = [0]

        for st in start_frames:
            jump = seq_len - overlap
            subseqs_buffer = []
            # The original image and data
            sub_seqs_vanilla = []
            for i in range(st, len(image_paths), jump):
                if i + seq_len <= len(image_paths):  # this will discard a few frames at the end
                    subseq = Subsequence(image_paths[i:i + seq_len], length=seq_len, seq=seq, type="vanilla", idx=i,
                                         idx_next=i + jump, gt_poses=gt_poses[i:i + seq_len])
                    sub_seqs_vanilla.append(subseq)
            subseqs_buffer += sub_seqs_vanilla

            if training and par.data_aug_transforms.enable:
                if par.data_aug_transforms.lr_flip:
                    subseq_flipped_buffer = []
                    H = np.diag([-1, 1, 1])  # reflection matrix, flip x, across the yz plane
                    for subseq in sub_seqs_vanilla:
                        flipped_gt_poses = [se3_math.T_from_Ct(H.dot(T[0:3, 0:3].dot(H.transpose())),
                                                               H.dot(T[0:3, 3])) for T in subseq.gt_poses]
                        subseq_flipped = Subsequence(subseq.image_paths, length=subseq.length, seq=seq,
                                                     type=subseq.type + "_flippedlr", idx=subseq.id,
                                                     idx_next=subseq.id_next, gt_poses=flipped_gt_poses)
                        subseq_flipped_buffer.append(subseq_flipped)
                    subseqs_buffer += subseq_flipped_buffer

                # Reverse, effectively doubles the number of examples
                if par.data_aug_transforms.reverse:
                    subseqs_rev_buffer = []
                    for subseq in subseqs_buffer:
                        subseq_rev = Subsequence(list(reversed(subseq.image_paths)), length=subseq.length, seq=seq,
                                                 type=subseq.type + "_reversed",
                                                 idx=subseq.id_next, idx_next=subseq.id,
                                                 gt_poses=np.flip(subseq.gt_poses, axis=0))
                        subseqs_rev_buffer.append(subseq_rev)
                    subseqs_buffer += subseqs_rev_buffer

            # collect the sub-sequences
            subseq_list += subseqs_buffer

        print('Folder %s finish in %.2g sec' % (seq, time.time() - start_t))

    return subseq_list


class SubseqDataset(Dataset):
    def __init__(self, subseqs, img_size=None, img_mean=None, img_std=(1, 1, 1),
                 minus_point_5=False, training=True):

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

        # log
        # logger.print("Transform parameters: ")
        # logger.print("pre_runtime_transformer:", self.pre_runtime_transformer)
        # logger.print("runtime_transformer:", self.runtime_transformer)
        # logger.print("minus_point_5:", self.minus_point_5)
        # logger.print("normalizer:", self.normalizer)

        # organize data
        self.subseqs = subseqs
        total_images = self.subseqs[0].length * len(subseqs)
        counter = 0
        start_t = time.time()
        for subseq in self.subseqs:
            for path in subseq.image_paths:
                if not image_cache.exists(path):
                    image_cache.store(path, self.pre_runtime_transformer(Image.open(path)))
                counter += 1
                print("Processed %d/%d (%.2f%%)" % (counter, total_images, counter / total_images * 100), end="\r")
        logger.print("Image preprocessing took %.2fs" % (time.time() - start_t))

    def __getitem__(self, index):
        subseq = self.subseqs[index]

        # get relative poses
        gt_rel_poses = []
        for i in range(1, len(subseq.gt_poses)):
            T_i_vkm1 = subseq.gt_poses[i - 1]
            T_i_vk = subseq.gt_poses[i]
            T_vkm1_k = se3_math.reorthogonalize_SE3(np.linalg.inv(T_i_vkm1).dot(T_i_vk))
            r_vk_vkm1_vkm1 = T_vkm1_k[0:3, 3]  # get the translation from T
            phi_vkm1_vk = se3_math.log_SO3(T_vkm1_k[0:3, 0:3])
            gt_rel_poses.append(np.concatenate([r_vk_vkm1_vkm1, phi_vkm1_vk, ]))

        gt_rel_poses = torch.FloatTensor(gt_rel_poses)

        image_sequence = []
        for img_path in subseq.image_paths:

            image = image_cache.get(img_path)
            if "flippedlr" in subseq.type:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = self.runtime_transformer(image)
            if self.minus_point_5:
                image = image - 0.5  # from [0, 1] -> [-0.5, 0.5]
            image = self.normalizer(image)
            image_sequence.append(image)
        image_sequence = torch.stack(image_sequence, 0)

        return (subseq.length, subseq.seq, subseq.type, subseq.id, subseq.id_next), image_sequence, gt_rel_poses

    @staticmethod
    def decode_batch_meta_info(batch_meta_info):
        seq_len_list = batch_meta_info[0]
        seq_list = batch_meta_info[1]
        type_list = batch_meta_info[2]
        id_list = batch_meta_info[3]
        id_next_list = batch_meta_info[4]

        # check batch dimension is consistent
        assert (len(seq_list) == len(seq_len_list))
        assert (len(seq_list) == len(type_list))
        assert (len(seq_list) == len(id_list))
        assert (len(seq_list) == len(id_next_list))

        return seq_len_list, seq_list, type_list, id_list, id_next_list

    def __len__(self):
        return len(self.subseqs)
