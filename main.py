import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import pandas as pd
import trainer
import argparse
from params import par
from model import DeepVO
from data_helper import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset, get_partition_data_info
from log import logger

np.set_printoptions(linewidth=1024)
logger.initialize(working_dir=par.results_dir, use_tensorboard=True)

arg_parser = argparse.ArgumentParser(description='Train E2E VIO')
arg_parser.add_argument('--gpu_id', type=int, nargs="+", help="select the GPU to perform training on")
gpu_ids = arg_parser.parse_args().gpu_id

train_description = input("Enter a description of this training run: ")
logger.print("Train description: ", train_description)
logger.tensorboard.add_text("description", train_description)

logger.log_parameters()
logger.log_source_files()

# set the visible GPUs
if gpu_ids:
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(i) for i in gpu_ids])
    logger.print("CUDA_VISIVLE_DEVICES: %s" % os.environ["CUDA_VISIBLE_DEVICES"])

# Prepare Data
if os.path.isfile(par.train_data_info_path) and os.path.isfile(par.valid_data_info_path):
    logger.print('Load data info from {}'.format(par.train_data_info_path))
    train_df = pd.read_pickle(par.train_data_info_path)
    valid_df = pd.read_pickle(par.valid_data_info_path)
else:
    logger.print('Create new data info')
    if par.partition != None:
        partition = par.partition
        train_df, valid_df = get_partition_data_info(partition, par.train_video, par.seq_len, overlap=1,
                                                     sample_times=par.sample_times, shuffle=True, sort=True)
    else:
        train_df = get_data_info(folder_list=par.train_video, seq_len_range=par.seq_len, overlap=1,
                                 sample_times=par.sample_times)
        valid_df = get_data_info(folder_list=par.valid_video, seq_len_range=par.seq_len, overlap=1,
                                 sample_times=par.sample_times)
    # save the data info
    train_df.to_pickle(par.train_data_info_path)
    valid_df.to_pickle(par.valid_data_info_path)

train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds,
                                     par.minus_point_5)
train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True)
valid_dataset = ImageSequenceDataset(valid_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds,
                                     par.minus_point_5)
valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

logger.print('Number of samples in training dataset: %d' % len(train_df.index))
logger.print('Number of samples in validation dataset: %d' % len(valid_df.index))
logger.print('=' * 50)

# Model
e2e_vio_model = DeepVO(par.img_h, par.img_w, par.batch_norm)
e2e_vio_model = e2e_vio_model.cuda()

# Load FlowNet weights pretrained with FlyingChairs
# NOTE: the pretrained model assumes image rgb values in range [-0.5, 0.5]
if par.pretrained_flownet and not par.resume:
    pretrained_w = torch.load(par.pretrained_flownet)
    logger.print('Load FlowNet pretrained model')
    # Use only conv-layer-part of FlowNet as CNN for DeepVO
    model_dict = e2e_vio_model.state_dict()
    update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
    model_dict.update(update_dict)
    e2e_vio_model.load_state_dict(model_dict)

# Create optimizer
optimizer = None
lr_scheduler = None
if par.optim['opt'] == 'Adam':
    optimizer = torch.optim.Adam(e2e_vio_model.parameters(), lr=0.001, betas=(0.9, 0.999))
elif par.optim['opt'] == 'Adagrad':
    optimizer = torch.optim.Adagrad(e2e_vio_model.parameters(), lr=par.optim['lr'])
elif par.optim['opt'] == 'Cosine':
    optimizer = torch.optim.SGD(e2e_vio_model.parameters(), lr=par.optim['lr'])
    T_iter = par.optim['T'] * len(train_dl)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_iter, eta_min=0, last_epoch=-1)

# Load trained DeepVO model and optimizer
if par.resume:
    e2e_vio_model.load_state_dict(torch.load(par.load_model_path))
    optimizer.load_state_dict(torch.load(par.load_optimizer_path))
    logger.print('Load model from: %s' % par.load_model_path)
    logger.print('Load optimizer from: %s' % par.load_optimizer_path)

# if to use more than one GPU
if par.n_gpu > 1:
    assert (torch.cuda.device_count() == par.n_gpu)
    e2e_vio_model = torch.nn.DataParallel(e2e_vio_model, device_ids=gpu_ids)

e2e_vio_trainer = trainer.Trainer(e2e_vio_model)

# Train
min_loss_t = 1e10
min_loss_v = 1e10
# e2e_vio_trainer.set_train_mode()
for epoch in range(par.epochs):
    st_t = time.time()
    logger.print('=' * 50)
    # Train
    e2e_vio_model.train()
    loss_mean = 0
    t_loss_list = []
    count = 0
    for _, t_x, t_y in train_dl:
        print("%d/%d (%.2f%%)" % (count, len(train_dl), 100 * count / len(train_dl)), end='\r')
        t_x = t_x.cuda(non_blocking=par.pin_mem)
        t_y = t_y.cuda(non_blocking=par.pin_mem)
        ls = e2e_vio_trainer.step(t_x, t_y, optimizer).data.cpu().numpy()
        t_loss_list.append(float(ls))
        loss_mean += float(ls)
        if par.optim == 'Cosine':
            lr_scheduler.step()
        count += 1
    logger.print('Train take {:.1f} sec'.format(time.time() - st_t))
    loss_mean /= len(train_dl)
    logger.tensorboard.add_scalar("train_loss/epochs", loss_mean, epoch)

    # Validation
    st_t = time.time()
    e2e_vio_model.eval()
    loss_mean_valid = 0
    v_loss_list = []
    for _, v_x, v_y in valid_dl:
        v_x = v_x.cuda(non_blocking=par.pin_mem)
        v_y = v_y.cuda(non_blocking=par.pin_mem)
        v_ls = e2e_vio_trainer.get_loss(v_x, v_y).data.cpu().numpy()
        v_loss_list.append(float(v_ls))
        loss_mean_valid += float(v_ls)
    logger.print('Valid take {:.1f} sec'.format(time.time() - st_t))
    loss_mean_valid /= len(valid_dl)
    logger.tensorboard.add_scalar("val_loss/epochs", loss_mean_valid, epoch)

    # Save records
    logger.print('Epoch {}\ntrain loss mean: {}, '
                 'std: {:.2f}\nvalid loss mean: {}, '
                 'std: {:.2f}\n'.format(epoch + 1, loss_mean,
                                        np.std(t_loss_list),
                                        loss_mean_valid,
                                        np.std(v_loss_list)))

    # Save model
    if (epoch + 1) % 5 == 0:
        logger.print('Save model at ep {}, checkpoint'.format(epoch + 1))  # use 4.6 sec
        torch.save(e2e_vio_model.state_dict(), par.save_model_path + '.checkpoint')
        torch.save(optimizer.state_dict(), par.save_optimizer_path + '.checkpoint')
    elif loss_mean_valid < min_loss_v:
        min_loss_v = loss_mean_valid
        logger.print('Save model at ep {}, mean of valid loss: {}'.format(epoch + 1, loss_mean_valid))  # use 4.6 sec
        torch.save(e2e_vio_model.state_dict(), par.save_model_path + '.valid')
        torch.save(optimizer.state_dict(), par.save_optimizer_path + '.valid')
    elif loss_mean < min_loss_t and epoch:
        min_loss_t = loss_mean
        logger.print('Save model at ep {}, mean of train loss: {}'.format(epoch + 1, loss_mean))
        torch.save(e2e_vio_model.state_dict(), par.save_model_path + '.train')
        torch.save(optimizer.state_dict(), par.save_optimizer_path + '.train')
