#-*- coding:utf-8 _*-  
""" 
@author: LiuZhen
@license: Apache Licence 
@file: train.py 
@time: 2020/06/30
@contact: liuzhen.pwd@gmail.com
@site:  
@software: PyCharm 

"""
import argparse
import logging
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from runx.logx import logx

from dataset.dataset import Dynamic_Scenes_Dataset
from graphs.models import AHDRNet
from utils.utils import *



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_arch', type=int, default=0,
                        help='model architecture')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='training batch size (default: 2)')
    parser.add_argument('--test_batch_size', type=int, default=2, metavar='N',
                        help='testing batch size (default: 2)')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--loss_func', type=int, default=0,
                        help='loss functions for training')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--lr_decay_interval', type=int, default=2000,
                        help='decay learning rate every N epochs(default: 100)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=443, metavar='S',
                        help='random seed (default: 443)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load', type=str, default=None,
                        help='load model from a .pth file')
    parser.add_argument('--init_weights', type=bool, default=True,
                        help='init model weights')
    parser.add_argument('--logdir', type=str, default=None,
                        help='target log directory')
    parser.add_argument("--dataset_dir", type=str, default='/data/dataset/HDR_Dynamic_Scenes_SIGGRAPH2017/Training',
                        help='dataset directory')
    parser.add_argument('--validation', type=float, default=10.0,
                        help='percent of the data that is used as validation (0-100)')
    return parser.parse_args()


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    epoch_loss = 0
    for batch_idx, batch_data in enumerate(train_loader):
        batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), batch_data['input2'].to(device)
        label = batch_data['label'].to(device)

        pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
        loss = criterion(pred, label)
        psnr = batch_PSNR(torch.clamp(pred, 0., 1.), label, 1.0)


        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(model.parameters(), 0.01)
        optimizer.step()

        iteration = (epoch - 1) * len(train_loader) + batch_idx
        if batch_idx % args.log_interval == 0:
            logx.msg('Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(batch_data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))
            # logx.add_scalar('Loss/train', loss.item(), iteration)
            logx.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], iteration)
            logx.add_scalar('psnr', psnr, iteration)
            # logx.add_image('input1', imgs[0, :3, :, :][[2, 1, 0], :, :] / 255.0, iteration)
            # logx.add_image('input2', imgs[0, 3:, :, :][[2, 1, 0], :, :] / 255.0, iteration)
            # logx.add_image('mask_true', true_masks[0] / 255.0, iteration)
            # logx.add_image('mask_pred', masks_pred[0] / 255.0, iteration)

        # capture metrics
        metrics = {'loss': loss.item()}
        logx.metric('train', metrics, iteration)

    # save_model
    save_dict = {
        'epoch': epoch + 1,
        'arch': 'AHDRNet',
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    logx.save_model(
        save_dict,
        epoch=epoch,
        metric=epoch_loss / len(train_loader),
        higher_better=False
    )


# def validation(args, model, device, val_loader, optimizer, epoch, criterion):
#     model.eval()
#     mask_type = torch.float32 if model.n_classes == 1 else torch.long
#     n_val = len(val_loader)  # the number of batch
#     val_loss = 0
#
#     for batch_data in val_loader:
#         imgs, true_masks = batch_data['image'], batch_data['mask']
#         imgs = imgs.to(device=device, dtype=torch.float32)
#         true_masks = true_masks.to(device=device, dtype=mask_type)
#
#         with torch.no_grad():
#             mask_pred = model(imgs)
#
#         if model.n_classes >= 1:
#             loss = criterion(mask_pred, true_masks).item()
#             val_loss += loss
#         else:
#             pred = torch.sigmoid(mask_pred)
#             pred = (pred > 0.5).float()
#             val_loss += dice_coeff(pred, true_masks).item()
#     val_loss /= n_val
#     logx.msg('\nValidation set: Average loss: {:.4f}\n'.format(val_loss))
#
#     # capture metrics
#     metrics = {'loss': val_loss}
#     logx.metric('val', metrics, epoch)
#
#     # save_model
#     save_dict = {
#         'epoch': epoch + 1,
#         'arch': 'UNet',
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict()
#     }
#
#     logx.save_model(
#         save_dict,
#         epoch=epoch,
#         metric=val_loss,
#         higher_better=False
#     )


def main():
    # Settings
    args = get_args()

    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # init logx
    logx.initialize(logdir=args.logdir, coolname=True,  tensorboard=True, hparams=vars(args))

    # random seed
    set_random_seed(args.seed)

    # dataset and dataloader
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([transforms.ToTensor(), normalize])


    train_dataset = Dynamic_Scenes_Dataset(root_dir=args.dataset_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # model architecture
    if args.model_arch == 0:
        model = AHDRNet()
    elif args.model_arch == 1:
        model = AHDRNet()
    else:
        logx.msg("This model is not yet implemented!\n")
        return

    if args.init_weights:
        init_parameters(model)
    if args.load:
        model.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logx.msg(f'Model loaded from {args.load}')
    model.to(device)

    # loss function and optimizer
    if args.loss_func == 0:
        criterion = nn.L1Loss()
    elif args.loss_func == 1:
        criterion = nn.MSELoss()
    else:
        logx.msg("Error loss functions.\n")
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())

    logx.msg(f'''Starting training:
        Model Paras:     {num_parameters}
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Loss function:   {args.loss_func}
        Learning rate:   {args.lr}
        Training size:   {len(train_loader)}
        Device:          {device.type}
        Dataset dir:     {args.dataset_dir}
        ''')

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        # validation(args, model, device, val_loader, optimizer, epoch, criterion)


if __name__ == '__main__':
    main()