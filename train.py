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
from graphs.origin_ahdrnet import AHDR
from utils.utils import *



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_arch', type=int, default=1,
                        help='model architecture')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='training batch size (default: 2)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 2)')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data (default: 8)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
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
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load', type=str, default=None,
                        help='load model from a .pth file')
    parser.add_argument('--init_weights', type=bool, default=True,
                        help='init model weights')
    parser.add_argument('--logdir', type=str, default=None,
                        help='target log directory')
    parser.add_argument("--dataset_dir", type=str, default='/data/dataset/HDR_Dynamic_Scenes_SIGGRAPH2017',
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
        pred = torch.clamp(pred, 0., 1.)

        loss = criterion(pred, label)
        psnr = batch_PSNR(pred, label, 1.0)
        # psnr = batch_PSNR(torch.clamp(pred, 0., 1.), label, 1.0)


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
            logx.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], iteration)
            logx.add_scalar('train/psnr', psnr, iteration)
            logx.add_image('train/input1', batch_ldr0[0][[2, 1, 0], :, :], iteration)
            logx.add_image('train/input2', batch_ldr1[0][[2, 1, 0], :, :], iteration)
            logx.add_image('train/input3', batch_ldr2[0][[2, 1, 0], :, :], iteration)
            logx.add_image('train/pred', pred[0][[2, 1, 0], :, :], iteration)
            logx.add_image('train/gt', label[0][[2, 1, 0], :, :], iteration)

        # capture metrics
        metrics = {'loss': loss.item()}
        logx.metric('train', metrics, iteration)


def validation(args, model, device, val_loader, optimizer, epoch, criterion):
    model.eval()
    n_val = len(val_loader)
    val_loss = 0
    val_psnr = 0
    for batch_idx, batch_data in enumerate(val_loader):
        batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), \
                                             batch_data['input2'].to(device)
        label = batch_data['label'].to(device)

        with torch.no_grad():
            pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
            pred = torch.clamp(pred, 0., 1.)

        loss = criterion(pred, label)
        psnr = batch_PSNR(pred, label, 1.0)
        logx.msg('Validation set: PSNR: {:.4f}'.format(psnr))

        iteration = (epoch - 1) * len(val_loader) + batch_idx
        if epoch % 100 == 0:
            logx.add_image('val/input1', batch_ldr0[0][[2, 1, 0], :, :], iteration)
            logx.add_image('val/input2', batch_ldr1[0][[2, 1, 0], :, :], iteration)
            logx.add_image('val/input3', batch_ldr2[0][[2, 1, 0], :, :], iteration)
            logx.add_image('val/pred', pred[0][[2, 1, 0], :, :], iteration)
            logx.add_image('val/gt', label[0][[2, 1, 0], :, :], iteration)

        val_loss += loss
        val_psnr += psnr

    val_loss /= n_val
    val_psnr /= n_val
    logx.msg('Validation set: Average loss: {:.4f}'.format(val_loss))
    logx.msg('Validation set: Average PSNR: {:.4f}\n'.format(val_psnr))

    # capture metrics
    metrics = {'psnr': val_psnr}
    logx.metric('val', metrics, epoch)
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
        metric=val_loss,
        higher_better=True
    )


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


    train_dataset = Dynamic_Scenes_Dataset(root_dir=args.dataset_dir, is_training=True, crop=True, crop_size=(256, 256))
    val_dataset = Dynamic_Scenes_Dataset(root_dir=args.dataset_dir, is_training=False, crop=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)


    # model architecture
    if args.model_arch == 0:
        model = AHDRNet()
    elif args.model_arch == 1:
        model = AHDR(6, 3, 64, 32)
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
    # # log graph
    # dummy_input = torch.from_numpy(np.random.rand(1, 6, 256, 256)).float().to(device)
    # logx.add_graph(model, input_to_model=(dummy_input, dummy_input, dummy_input))

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
        validation(args, model, device, val_loader, optimizer, epoch, criterion)


if __name__ == '__main__':
    main()