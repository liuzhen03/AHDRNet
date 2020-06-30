#-*- coding:utf-8 _*-  
""" 
@author: LiuZhen
@license: Apache Licence 
@file: utils.py 
@time: 2020/06/30
@contact: liuzhen.pwd@gmail.com
@site:  
@software: PyCharm 

"""
import numpy as np
import os, glob
import cv2
import imageio
from math import log10
import torch
import torch.nn as nn
import torch.nn.init as init
from skimage.measure.simple_metrics import compare_psnr
imageio.plugins.freeimage.download()


def list_all_files_sorted(folderName, extension=""):
    return sorted(glob.glob(os.path.join(folderName, "*" + extension)))


def ReadExpoTimes(fileName):
    return np.power(2, np.loadtxt(fileName))


def ReadImages(fileNames):
    imgs = []
    for imgStr in fileNames:
        img = cv2.imread(imgStr, -1)

        # equivalent to im2single from Matlab
        img = img / 2 ** 16
        img = np.float32(img)

        img.clip(0, 1)

        imgs.append(img)
    return np.array(imgs)


def ReadLabel(fileName):
    label = imageio.imread(os.path.join(fileName, 'HDRImg.hdr'), 'hdr')
    label = label[:, :, [2, 1, 0]]  ##cv2
    return label


def LDR_to_HDR(imgs, expo, gamma):
    return (imgs ** gamma) / expo


def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)


def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_parameters(net):
    """Init layer parameters"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


def set_random_seed(seed):
    """Set random seed for reproduce"""
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

