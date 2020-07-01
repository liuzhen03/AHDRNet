#-*- coding:utf-8 _*-  
""" 
@author: LiuZhen
@license: Apache Licence 
@file: dataset.py 
@time: 2020/06/30
@contact: liuzhen.pwd@gmail.com
@site:  
@software: PyCharm 

"""
import torch
import numpy as np
from torch.utils.data import Dataset
import os
from utils.utils import *

class Dynamic_Scenes_Dataset(Dataset):
    def __init__(self, root_dir, is_training=True, crop=True, crop_size=None):
        self.root_dir = root_dir
        self.is_training = is_training
        self.crop = crop
        self.crop_size = crop_size

        # scenes dir
        if not self.is_training:
            self.scenes_dir = os.path.join(self.root_dir, 'Test/PAPER')
            self.scenes_dir_list = os.listdir(self.scenes_dir)
        else:
            self.scenes_dir = os.path.join(self.root_dir, 'Training')
            self.scenes_dir_list = os.listdir(self.scenes_dir)
        self.image_list = []
        for scene in range(len(self.scenes_dir_list)):
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_dir_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_dir_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_dir_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        # Read exposure times in one scene
        expoTimes = ReadExpoTimes(self.image_list[index][0])
        # Read LDR image in one scene
        ldr_images = ReadImages(self.image_list[index][1])
        # Read HDR label
        label = ReadLabel(self.image_list[index][2])
        # ldr images process
        pre_img0 = LDR_to_HDR(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = LDR_to_HDR(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = LDR_to_HDR(ldr_images[2], expoTimes[2], 2.2)

        pre_img0 = np.concatenate((ldr_images[0], pre_img0), 2)
        pre_img1 = np.concatenate((ldr_images[1], pre_img1), 2)
        pre_img2 = np.concatenate((ldr_images[2], pre_img2), 2)


        # hdr label process
        label = range_compressor(label)

        # data argument
        if self.crop:
            H, W, _ = ldr_images[0].shape
            x = np.random.randint(0, H - self.crop_size[0] - 1)
            y = np.random.randint(0, W - self.crop_size[1] - 1)

            img0 = pre_img0[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
            label = label[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
        else:
            img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
            label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'label': label}

        return sample

    def __len__(self):
        return len(self.scenes_dir_list)