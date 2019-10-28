"""Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50"""
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2018/3/27
# 以下是调用方法 (命令行)
# CUDA_VISIBLE_DEVICES=0 python contentFeature.py --database=KoNViD-1k --frame_batch_size=64
# CUDA_VISIBLE_DEVICES=1 python contentFeature.py --database=CVD2014 --frame_batch_size=32
# CUDA_VISIBLE_DEVICES=0 python contentFeature.py --database=LIVE-Qualcomm --frame_batch_size=8

import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2 as cv
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
from argparse import ArgumentParser
import scipy.io as scio
import pandas as pd
import sys
import threading


# 从原始数据集中，提取特征
class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, score, video_format='RGB', width=None, height=None):
        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name), self.height, self.width, inputdict={'-pix_fmt':'yuvj420p'})
        else:
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
        video_score = self.score[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([video_length, video_channel,  video_height, video_width])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        sample = {'video': transformed_video,
                  'score': video_score}

        return sample

## 用来提取初级特征的网络，提出来的特征是均值与方差
class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c   从当中提出feature
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std



# 计算2D的标准差
def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)

# 提取特征的步骤,把单个视频的所有帧的特征都提出来,拼接到一起
def ExtractFeatVid(video_data, frame_batch_size=64, device='cuda'):
    """feature extraction"""
    # transform the video
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    video_length = video_data.shape[0]
    video_channel = video_data.shape[3]
    video_height = video_data.shape[1]
    video_width = video_data.shape[2]
    transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
    for frame_idx in range(video_data.shape[0]):
        frame = video_data[frame_idx]
        frame = Image.fromarray(frame)
        frame = transform(frame)
        transformed_video[frame_idx] = frame

    # extract
    extractor = ResNet50().to(device)
    video_length = transformed_video.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    while frame_end < video_length:
        batch = transformed_video[frame_start:frame_end].to(device)
        features_mean, features_std = extractor(batch)
        output1 = torch.cat((output1, features_mean), 0)
        output2 = torch.cat((output2, features_std), 0)
        frame_end += frame_batch_size
        frame_start += frame_batch_size
    last_batch = transformed_video[frame_start:video_length].to(device)
    features_mean, features_std = extractor(last_batch)
    output1 = torch.cat((output1, features_mean), 0)
    output2 = torch.cat((output2, features_std), 0)
    output = torch.cat((output1, output2), 1).squeeze()

    return output



# for testing gpu mem consumption
class Consumer(threading.Thread):
    def __init__(self, sem):
        super().__init__()
        self.sem = sem
        return
    def run(self):
        self.sem.acquire()
        videoName = '/home/winston/workSpace/PycharmProjects/VQA/Ours/Datasets/VQA/KoNViD/KoNViD_1k_videos/3337642103.mp4'
        videoClip = skvideo.io.vread(videoName)
        feat = ExtractFeatVid(videoClip)
        self.sem.release()
        return

if __name__ == '__main__':
    # 9.7 GB GPU consumption
    videoName = '/home/winston/workSpace/PycharmProjects/VQA/Ours/Datasets/VQA/KoNViD/KoNViD_1k_videos/3337642103.mp4'
    videoClip = skvideo.io.vread(videoName)
    feat = ExtractFeatVid(videoClip)


    '''
    GTXGraphicCard = threading.Semaphore(1)
    c1 = Consumer(GTXGraphicCard)
    c1.start()
    '''


