"""Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50"""
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2018/3/27
# 以下是调用方法 (命令行)
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=KoNViD-1k --frame_batch_size=64
# CUDA_VISIBLE_DEVICES=1 python CNNfeatures.py --database=CVD2014 --frame_batch_size=32
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=LIVE-Qualcomm --frame_batch_size=8

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
def get_features(video_data, frame_batch_size=64, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    while frame_end < video_length:
        batch = video_data[frame_start:frame_end].to(device)
        features_mean, features_std = extractor(batch)
        output1 = torch.cat((output1, features_mean), 0)
        output2 = torch.cat((output2, features_std), 0)
        frame_end += frame_batch_size
        frame_start += frame_batch_size
    last_batch = video_data[frame_start:video_length].to(device)
    features_mean, features_std = extractor(last_batch)
    output1 = torch.cat((output1, features_mean), 0)
    output2 = torch.cat((output2, features_std), 0)
    output = torch.cat((output1, output2), 1).squeeze()

    return output


## 本文作者的主函数
def originalMain():
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument('--database', default='KoNViD-1k', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=64,
                        help='frame batch size for feature extraction (default: 64)')

    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    if args.database == 'KoNViD-1k':
        videos_dir = '/home/winston/workSpace/PycharmProjects/VSFA/Datasets/VQA/KoNViD/KoNViD_1k_videos/'  # videos dir
        features_dir = 'CNN_features_KoNViD-1k/'  # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        videos_dir = '/media/ldq/Research/Data/CVD2014/'
        features_dir = 'CNN_features_CVD2014/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-Qualcomm':
        videos_dir = '/media/ldq/Others/Data/12.LIVE-Qualcomm Mobile In-Capture Video Quality Database/'
        features_dir = 'CNN_features_LIVE-Qualcomm/'
        datainfo = 'data/LIVE-Qualcomminfo.mat'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    # 读取数据集信息并解析
    Info = h5py.File(datainfo)
    video_names = [Info[Info['video_names'][0, :][i]].value.tobytes()[::2].decode() for i in range(len(Info['video_names'][0, :]))]

    scores = Info['scores'][0, :]
    video_format = Info['video_format'].value.tobytes()[::2].decode()
    width = int(Info['width'][0])
    height = int(Info['height'][0])

    if args.database == 'KoNViD-1k':
        ## 进行文件名的转换
        # 读取pandas,创建字典进行映射
        df = pd.read_csv('/home/winston/workSpace/PycharmProjects/VSFA/Datasets/VQA/KoNViD/KoNViD_1k_attributes.csv')
        li = [str(x) + '.mp4' for x in df['flickr_id']]
        fnMap = dict(zip( df['file_name'],li))
        video_names = [fnMap[x] for x in video_names]


    dataset = VideoDataset(videos_dir, video_names, scores, video_format, width, height)


    for i in range(len(dataset)):
        # 对于每一个视频，进行特征提取，将特征与打分存在一起
        current_data = dataset[i]
        current_video = current_data['video']
        current_score = current_data['score']
        print('Video {}: length {}'.format(i, current_video.shape[0]))
        features = get_features(current_video, args.frame_batch_size, device)
        np.save(features_dir + str(i) + '_resnet-50_res5c', features.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_score)

# 自己写的函数，用于测试整体读入视频的性能
def imgs2Video():
    Dir = '/home/winston/Datasets/Tracking/OriginalInterp2/OTB100/Doll/img/'
    video = cv.VideoWriter('/home/winston/Desktop/demo.avi', cv.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25, (400,300))
    l = os.listdir(Dir)
    l.sort()
    counter = 0
    for item in l:
        img = cv.imread(os.path.join(Dir,item))  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
        video.write(img)  # 把图片写进视频
        counter += 1
        print(counter)
    video.release()













if __name__ == "__main__":
    imgs2Video()






def decode():
    return 1