# Torch
import torch
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
from torch.nn import init
import torch.functional as F
import tensorboardX as tbx
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models

import numpy as np
from scipy import stats
from tensorboardX import SummaryWriter

# Traditional
import os
import numpy as np
import time
from datetime import datetime
import math
from tqdm import tqdm
from PIL import Image
import skvideo.io

# Program
import Config as conf
import NetBricks.ljchopt1 as bricks
import Utils.common as tools


""" =================================================================================================
                                            数据集函数与视频数据加载读取

    1. 可以尝试都作为四维度矩阵读入 (为了快速实验)
    2. 端到端

    =================================================================================================
"""
# 提取特征用.  (视频所有帧,分数)
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

# 训练、测试数据集。(视频所有帧特征,长度，分数)
class VQADataset(Dataset):
    def __init__(self, features_dir='CNN_features_KoNViD-1k/', index=None, max_len=240, feat_dim=4096, scale=1):
        super(VQADataset, self).__init__()
        self.features = np.zeros((len(index), max_len, feat_dim))
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        for i in range(len(index)):
            features = np.load(features_dir + str(index[i]) + '_resnet-50_res5c.npy',allow_pickle=True)
            self.length[i] = features.shape[0]
            self.features[i, :features.shape[0], :] = features
            self.mos[i] = np.load(features_dir + str(index[i]) + '_score.npy')  #
        self.scale = scale  #
        self.label = self.mos / self.scale  # label normalization 需要对标签进行标准化

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = (self.features[idx], self.length[idx], self.label[idx])
        return sample

""" =================================================================================================
                                            重要网络组件

     

    =================================================================================================
"""
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

## 用来提取初级特征的网络(backbone)，提出来的特征是均值与方差
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

# 中间FC网络,用于降维. linear + non-linear
class ANN(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        # 全连接一共几层,本文一层
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)  #
        self.dropout = nn.Dropout(p=dropout_p)  #
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers-1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input

# VSFA的创新点，时间域pooling：
def TP(q, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling,基于时间域的pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device) # min pooling的占位符
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1) # min pooling
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1) # padding在右边
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l










""" =================================================================================================
                                            各种模型的尝试



    =================================================================================================
"""


# VSFA模型主体，主体框架
class VSFA(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, hidden_size=32):
        super(VSFA, self).__init__()
        self.hidden_size = hidden_size
        self.ann = ANN(input_size, reduced_size, 1)
        self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.q = nn.Linear(hidden_size, 1)

    def forward(self, input, input_length):
        # 降维
        input = self.ann(input)
        # GRU outputs -- [N,T,单双向×hidden_size]
        outputs, _ = self.rnn(input, self._get_initial_state(input.size(0), input.device))
        ## 将GRU输出的向量来输出为output
        q = self.q(outputs)         # 基于帧的分数
        score = torch.zeros_like(input_length, device=q.device)  #
        ## score 的 batch-wise 循环-- temporal pooling
        for i in range(input_length.shape[0]):
            qi = q[i, :np.int(input_length[i].numpy())] # q[N,T]
            qi = TP(qi)
            score[i] = torch.mean(qi)  # video overall quality
        return score

    # RNN的初始状态
    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0












def main():

    return 0


if __name__ == '__main__':
    main()


