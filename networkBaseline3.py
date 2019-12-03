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
from torch.nn import utils as nn_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models
import h5py
import numpy as np
import json
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
import Config.confBaseline3 as conf
import Utils.common as tools
import pickle
import lmdb

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
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name), self.height, self.width,
                                          inputdict={'-pix_fmt': 'yuvj420p'})
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
        transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        sample = {'video': transformed_video,
                  'score': video_score}

        return sample


# 训练、测试数据集。(视频所有帧特征,长度，分数)
class VQADataset_Old(Dataset):
    def __init__(self, features_dir='CNN_features_KoNViD-1k/', index=None, max_len=240, feat_dim=4096, scale=1):
        super(VQADataset, self).__init__()
        self.features = np.zeros((len(index), max_len, feat_dim))
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        for i in range(len(index)):
            features = np.load(features_dir + str(index[i]) + '_resnet-50_res5c.npy', allow_pickle=True)
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

# useless
class newLoss(nn.Module):
    def __init__(self, lamda=2):
        super(newLoss, self).__init__()
        self.lamda = lamda

    def forward(self, x, y):
        delta = torch.abs(x - y) * 1000
        l = torch.mean(torch.pow(delta, self.lamda))
        return l


# The Training and testing datasets
class VQADataset(Dataset):
    def __init__(self, videoNameList, max_len=conf.MAX_TIME_LEN):
        super(VQADataset, self).__init__()
        # videoNameList is full name list
        self.videoSet = videoNameList;
        videoNameList.sort()
        self.len = len(self.videoSet)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        vdSampleDir = self.videoSet[idx]
        if np.random.rand() < 0.5:
            with open(os.path.join(vdSampleDir, 'sample.pkl'), 'rb') as f:
                sample = pickle.load(f)
        else:
            with open(os.path.join(vdSampleDir, 'sampleR.pkl'), 'rb') as f:
                sample = pickle.load(f)

        # feat

        motionFeat  = sample['motionFeat']
        distortFeat = sample['distortionFeat']
        contentFeat = sample['contentFeat']

        label = sample['mos'] / sample['mosScale']
        vidLen = sample['vidLen']

        return motionFeat, distortFeat, contentFeat, label, vidLen

''''''
# The Training and testing datasets
class VQADatasetH5(Dataset):
    def __init__(self, lmdbPath=os.path.join(conf.TRAINING_SAMPLE_BASEPATH, conf.DATASET_NAME, 'fastRecord.hdf5'),
                 max_len=conf.MAX_TIME_LEN):
        super(VQADatasetH5, self).__init__()
        # videoNameList is full name list
        self.h5pyf = h5py.File(lmdbPath, "r")
        print('The data prep is finished')

    def __len__(self):
        return len(self.h5pyf.keys())

    def __getitem__(self, idx):
        if idx >= self.__len__():
            return None
        sampleItem = self.h5pyf[str(idx)]

        # feat
        cubes = sampleItem['cubes'].value

        cube = cubes[0, :, :, :, :]
        #
        distortFeat = sampleItem['distortionFeat'].value
        contentFeat = sampleItem['contentFeat'].value
        label = sampleItem['mos'].value / sampleItem['mosScale'].value
        vidLen = sampleItem['vidLen'].value

        return cube, distortFeat, contentFeat, label, vidLen


""" =================================================================================================
                                            重要网络组件



    =================================================================================================
"""


# 初始化权值
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)


# 计算2D的标准差
def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)

def bestCriterion(SROCC,KROCC,PLCC,RMSE):
    return 2*SROCC + PLCC


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
        self.ss = input_size
        self.fc0 = nn.Linear(input_size, reduced_size)  #
        self.dropout = nn.Dropout(p=dropout_p)  #
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers - 1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input


# VSFA的创新点，时间域pooling：
def TP(q, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling,基于时间域的pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf') * torch.ones((1, 1, tau - 1)).to(q.device)  # min pooling的占位符
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)  # min pooling
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)  # padding在右边
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l

def TP2(q,param, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling,基于时间域的pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    length = q.shape[0]
    ### adaptive memory start

    STEEP_MAX = 30;         # 边缘衰减
    STEEP_MIN = 1;
    INTERVAL_MAX = 30;      # 平台长度
    INTERVAL_MIN = 3;

    steepFactor = param[0]
    intervalFactor = param[1]
    steep = STEEP_MIN + steepFactor * (STEEP_MAX - STEEP_MIN);
    interval = INTERVAL_MIN + intervalFactor * (INTERVAL_MAX - INTERVAL_MIN) # 单向

    ## 生成mask
    length = q.shape[-1]
    grid = torch.range(start=0,end=length-1).cuda()
    qMemStd = q - torch.min(q)
    memList = list()
    memList.append(q[0,0,0].view(1,1,1))
    softList = list()
    negQ = torch.squeeze(-q)
    expQ = torch.squeeze(torch.exp(negQ))
    qExpQ = torch.squeeze(q) * expQ
    for i in range(0,int(qMemStd.shape[-1])):
        if i is not  0:
            mask = (1 + torch.exp(-steep * (-torch.abs((grid - i)) + interval)))
            qMemMasked = torch.squeeze(qMemStd) * mask
            idx = torch.argmin(qMemMasked[:i + 1]) # weighted minpooling
            memList.append(q[0,0,idx].view(1,1,1))
        if i is not int(qMemStd.shape[-1]-1):
            left = torch.max(torch.zeros_like(interval).int(),i-torch.round(interval).int());right = torch.min( (torch.ones_like(interval) * length).int().cuda(),i + torch.round(interval).int())
            qCurr = (torch.sum(qExpQ[i:right+1])) / (torch.sum(expQ[i:right+1]))
            softList.append(qCurr)
            #sdf = 43

    softList.append(q[0,0,length-1].view([]))
    l = torch.cat(memList,dim=-1)
    m = torch.stack(softList); m = m.view([1,1,-1])

    ### adaptive memory end
    '''
    # l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)  # min pooling
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)  # padding在右边
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    '''
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
        q = self.q(outputs)  # 基于帧的分数
        score = torch.zeros_like(input_length, device=q.device)  #
        ## score 的 batch-wise 循环-- temporal pooling
        for i in range(input_length.shape[0]):
            qi = q[i, :np.int(input_length[i].numpy())]  # q[N,T]
            qi = TP(qi)
            score[i] = torch.mean(qi)  # video overall quality
        return score

    # RNN的初始状态
    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


# 本人算法尝试1
'''
记得做数据级的时候，把feat content 以及feat distortion的维度减2

'''

# conv3d
class LJCH0(nn.Module):
    def __init__(self, maxLen):
        super(LJCH1, self).__init__()
        ## 呵呵 附庸风雅
        self.reduced_size = 128
        self.hidden_size = 32
        # frame wise conv
        TIME_INTERVAL = conf.ADJACENT_INTERVAL
        self.ann = ANN(4608, self.reduced_size, 1)
        self.rnn = nn.GRU(self.reduced_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.time_interval = TIME_INTERVAL
        self.stFeat  =  SpatialTemporalFeat(TIME_INTERVAL)
        self.q = nn.Linear(self.hidden_size * 2, 1)
        self.maxLen = maxLen

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        return h0

    def forward(self, cube, inputLength, featContent, featDistort):
        T = self.maxLen  # int(torch.max(inputLength).detach().cpu().numpy())
        li = list()
        for i in range(self.time_interval // 2, T - self.time_interval // 2):
            # print(i)
            videoClip = cube[:, i - self.time_interval // 2:i + self.time_interval // 2 + 1, :, :, :]  # (N, C, D, H, W)
            feat = self.stFeat(videoClip)
            feat = torch.unsqueeze(feat, 1)

            li.append(feat)
        spatialTemporalFeat = torch.cat(li, dim=1)

        ## feature concatenation anS feat reduce ==> LENGTH OF MAX_LEN !!!
        totalFeat = torch.cat([spatialTemporalFeat, featContent, featDistort], dim=-1)

        scores = self.ann(totalFeat)
        outputs, _ = self.rnn(scores, self._get_initial_state(scores.size(0), scores.device))
        q = self.q(outputs)  # 基于帧的分数
        score = torch.zeros([cube.shape[0]]).cuda()  # batch-wise
        ## score 的 batch-wise 循环-- temporal pooling
        for i in range(cube.shape[0]):
            qi = q[i, :int(inputLength[i]-2)]  # q[N,T]
            qi = TP(qi)
            score[i] = torch.mean(qi)  # video overall quality
        return score

# 这个多尺度卷积效果不好，多尺度卷积没有理由
class LJCH1(nn.Module):
    def __init__(self, maxLen):
        super(LJCH1, self).__init__()
        ## 呵呵 附庸风雅
        self.reduced_size = 128
        self.hidden_size = 32
        # frame wise conv
        TIME_INTERVAL = conf.ADJACENT_INTERVAL
        self.ann = ANN(4608 + 256, self.reduced_size, 1)
        self.rnn = nn.GRU(self.reduced_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.time_interval = TIME_INTERVAL
        self.q = nn.Linear(self.hidden_size * 2, 1)
        self.maxLen = maxLen
        self.msr = MultiScaleRecall()

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        return h0

    def forward(self, motionFeat, inputLength, featContent, featDistort):
        totalFeat = torch.cat([featContent, featDistort,motionFeat], dim=-1)
        scores = self.ann(totalFeat)
        outputs, _ = self.rnn(scores, self._get_initial_state(scores.size(0), scores.device))
        q = self.q(outputs)  # 基于帧的分数
        ## score 的 batch-wise 循环-- temporal pooling inputLength# [16]
        resScore = self.msr(q,inputLength- 2 * self.time_interval // 2 - 1)
        return resScore




class memPramReg(nn.Module):
    def __init__(self, input_size=64):
        super(memPramReg, self).__init__()
        self.inputSize = input_size
        self.adaptiveMemRegression = nn.Sequential(nn.Linear(self.inputSize, 32),
                                                   nn.ReLU(),

                                                   nn.Dropout(),
                                                   nn.Linear(32, 2)
                                                   )
    def forward(self, inp): # [N, input_size]
        factor = self.adaptiveMemRegression(inp)
        return factor
        # 取值范围的约束？



'''
class LJCH2(nn.Module):
    def __init__(self, maxLen):
        super(LJCH2, self).__init__()
        ## 呵呵 附庸风雅
        self.reduced_size = 128
        self.hidden_size = 32
        # frame wise conv
        TIME_INTERVAL = conf.ADJACENT_INTERVAL
        self.ann = ANN(4608 + 256, self.reduced_size, 1)
        self.rnn = nn.GRU(self.reduced_size, self.hidden_size, batch_first=True, bidirectional=True,dropout=0.5)
        self.time_interval = TIME_INTERVAL
        self.q = nn.Linear(self.hidden_size * 2, 1)
        self.maxLen = maxLen
        self.memPramReg = memPramReg()


    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        return h0




    def forward(self, motionFeat, inputLength, featContent, featDistort):
        totalFeat = torch.cat([featContent, featDistort,motionFeat], dim=-1)
        scores = self.ann(totalFeat)
        validLen = inputLength - 2 * (self.time_interval // 2) - 1 # idx = len-1
        pack = nn_utils.rnn.pack_padded_sequence(scores, validLen, batch_first=True,enforce_sorted=False)


        outputs, hid = self.rnn(pack, self._get_initial_state(scores.size(0), scores.device))
        outputs, inpLen = nn_utils.rnn.pad_packed_sequence(outputs,batch_first=True)


       
        # 抽取最后状态或者开头状态,用它们来回归印象参数 -- content adaptive memory 如果不行再试试单向的

        lastHidForwardLi = list()
        lastHidBackwardLi = list()

        for i in range(outputs.shape[0]):
            lastHidForwardLi.append(outputs[i,int(validLen[i]-1),:32])
            lastHidBackwardLi.append(outputs[i,0,32:])
        lastHidForward = torch.stack(lastHidForwardLi)
        lastHidBackward = torch.stack(lastHidBackwardLi)


        # 计算每帧的分数
        q = self.q(outputs)  # 基于帧的分数
        param = torch.sigmoid(self.memPramReg(torch.cat([lastHidForward,lastHidBackward],dim= -1)))
        #





        ## score 的 batch-wise 循环-- temporal pooling inputLength# [16]
        score = torch.zeros([motionFeat.shape[0]]).cuda()
        for i in range(motionFeat.shape[0]):
            qi = q[i, :int(validLen[i])]  # q[N,T]
            qi = TP2(qi,param[i,:])
            score[i] = torch.mean(qi)  # video overall quality
        return sco
        
        
def TP20(q,param, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling,基于时间域的pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    length = q.shape[0]
    ### adaptive memory start

    STEEP_MAX = 30;         # 边缘衰减
    STEEP_MIN = 1;
    INTERVAL_MAX = 30;      # 平台长度
    INTERVAL_MIN = 3;

    steepFactor = param[0]
    intervalFactor = param[1]
    steep = STEEP_MIN + steepFactor * (STEEP_MAX - STEEP_MIN);
    interval = INTERVAL_MIN + intervalFactor * (INTERVAL_MAX - INTERVAL_MIN) # 单向

    ## 生成mask
    length = q.shape[-1]
    grid = torch.range(start=0,end=length-1).cuda()

    qMemStd = q - torch.min(q)

    memList = list()
    memList.append(q[0,0,0].view(1,1,1))
    for i in range(1,int(qMemStd.shape[-1])):
        mask = (1 + torch.exp(-steep * (-torch.abs((grid - i)) + interval)))
        qMemMasked = torch.squeeze(qMemStd) * mask
        idx = torch.argmin(qMemMasked[:i + 1]) # weighted minpooling
        memList.append(q[0,0,idx].view(1,1,1))
    l = torch.cat(memList,dim=-1)

    ### adaptive memory end

    # l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)  # min pooling
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)  # padding在右边
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l
class LJCH20(nn.Module):
    def __init__(self, maxLen):
        super(LJCH20, self).__init__()
        ## 呵呵 附庸风雅
        self.reduced_size = 128
        self.hidden_size = 32
        # frame wise conv
        TIME_INTERVAL = conf.ADJACENT_INTERVAL
        self.ann = ANN(4608 + 256, self.reduced_size, 1)
        self.rnn = nn.GRU(self.reduced_size, self.hidden_size, batch_first=True, bidirectional=True,dropout=0.5)
        self.time_interval = TIME_INTERVAL
        self.q = nn.Linear(self.hidden_size * 2, 1)
        self.maxLen = maxLen
        self.memPramReg = memPramReg()


    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        return h0




    def forward(self, motionFeat, inputLength, featContent, featDistort):
        totalFeat = torch.cat([featContent, featDistort,motionFeat], dim=-1)
        scores = self.ann(totalFeat)
        validLen = inputLength - 2 * (self.time_interval // 2) - 1 # idx = len-1
        pack = nn_utils.rnn.pack_padded_sequence(scores, validLen, batch_first=True,enforce_sorted=False)


        outputs, hid = self.rnn(pack, self._get_initial_state(scores.size(0), scores.device))
        outputs, inpLen = nn_utils.rnn.pad_packed_sequence(outputs,batch_first=True)


      
        # 抽取最后状态或者开头状态,用它们来回归印象参数 -- content adaptive memory 如果不行再试试单向的

        lastHidForwardLi = list()
        lastHidBackwardLi = list()

        for i in range(outputs.shape[0]):
            lastHidForwardLi.append(outputs[i,int(validLen[i]-1),:32])
            lastHidBackwardLi.append(outputs[i,0,32:])
        lastHidForward = torch.stack(lastHidForwardLi)
        lastHidBackward = torch.stack(lastHidBackwardLi)


        # 计算每帧的分数
        q = self.q(outputs)  # 基于帧的分数
        param = torch.sigmoid(self.memPramReg(torch.cat([lastHidForward,lastHidBackward],dim= -1)))
        #





        ## score 的 batch-wise 循环-- temporal pooling inputLength# [16]
        score = torch.zeros([motionFeat.shape[0]]).cuda()
        for i in range(motionFeat.shape[0]):
            qi = q[i, :int(validLen[i])]  # q[N,T]
            qi = TP20(qi,param[i,:])
            score[i] = torch.mean(qi)  # video overall quality
        return score
        
        
class LJCH21(nn.Module):
    def __init__(self, maxLen):
        super(LJCH2, self).__init__()
        ## 呵呵 附庸风雅
        self.reduced_size = 128
        self.hidden_size = 32
        # frame wise conv
        TIME_INTERVAL = conf.ADJACENT_INTERVAL
        self.ann = ANN(4608 + 256, self.reduced_size, 1)
        self.rnn = nn.GRU(self.reduced_size, self.hidden_size, batch_first=True, bidirectional=True,dropout=0.5)
        self.time_interval = TIME_INTERVAL
        self.q = nn.Linear(self.hidden_size * 2, 1)
        self.maxLen = maxLen
        self.wq =nn.Sequential(nn.Linear(self.hidden_size * 2, 16),nn.LeakyReLU(),nn.Dropout(),nn.Linear(16, 1))

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        return h0




    def forward(self, motionFeat, inputLength, featContent, featDistort):
        totalFeat = torch.cat([featContent, featDistort,motionFeat], dim=-1)
        scores = self.ann(totalFeat)
        validLen = inputLength - 2 * (self.time_interval // 2) - 1 # idx = len-1
        pack = nn_utils.rnn.pack_padded_sequence(scores, validLen, batch_first=True,enforce_sorted=False)


        outputs, hid = self.rnn(pack, self._get_initial_state(scores.size(0), scores.device))
        outputs, inpLen = nn_utils.rnn.pad_packed_sequence(outputs,batch_first=True)



        # 计算每帧的分数
        q = self.q(outputs)  # 基于帧的分数

        wq = self.wq(outputs)


        ## score 的 batch-wise 循环-- temporal pooling inputLength# [16]
        score = torch.zeros([motionFeat.shape[0]]).cuda()
        for i in range(motionFeat.shape[0]):
            qi = q[i, :int(validLen[i])]  # q[N,T]
            qi = TP(qi)
            w = F.softmax(torch.squeeze(wq[i,:int(validLen[i]),:]),dim=0)
            score[i] = torch.sum(w*torch.squeeze(qi)).view([-1])  # video overall quality
        return score

'''

def TP20(q,param, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling,基于时间域的pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    length = q.shape[0]
    ### adaptive memory start

    STEEP_MAX = 30;         # 边缘衰减
    STEEP_MIN = 1;
    INTERVAL_MAX = 30;      # 平台长度
    INTERVAL_MIN = 3;

    steepFactor = param[0]
    intervalFactor = param[1]
    steep = STEEP_MIN + steepFactor * (STEEP_MAX - STEEP_MIN);
    interval = INTERVAL_MIN + intervalFactor * (INTERVAL_MAX - INTERVAL_MIN) # 单向

    ## 生成mask
    length = q.shape[-1]
    grid = torch.range(start=0,end=length-1).cuda()

    qMemStd = q - torch.min(q)

    memList = list()
    memList.append(q[0,0,0].view(1,1,1))
    for i in range(1,int(qMemStd.shape[-1])):
        mask = (1 + torch.exp(-steep * (-torch.abs((grid - i)) + interval)))
        qMemMasked = torch.squeeze(qMemStd) * mask
        idx = torch.argmin(qMemMasked[:i + 1]) # weighted minpooling
        memList.append(q[0,0,idx].view(1,1,1))
    l = torch.cat(memList,dim=-1)

    ### adaptive memory end

    # l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)  # min pooling
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)  # padding在右边
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l

class LJCH2(nn.Module):
    def __init__(self, maxLen):
        super(LJCH2, self).__init__()
        ## 呵呵 附庸风雅
        self.reduced_size = 128
        self.hidden_size = 32
        # frame wise conv
        TIME_INTERVAL = conf.ADJACENT_INTERVAL
        self.ann = ANN(4608 + 256, self.reduced_size, 1)
        self.rnn = nn.GRU(self.reduced_size, self.hidden_size, batch_first=True, bidirectional=True, dropout=0.5)
        self.time_interval = TIME_INTERVAL
        self.q = nn.Linear(self.hidden_size * 2, 1)
        self.maxLen = maxLen

        self.adaptiveMemRegression = nn.Sequential(nn.Linear(64, 32),
                                               nn.ReLU(),
                                               nn.Dropout(),
                                               nn.Linear(32, 5) # multi-scale
                                               )
    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        return h0

    def forward(self, motionFeat, inputLength, featContent, featDistort):
        totalFeat = torch.cat([featContent, featDistort, motionFeat], dim=-1)
        scores = self.ann(totalFeat)
        validLen = inputLength - 2 * (self.time_interval // 2) - 1  # idx = len-1
        pack = nn_utils.rnn.pack_padded_sequence(scores, validLen, batch_first=True, enforce_sorted=False)

        outputs, hid = self.rnn(pack, self._get_initial_state(scores.size(0), scores.device))
        outputs, inpLen = nn_utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # 抽取最后状态或者开头状态,用它们来回归印象参数 -- content adaptive memory 如果不行再试试单向的

        lastHidForwardLi = list()
        lastHidBackwardLi = list()

        for i in range(outputs.shape[0]):
            lastHidForwardLi.append(outputs[i, int(validLen[i] - 1), :32])
            lastHidBackwardLi.append(outputs[i, 0, 32:])
        lastHidForward = torch.stack(lastHidForwardLi)
        lastHidBackward = torch.stack(lastHidBackwardLi)

        # 计算每帧的分数
        q = self.q(outputs)  # 基于帧的分数
        param = torch.softmax(self.adaptiveMemRegression(torch.cat([lastHidForward, lastHidBackward], dim=-1)), dim=-1)
        #

        ## multi-len-mem(需要尝试一下softmax或者是0/1)
        ## tau = 8
        tau = 8
        score1 = torch.zeros([motionFeat.shape[0]]).cuda()
        for i in range(motionFeat.shape[0]):
            qi = q[i, :int(validLen[i])]  # q[N,T]
            qi = TP(qi,tau)
            score1[i] = torch.mean(qi)  # video overall quality

        ##
        ## tau = 10
        tau = 10
        score2 = torch.zeros([motionFeat.shape[0]]).cuda()
        for i in range(motionFeat.shape[0]):
            qi = q[i, :int(validLen[i])]  # q[N,T]
            qi = TP(qi,tau)
            score2[i] = torch.mean(qi)  # video overall quality

        ##
        ## tau = 12
        tau = 12
        score3 = torch.zeros([motionFeat.shape[0]]).cuda()
        for i in range(motionFeat.shape[0]):
            qi = q[i, :int(validLen[i])]  # q[N,T]
            qi = TP(qi,tau)
            score3[i] = torch.mean(qi)  # video overall quality

        ##
        ## tau = 14
        tau = 14
        score4 = torch.zeros([motionFeat.shape[0]]).cuda()
        for i in range(motionFeat.shape[0]):
            qi = q[i, :int(validLen[i])]  # q[N,T]
            qi = TP(qi,tau)
            score4[i] = torch.mean(qi)  # video overall quality

        ##
        ## tau = 16
        tau = 16
        score5 = torch.zeros([motionFeat.shape[0]]).cuda()
        for i in range(motionFeat.shape[0]):
            qi = q[i, :int(validLen[i])]  # q[N,T]
            qi = TP(qi,tau)
            score5[i] = torch.mean(qi)  # video overall quality
        '''
        ## softfusion
        scores = torch.stack([score1,score2,score3,score4,score5],dim=-1)
        finalScores = torch.sum(scores * param,dim=-1)
        '''
        ## hard choice
        scores = torch.stack([score1, score2, score3, score4, score5], dim=-1)
        labels = torch.argmax(param, dim=1)
        one_hot = torch.zeros(param.shape).cuda()
        one_hot[torch.arange(param.shape[0]), labels] = 1
        reverted = torch.argmax(one_hot, dim=1)
        assert (labels == reverted).all().item()
        finalScores = torch.sum(one_hot * scores, dim=-1)
        return  finalScores

# 本人算法尝试1
class SpatialTemporalFeat(nn.Module):
    def __init__(self, ti):
        super(SpatialTemporalFeat, self).__init__()
        # frame wise conv
        self.time_interval = ti
        self.conv3d_1 = Conv3DShared(depth=self.time_interval, timeInterval=self.time_interval, padding=[0, 1, 1])
        self.conv3d_2 = Conv3DShared(depth=1, timeInterval=self.time_interval, padding=[0, 1, 1])

        # cross frame conv
        self.conv3d_3 = Conv3D(depth=1, timeInterval=self.time_interval, padding=[0, 1, 1])
        self.conv3d_4 = Conv3D(depth=1, timeInterval=self.time_interval, padding=[0, 0, 0])

        # regular networks
        self.conv1 = Convlayer2D(self.time_interval, 64)
        self.conv2 = Convlayer2D(64, 128)
        self.conv3 = Convlayer2D(128, 256)
        self.conv4 = Convlayer2D(256, 512)

        # fc
        self.fc1 = nn.Linear(512, 512)

    def forward(self, inpu):
        out = self.conv3d_1(inpu)  # 3
        out = self.conv3d_2(out)  # 5
        out = self.conv3d_3(out)  # 7
        out = self.conv3d_4(out)  # 9
        out = torch.squeeze(out, dim=2)
        out1 = self.conv1(out)  # 11

        out1 = F.max_pool2d(out1, kernel_size=[2, 2])
        out2 = self.conv2(out1)  # 16
        out2 = F.max_pool2d(out2, kernel_size=[2, 2])
        out3 = self.conv3(out2)  # 21
        out4 = F.max_pool2d(out3, kernel_size=[2, 2])
        out5 = self.conv4(out4)  # 26
        feat = F.dropout(global_avg_pool2d(out5))
        feat = torch.squeeze(feat, dim=-1)
        feat = torch.squeeze(feat, dim=-1)
        fc1 = self.fc1(feat)
        return fc1

###### 比较loss
class ComparingLoss(nn.Module):
    def __init__(self):
        super(ComparingLoss, self).__init__()
        self.accept = True

    def forward(self,x,label):
        '''

        :param x: [T,]
        :param y: [T,]
        :return:
        '''
        T = x.shape[0]
        sum = torch.zeros([1])
        for i in range(T-1):
            for j in range(i+1,T):
                sum += ((x[i] > x[j]) ^ (label[i] > label[j])) * ( torch.abs((x[i] - label[i])) + torch.abs((x[j] - label[j])))
        sum /= T
        return sum
###### 多尺度时间域融合
class MultiScaleRecall(nn.Module):
    '''
        multiscale recall scoring 新型的时域增强
    '''
    # input is : [3,240]
    def __init__(self):
        super(MultiScaleRecall, self).__init__()
        # 全连接一共几层,本文一层
        self.w1 = nn.Parameter(torch.FloatTensor(np.random.normal(0,1e-2,[1,1,3])),requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(np.random.normal(0,1e-2,[1,1,5])),requires_grad=True)
        self.w3 = nn.Parameter(torch.FloatTensor(np.random.normal(0,1e-2,[1,1,7])),requires_grad=True)
        # self.wFusion = nn.Parameter(torch.FloatTensor(np.random.normal(0,1e-2,[3+5+7,3+5+7])),requires_grad=True)
        # self.wFusion1 = nn.Parameter(torch.FloatTensor(np.random.normal(0,1e-2,[3+5+7,3])),requires_grad=True)
    def forward(self,x,length):
        finalQScoresList =  list()
        for i in range(int(length.shape[0])):
            lenn = length[i]
            # for every batch
            q = x[i,:int(lenn)] # (221,) crop the valid time interval
            q = q.view([1,1,-1])
            # fusion weight for conv3
            qw1 = F.softmax(F.conv1d(q,self.w1,padding=1),dim=-1)
            # fusion weight for conv5
            qw2 = F.softmax(F.conv1d(q,self.w2,padding=2),dim=-1)
            # fusion weight for conv7
            qw3 = F.softmax(F.conv1d(q,self.w3,padding=3),dim=-1)

            # kernel weight distribution
            kernelInfo = torch.cat([self.w1.view([1,-1]),self.w2.view([1,-1]),self.w3.view([1,-1])],dim=-1)

            # learnt form the kernel weight distribution
            # fusionRate = torch.softmax(F.leaky_relu(torch.mm(kernelInfo,self.wFusion).squeeze()),dim=-1)

            # final fused score
            # fusedScore = fusionRate[0]*(torch.sum(qw1 * q)) + fusionRate[1]*(torch.sum(qw2 * q)) + fusionRate[2]*(torch.sum(qw3 * q))
            fusedScore = (torch.sum(qw1 * q))/3 + (torch.sum(qw2 * q))/3 + (torch.sum(qw3 * q)) / 3
            fusedScore = fusedScore.view([1,-1])
            finalQScoresList.append(fusedScore)
        finalQScore = torch.cat(finalQScoresList,dim=0)
        return finalQScore



def global_avg_pool2d(x):
    """2D global standard variation pooling"""
    return torch.mean(x.view(x.size()[0], x.size()[1], -1, 1),
                      dim=2, keepdim=True)


class Convlayer2D(nn.Module):
    '''
        2D卷积
    '''

    def __weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv3d') != -1:
            torch.nn.init.kaiming_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)
        if classname.find('Conv2d') != -1:
            torch.nn.init.kaiming_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)

    def __init__(self, inc, outc, h=3, w=3, padding=[1, 1], stride=[1, 1], isBn=True, activation=F.relu):
        super(Convlayer2D, self).__init__()
        self.conv = nn.Conv2d(inc, outc, [h, w], stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(outc)
        self.activation = activation
        self.isBn = isBn
        self.apply(self.__weights_init)

    def forward(self, x):  # conv - bn - relu
        y = self.conv(x)  #
        if self.isBn:
            y = self.bn(y)
        if self.activation is not None:
            y = self.activation(y)
        return y


class Conv3DShared(nn.Module):
    '''
        用来对相邻的每一个帧进行提取特征
    '''

    def __init__(self, depth=3, timeInterval=3, padding=[0, 1, 1]):
        super(Conv3DShared, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(np.random.normal(loc=0, scale=0.1, size=[1, 1, depth, 3,
                                                                                              3])))  # outc, inc,d,h,w size=[1,1,3,3,3]=size=[3,1,3,3,3]
        self.bias = nn.Parameter(torch.FloatTensor(np.zeros([1])))
        self.timeInterval = timeInterval
        self.padding = padding
        self.depth = depth
        self.bn = nn.BatchNorm3d(
            self.timeInterval)  # num_features:时间域个数[1]  Input: :math:`(N, C, D, H, W)   - Output: :math:`(N, C, D, H, W)` (same shape as input)

    def forward(self, x):  # conv - bn - relu
        w = self.weight.repeat([self.timeInterval, 1, 1, 1, 1])
        b = self.bias.repeat([self.timeInterval])
        y = F.conv3d(x, w, b, groups=self.timeInterval, padding=self.padding)
        y = self.bn(y)
        y = F.relu(y)
        return y


class Conv3D(nn.Module):
    '''
        用来跨帧提取特征
    '''

    def __init__(self, depth=3, timeInterval=3, padding=[1, 1, 1]):
        super(Conv3D, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(np.random.normal(loc=0, scale=0.1,
                                                                      size=[timeInterval, timeInterval, depth, 3,
                                                                            3])))  # outc, inc,h,w,c  size=[1,1,3,3,3]=size=[3,1,3,3,3]
        self.bias = nn.Parameter(torch.FloatTensor(np.zeros([timeInterval])))
        self.timeInterval = timeInterval
        self.padding = padding
        self.depth = depth
        self.bn = nn.BatchNorm3d(
            self.timeInterval)  # num_features:时间域个数[1]  Input: :math:`(N, C, D, H, W)   - Output: :math:`(N, C, D, H, W)` (same shape as input)

    def forward(self, x):  # conv - bn - relu
        y = F.conv3d(x, self.weight, self.bias, padding=self.padding)
        y = self.bn(y)
        y = F.relu(y)
        return y



def main():  # NCHW  | NCDHW   cin cout hwd | [batch_num, time_interval, channel, height, width]
    max_len = 580
    bn = 2

    # The set of the target shape
    s = torch.rand([bn, max_len, 3, 64, 64]).cuda()
    inputLen = torch.randint(200, 400, [2, 1])
    featContent = torch.rand([bn, max_len, 4096]).cuda()  # 4096
    featDistort = torch.rand([bn, max_len, 512]).cuda()  # 512

    # cube,inputLength,featContent,featDistort):
    featContent = featContent[:, 1:-1, :]
    featDistort = featDistort[:, 1:-1, :]

    net = LJCH1().cuda()

    scores = net(s, inputLen, featContent, featDistort)
    print(scores.cpu().numpy())

    return 0


def test1():
    basePath = '/home/winston/workSpace/PycharmProjects/VQA/Ours/TrainingSamples/KoNViD'
    li = os.listdir(basePath)
    li = [os.path.join(basePath, x) for x in li]
    ss = VQADataset(li)
    net = LJCH1()
    N = 1000
    s = time.time()
    for i in range(N):
        cube, distortFeat, contentFeat, label, vidLen = ss.getitem__(2)
        net(cube, vidLen, contentFeat, distortFeat)
    e = time.time()
    t = e - s
    ss = t / N
    print('----------------------')
    print(ss)
    return 3


def test2():
    d = VQADatasetH5()
    a, b, c, d, s = d.getitem__(3)
    return


if __name__ == '__main__':
    test2()


'''
前着你那个

hid[1,3,:] # outputs[3,0,32:] batch3号的最后一个元素的去向
tensor([-0.9136,  0.8916,  0.0931,  0.2071, -0.5570, -0.2541, -0.6761,  0.9255,
        -0.8166,  0.7803, -0.9827,  0.4119,  0.9509, -0.5924, -0.7002,  0.8857,
        -0.9185, -0.4454,  0.1220, -0.9689,  0.9960, -0.9828, -0.6659, -0.8052,
         0.8664, -0.9838,  0.9877,  0.9687, -0.3757, -0.1598, -0.3387, -0.8694],
       device='cuda:0', grad_fn=<SliceBackward>)
hid[0,3,:] # outputs[3,188,:32] batch3号的最后一个元素的去向


'''