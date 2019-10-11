import os
import random
from network import *
import datetime
import Utils.common as comm


'''################################################################################################
                                    学习超参数

#################################################################################################
'''
## 本次实验名称(model + expid) 所以做十次实验要调整10次ID
MODEL_NAME = 'LJCH1'
## Dataset的名称
DATASET_NAME = 'KoNViD-1k'
## 学习率
LR = 1e-4#3
## 学习率下降间隔
LR_INTERVAL = 2000
## 学习率下降的比率
LR_DECAY_FACOTOR = 0.9
## L2 正则约束系数
WEIGHT_DECAY= 0.0 #0.000000
## 批训练大小，训练
BATCH_SIZE = 8
## 批训练大小，测试
VALTEST_BATCHSIZE = 24

## 训练是按照Epoch，还是ITers
TRAIN_EPOCH_OR_ITERS = 'epoch'
## 加载模型所用的线程数
NUM_WORKERS = 3
## 最大训练EPOCH次数
MAX_Epoch = 1000
## 最大训练迭代数目
MAX_ITERATIONS = 65000
## 训练的时候不测试
NO_TEST_DURING_TRAINING = False

## 数据划分设置
TRAIN_RATE = 3
VAL_RATE   = 1
TEST_RATE  = 1
TRAIN_RATIO = TRAIN_RATE / (TRAIN_RATE + VAL_RATE + TEST_RATE)
VAL_RATIO = VAL_RATE / (TRAIN_RATE + VAL_RATE + TEST_RATE)
TEST_RATIO = TEST_RATE / (TRAIN_RATE + VAL_RATE + TEST_RATE)

## 进行模型参数的偷偷保留
# STEALTH_MODE_MODEL_ON = True

'''################################################################################################
                                    路径配置

#################################################################################################
'''
# 路径配置
_PROJECT_BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_PATH = os.path.join(_PROJECT_BASEPATH,'Datasets','Neijing/blendDS')
DATASETS_TRAIN_PATH = os.path.join(DATASETS_PATH,'train')
DATASETS_VAL_PATH = os.path.join(DATASETS_PATH,'val')
TRAIN_FN = os.path.join(DATASETS_PATH, 'Train.tfrecord')
VAL_FN = os.path.join(DATASETS_PATH, 'Val.tfrecord')
PRETRAINED_VGG19 = os.path.join(_PROJECT_BASEPATH,'/pretrainedMod/vgg_19.ckpt')
PRETRAINED_Resnet50 = os.path.join(_PROJECT_BASEPATH,'/pretrainedMod/resnet_v2_50.ckpt')
PRETRAINED_VGG19NPY = os.path.join(_PROJECT_BASEPATH,'/pretrainedMod/vgg16.npy')
MODEL_PATH = os.path.join(_PROJECT_BASEPATH,'Models','VQA_%s','model') % MODEL_NAME
LOG_PATH = os.path.join(_PROJECT_BASEPATH,'Models','VQA_%s','log') % MODEL_NAME
STEALTH_MODE_MODEL_PATH = os.path.join(_PROJECT_BASEPATH,'Models','VQA_%s','snapshots') % MODEL_NAME
RESULT_PATH =  os.path.join(_PROJECT_BASEPATH,'Results','VQA_%s','result') % MODEL_NAME

# 网络结构



'''################################################################################################
                                    网络模型参数

#################################################################################################
'''
FIANL_CLASSES_NUM = 5
SAMPLE_H = 300
SAMPLE_W = 300
STD_INPUT_H = 224
STD_INPUT_W = 224



'''################################################################################################
                                         其他参数

#################################################################################################
'''
PRINT_INTERVAL = 50
SAVE_INTERVAL = 1000
SUMMARY_INTERVAL = 50
VALIDATION_INTERVAL = 500
STEALTH_INTERVAL = 1000


GPU_FLAG = True
GPUS = 0
SEED = random.randint(1, 900000)




