import os
import random
# from network import *
import datetime




'''################################################################################################
                                    功能性函数

#################################################################################################'''

def initConfig(tr,vb):
    global testRound
    testRound = tr
    global  verbose
    verbose = vb
    global MODEL_NAME,MODEL_PATH,BEST_PERFORMANCE_MODEL_PATH,LOG_TRAIN_PATH,LOG_VAL_PATH,LOG_TEST_PATH
    MODEL_NAME = 'Baseline1_%d' % testRound
    MODEL_PATH = os.path.join(_PROJECT_BASEPATH,'Models','VQA_%s','model') % MODEL_NAME
    BEST_PERFORMANCE_MODEL_PATH = os.path.join(_PROJECT_BASEPATH,'Models','VQA_%s','bestPerformance') % MODEL_NAME
    LOG_TRAIN_PATH = os.path.join(_PROJECT_BASEPATH,'Models','VQA_%s','log') % MODEL_NAME
    LOG_VAL_PATH = os.path.join(_PROJECT_BASEPATH,'Models','VQA_%s','log','val') % MODEL_NAME
    LOG_TEST_PATH = os.path.join(_PROJECT_BASEPATH,'Models','VQA_%s','log','test') % MODEL_NAME

testRound = 888
verbose = 888

'''################################################################################################
                                    学习超参数

#################################################################################################
'''
## 本次实验名称(model + expid) 所以做十次实验要调整10次ID
MODEL_NAME = 'Baseline1_%d' % testRound
## Dataset的名称
DATASET_NAME = 'KoNViD'
## Distortion Feat 提取算法
DISTORTION_ALGORITHM_NAME = 'DeepIQA'
DISTORTION_ALG_FUNC_HANDLE = None
## 学习率
LR = 1e-4#3
## 学习率下降间隔
LR_INTERVAL = 2000
## 学习率下降的比率
LR_DECAY_FACOTOR = 0.9
## L2 正则约束系数
WEIGHT_DECAY= 0.0 #0.000000
## 批训练大小，训练
BATCH_SIZE = 9
## 梯度累计步骤
GRAD_ACCUM = 3
## 批训练大小，测试
VALTEST_BATCHSIZE = 24

## 训练是按照Epoch，还是ITers
TRAIN_EPOCH_OR_ITERS = 'epoch'
## 加载模型所用的线程数
NUM_WORKERS = 3
## 最大训练EPOCH次数
MAX_Epoch = 10
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
DATASET_VIDEOS_PATH = '???'   # for videoSets
DATASET_INFO_PATH   = '???'
if DATASET_NAME == 'KoNViD':
    DATASET_VIDEOS_PATH = os.path.join(_PROJECT_BASEPATH,'Datasets','VQA','KoNViD','KoNViD_1k_videos')  # where all the videos are
    DATASET_INFO_PATH = os.path.join(_PROJECT_BASEPATH,'DatasetInfo','KoNViD-1kinfo.mat')                  # the benchmark information
    MAX_TIME_LEN = 240
else:
    pass

TRAINING_SAMPLE_BASEPATH = os.path.join(_PROJECT_BASEPATH,'TrainingSamples')
MODEL_PATH = os.path.join(_PROJECT_BASEPATH,'Models','VQA_%s','model') % MODEL_NAME
BEST_PERFORMANCE_MODEL_PATH = os.path.join(_PROJECT_BASEPATH,'Models','VQA_%s','bestPerformance') % MODEL_NAME
LOG_TRAIN_PATH = os.path.join(_PROJECT_BASEPATH,'Models','VQA_%s','log') % MODEL_NAME
LOG_VAL_PATH = os.path.join(_PROJECT_BASEPATH,'Models','VQA_%s','log','val') % MODEL_NAME
LOG_TEST_PATH = os.path.join(_PROJECT_BASEPATH,'Models','VQA_%s','log','test') % MODEL_NAME
PRETRAINED_MODELS_DIR = os.path.join(_PROJECT_BASEPATH,'pretrainedModels')

PARTITION_TABLE = 'PartitionTabel.pkl'



# 网络结构


'''################################################################################################
                                    网络模型参数

#################################################################################################
'''
# 时空能量矩阵的融合比例
FUSION_RATE=0.5

# 时空cube采样块的大小
SAMPLING_SIZE = 64

# 每张图中选择的采样块的数量
SAMPLING_NUM = 4

# guided pooling的策略
STRATEGY ='fourRand'        # 'fourTop','uniRand','topNum'

# spatial temporal ajacent interval
ADJACENT_INTERVAL = 3

'''################################################################################################
                                         其他参数

#################################################################################################
'''
# 显示训练的过程
PRINT_INTERVAL = 50
SAVE_INTERVAL = 1000
SUMMARY_INTERVAL = 50
VALIDATION_INTERVAL = 500
STEALTH_INTERVAL = 1000


GPU_FLAG = True
GPUS = 0
SEED = random.randint(1, 900000)

# 提取特征的流程控制
FLG_EXTRACT_MAPCUBES = True
FLG_OVERWRITE_MAPCUBES = False
FLG_EXTRACT_FEAT_DISTORTION = True
FLG_OVERWRITE_FEAT_DISTORTION = False
FLG_EXTRACT_FEAT_CONTENT    = True
FLG_OVERWRITE_FEAT_CONTENT = False
FLG_MERGE_MAP_CUBE_AND_FEATS = True
FLG_OVERWRITE_MERGE_ALL = True

'''################################################################################################
                                         其他参数自动配置生成区

#################################################################################################
'''


