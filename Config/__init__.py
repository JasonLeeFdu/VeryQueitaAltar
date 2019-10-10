import os
import random


_PROJECT_BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_PATH = os.path.join(_PROJECT_BASEPATH,'Datasets')


# 网络结构
MAX_PRED_LEN = 9
STRING_LEN = MAX_PRED_LEN + 1
CHAR_KIND = 12
MISS_LOSS_POS =12


# For Training
MODEL_PATH = os.path.join(_PROJECT_BASEPATH,'model')
LOG_PATH = os.path.join(_PROJECT_BASEPATH,'log')
NUM_WORKERS = 3
MAX_Epoch = 101
SAVE_INTERVAL = 1000
PRINT_INTERVAL = 100
SUM_INTERVAL = 100
VAL_INTERVAL = 15000
GPU_FLAG = True
GPUS = 0
SEED = random.randint(1, 900000)
LR_INTERVAL = 20
WEIGHT_DECAY= 5e-3# 1e-3 #0.000000
MOMENTUM = 0.9

PRELOAD = False


## Hyper parameters concerning with training performance and Gradient Deminish or ex
## GRADIENT_CLIP = 0.1                     　# small
LR = 1e-3#3                                　# small 0.0005

BATCH_SIZE = 8                              # X
VALTEST_BATCHSIZE = 16;





#######
WEIGHT_INIT_STDDEV_FACTOR = 1.3                # big
WEIGHT_INIT_MEAN_FACTOR = 0
SUMMARY_SCALAR_FIX  = 3e-3
GRADIENT_CLIP_THETA = 0.1

## 模型测试
# TEST_MODEL_PATH = '/home/winston/workSpace/PycharmProjects/Foundation/AutoEncoder/testModels'




