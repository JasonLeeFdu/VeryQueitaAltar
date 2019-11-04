## image IO
#
import warnings

warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import tensorboardX as tbx

# Program

import Utils.common as tools



import scipy.io as scio
import time
import pickle
import argparse




import networkBaseline1 as nt
import Config.confBaseline1 as conf
from networkBaseline1 import weights_init


'''
random crop on bigger cube
flip

'''




parser = argparse.ArgumentParser()

parser.add_argument('--verbose', '-v',type=int, help='是否显示训练信息',default=1)
parser.add_argument('--testRound', '-t',type=int, help='测试第几轮',default=0)

args = parser.parse_args()
verbose   = args.verbose
testRound   = args.testRound
conf.initConfig(testRound,verbose)



def originalVSFAMain():
    # 根据不同的数据集来设定不同的 信息与feature
    if conf.DATASET_NAME == 'KoNViD-1k':
        features_dir = 'CNN_features_KoNViD-1k/'  # features dir
        datainfo = 'DatasetInfo/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if conf.DATASET_NAME == 'CVD2014':
        features_dir = 'CNN_features_CVD2014/'
        datainfo = 'DatasetInfo/CVD2014info.mat'
    if conf.DATASET_NAME == 'LIVE-Qualcomm':
        features_dir = 'CNN_features_LIVE-Qualcomm/'
        datainfo = 'DatasetInfo/LIVE-Qualcomminfo.mat'
        conf.MAX_Epoch = 15000  # need more training to converge

    # 路径的设置与保存
    tools.securePath('Models')
    tools.securePath('Results')
    tools.securePath(conf.MODEL_PATH)
    tools.securePath(conf.BEST_PERFORMANCE_MODEL_PATH)
    save_result_file = os.path.join(conf.BEST_PERFORMANCE_MODEL_PATH, 'statistic.pkl')

    writer = tbx.SummaryWriter(log_dir=conf.LOG_TRAIN_PATH)

    # 提示信息显示

    print('=============================== 本次训练信息 ==============================')
    print('实验模型名称: {}'.format(conf.MODEL_NAME))
    print('数据集' + conf.DATASET_NAME)
    print('模型名称' + conf.MODEL_NAME)
    print('==========================================================================')

    # 读取统计的Dataset信息       # exp_id = 0
    Info = scio.loadmat(conf.DATASET_INFO_PATH)
    max_len = int(Info['max_len'][0])

    # 将数据集的分数进行标准化(有这么做的)
    scale = Info['scores'][0, :].max()  # label normalization factor

    # 数据集划分，把数据集分成 训练：验证：测试  3 : 1 : 1
    N = Info['videoNum'][0][0]
    TrainN = int(N * conf.TRAIN_RATIO)
    ValN = int(N * conf.VAL_RATIO)
    if not os.path.exists(conf.PARTITION_TABLE_TOTAL_EXP):
        li = list()
        for i in range(1000):
            arr = np.random.permutation(N)
            arr = arr.reshape([N, 1])
            li.append(arr)
        Array = np.concatenate(li, axis=1)
        with open(conf.PARTITION_TABLE_TOTAL_EXP, 'wb') as f:
            pickle.dump(Array, f)

    else:
        with open(conf.PARTITION_TABLE_TOTAL_EXP, 'rb') as f:
            Array = pickle.load(f)

    train_index = Array[:TrainN, conf.testRound]
    val_index = Array[TrainN:ValN + TrainN, conf.testRound]
    test_index = Array[ValN + TrainN:, conf.testRound]

    vnameSet = os.listdir(conf.DATASET_VIDEOS_PATH)
    vnameSet = [os.path.join(conf.TRAINING_SAMPLE_BASEPATH, conf.DATASET_NAME, x[:-4]) for x in vnameSet]

    trainSet = [vnameSet[x] for x in train_index]
    valSet = [vnameSet[x] for x in val_index]
    testSet = [vnameSet[x] for x in test_index]

    # 数据集制作   videoNameList,max_len = conf.MAX_TIME_LEN)
    train_dataset = nt.VQADataset(trainSet, max_len=conf.MAX_TIME_LEN)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=conf.BATCH_SIZE, shuffle=True,
                                               num_workers=5)
    val_dataset = nt.VQADataset(valSet, max_len=conf.MAX_TIME_LEN)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=5)
    if conf.TEST_RATIO > 0:
        test_dataset = nt.VQADataset(testSet, max_len=conf.MAX_TIME_LEN)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=5)

    # 网络载入并设定损失与优化
    device = torch.device("cuda")
    model = nt.LJCH1(max_len).cuda()  #
    model.apply(weights_init)
    criterion = nn.L1Loss().cuda()  # 本文采用 L1 loss
    best_val_criterion = -1  # 选取模型是采用验证集里面，表现最好的那一个SROCC min
    modelSaved, Epoch, Iter, GlobalIter = tools.loadLatestCheckpoint(modelPath=conf.MODEL_PATH, fnCore='model')
    if modelSaved is not None:
        model = modelSaved
        if conf.verbose == 1:
            print('The model has been trained in Epoch:%d, GlobalIteration:%d' % (Epoch, GlobalIter));
            print('')
    else:
        if conf.verbose == 1:
            print('Brand new model');
            print('')
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LR, weight_decay=conf.WEIGHT_DECAY)
    for epoch in range(Epoch + 1, conf.MAX_Epoch):  # def forward(self, cube,inputLength,featContent,featDistort):
        # Train for 1 epoch
        if conf.verbose == 1:
            print('--------------------------- EPOCH:' + str(epoch+1) + '/' + str(
                conf.MAX_Epoch) + ' ---------------------------')
        model.train()
        L = 0
        ii = -1
        y_pred1 = np.zeros(len(val_index))
        y_val1 = np.zeros(len(val_index))
        optimizer.zero_grad()

        for i, (cube, distortFeat, contentFeat, label, vidLen) in enumerate(train_loader):
            ii = i
            s = time.time()
            # y_val1[i] = scale * label.numpy()  #
            cube = cube.cuda().float()
            distortFeat = distortFeat.cuda().float()
            contentFeat = contentFeat.cuda().float()
            label = label.cuda().float().squeeze()
            vidLen = vidLen.cuda().float()

            outputs = model(cube, vidLen, contentFeat, distortFeat)
            # y_pred1[i] = scale * outputs[0].to('cpu').numpy()
            loss = criterion(outputs, label)
            goupi = loss.detach().cpu().numpy()
            # 2.1 loss regularization
            loss = loss / conf.GRAD_ACCUM
            # 2.2 back propagation and accumulation
            loss.backward()
            # 3. update parameters of net
            if ((i + 1) % conf.GRAD_ACCUM) == 0:
                # optimizer the net
                optimizer.step()  # update parameters of net
                optimizer.zero_grad()  # reset gradient

            L = L + loss.item() * conf.GRAD_ACCUM

            if i % 10 == 0 and conf.verbose == 1:
                print('Iter: %d, Loss: %f' % (i, goupi))
                print('Outputs: \t', end='');
                print(outputs.detach().cpu().numpy())
                print('Label: \t', end='');
                print(label.detach().cpu().numpy());
                print('')
        ## save
        tools.saveCheckpoint(netModel=model, epoch=epoch, iterr=ii, glbiter=ii * epoch + ii, savingPath=conf.MODEL_PATH)
        ## the remain unupdated grad
        if ((ii + 1) % conf.GRAD_ACCUM) != 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient
        train_loss = L / (i + 1)
        ##

        model.eval()

        ''' 
        # train-val
        y_pred = np.zeros(len(val_index))
        y_val = np.zeros(len(val_index))
        L = 0

        with torch.no_grad():
            for i, (cube, distortFeat, contentFeat, label, vidLen) in enumerate(val_loader):
                y_val[i] = scale * label.numpy()  #
                cube = cube.to(device).float()
                distortFeat = distortFeat.to(device).float()
                contentFeat = contentFeat.to(device).float()
                label = label.to(device).float()
                vidLen = vidLen.to(device).float()

                outputs = model(cube, vidLen, contentFeat, distortFeat)
                y_pred[i] = scale * outputs[0].to('cpu').numpy()
                loss = criterion(outputs, label)
                L = L + loss.item()
        '''

        # Val
        y_pred = np.zeros(len(val_index))
        y_val = np.zeros(len(val_index))
        L = 0
        with torch.no_grad():
            for i, (cube, distortFeat, contentFeat, label, vidLen) in enumerate(val_loader):
                y_val[i] = scale * label.to('cpu').numpy()  #
                cube = cube.to(device).float()
                distortFeat = distortFeat.to(device).float()
                contentFeat = contentFeat.to(device).float()
                label = label.to(device).float()
                vidLen = vidLen.to(device).float()
                outputs = model(cube, vidLen, contentFeat, distortFeat)
                y_pred[i] = scale * outputs[0].to('cpu').numpy()
                loss = criterion(outputs, label)
                L = L + loss.item()
        val_loss = L / (i + 1)
        val_PLCC = stats.pearsonr(y_pred, y_val)[0]
        val_SROCC = stats.spearmanr(y_pred, y_val)[0]
        val_RMSE = np.sqrt(((y_pred - y_val) ** 2).mean())
        val_KROCC = stats.stats.kendalltau(y_pred, y_val)[0]

        # Test
        if conf.TEST_RATIO > 0 and not conf.NO_TEST_DURING_TRAINING:
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            with torch.no_grad():
                for i, (cube, distortFeat, contentFeat, label, vidLen) in enumerate(test_loader):
                    y_test[i] = scale * label.numpy()  #
                    cube = cube.to(device).float()
                    distortFeat = distortFeat.to(device).float()
                    contentFeat = contentFeat.to(device).float()
                    label = label.to(device).float()
                    vidLen = vidLen.to(device).float()
                    outputs = model(cube, vidLen, contentFeat, distortFeat)
                    y_pred[i] = scale * outputs[0].to('cpu').numpy()
                    loss = criterion(outputs, label)
                    L = L + loss.item()
            test_loss = L / (i + 1)
            PLCC = stats.pearsonr(y_pred, y_test)[0]
            SROCC = stats.spearmanr(y_pred, y_test)[0]
            RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
            KROCC = stats.stats.kendalltau(y_pred, y_test)[0]

        # TensorboardX - epoch wise
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("SROCC/val", val_SROCC, epoch)
        writer.add_scalar("KROCC/val", val_KROCC, epoch)
        writer.add_scalar("PLCC/val", val_PLCC, epoch)
        writer.add_scalar("RMSE/val", val_RMSE, epoch)

        '''
        writer.add_scalar("loss/trainval", trainval_loss, epoch)
        writer.add_scalar("SROCC/trainval", trainval_SROCC, epoch)
        writer.add_scalar("KROCC/trainval", trainval_KROCC, epoch)
        writer.add_scalar("PLCC/trainval", trainval_PLCC, epoch)
        writer.add_scalar("RMSE/trainval", trainval_RMSE, epoch)
        '''

        if conf.TEST_RATIO > 0 and not conf.NO_TEST_DURING_TRAINING:
            writer.add_scalar("loss/test", test_loss, epoch)
            writer.add_scalar("SROCC/test", SROCC, epoch)
            writer.add_scalar("KROCC/test", KROCC, epoch)
            writer.add_scalar("PLCC/test", PLCC, epoch)
            writer.add_scalar("RMSE/test", RMSE, epoch)

        # 选择验证效果好的模型保存(由于是基于epoch的，所以比较科学)
        if SROCC > best_val_criterion and epoch > conf.MAX_Epoch / 6:  ##$$##
            if conf.verbose == 1:
                print("实验{}: 更新最佳参数，位于 Epoch {}".format(conf.MODEL_NAME, epoch))
                print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            if conf.TEST_RATIO > 0 and not conf.NO_TEST_DURING_TRAINING and  conf.verbose == 1:
                print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"  #
                      .format(test_loss, SROCC, KROCC, PLCC, RMSE))
            # 保存相关的模型
            tools.saveCheckpoint(netModel=model, epoch=epoch, iterr=ii, glbiter=ii * epoch + ii,
                                 savingPath=conf.BEST_PERFORMANCE_MODEL_PATH, defaultFileName='bestModel.mdl')
            saveDict = dict()
            saveDict['y_pred'] = y_pred
            saveDict['y_label'] = y_test
            saveDict['test_loss'] = test_loss
            saveDict['SROCC'] = SROCC
            saveDict['KROCC'] = KROCC
            saveDict['PLCC'] = PLCC
            saveDict['RMSE'] = RMSE
            saveDict['test_index'] = test_index
            tools.savePickle(saveDict, save_result_file)
            best_val_criterion = SROCC  # update best val SROCC     ##$$##

    # Test
    if conf.TEST_RATIO > 0:
        # reload and test
        model,_ = tools.loadSpecificCheckpointNetState1(None, None, None, conf.BEST_PERFORMANCE_MODEL_PATH, 'bestModel.mdl')
        model.eval()
        with torch.no_grad():
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            for i, (cube, distortFeat, contentFeat, label, vidLen) in enumerate(test_loader):
                y_test[i] = scale * label.numpy()  #
                cube = cube.to(device).float()
                distortFeat = distortFeat.to(device).float()
                contentFeat = contentFeat.to(device).float()
                label = label.to(device).float()
                vidLen = vidLen.to(device).float()
                outputs = model(cube, vidLen, contentFeat, distortFeat)
                y_pred[i] = scale * outputs[0].to('cpu').numpy()
                loss = criterion(outputs, label)
                L = L + loss.item()
        test_loss = L / (i + 1)
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        print("第{}次实验，  最终算法的测试：".format(conf.testRound))
        print(" test loss={:.4f}, testSROCC={:.4f}, testKROCC={:.4f}, testPLCC={:.4f}, testRMSE={:.4f}".format(test_loss, SROCC, KROCC, PLCC, RMSE))
        with torch.no_grad():
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            for i, (cube, distortFeat, contentFeat, label, vidLen) in enumerate(val_loader):
                y_test[i] = scale * label.numpy()  #
                cube = cube.to(device).float()
                distortFeat = distortFeat.to(device).float()
                contentFeat = contentFeat.to(device).float()
                label = label.to(device).float()
                vidLen = vidLen.to(device).float()
                outputs = model(cube, vidLen, contentFeat, distortFeat)
                y_pred[i] = scale * outputs[0].to('cpu').numpy()
                loss = criterion(outputs, label)
                L = L + loss.item()
        test_loss = L / (i + 1)
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        print("val loss={:.4f}, valSROCC={:.4f}, valKROCC={:.4f}, valPLCC={:.4f}, valRMSE={:.4f}"
              .format(test_loss, SROCC, KROCC, PLCC, RMSE))


def main():
    return 0


if __name__ == '__main__':
    originalVSFAMain()

"""
"""






