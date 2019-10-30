## image IO
import warnings

warnings.filterwarnings('ignore')
import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from scipy import stats
from tensorboardX import SummaryWriter
import network as nt
import tensorboardX as tbx

# Program
import Config as conf
import Utils.common as tools
import scipy.io as scio
from network import weights_init, newLoss
import time

'''
random crop on bigger cube
flip

'''


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
    tools.securePath(conf.RESULT_PATH)
    trained_model_file = os.path.join(conf.MODEL_PATH, 'trainedParams')
    save_result_file = conf.RESULT_PATH

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
    arr = np.random.permutation(N)
    train_index = arr[:TrainN]
    val_index = arr[TrainN:ValN + TrainN]
    test_index = arr[ValN + TrainN:]

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
    modelSaved, Epoch, Iter, GlobalIter = tools.loadLatestCheckpoint(fnCore='model')
    if modelSaved is not None:
        model = modelSaved
        print('The model has been trained in Epoch:%d, GlobalIteration:%d' % (Epoch, GlobalIter));
        print('')
    else:
        print('Brand new model');
        print('')
    # The optimizer should be after the real load in models, therefore the weight is updated. #NO updating#,#weird ghost network layerss#
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LR, weight_decay=conf.WEIGHT_DECAY)
    for epoch in range(Epoch + 1, conf.MAX_Epoch):  # def forward(self, cube,inputLength,featContent,featDistort):
        # Train for 1 epoch
        print('--------------------------- EPOCH:' + str(epoch) + '/' + str(
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
                # optimizer the `net
                optimizer.step()  # update parameters of net
                optimizer.zero_grad()  # reset gradient

            L = L + loss.item() * conf.GRAD_ACCUM

            if i % 10 == 0:
                print('Iter: %d, Loss: %f' % (i, goupi))
                print('Outputs: \t', end='');
                print(outputs.detach().cpu().numpy())
                print('Label: \t', end='');
                print(label.detach().cpu().numpy());
                print('')
        ## save
        tools.saveCheckpoint(netModel=model, epoch=epoch, iterr=ii, glbiter=ii * epoch + ii)
        ## the remain unupdated grad
        if ((ii + 1) % conf.GRAD_ACCUM) != 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient
        train_loss = L / (i + 1)

        model.eval()


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
        writer.add_scalar("loss/train", test_loss, epoch)
        writer.add_scalar("SROCC/val", SROCC, epoch)
        writer.add_scalar("KROCC/val", KROCC, epoch)
        writer.add_scalar("PLCC/val", PLCC, epoch)
        writer.add_scalar("RMSE/val", RMSE, epoch)
        '''
        writer.add_scalar("loss/trainval", trainval_loss, epoch)
        writer.add_scalar("SROCC/trainval", trainval_SROCC, epoch)             print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .for mat(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            if conf.TEST_RATIO > 0 and not conf.NO_TEST_DURING_TRAINING:
                print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(test_loss, SROCC, KROCC, PLCC, RMSE))
                np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
            torch.save(model.state_dict(), trained_model_file)
            best_val_criterion = val_SROCC  # update best val SROCC
    '''
    # Test
    if conf.TEST_RATIO > 0:
        model.load_state_dict(torch.load(trained_model_file))  #
        model.eval()
        with torch.no_grad():
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            for i, (features, length, label) in enumerate(test_loader):
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
        print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
              .format(test_loss, SROCC, KROCC, PLCC, RMSE))
        # np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))


def originalCNNFeatExtractMain():
    dfgdsf = 4


#
def main():
    return 0


if __name__ == '__main__':
    originalVSFAMain()
