## image IO
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

# Program
import Config as conf
import Utils.common as tools



torch.cuda.current_device()
torch.cuda._initialized = True





def originalVSFAMain():
    # 根据不同的数据集来设定不同的 信息与feature
    if conf.DATASET_NAME == 'KoNViD-1k':
        features_dir = 'CNN_features_KoNViD-1k/'  # features dir
        datainfo = 'Data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if conf.DATASET_NAME == 'CVD2014':
        features_dir = 'CNN_features_CVD2014/'
        datainfo = 'Data/CVD2014info.mat'
    if conf.DATASET_NAME == 'LIVE-Qualcomm':
        features_dir = 'CNN_features_LIVE-Qualcomm/'
        datainfo = 'Data/LIVE-Qualcomminfo.mat'
        conf.MAX_Epoch = 15000  # need more training to converge


    # 路径的设置与保存
    tools.securePath('Models')
    tools.securePath('Results')
    tools.securePath(conf.MODEL_PATH)
    tools.securePath(conf.RESULT_PATH)
    trained_model_file = os.path.join(conf.MODEL_PATH,'trainedParams')
    save_result_file = conf.RESULT_PATH
    writer = SummaryWriter(log_dir=conf.LOG_PATH)


    # 提示信息显示
    print('=============================== 本次训练信息 ==============================')
    print('实验模型名称: {}'.format(conf.MODEL_NAME))
    print('数据集' + conf.DATASET_NAME)
    print('模型名称' + conf.MODEL_NAME)
    print('==========================================================================')



    # 读取统计的Dataset信息       # exp_id = 0
    Info = h5py.File(datainfo)  # index, ref_ids.二者的关系。ref_ids是视频的id，index是若干次id的random permute
    max_len = int(Info['max_len'][0])



    # 将数据集的分数进行标准化(有这么做的)
    scale = Info['scores'][0, :].max()  # label normalization factor

    # 数据集划分，把数据集分成 训练：验证：测试  3 : 1 : 1
    N = Info['index'].shape[0]
    TrainN = int(N * conf.TRAIN_RATIO)
    ValN = int(N * conf.VAL_RATIO)
    arr = np.random.permutation(N)
    train_index = arr[:TrainN]
    val_index   = arr[TrainN:ValN + TrainN]
    test_index  = arr[ValN + TrainN:]



    # 数据集制作
    train_dataset = nt.VQADataset(features_dir, train_index, max_len, scale=scale)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=conf.BATCH_SIZE, shuffle=True)
    val_dataset = nt.VQADataset(features_dir, val_index, max_len, scale=scale)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset)
    if conf.TEST_RATIO > 0:
        test_dataset = nt.VQADataset(features_dir, test_index, max_len, scale=scale)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

    # 网络载入并设定损失与优化
    device = torch.device("cuda")
    model = nt.VSFA().to(device)  #
    criterion = nn.L1Loss()  # 本文采用 L1 loss
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LR, weight_decay=conf.WEIGHT_DECAY)
    best_val_criterion = -1  # 选取模型是采用验证集里面，表现最好的那一个SROCC min


    for epoch in range(conf.MAX_Epoch):
        # Train for 1 epoch
        print('EPOCH:' + str(epoch + 1) + '/' + str(conf.MAX_Epoch))
        model.train()
        L = 0
        for i, (features, length, label) in enumerate(train_loader):
            features = features.float().cuda()
            label = label.float().cuda()
            optimizer.zero_grad()  #
            outputs = model(features, length.float())
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            L = L + loss.item()
            if i % 10 == 0:
                print('Iter: %d, Loss: %f' % (i, loss))
        train_loss = L / (i + 1)

        model.eval()
        # Val
        y_pred = np.zeros(len(val_index))
        y_val = np.zeros(len(val_index))
        L = 0
        with torch.no_grad():
            for i, (features, length, label) in enumerate(val_loader):
                y_val[i] = scale * label.numpy()  #
                features = features.to(device).float()
                label = label.to(device).float()
                outputs = model(features, length.float())
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
                for i, (features, length, label) in enumerate(test_loader):
                    y_test[i] = scale * label.numpy()  #
                    features = features.to(device).float()
                    label = label.to(device).float()
                    outputs = model(features, length.float())
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
        if conf.TEST_RATIO > 0 and not conf.NO_TEST_DURING_TRAINING:
            writer.add_scalar("loss/test", test_loss, epoch)
            writer.add_scalar("SROCC/test", SROCC, epoch)
            writer.add_scalar("KROCC/test", KROCC, epoch)
            writer.add_scalar("PLCC/test", PLCC, epoch)
            writer.add_scalar("RMSE/test", RMSE, epoch)

        # 选择验证效果好的模型保存(由于是基于epoch的，所以比较科学)
        if val_SROCC > best_val_criterion  and epoch > conf.MAX_Epoch / 6:
            print("实验{}: 更新最佳参数，位于 Epoch {}".format(conf.MODEL_NAME, epoch))
            print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            if conf.TEST_RATIO > 0 and not conf.NO_TEST_DURING_TRAINING:
                print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(test_loss, SROCC, KROCC, PLCC, RMSE))
                np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
            torch.save(model.state_dict(), trained_model_file)
            best_val_criterion = val_SROCC  # update best val SROCC

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
                features = features.to(device).float()
                label = label.to(device).float()
                outputs = model(features, length.float())
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
        np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))


def originalCNNFeatExtractMain():
    dfgdsf = 4


#
def main():
    return 0


if __name__ == '__main__':
    originalVSFAMain()
















