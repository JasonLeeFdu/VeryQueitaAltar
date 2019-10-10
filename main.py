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
from argparse import ArgumentParser
import network as nt

# Program
import Config as conf
import NetBricks.ljchopt1 as bricks
import Utils.common as tools









def originalVSFAMain():
    '''
        超参数：


    '''
    parser = ArgumentParser(description='"VSFA: Quality Assessment of In-the-Wild Videos')
    # 学习率
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate (default: 0.00001)')
    # 批大小
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    # EPOCH
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of epochs to train (default: 3000)')
    # 数据集Datasets
    parser.add_argument('--database', default='KoNViD-1k', type=str,
                        help='database name (default: KoNViD-1k)')
    # 模型名称
    parser.add_argument('--model', default='VSFA', type=str,
                        help='model name (default: VSFA)')
    # 实验ID
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    # 集合划分，test比率
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='test ratio (default: 0.2)')
    # 集合划分，val比率
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='val ratio (default: 0.2)')
    # Weight Decay
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    # 在训练的时候不测试
    parser.add_argument("--notest_during_training", action='store_true',
                        help='flag whether to test during training')
    # 是否启用Tensorboard
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to enable TensorBoard visualization')
    # tensorboard的路径
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    # 禁用GPU
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    # 根据不同的数据集来设定不同的 信息与feature
    if args.database == 'KoNViD-1k':
        features_dir = 'CNN_features_KoNViD-1k/'  # features dir
        datainfo = 'Data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        features_dir = 'CNN_features_CVD2014/'
        datainfo = 'Data/CVD2014info.mat'
    if args.database == 'LIVE-Qualcomm':
        features_dir = 'CNN_features_LIVE-Qualcomm/'
        datainfo = 'Data/LIVE-Qualcomminfo.mat'
        args.epochs = 15000  # need more training to converge

    # 路径的设置与保存
    tools.securePath('Models')
    tools.securePath('Results')
    trained_model_file = os.path.join(conf.MODEL_PATH,'trainedParams')
    save_result_file = conf.RESULT_PATH
    if not args.disable_visualization:  # Tensorboard Visualization
        writer = SummaryWriter(log_dir=conf.LOG_PATH)

    print('=============================== 本次训练信息 ==============================')
    print('实验 ID: {}'.format(args.exp_id))
    print('数据集' + args.database)
    print('模型名称' + args.model)
    print('==========================================================================')
    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    # 读取统计的Dataset信息
    Info = h5py.File(datainfo)  # index, ref_ids.二者的关系。ref_ids是视频的id，index是若干次id的random permute
    index = Info['index']
    index = index[:, args.exp_id % index.shape[1]]  # np.random.permutation(N)
    ref_ids = Info['ref_ids'][0, :]  #
    max_len = int(Info['max_len'][0])
    # randpermute切割
    trainindex = index[0:int(np.ceil((1 - args.test_ratio - args.val_ratio) * len(index)))]
    testindex = index[int(np.ceil((1 - args.test_ratio) * len(index))):len(index)]
    train_index, val_index, test_index = [], [], []

    for i in range(len(ref_ids)):
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            test_index.append(i) if (ref_ids[i] in testindex) else \
                val_index.append(i)

    # 有这么做的
    scale = Info['scores'][0, :].max()  # label normalization factor

    train_dataset = nt.VQADataset(features_dir, train_index, max_len, scale=scale)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = nt.VQADataset(features_dir, val_index, max_len, scale=scale)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset)
    if args.test_ratio > 0:
        test_dataset = nt.VQADataset(features_dir, test_index, max_len, scale=scale)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

    model = nt.VSFA().to(device)  #

    criterion = nn.L1Loss()  # L1 loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_criterion = -1  # SROCC min

    for epoch in range(args.epochs):
        # Train
        model.train()
        L = 0
        for i, (features, length, label) in enumerate(train_loader):
            features = features.to(device).float()
            label = label.to(device).float()
            optimizer.zero_grad()  #
            outputs = model(features, length.float())
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            L = L + loss.item()
            if i % 10 == 0:
                print('Iter %d :, Loss %f:' % (i, loss))
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
        if args.test_ratio > 0 and not args.notest_during_training:
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

        if not args.disable_visualization:  # record training curves
            writer.add_scalar("loss/train", train_loss, epoch)  #
            writer.add_scalar("loss/val", val_loss, epoch)  #
            writer.add_scalar("SROCC/val", val_SROCC, epoch)  #
            writer.add_scalar("KROCC/val", val_KROCC, epoch)  #
            writer.add_scalar("PLCC/val", val_PLCC, epoch)  #
            writer.add_scalar("RMSE/val", val_RMSE, epoch)  #
            if args.test_ratio > 0 and not args.notest_during_training:
                writer.add_scalar("loss/test", test_loss, epoch)  #
                writer.add_scalar("SROCC/test", SROCC, epoch)  #
                writer.add_scalar("KROCC/test", KROCC, epoch)  #
                writer.add_scalar("PLCC/test", PLCC, epoch)  #
                writer.add_scalar("RMSE/test", RMSE, epoch)  #

        # Update the model with the best val_SROCC
        # when epoch is larger than args.epochs/6
        # This is to avoid the situation that the model will not be updated
        # due to the impact of randomly initializations of the networks
        if val_SROCC > best_val_criterion and epoch > args.epochs / 6:
            print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(args.exp_id, epoch))
            print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            if args.test_ratio > 0 and not args.notest_during_training:
                print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(test_loss, SROCC, KROCC, PLCC, RMSE))
                np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
            torch.save(model.state_dict(), trained_model_file)
            best_val_criterion = val_SROCC  # update best val SROCC

    # Test
    if args.test_ratio > 0:
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
















