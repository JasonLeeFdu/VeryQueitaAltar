## image IO
import imageio
import skimage

# tf
import  tensorflow as tf

# Traditional
import os
import numpy as np
import time
import cv2 as cv
from datetime import datetime
import math
from tqdm import tqdm

# Program
import Config as conf
import Utils.common as tools
import recordMaker as rm
from network import *
import matplotlib.pyplot as plot
import tensorflow.contrib.slim as slim
import argparse
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    default='data')

parser.add_argument('--test_dir',
                    default='data')

parser.add_argument('--model_dir',
                    default='model/multi_task_0812')

parser.add_argument('--restore_dir',
                    default='model/multi_task/model99.ckpt')

parser.add_argument('--restore',
                    default=False
                    )

parser.add_argument('--epochs',
                    type=int,
                    default=202)

parser.add_argument('--peochs_per_eval',
                    type=int,
                    default=1)

parser.add_argument('--logdir',
                    default='model/multi_task_0812/logs')

parser.add_argument('--batch_size',
                    type=int,
                    default=12)

parser.add_argument('--is_cross_entropy',
                    action='store_true',
                    default=True)

parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-4)

parser.add_argument('--decay_rate',
                    type=float,
                    default=0.316)


parser.add_argument('--decay_step',
                    type=int,
                    default=50)

parser.add_argument('--weight',
                    nargs='+',
                    type=float,
                    default=[1.0,1.0])

parser.add_argument('--random_seed',
                    type=int,
                    default=1234)

flags = parser.parse_args()

'''
输入归一化的问题                            => done
L2 norm of weight-weight decay           => done
L2 gamma beta                            => done
学习率的问题                               => done
保证batchnorm中的gamma beta 问题           => done (还是采用了gradient apply)


precision recall IOU 等专用医用数据
医用数据代码对接
医用独立分类接口

tensorboard                              => done
断点续传                                  => done
确保trian test 一致问题                    => done
实现train val                            => done

'''



def trainMaskModel():
    # 目录
    tools.securePath(conf.LOG_PATH)
    tools.securePath(conf.MODEL_PATH)
    tools.securePath(conf.STEALTH_MODE_MODEL_PATH)
    with tf.Session() as sess:
        ## summarize 的名字
        trainWriter = tf.summary.FileWriter(os.path.join(conf.LOG_PATH, 'train'), sess.graph)
        testWriter = tf.summary.FileWriter(os.path.join(conf.LOG_PATH, 'val'), sess.graph)

        # 输入数据 input data
        trainSwitch = tf.placeholder(dtype=tf.bool)
        coord = tf.train.Coordinator()
        imageTrain, labelTrain, clsLabelTrain = rm.readAndDecode(conf.TRAIN_FN)
        imageVal, labelVal, clsLabelVal = rm.readAndDecode(conf.VAL_FN)
        smpNumTrain = tools.countTfRecordSamples(conf.TRAIN_FN)
        smpNumVal = tools.countTfRecordSamples(conf.VAL_FN)
        iterPEpochTrain = tools.itersPerEpoch(smpNumTrain,conf.BATCH_SIZE)
        iterPEpochTest = tools.itersPerEpoch(smpNumVal,conf.VALTEST_BATCHSIZE)




        imageTrainB, labelTrainB, clsLabelTrainB = tf.train.shuffle_batch([imageTrain, labelTrain, clsLabelTrain],
                                                                          batch_size=conf.BATCH_SIZE,
                                                                          capacity=512,
                                                                          min_after_dequeue=256, num_threads=3)
        imageValB, labelValB, clsLabelValB = tf.train.shuffle_batch([imageVal, labelVal, clsLabelVal],
                                                                    batch_size=conf.VALTEST_BATCHSIZE,
                                                                    capacity=512,
                                                                    min_after_dequeue=256, num_threads=3)

        # 用于读取数据的多个线程(队列读取线程)的协调，否则会在线程暂停的时候报错
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 输入管道的架设(训练验证分支)
        image = tf.placeholder_with_default(input=imageTrainB, shape=[None, 224, 224, 3])
        label = tf.placeholder_with_default(input=labelTrainB, shape=[None, 224, 224, conf.FIANL_CLASSES_NUM])
        clsLabel = tf.placeholder_with_default(input=clsLabelTrainB, shape=[None, 1, 1, conf.FIANL_CLASSES_NUM])

        # 构建网络

        lev1, lev2, lev3, lev4, fusedSingleClsPred, fusedMultiClsPred = conf.FUNC_HANDEL(image, trainSwitch,conf.LOAD_PRETRAIN,sess)
        lossFunction,iouAvg = conf.LOSS_HANDLE(lev1, lev2, lev3, lev4, fusedSingleClsPred, fusedMultiClsPred, label)



        # 正则化项

        weights_var = tf.trainable_variables()
        weights_var_withoutNorm = [x for x in weights_var if
                                   ((x.name.find('gamma') == -1) and (x.name.find('beta') == -1))]
        l2NormLoss = conf.WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in weights_var_withoutNorm]) / len(
            weights_var_withoutNorm)

        # 统一loss
        loss = lossFunction + l2NormLoss

        # 记录哪一些重要的节点
        tf.summary.image("input", image)
        tf.summary.image("label", tf.expand_dims(label[:, :, :, 1], axis=-1))
        tf.summary.scalar("Loss(networkAndL2)", loss)

        # 全局训练次数
        globalStep = tf.Variable(0, name='global_step', trainable=False)

        # 不同层设置不同的学习率

        learning_rate = tf.train .exponential_decay(conf.LR, tf.train.get_or_create_global_step(), conf.LR_INTERVAL, conf.LR_DECAY_FACOTOR,
                                                  staircase=True)
        #learning_rate = tf.train.exponential_decay(flags.learning_rate, globalStep,
                                                   #tf.cast(2428 / flags.batch_size * flags.decay_step, tf.int32),
                                                   #flags.decay_rate, staircase=True)
        #learning_rate = tf.train.exponential_decay(conf.LR, tf.train.get_or_create_global_step(), (conf.MAX_ITERATIONS / 40), 0.9,
        #staircase=True)

        if not conf.LR_PRETRAIN_DIFFERENT:
            optimizer = tf.train.AdamOptimizer(learning_rate)
            #optimizer = tf.train.MomentumOptimizer(learning_rate,0.9)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # 为了保证bn 顺利工作
            with tf.control_dependencies(update_ops):
                trainOpts = optimizer.minimize(loss, tf.train.get_or_create_global_step())
                optLr2 = optimizer._lr
                #optLr2 = loss
        else:
            weights_var_all = tf.trainable_variables()
            # weights_var_all_withoutNorm = [x for x in weights_var_all if ((x.name.find('gamma') == -1) and (x.name.find('beta') == -1))]
            weights_var_resnet = tf.trainable_variables(scope=conf.PRETRAIN_SCOPE)
            weights_var_resnet = weights_var_resnet[:-conf.RESERVE_LEVEL] # resnet50 weights_var_resnet[:-2]  # VGG19-weights_var_resnet[:-6]
            # weights_var_resnet_withoutNorm = [x for x in weights_var_resnet if ((x.name.find('gamma') == -1) and (x.name.find('beta') == -1))]
            weights_var_FPNBridge = [x for x in weights_var_all if x not in weights_var_resnet]
            # weights_var_FPNBridge_withoutNorm = [x for x in weights_var_all_withoutNorm if x not in weights_var_resnet_withoutNorm]
            var_list1 = weights_var_resnet  # 模型参数
            #####################

            var_list2 = weights_var_FPNBridge
            opt1 = tf.train.AdamOptimizer(learning_rate * conf.PRETRAIN_DECAY_RATE)
            optLr1 = opt1._lr
            opt2 = tf.train.AdamOptimizer(learning_rate)
            optLr2 = opt2._lr
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # 为了保证bn 顺利工作
            with tf.control_dependencies(update_ops):
                grads = tf.gradients(loss, var_list1 + var_list2)
                grads1 = grads[:len(var_list1)]
                grads2 = grads[len(var_list1):]
                train_op1 = opt1.apply_gradients(zip(grads1, var_list1), global_step=globalStep)  # only once
                train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
                trainOpts = tf.group(train_op1, train_op2)

        # 全局变量赋值
        initOpt = tf.global_variables_initializer()
        sess.run(initOpt)

        # 载入之前训练的模型
        saver = tf.train.Saver(max_to_keep=10)
        ckpt = tf.train.get_checkpoint_state(conf.MODEL_PATH)
        start_it = 1
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_str = ckpt.model_checkpoint_path
            start_it = int(ckpt_str[ckpt_str.find('-') + 1:]) + 1
            print('Continue training at Iter %d' % start_it)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No training model found, start from iter 1')

        # 训练循环
        print('开始训练')
        AvgFreq = 0
        Avgloss = 0
        mergedSummOpt = tf.summary.merge_all()
        mI = 0;ctrl_1 = False
        if conf.TRAIN_EPOCH_OR_ITERS == 'iter':
            mI = conf.MAX_ITERATIONS
            mE = tools.currentEpoch(conf.MAX_ITERATIONS, iterPEpochTrain)

        elif conf.TRAIN_EPOCH_OR_ITERS == 'epoch':
            mI = conf.MAX_Epoch * iterPEpochTrain
            mE = conf.MAX_Epoch

        for iter1 in np.arange(start_it, mI+1):
            ctrl_1 = True
            startTime = time.time()  # 统计
            _, lossData, lr2, summ = sess.run([trainOpts, loss, optLr2, mergedSummOpt],
                                              feed_dict={trainSwitch: True})  # 训练，注意sess.run 的第一个参数是一个fetch list
            endTime = time.time()
            # 更新统计变量
            AvgFreq += endTime - startTime
            Avgloss += lossData

            ## 显示
            if (iter1 % conf.PRINT_INTERVAL == 0) and (iter1 != 0):  # 显示  gs
                AvgFreq = (conf.PRINT_INTERVAL * conf.BATCH_SIZE) / AvgFreq
                Avgloss = Avgloss / conf.PRINT_INTERVAL
                epochTrain = tools.currentEpoch(iter1,iterPEpochTrain)
                format_str = '%s: Epoch: %d(%d), Iters %d(%d), average loss(255) = %.7f, average frequency = %.3f(HZ), LR = %.10f'
                if iter1 % conf.SAVE_INTERVAL == 0:
                    print(format_str % (datetime.now(), epochTrain, mE, iter1, mI, Avgloss, AvgFreq, lr2), end='')
                else:
                    print(format_str % (datetime.now(), epochTrain, mE, iter1, mI, Avgloss, AvgFreq, lr2))
                AvgFreq = 0
                Avgloss = 0
            ## 保存
            if (iter1 % conf.SAVE_INTERVAL == 0) and (iter1 != 0):  # 显示
                saver.save(sess, os.path.join(conf.MODEL_PATH, 'model.ckpt'), global_step=iter1)
                print(' ... Iter %d model saved! ' % iter1)
            ## summary
            if iter1 % conf.SUMMARY_INTERVAL == 0 :  # 显示
                trainWriter.add_summary(summ, iter1)
            ## test and summary
            if iter1 % conf.VALIDATION_INTERVAL == 0:  # 显示
                trainWriter.add_summary(summ, iter1)
                valLoss = 0.0
                for i in range(iterPEpochTest):
                    ima, lab, cls = sess.run([imageValB, labelValB, clsLabelValB])
                    iouavg,lossData,summ = sess.run([iouAvg,loss, mergedSummOpt],
                                              feed_dict={trainSwitch: False, image: ima, label: lab,
                                                         clsLabel: cls})  # 训练，注意sess.run 的第一个参数是一个fetch list
                    valLoss += iouavg
                    if i == 0:
                        testWriter.add_summary(summ, iter1)
                valLoss /= iterPEpochTest
                epoch = tools.currentEpoch(iter1, iterPEpochTrain)
                print('VALIDATION @ Epoch {},  IOU AVG:{} '.format(epoch, valLoss))
            ## stealth
            if (iter1 % conf.STEALTH_INTERVAL == 0) and (iter1 != 0) and (conf.STEALTH_MODE_MODEL_ON):  # 显示
                srcDir = conf.MODEL_PATH
                li = os.listdir(srcDir);li.sort()
                targets = li[-3:]
                for elem in targets:
                    srcFn = os.path.join(srcDir,elem)
                    dstFn = os.path.join(conf.STEALTH_MODE_MODEL_PATH,elem)
                    shutil.copyfile(srcFn,dstFn)

        if ctrl_1:
            # When finish the last iteration
            saver.save(sess, os.path.join(conf.MODEL_PATH, 'model.ckpt'), global_step=mI)
            print(' Last model saved! Finish training.')
        else:
            print(' No iteration required. Already trained.')

        # close the queue reading threads// used for multi-queue readers
        coord.request_stop()
        coord.join(threads)



def trainClsModel():
    # 目录
    tools.securePath(conf.LOG_PATH)
    tools.securePath(conf.MODEL_PATH)
    with tf.Session() as sess:
        ## summarize 的名字
        trainWriter = tf.summary.FileWriter(os.path.join(conf.LOG_PATH, 'train'), sess.graph)
        testWriter = tf.summary.FileWriter(os.path.join(conf.LOG_PATH, 'val'), sess.graph)

        # 输入数据 input data
        trainSwitch = tf.placeholder(dtype=tf.bool)
        coord = tf.train.Coordinator()
        imageTrain, labelTrain, clsLabelTrain = rm.readAndDecode(conf.TRAIN_FN)
        imageVal, labelVal, clsLabelVal = rm.readAndDecode(conf.VAL_FN)
        smpNumTrain = tools.countTfRecordSamples(conf.TRAIN_FN)
        smpNumVal = tools.countTfRecordSamples(conf.VAL_FN)
        iterPEpochTrain = tools.itersPerEpoch(smpNumTrain,conf.BATCH_SIZE)
        iterPEpochTest = tools.itersPerEpoch(smpNumVal,conf.VALTEST_BATCHSIZE)




        imageTrainB, labelTrainB, clsLabelTrainB = tf.train.shuffle_batch([imageTrain, labelTrain, clsLabelTrain],
                                                                          batch_size=conf.BATCH_SIZE,
                                                                          capacity=512,
                                                                          min_after_dequeue=256, num_threads=3)
        imageValB, labelValB, clsLabelValB = tf.train.shuffle_batch([imageVal, labelVal, clsLabelVal],
                                                                    batch_size=conf.VALTEST_BATCHSIZE,
                                                                    capacity=512,
                                                                    min_after_dequeue=256, num_threads=3)

        # 用于读取数据的多个线程(队列读取线程)的协调，否则会在线程暂停的时候报错
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 输入管道的架设(训练验证分支)
        image = tf.placeholder_with_default(input=imageTrainB, shape=[None, 224, 224, 3])
        label = tf.placeholder_with_default(input=labelTrainB, shape=[None, 224, 224, conf.FIANL_CLASSES_NUM])
        clsLabel = tf.placeholder_with_default(input=clsLabelTrainB, shape=[None, 1, 1, conf.FIANL_CLASSES_NUM])

        # 构建网络
        logitsClass, predCls = conf.FUNC_HANDEL(image, trainSwitch,conf.LOAD_PRETRAIN,sess)
        lossFunction,precisionCls = conf.LOSS_HANDLE( logitsClass, clsLabel)



        #if conf.LOAD_PRETRAIN:
        #    loadPretrainedResnetVGG19(sess)
        # loadPretrainedResnetVGG19(sess)

        # 正则化项

        weights_var = tf.trainable_variables()
        weights_var_withoutNorm = [x for x in weights_var if
                                   ((x.name.find('gamma') == -1) and (x.name.find('beta') == -1))]
        l2NormLoss = conf.WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in weights_var_withoutNorm]) / len(
            weights_var_withoutNorm)

        # 统一loss
        loss = lossFunction + l2NormLoss

        # 记录哪一些重要的节点
        tf.summary.image("input", image)
        tf.summary.image("label", tf.expand_dims(label[:, :, :, 1], axis=-1))
        tf.summary.scalar("Loss(networkAndL2)", loss)

        # 全局训练次数
        globalStep = tf.Variable(0, name='global_step', trainable=False)

        # 不同层设置不同的学习率

        learning_rate = tf.train .exponential_decay(conf.LR, tf.train.get_or_create_global_step(), conf.LR_INTERVAL, conf.LR_DECAY_FACOTOR,
                                                  staircase=True)
        #learning_rate = tf.train.exponential_decay(flags.learning_rate, globalStep,
                                                   #tf.cast(2428 / flags.batch_size * flags.decay_step, tf.int32),
                                                   #flags.decay_rate, staircase=True)
        #learning_rate = tf.train.exponential_decay(conf.LR, tf.train.get_or_create_global_step(), (conf.MAX_ITERATIONS / 40), 0.9,
        #staircase=True)

        if not conf.LR_PRETRAIN_DIFFERENT:
            optimizer = tf.train.AdamOptimizer(learning_rate)
            #optimizer = tf.train.MomentumOptimizer(learning_rate,0.9)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # 为了保证bn 顺利工作
            with tf.control_dependencies(update_ops):
                trainOpts = optimizer.minimize(loss, tf.train.get_or_create_global_step())
                optLr2 = optimizer._lr
                #optLr2 = loss
        else:
            weights_var_all = tf.trainable_variables()
            # weights_var_all_withoutNorm = [x for x in weights_var_all if ((x.name.find('gamma') == -1) and (x.name.find('beta') == -1))]
            weights_var_resnet = tf.trainable_variables(scope=conf.PRETRAIN_SCOPE)
            weights_var_resnet = weights_var_resnet[:-conf.RESERVE_LEVEL] # resnet50 weights_var_resnet[:-2]  # VGG19-weights_var_resnet[:-6]
            # weights_var_resnet_withoutNorm = [x for x in weights_var_resnet if ((x.name.find('gamma') == -1) and (x.name.find('beta') == -1))]
            weights_var_FPNBridge = [x for x in weights_var_all if x not in weights_var_resnet]
            # weights_var_FPNBridge_withoutNorm = [x for x in weights_var_all_withoutNorm if x not in weights_var_resnet_withoutNorm]
            var_list1 = weights_var_resnet  # 模型参数
            #####################

            var_list2 = weights_var_FPNBridge
            opt1 = tf.train.AdamOptimizer(learning_rate * conf.PRETRAIN_DECAY_RATE)
            optLr1 = opt1._lr
            opt2 = tf.train.AdamOptimizer(learning_rate)
            optLr2 = opt2._lr
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # 为了保证bn 顺利工作
            with tf.control_dependencies(update_ops):
                grads = tf.gradients(loss, var_list1 + var_list2)
                grads1 = grads[:len(var_list1)]
                grads2 = grads[len(var_list1):]
                train_op1 = opt1.apply_gradients(zip(grads1, var_list1), global_step=globalStep)  # only once
                train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
                trainOpts = tf.group(train_op1, train_op2)

        # 全局变量赋值
        initOpt = tf.global_variables_initializer()
        sess.run(initOpt)

        # 载入之前训练的模型
        saver = tf.train.Saver(max_to_keep=10)
        ckpt = tf.train.get_checkpoint_state(conf.MODEL_PATH)
        start_it = 1
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_str = ckpt.model_checkpoint_path
            start_it = int(ckpt_str[ckpt_str.find('-') + 1:]) + 1
            print('Continue training at Iter %d' % start_it)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No training model found, start from iter 1')

        # 训练循环
        print('开始训练')
        AvgFreq = 0
        Avgloss = 0
        mergedSummOpt = tf.summary.merge_all()
        mI = 0;ctrl_1 = False
        if conf.TRAIN_EPOCH_OR_ITERS == 'iter':
            mI = conf.MAX_ITERATIONS
            mE = tools.currentEpoch(conf.MAX_ITERATIONS, iterPEpochTrain)

        elif conf.TRAIN_EPOCH_OR_ITERS == 'epoch':
            mI = conf.MAX_Epoch * iterPEpochTrain
            mE = conf.MAX_Epoch

        for iter1 in np.arange(start_it, mI+1):
            ctrl_1 = True
            startTime = time.time()  # 统计
            _, lossData, lr2, summ = sess.run([trainOpts, loss, optLr2, mergedSummOpt],
                                              feed_dict={trainSwitch: True})  # 训练，注意sess.run 的第一个参数是一个fetch list
            endTime = time.time()
            # 更新统计变量
            AvgFreq += endTime - startTime
            Avgloss += lossData

            ## 显示
            if (iter1 % conf.PRINT_INTERVAL == 0) and (iter1 != 0):  # 显示  gs
                AvgFreq = (conf.PRINT_INTERVAL * conf.BATCH_SIZE) / AvgFreq
                Avgloss = Avgloss / conf.PRINT_INTERVAL
                epochTrain = tools.currentEpoch(iter1,iterPEpochTrain)
                format_str = '%s: Epoch: %d(%d), Iters %d(%d), average loss(255) = %.7f, average frequency = %.3f(HZ), LR = %.10f'
                if iter1 % conf.SAVE_INTERVAL == 0:
                    print(format_str % (datetime.now(), epochTrain, mE, iter1, mI, Avgloss, AvgFreq, lr2), end='')
                else:
                    print(format_str % (datetime.now(), epochTrain, mE, iter1, mI, Avgloss, AvgFreq, lr2))
                AvgFreq = 0
                Avgloss = 0
            ## 保存
            if (iter1 % conf.SAVE_INTERVAL == 0) and (iter1 != 0):  # 显示
                saver.save(sess, os.path.join(conf.MODEL_PATH, 'model.ckpt'), global_step=iter1)
                print(' ... Iter %d model saved! ' % iter1)
            ## summary
            if iter1 % conf.SUMMARY_INTERVAL == 0 :  # 显示
                trainWriter.add_summary(summ, iter1)
            ## test and summary

            if iter1 % conf.VALIDATION_INTERVAL == 0:  # 显示
                trainWriter.add_summary(summ, iter1)
                valLoss = 0.0
                for i in range(iterPEpochTest):
                    ima, lab, cls = sess.run([imageValB, labelValB, clsLabelValB])
                    pc,lossData,summ = sess.run([precisionCls,loss, mergedSummOpt],
                                              feed_dict={trainSwitch: False, image: ima, label: lab,
                                                         clsLabel: cls})  # 训练，注意sess.run 的第一个参数是一个fetch list
                    valLoss += pc
                    if i == 0:
                        testWriter.add_summary(summ, iter1)
                valLoss /= iterPEpochTest
                epoch = tools.currentEpoch(iter1, iterPEpochTrain)
                print('VALIDATION @ Epoch {},  Precision:{} '.format(epoch, valLoss))

        if ctrl_1:
            # When finish the last iteration
            saver.save(sess, os.path.join(conf.MODEL_PATH, 'model.ckpt'), global_step=mI)
            print(' Last model saved! Finish training.')
        else:
            print(' No iteration required. Already trained.')

        # close the queue reading threads// used for multi-queue readers
        coord.request_stop()
        coord.join(threads)




def main():
    trainMaskModel()



if __name__ == '__main__':
    main()





























































































































































