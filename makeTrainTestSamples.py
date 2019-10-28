# 你可以叫我御姐
import h5py

import skvideo.io
import os
import scipy.io as scio
import json
import numpy as np
import skvideo.io
import torch
import pickle
import Config as conf
import Utils.common as tools
from energyGuidedPooling import EnergyGuidedPooling
from contentFeature import ExtractFeatVid as contentFeatureExtractor
from  distortionFeatureDeepIQA import ExtractFeatVid
import lmdb

def buildUpSampleFileSystem(trainingSampleBasePath,benchmarkName):
    tools.securePath(os.path.join(trainingSampleBasePath,benchmarkName))
    return

# build up the trainig samples: map cubes  contentFeat distortionFeat
def BuildUpTrainingSamples():
    # securePath
    basePath = conf.TRAINING_SAMPLE_BASEPATH
    benchmarkName = conf.DATASET_NAME
    buildUpSampleFileSystem(basePath,benchmarkName)
    videoPath = conf.DATASET_VIDEOS_PATH
    videoSets = os.listdir(videoPath);videoSets.sort() # the sorted video names are also the order of the samples
    # load the video info
    infoMat = scio.loadmat(conf.DATASET_INFO_PATH)
    keyFn   = infoMat['video_names'].reshape([-1]).tolist(); keyFn = [str(x)[2:-2] for x in keyFn]
    valueMap   = infoMat['scores'].astype(np.float32)
    scoreScale = np.max(valueMap)

    # Firstly, the map, cubes and mos
    if conf.FLG_EXTRACT_MAPCUBES:
        counter = 0
        for videoName in videoSets:
            # For each video
            videoFeatsDir = os.path.join(basePath, benchmarkName, videoName[:-4])  # where to save these features
            if os.path.exists(os.path.join(videoFeatsDir,'_energyMapNCubes.pkl')) and not conf.FLG_OVERWRITE_MAPCUBES:
                counter += 1
                print(videoName + ' info, maps and cubes have already been extracted! (%f%%)' % (100 * counter / len(videoSets)))
                continue

            counter += 1
            res1 = dict()
            tools.securePath(videoFeatsDir)
            # load the video and calculate
            videoClip = skvideo.io.vread(os.path.join(videoPath,videoName))
            se, te, cubes = EnergyGuidedPooling(videoClip)
            res1['se'] = se
            res1['te'] = te
            # post process(only align)
            maxlen = conf.MAX_TIME_LEN
            canvas = np.zeros([maxlen,3,conf.SAMPLING_SIZE,conf.SAMPLING_SIZE],np.float32)
            newCubes = []
            for i in cubes:
                fore = i[:, :, :, :] / 255.0  # regularize
                canvas[:fore.shape[0],:,:,:] = fore
                newCubes.append(canvas)
            for i in newCubes:
                if i.shape[0] != 240:
                    d = 67
            res1['cubes']       = newCubes
            res1['vidLen']      = videoClip.shape[0]
            res1['mos']         = valueMap[keyFn.index(videoName[:-4])]
            res1['mosScale']    = scoreScale
            with open(os.path.join(videoFeatsDir,'_energyMapNCubes.pkl'), 'wb') as f: # final is '3242353245_sample.pkl'
                pickle.dump(res1, f)
            print(videoName + ' info, maps and cubes are extracted! (%f%%)' % (100*counter/len(videoSets)))
        torch.cuda.empty_cache()
    # for the distortion feature
    if conf.FLG_EXTRACT_FEAT_DISTORTION:
        counter = 0
        for videoName in videoSets:
            # For each video
            videoFeatsDir = os.path.join(basePath, benchmarkName, videoName[:-4])  # where to save these features
            if os.path.exists(os.path.join(videoFeatsDir, '_distortionFeat.pkl')) and not conf.FLG_OVERWRITE_FEAT_DISTORTION:
                counter += 1
                print(videoName + ' distortion feat has already been extracted! (%f%%)' % (
                            100 * counter / len(videoSets)))
                continue

            counter += 1
            res2 = dict()
            tools.securePath(videoFeatsDir)
            # load the video and calculate
            videoClip = skvideo.io.vread(os.path.join(videoPath, videoName))
            if conf.DISTORTION_ALGORITHM_NAME == 'DeepIQA':
                feat      = ExtractFeatVid(videoClip)
                # post process(cut and align)
                maxlen = conf.MAX_TIME_LEN
                interv = conf.ADJACENT_INTERVAL // 2
                newFeat = np.zeros([maxlen - 2 * interv, 512], np.float32)
                newFeat[:feat.shape[0] - 2 * interv, :] = feat[interv:-interv, :]
            else:
                newFeat = []
            res2['distortionFeat'] = newFeat
            with open(os.path.join(videoFeatsDir,'_distortionFeat.pkl'), 'wb') as f: # final is '3242353245_sample.pkl'
                pickle.dump(res2, f)
            print(videoName + ' distortion feat is extracted! (%f%%)' % (100*counter/len(videoSets)))
        torch.cuda.empty_cache()

    # for the content feature
    if conf.FLG_EXTRACT_FEAT_CONTENT:
        counter = 0
        for videoName in videoSets:
            videoFeatsDir = os.path.join(basePath, benchmarkName, videoName[:-4])  # where to save these features
            if os.path.exists(os.path.join(videoFeatsDir, '_contentFeat.pkl')) and not conf.FLG_OVERWRITE_FEAT_CONTENT:
                counter += 1
                print(videoName + ' content feat has already been extracted! (%f%%)' % (
                        100 * counter / len(videoSets)))
                continue
            # For each video
            counter += 1
            res3 = dict()
            videoFeatsDir = os.path.join(
                os.path.join(basePath, benchmarkName, videoName[:-4]))  # where to save these features
            tools.securePath(videoFeatsDir)
            # load the video and calculate
            videoClip = skvideo.io.vread(os.path.join(videoPath, videoName))
            feat = contentFeatureExtractor(videoClip)
            # post process(cut and align)
            maxlen = conf.MAX_TIME_LEN
            interv = conf.ADJACENT_INTERVAL // 2
            newFeat = np.zeros([maxlen - 2*interv, 4096], np.float32)
            newFeat[:feat.shape[0] - 2*interv, :] = feat[interv:-interv, :].cpu().numpy()
            if newFeat.shape[0] != 238:
                d = 67
            res3['contentFeat'] = newFeat
            with open(os.path.join(videoFeatsDir, '_contentFeat.pkl'),
                      'wb') as f:
                pickle.dump(res3, f)
            print(videoName + ' content feat is extracted! (%f%%)' % (100 * counter / len(videoSets)))
        torch.cuda.empty_cache()

    print('================================= Feature Extractions are Done! =================================')
    # merging the samples
    if conf.FLG_MERGE_MAP_CUBE_AND_FEATS:
        alreadyExtractedVideos = os.listdir(os.path.join(basePath, benchmarkName))
        counter = 0
        for vname in alreadyExtractedVideos:
            directoryName = os.path.join(basePath, benchmarkName,vname)
            if os.path.exists(os.path.join(directoryName, 'sample.pkl')) and not conf.FLG_OVERWRITE_MERGE_ALL:
                counter += 1
                print('Sample ' + vname + ' has already been build! (%f%%)' % (100 * counter / len(alreadyExtractedVideos)))
                continue
            counter += 1
            with open(os.path.join(directoryName,'_energyMapNCubes.pkl'), 'rb') as f:
                mapCubeDict = pickle.load(f)

            with open(os.path.join(directoryName,'_distortionFeat.pkl'), 'rb') as f:
                distortFeatDict = pickle.load(f)

            with open(os.path.join(directoryName,'_contentFeat.pkl'), 'rb') as f:
                contentFeatDict = pickle.load(f)
            combinedDict = dict()
            combinedDict.update(mapCubeDict)
            combinedDict.update(distortFeatDict)
            combinedDict.update(contentFeatDict)
            with open(os.path.join(directoryName, 'sample.pkl'),
                      'wb') as f:
                pickle.dump(combinedDict, f)
            print('Sample ' + vname + ' is build!  (%f%%)' % (100 * counter / len(alreadyExtractedVideos)))




def makeH5py():
    videoPath = conf.DATASET_VIDEOS_PATH
    videoSets = os.listdir(videoPath)
    videoSets.sort()
    interval = 100;

    h5pyf = h5py.File(os.path.join(conf.TRAINING_SAMPLE_BASEPATH, conf.DATASET_NAME,'fastRecord.hdf5'), "w")

    for i in range(len(videoSets)):
        vn = videoSets[i]
        directoryName = os.path.join(conf.TRAINING_SAMPLE_BASEPATH, conf.DATASET_NAME, vn[:-4])
        with open(os.path.join(directoryName,'sample.pkl'), 'rb') as f:
            sampleDict = pickle.load(f)

        # 第i号样本的数据
        dsi = h5pyf.create_group(str(i))
        for keys in sampleDict.keys():
            dsi[keys] = sampleDict[keys]

        print(i)
    h5pyf.close()
    print('数据写入完毕')
    return






def testHD5F():
    ## access by its attribute
    f = h5py.File("myh5py.hdf5", "w")

    # 创建组bar1,组bar2，数据集dset
    g1 = f.create_group("bar1")
    g2 = f.create_group("bar2")
    d = f.create_dataset("dset", data=np.arange(10))

    # 在bar1组里面创建一个组car1和一个数据集dset1。
    c1 = g1.create_group("car1")
    d1 = g1.create_dataset("dset1", data=np.arange(10))

    # 在bar2组里面创建一个组car2和一个数据集dset2
    c2 = g2.create_group("car2")
    d2 = g2.create_dataset("dset2", data=np.arange(10))

    # 根目录下的组和数据集
    print(".............")
    for key in f.keys():
        print(f[key].name)

    # bar1这个组下面的组和数据集
    print(".............")
    for key in g1.keys():
        print(g1[key].name)

    # bar2这个组下面的组和数据集
    print(".............")
    for key in g2.keys():
        print(g2[key].name)

    # 那么car1组和car2组下面都有什么
    print(".............")
    print(c1.keys())
    print(c2.keys())
    print(g1.keys())
    df = 342





def main():
    makeH5py()
    return 43

if __name__ == '__main__':
    main()