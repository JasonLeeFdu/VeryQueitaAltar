import numpy as np
import matplotlib.pylab as plt
import cv2 as cv
import skvideo.io
import torch
import torch.nn.functional as F
import copy
import pickle
import Config as conf


'''
网络输入帧的归一化


'''



# CONSTANT:

KERNEL_WIDTH    = 15
SIGMA           = 3             # key


# imshow

def imshow(img):
    if np.max(img) < 1.1:
        img = img * 255
    img = img.astype(np.uint8)
    plt.imshow(img)
    plt.show()


# Gaussian kernel generator
def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

# Do the convolution operation on every video frame.(By spatial energy map)
def convTheFrames_I(Vd,Kn,isCudaRes=True):
    '''

    :param Vd: [(Times, Height, Width, Channel)]                            NP
    :param Kn: [(WidthK,HeightK)]                                           NP
    :return:   [(Times, Height-HeightK//2, Width-WidthK//2, Channe)]        PT
    '''
    nFrames = Vd.shape[0];nChannel = Vd.shape[-1]
    Kn = np.expand_dims(Kn,0); Kn = np.expand_dims(Kn,0)
    Kn = np.tile(Kn,reps=[nFrames*nChannel,1,1,1])
    Kn_pt = torch.FloatTensor(Kn).cuda()
    Vdd = np.transpose(Vd, [0, 3, 1, 2])
    Vdd = np.reshape(Vdd, [1,Vdd.shape[0] * Vdd.shape[1], Vdd.shape[2], Vdd.shape[3]])
    Vd_pt = torch.FloatTensor(Vdd).cuda()
    res   = torch.clamp(torch.round(F.conv2d(Vd_pt,Kn_pt,groups=nFrames*nChannel)),0,255).detach()
    res   = res.view([Vd.shape[0],Vd.shape[-1], res.shape[2], res.shape[3]]) # 240 3 554 974
    res   = torch.transpose(res,1,2)
    res   = torch.transpose(res,2,3)
    if not isCudaRes:
        res = res.cpu().numpy()
    return res



# Softmax on a 2-D plane (By both map generation)
def featMapSoftMax_Pt(feat):
    ##@@##
    expFeat = torch.exp(torch.clamp(feat,max=70))
    factor = torch.sum(expFeat)
    res   = expFeat / factor
    return res


# padding original frames for later convolution (By spatial energy map)
def paddingFrames(oriVd,paddingWidth):
    paddingList = list()
    for i in range(oriVd.shape[0]):
        paddedFrame = cv.copyMakeBorder(oriVd[i, :, :, :], paddingWidth, paddingWidth, paddingWidth, paddingWidth,
                                        cv.BORDER_REFLECT)
        paddedFrame = np.expand_dims(paddedFrame, axis=0)
        paddingList.append(paddedFrame)
    paddedVideo = np.concatenate(paddingList, axis=0)
    return paddedVideo



## Solve the spatial energy map (fixing the center gaussian kernel get a totally different energy map)
def SolveSpatialEnergyMap(videoClip,keepPt = True):
    paddedVd = paddingFrames(videoClip,KERNEL_WIDTH//2)
    oriVd = torch.FloatTensor(videoClip).cuda()             #pt
    kernel = fspecial_gauss(KERNEL_WIDTH, SIGMA)
    gaussedVideo = convTheFrames_I(paddedVd, kernel)        #pt
    spatialEnergy = torch.mean(torch.mean(torch.abs(oriVd - gaussedVideo), dim=0), dim=-1)
    spatialEnergy = featMapSoftMax_Pt(spatialEnergy)
    if not keepPt:
        spatialEnergy = spatialEnergy.detach().cpu().numpy() #numpy
    else:
        spatialEnergy = spatialEnergy.detach() #pt
    torch.cuda.empty_cache()
    return  spatialEnergy



## Solve the temporal energy map
def SolveTemporalEnergyMap(videoClip,keepPt = True):
    videoClip = torch.FloatTensor(videoClip)
    vc1 = videoClip[:-1,:,:,:]
    vc2 = videoClip[1:,:,:,:]
    vcTemporalDiff = torch.mean(vc2 - vc1,dim=-1)
    temporalEnergy  = torch.std(vcTemporalDiff,dim=0)
    temporalEnergy = featMapSoftMax_Pt(temporalEnergy)

    if not keepPt:
        temporalEnergy = temporalEnergy.detach().cpu().numpy()    # numpy
    else:
        temporalEnergy  = temporalEnergy.detach()                  # pt
    torch.cuda.empty_cache()
    return temporalEnergy


# write the pooled cube into video
def writeCude(cube,fn='test.avi'):
    skvideo.io.vwrite(fn,cube)
    return

# plot 2D map (To 3D and np.uint8)
def plotRectOn2DMap(I, boxes): # boxes  ordered as matrix
    if np.max(I) < 1.2:
        I = I * 255;I = I.astype(np.uint8)
    II = copy.deepcopy(I)
    if len(I.shape)==2:
        II = np.expand_dims(II,axis=-1)
    width = 3
    color = [0, 0, 255]
    [h, w, c] = II.shape
    if c == 1:
        II = np.tile(II,reps=[1,1,3])
    mask = np.zeros([h, w])

    for box in boxes:
        # adjust
        newBox = box
        box = [newBox[2],newBox[0],newBox[3],newBox[1]]
        # | left
        mask[box[1]: box[3], box[0]: box[0] + width] = 1
        # - top
        mask[box[1]: box[1] + width, box[0]: box[2]] = 1
        # | right
        mask[box[1]: box[3], box[2] - width: box[2]] = 1
        # _ bottom
        mask[box[3] - width: box[3], box[0]: box[2]] = 1

    im_over = np.zeros(II.shape)
    im_over[:, :, 0] = (1 - mask) * II[:, :, 0] + mask * color[0]
    im_over[:, :, 1] = (1 - mask) * II[:, :, 1] + mask * color[1]
    im_over[:, :, 2] = (1 - mask) * II[:, :, 2] + mask * color[2]
    return im_over


# Fusing these maps
def fuseTeSeMap(seMap,teMap,fusionRate=0.5):
    totalEnergyMap = fusionRate * seMap + (1-fusionRate) * teMap
    return  totalEnergyMap


# get The energy guided pooling
def energyGuidedPoolingCore(seMap,teMap,fusionRate=conf.FUSION_RATE,sampleSize =conf.SAMPLING_SIZE ,sampleNum = conf.SAMPLING_NUM, strategy='fourTop'):
    '''
    :param teMap: temporal energy
    :param seMap:
    :param sampleSize:
    :param fusionRate:
    :param sampleNum:
    :param strategy:  'fourTop','uniRand','topNum'
    :return:
    '''

    totalEnergyMap = fuseTeSeMap(seMap,teMap,fusionRate=fusionRate)
    h,w = totalEnergyMap.shape
    hGrid = h // sampleSize
    wGrid = w // sampleSize
    if strategy == 'fourTop': # where sampleNum doesn't work
        hGridMid = hGrid // 2
        wGridMid = wGrid // 2
        pooledMap = np.zeros([hGrid,wGrid],np.float32)
        # calcu-energy
        for i in range(hGrid):
            for j in range(wGrid):
                arr = totalEnergyMap[i*sampleSize:(i+1)*sampleSize,j*sampleSize:(j+1)*sampleSize]
                aptitute = np.sum(arr)
                pooledMap[i,j] = aptitute

        resList = list()

        # top left
        maxV = -1.0
        maxIJ = [-1, -1]
        for i in range(hGridMid):
            for j in range(wGridMid):
                if pooledMap[i,j] > maxV:
                    maxV = pooledMap[i,j]
                    maxIJ = [i,j]
        resList.append(maxIJ)

        # top-right
        maxV = -1.0
        maxIJ = [-1, -1]
        for i in range(hGridMid):
            for j in range(wGridMid,wGrid):
                if pooledMap[i, j] > maxV:
                    maxV = pooledMap[i, j]
                    maxIJ = [i, j]
        resList.append(maxIJ)

        # down-left
        maxV = -1.0
        maxIJ = [-1, -1]
        for i in range(hGridMid,hGrid):
            for j in range(wGridMid):
                if pooledMap[i, j] > maxV:
                    maxV = pooledMap[i, j]
                    maxIJ = [i, j]
        resList.append(maxIJ)

        # down-right
        maxV = -1.0
        maxIJ = [-1, -1]
        for i in range(hGridMid,hGrid):
            for j in range(wGridMid,wGrid):
                if pooledMap[i, j] > maxV:
                    maxV = pooledMap[i, j]
                    maxIJ = [i, j]
        resList.append(maxIJ)
        resList = [ [x1*sampleSize,(x1+1)*sampleSize,x2*sampleSize,(x2+1)*sampleSize] for [x1,x2] in resList]
        return resList

    elif strategy == 'uniRand':
        idxs = np.random.permutation(hGrid*wGrid)
        resList = list()
        for i in range(sampleNum):
            idx= idxs[i]
            hg = idx // wGrid
            wg = idx %  wGrid
            resList.append([hg,wg])
        resList = [[x1 * sampleSize, (x1 + 1) * sampleSize, x2 * sampleSize, (x2 + 1) * sampleSize] for [x1, x2] in
                   resList]
        return  resList
    elif strategy == 'topNum':
        # calcu-energy
        pooledMap = np.zeros([hGrid,wGrid],np.float32)
        resList = list()
        for i in range(hGrid):
            for j in range(wGrid):
                arr = totalEnergyMap[i * sampleSize:(i + 1) * sampleSize, j * sampleSize:(j + 1) * sampleSize]
                aptitute = np.sum(arr)
                pooledMap[i, j] = aptitute

        pooledMapFlat = pooledMap.reshape([-1])
        idxs = (-pooledMapFlat).argsort()
        for i in range(sampleNum):
            idd = idxs[i]
            hg = idd // wGrid
            wg = idd % wGrid
            resList.append([hg,wg])
        resList = [[x1 * sampleSize, (x1 + 1) * sampleSize, x2 * sampleSize, (x2 + 1) * sampleSize] for [x1, x2] in
                   resList]
        return resList
    return []




# Gamma correction
def gammaAdjust(mapp,gamma=0.04):
    max = np.max(mapp)
    mapp = mapp / max
    newMap = np.clip(np.power(mapp, gamma),0,1)
    return  newMap




def EnergyGuidedPooling(videoClip):
    se = SolveSpatialEnergyMap(videoClip).cpu().numpy()
    te = SolveTemporalEnergyMap(videoClip).cpu().numpy()
    resList = energyGuidedPoolingCore(se, te)
    cubes = list()
    ## crop and save
    for rect in resList:
        cube = videoClip[:, rect[0]:rect[1], rect[2]:rect[3], :]
        cube = np.transpose(cube,[0,3,1,2])
        cubes.append(cube)
    return  se,te,cubes  # return the list of the cubes



def main():
    # 6.8G GPU Consumption
    videoName = '/home/winston/workSpace/PycharmProjects/VQA/Ours/Datasets/VQA/KoNViD/KoNViD_1k_videos/3337642103.mp4'
    videoClip = skvideo.io.vread(videoName)
    se, te, cubes = EnergyGuidedPooling(videoClip)




if __name__ == '__main__':
    main()









'''
test code:


1.
msg = skvideo.io.ffprobe(videoName)
    width = int(msg['video']['@width'])
    height = int(msg['video']['@height'])
    nFrames = int(msg['video']['@nb_frames'])

2. for test


def test1():
    ###　global energy map
    # read video
    videoName = '/home/winston/workSpace/PycharmProjects/VQA/Ours/Datasets/VQA/KoNViD/KoNViD_1k_videos/3337642103.mp4'
    videoClip = skvideo.io.vread(videoName)

    # Solve the guiding map (in GPU)
    se = SolveSpatialEnergyMap(videoClip).cpu().numpy()
    te = SolveTemporalEnergyMap(videoClip).cpu().numpy()

    totalEMap = fuseTeSeMap(se,te)
    totalEMapToned = gammaAdjust(totalEMap)
    resList = energyGuidedPoolingCore(se,te)
    rectedEnergyMap = plotRectOn2DMap(totalEMapToned,resList)
    imshow(rectedEnergyMap)


def test2():
    
    test savings
    

    ###　global energy map
    # read video
    videoName = '/home/winston/workSpace/PycharmProjects/VQA/Ours/Datasets/VQA/KoNViD/KoNViD_1k_videos/3337642103.mp4'
    videoClip = skvideo.io.vread(videoName)

    # Solve the guiding map (in GPU)
    se = SolveSpatialEnergyMap(videoClip).cpu().numpy()
    te = SolveTemporalEnergyMap(videoClip).cpu().numpy()


    totalEMap = fuseTeSeMap(se,te)
    totalEMapToned = gammaAdjust(totalEMap)
    resList = energyGuidedPoolingCore(se,te)

    sav = dict()
    # sav[''] = resList

    cubes = list()
    ## crop and save
    for rect in resList:
        cube = videoClip[:, rect[0]:rect[1], rect[2]:rect[3], :]
        cubes.append(cube)

    sav['cubes'] = cubes
    sav['EnergyMap'] = totalEMap
    ## for testing
    sav['otherFeatsandLabels'] = np.random.rand(300,8012)

    # 38M perimage
    with open('testMem.pkl', 'wb') as f:
        pickle.dump(sav, f)
    
    with open('testMem.pkl', 'rb') as f:
        data = pickle.load(f)
    
    ytef = 532
'''