import os
import Config as conf #
import pickle
import io
from thop import  profile

'''
为确保泛化性，不能与配置文件耦合

'''


def securePath(path):
    pathList = list()
    while (not os.path.exists(path)) and (path!=''):
        pathList.append(path)
        path = os.path.dirname(path)
    pathList.reverse()
    for elem in pathList:
        os.mkdir(elem)

    return


def secureSoftLink(src,dst):
    ## src dst
    if not os.path.exists(dst):
        os.symlink(src,dst)
    return



def saveCheckpoint(netModel, epoch, iterr, glbiter, fnCore='model',savingPath='',defaultFileName = None):
    ##net_state = netModel.state_dict()
    res = dict()
    ##res['NetState'] = net_state
    res['NetState'] = netModel
    res['Epoch'] = epoch
    res['Iter'] = iterr
    res['GlobalIter'] = glbiter
    if defaultFileName is None:
        fn = fnCore + '_' + str(epoch) + '_' + str(iterr) + '.mdl'
    else:
        fn = defaultFileName
    pfn = os.path.join(savingPath, fn)
    pfnFile = io.open(pfn, mode='wb')
    pickle.dump(res, pfnFile)


def savePickle(obj,fn):
    with open(fn,'wb') as f:
        pickle.dump(obj,f)
    return

def loadPickle(fn):
    with open(fn,'rb') as f:
        res = pickle.load(f)
    return  res


def loadSpecificCheckpointNetState1(epoch, iterr, fnCore='model',savingPath='',defaultFileName = None):
    if defaultFileName is None:
        fn = fnCore + '_' + str(epoch) + '_' + str(iterr) + '.mdl'
        pfn = os.path.join(conf.MODEL_PATH, fn)
    else:
        pfn = os.path.join(savingPath,defaultFileName)
    with open(pfn,'rb') as f:
        res = pickle.load(f)
    net_state = res['NetState']
    globalIter = res['GlobalIter']
    return net_state, globalIter


def loadLatestCheckpoint(modelPath,fnCore='model'):
    # return net_status epoch iterr
    candidateCpSet = os.listdir(modelPath)
    candidateCpSet = [x for x in candidateCpSet if x.startswith(fnCore) and x.endswith('.mdl')]
    if len(candidateCpSet) == 0:
        return None, 0, 0, 0
    ref = [x.split('.')[0] for x in candidateCpSet]
    ref1 = [x.split('_')[1] for x in ref]
    ref2 = [x.split('_')[2] for x in ref]
    factor = 10 ** len(sorted(ref2, key=lambda k: len(k), reverse=True)[0])
    ref1 = [int(x) for x in ref1]
    ref2 = [int(x) for x in ref2]
    reff = list(zip(ref1, ref2))
    reff = [x[0] * factor + x[1] for x in reff]
    idx = reff.index(max(reff))
    latestCpFn = candidateCpSet[idx]
    latestCpFnFIO = io.open(os.path.join(modelPath, latestCpFn), 'rb')
    res = pickle.load(latestCpFnFIO)
    return res['NetState'], res['Epoch'], res['Iter'], res['GlobalIter']

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def loadModelsByPath(path):
    file = io.open(path, 'rb')
    res = pickle.load(file)
    return res['NetState'], res['Epoch'], res['Iter'], res['GlobalIter']

def loadStandAloneModelsByPath(path):
    file = io.open(path, 'rb')
    res = pickle.load(file)
    return res['NET']

def netDict2Net(dirName,net):
    print('Converting model paramdict to standalone network.')
    fnSet = os.listdir(dirName)
    fnSet = [x for x in fnSet if x[-4:]=='.mdl']
    dstDir = 'ConvertedStandaloneMethods'
    securePath(os.path.join(dirName,dstDir))
    for fn in fnSet:
        latestCpFnFIO = io.open(os.path.join(dirName, fn), 'rb')
        res = pickle.load(latestCpFnFIO)
        net.load_state_dict(res['NetState'])
        res['NET'] = net
        pfn = os.path.join(dirName,dstDir,fn)
        pfnFile = io.open(pfn, mode='wb')
        pickle.dump(res, pfnFile)
        print('Done: %s'% pfn)

def sizeofNet(net,inputs):
    # GFLOPs = 10 ^ 9 FLOPs  flops, params = sizeofNet(net,[torch.rand(1,3,128,128).cuda()])

    flops,params = profile(net,inputs=inputs)
    return flops,params


