import torch
import torch.nn as nn
import torch.nn.functional as F

### ===========================================================================
'''
   最新更新时间: 20190714

'''


### ==========  LOSS  ========================================================

class softmaxedLoss(nn.Module):
    def __init__(self,dim=-1):
        super(softmaxedLoss, self).__init__()
        self.dim = dim

    def forward(self,softMaxedPred,label):
        # softMaxedPred:[D1,D2,...,Dn,DclsNum]: already softmaxed
        # label  :[D1,D2,...,Dn]
        shape  = softMaxedPred.shape
        if self.dim == -1:
            onehot = torch.zeros(shape).scatter_(len(shape)-1,torch.unsqueeze(label,dim=-1) , 1)
        else:
            onehot = torch.zeros(shape).scatter_(self.dim,torch.unsqueeze(label,dim=self.dim) , 1)
        coef   = -torch.log(softMaxedPred)
        lossPoints = torch.sum(torch.mul(onehot,coef))
        return  lossPoints
class softmaxingLoss(nn.Module):
    def __init__(self,dim=-1):
        super(softmaxingLoss, self).__init__()
        self.dim = dim

    def forward(self,rawPred,label):
        # softMaxedPred:[D1,D2,...,Dn,DclsNum]: already softmaxed
        # label  :[D1,D2,...,Dn]
        shape  = rawPred.shape
        softMaxedPred = F.softmax(rawPred,dim=self.dim)
        if self.dim == -1:
            onehot = torch.zeros(shape).scatter_(len(shape)-1,torch.unsqueeze(label,dim=-1) , 1)
        else:
            onehot = torch.zeros(shape).scatter_(self.dim,torch.unsqueeze(label,dim=self.dim) , 1)
        coef   = -torch.log(softMaxedPred)
        lossPoints = torch.sum(torch.mul(onehot,coef))
        return  lossPoints




### ===========================================================================
# nTimes,inC,outC,name=''
class ResBlock(nn.Module):
    def __init__(self,nTimes,inC,outC,name=''):
        super(ResBlock, self).__init__()
        pivot = nTimes // 2
        self.bone = nn.Sequential()
        for i in range(nTimes):
            if i < pivot:
                self.bone.add_module(name+'_c'+str(i),nn.Conv2d(inC, inC, kernel_size=(3, 3), padding=1))
                self.bone.add_module(name+'_b'+str(i),nn.BatchNorm2d(inC))
                self.bone.add_module(name+'_r'+str(i),nn.ReLU())
            elif i == pivot:
                self.bone.add_module(name + '_c' + str(i), nn.Conv2d(inC, outC, kernel_size=(3, 3), padding=1))
                self.bone.add_module(name + '_b' + str(i), nn.BatchNorm2d(outC))
                self.bone.add_module(name + '_r' + str(i), nn.ReLU())
            elif i < nTimes-1:
                self.bone.add_module(name+'_c'+str(i),nn.Conv2d(outC, outC, kernel_size=(3, 3), padding=1))
                self.bone.add_module(name + '_b' + str(i), nn.BatchNorm2d(outC))
                self.bone.add_module(name + '_r' + str(i), nn.ReLU())
            else:
                self.bone.add_module(name+'_c'+str(i),nn.Conv2d(outC, outC, kernel_size=(3, 3), padding=1))
    def forward(self,x):
        ress = self.bone(x) + x
        return ress
    def initWeightSet(self):
        def __weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity ='leaky_relu')
                m.bias.data.zero_()
        self.apply(__weights_init)  # 初始化的一种方法

# nTimes,inoutC,name=''
class RBF(nn.Module):
    def __init__(self,nTimes,inoutC,name=''):
        super(RBF, self).__init__()
        pivot = nTimes // 2
        self.bone = nn.Sequential()
        self.rateGenerator = nn.Conv2d(2*inoutC,2,kernel_size=(3,3), padding=1) # 它的输出通道数可以是所有的
        self.fusionRatioStd = nn.Softmax(dim=1)
        for i in range(nTimes):
            if i < pivot:
                self.bone.add_module(name + '_c' + str(i), nn.Conv2d(inoutC, inoutC, kernel_size=(3, 3), padding=1))
                self.bone.add_module(name + '_b' + str(i), nn.BatchNorm2d(inoutC))
                self.bone.add_module(name + '_r' + str(i), nn.ReLU())
            elif i == pivot:
                self.bone.add_module(name + '_c' + str(i), nn.Conv2d(inoutC, inoutC, kernel_size=(3, 3), padding=1))
                self.bone.add_module(name + '_b' + str(i), nn.BatchNorm2d(inoutC))
                self.bone.add_module(name + '_r' + str(i), nn.ReLU())
            else:
                self.bone.add_module(name + '_c' + str(i), nn.Conv2d(inoutC, inoutC, kernel_size=(3, 3), padding=1))
                self.bone.add_module(name + '_b' + str(i), nn.BatchNorm2d(inoutC))
                self.bone.add_module(name + '_r' + str(i), nn.ReLU())

    def forward(self, x):
        x1 = self.bone(x)
        xx1 = torch.cat([x,x1],dim=1)
        rawFusionRatio = self.rateGenerator(xx1)  # => [N,2,H,W]
        fusionRatio = self.fusionRatioStd(rawFusionRatio)
        out = torch.mul(x,torch.unsqueeze(fusionRatio[:,0,:,:],dim=1)) + torch.mul(x1,torch.unsqueeze(fusionRatio[:,1,:,:],dim=1))
        return out
    def initWeightSet(self):
        def __weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

        self.apply(__weights_init)  # 初始化的一种方法

# nTimesSmall,nTimesBig,inoutC,name=''
class RFC(nn.Module):
    def __init__(self,nTimesSmall,nTimesBig,inoutC,name=''):
        super(RFC, self).__init__()
        self.bone33 = nn.Sequential()
        self.bone55 = nn.Sequential()
        for i in range(nTimesSmall):
            self.bone33.add_module(name + '_33_c' + str(i), nn.Conv2d(inoutC, inoutC, kernel_size=(3, 3), padding=1))
            self.bone33.add_module(name + '_b' + str(i), nn.BatchNorm2d(inoutC))
            self.bone33.add_module(name + '_r' + str(i), nn.ReLU())
        for i in range(nTimesBig):
            self.bone55.add_module(name + '_55_c' + str(i), nn.Conv2d(inoutC, inoutC, kernel_size=(5, 5), padding=2))
            self.bone55.add_module(name + '_b' + str(i), nn.BatchNorm2d(inoutC))
            self.bone55.add_module(name + '_r' + str(i), nn.ReLU())
        self.fc = nn.Linear(2*inoutC,inoutC)
        self.dp = nn.Dropout()
    def forward(self, x):
        w = x.shape[3]
        h = x.shape[2]
        x33 = self.bone33(x)
        x55 = self.bone55(x)
        xcat = torch.cat([x33,x55],dim = 1)  # 0--inoutC-1    inoutC -- 2*inoutC-1
        avgpoolFeat = torch.mean(xcat, dim=[2, 3])
        rawSoftChoice = self.dp(self.fc(avgpoolFeat))  # [N,inoutC] & 0.3453
        rawSoftChoice = torch.unsqueeze(rawSoftChoice, dim=2);rawSoftChoice = torch.unsqueeze(rawSoftChoice,dim=3);
        rawSoftChoice = rawSoftChoice.repeat(1,1,h,w)
        selected = torch.where(rawSoftChoice>0,x33,x55)
        return selected
    def initWeightSet(self):
        def __weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity ='leaky_relu')
                m.bias.data.zero_()
        self.apply(__weights_init)  # 初始化的一种方法

# inoutC,name=''
class Inception(nn.Module):
    def __init__(self,inoutC,name=''):
        super(Inception, self).__init__()
        self.bone33 = nn.Sequential()
        self.bone33.add_module(name + '_33_c', nn.Conv2d(inoutC, inoutC, kernel_size=(3, 3), padding=1))
        self.bone33.add_module(name + '_33_b', nn.BatchNorm2d(inoutC))
        self.bone33.add_module(name + '_33_r', nn.ReLU())
        self.bone11 = nn.Sequential()
        self.bone11.add_module(name + '_33_c', nn.Conv2d(inoutC, inoutC, kernel_size=(1, 1), padding=0))
        self.bone11.add_module(name + '_33_b', nn.BatchNorm2d(inoutC))
        self.bone11.add_module(name + '_33_r', nn.ReLU())
        self.bone55 = nn.Sequential()
        self.bone55.add_module(name + '_33_c', nn.Conv2d(inoutC, inoutC, kernel_size=(5, 1), padding=(2,0)))
        self.bone55.add_module(name + '_33_c', nn.Conv2d(inoutC, inoutC, kernel_size=(1, 5), padding=(0,2)))
        self.bone55.add_module(name + '_33_b', nn.BatchNorm2d(inoutC))
        self.bone55.add_module(name + '_33_r', nn.ReLU())
        self.fusion = nn.Conv2d(3*inoutC, inoutC, kernel_size=(3, 3), padding=1)

    def forward(self,x):
        x1 = self.bone11(x)
        x3 = self.bone33(x)
        x5 = self.bone55(x)
        x135 = torch.cat([x1,x3,x5],dim=1)
        out = self.fusion(x135)
        return out
    #

# as nn.Con2d
class OldConvolutionBN(nn.Module):# CBR
    def __init__(self,inc,outc,kernel_size,stride = 1,padding=0,dilation=1):
        super(OldConvolutionBN, self).__init__()
        self.conv = nn.Conv2d(inc,outc,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        self.bn   = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        return y
class OldConvolution(nn.Module):# CBR
    def __init__(self,inc,outc,kernel_size,stride = 1,padding=0,dilation=1):
        super(OldConvolution, self).__init__()
        self.conv = nn.Conv2d(inc,outc,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        return y
### ===========================================================================


