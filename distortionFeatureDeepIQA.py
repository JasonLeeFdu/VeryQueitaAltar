from numpy.lib.stride_tricks import as_strided
from chainer import serializers
import six
import numbers
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np
import skvideo.io

'''


DeepIQA TIP2018

'''

# model defination
class NR_IQA_Model(chainer.Chain):
    def __init__(self, top="patchwise"):
        super(NR_IQA_Model, self).__init__(
            conv1=L.Convolution2D(3, 32, 3, pad=1),
            conv2=L.Convolution2D(32, 32, 3, pad=1),

            conv3=L.Convolution2D(32, 64, 3, pad=1),
            conv4=L.Convolution2D(64, 64, 3, pad=1),

            conv5=L.Convolution2D(64, 128, 3, pad=1),
            conv6=L.Convolution2D(128, 128, 3, pad=1),

            conv7=L.Convolution2D(128, 256, 3, pad=1),
            conv8=L.Convolution2D(256, 256, 3, pad=1),

            conv9=L.Convolution2D(256, 512, 3, pad=1),
            conv10=L.Convolution2D(512, 512, 3, pad=1),

            fc1=L.Linear(512, 512),
            fc2=L.Linear(512, 1),

            fc1_a=L.Linear(512, 512),
            fc2_a=L.Linear(512, 1)
        )
        self.top = top

    def forward(self, x_data, y_data, train=True, n_patches=32):
        if not isinstance(x_data, Variable):
            x = Variable(x_data)
        else:
            x = x_data
            x_data = x.data
        self.n_images = y_data.shape[0]
        self.n_patches = x_data.shape[0]
        self.n_patches_per_image = self.n_patches / self.n_images

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.max_pooling_2d(h, 2)
        h_ = h
        self.feat = self.fc1(h_)

## initialization
MODEL_PATH = './pretrainedModels/nr_tid_patchwise.model'
chainer.global_config.train = False
chainer.global_config.cudnn_deterministic = True
model = NR_IQA_Model()
cuda.cudnn_enabled = True
cuda.check_cuda_available()
xp = cuda.cupy
serializers.load_hdf5(MODEL_PATH, model) # load the pretrained Deep IQA model
model.to_gpu()

# image to patches
def extract_patches(arr, patch_shape=(32,32,3), extraction_step=32):
    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches

def ExtractFeatImg(im,model_=model):
    patches = extract_patches(im)
    X = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2)) # for [NCHW]
    y = list()
    batchsize = min(2000, X.shape[0])
    t = xp.zeros((1, 1), np.float32)
    for i in six.moves.range(0, X.shape[0], batchsize):
        X_batch = X[i:i + batchsize]
        X_batch = xp.array(X_batch.astype(np.float32))
        model_.forward(X_batch, t, False, X_batch.shape[0])
        y.append(xp.asnumpy(model_.feat.data))
    y = np.concatenate(y,axis=0) # axis=0 patchwise concat
    y = np.mean(y,axis=0)
    y = np.expand_dims(y,axis=0)
    return y

def ExtractFeatVid(vd,model_=model):
    '''

    :param vd:      [T,H,W,C]
    :return:feat    [T,512]
    '''
    featList= list()
    T = vd.shape[0]
    for i in range(T):
        im = vd[i,:,:,:]
        feat = ExtractFeatImg(im,model_)
        featList.append(feat)
    feat = np.concatenate(featList,axis=0)
    return feat







if __name__ == '__main__':
    # 1.5GB GPU usage
    videoName = '/home/winston/workSpace/PycharmProjects/VQA/Ours/Datasets/VQA/KoNViD/KoNViD_1k_videos/3337642103.mp4'
    videoClip = skvideo.io.vread(videoName)
    feat = ExtractFeatVid(videoClip)
    f = 5342




