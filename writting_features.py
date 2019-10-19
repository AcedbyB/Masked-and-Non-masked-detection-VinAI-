import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import sklearn
import cv2
from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from PIL import Image
import joblib
from joblib import dump,load
import glob
import importlib
import scipy.io
from sklearn.metrics import accuracy_score




net = caffe.Net( '/home/tb/caffe/models/bvlc_alexnet/deploy.prototxt',
                 '/home/tb/caffe/models/bvlc_alexnet/BVLC_AlexNet.caffemodel',
                 caffe.TEST)
imgfilepath = glob.glob('/home/tb/Desktop/mtcnn/NonMask_train/*.jpg')

data = []


transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('/home/tb/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
net.blobs['data'].reshape(1,3,227,227)

cnt = 0
for t in imgfilepath:
    cnt += 1
    print(cnt)
    i = cv2.imread(t)
    i = transformer.preprocess('data', i)
    net.blobs['data'].data[...] = i
    # net.blobs['data'].data[...] = transformer.preprocess('data', i)

    # note we can change the batch size on-the-fly
    # since we classify only one image, we change batch size from 10 to 1

    # compute

    out = net.forward()
    fc7 = net.blobs['fc7'].data[0].copy()
    data.append(fc7)

scipy.io.savemat('/home/tb/Desktop/mtcnn/extracted_features_nonmask.mat',{ 'features': data})





