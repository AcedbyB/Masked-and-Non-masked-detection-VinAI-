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
import demo
import scipy.io



image_list = glob.glob('/home/group-3/Downloads/cropped_10000/*.jpg')
caffe.set_mode_gpu()
cnt = 0


net = caffe.Net( '/home/group-3/caffe/models/bvlc_alexnet/deploy.prototxt',
                 '/home/group-3/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel',
                 caffe.TEST)

final_data = []


net.blobs['data'].reshape(1,3,227,227)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('/home/group-3/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
# transformer.set_channel_swap('data', (2,1,0))
# transformer.set_raw_scale('data', 255.0)


# # load input and configure preprocessing
for t in image_list:
    cnt +=1
    i = cv2.imread(t)
    print(t)
    i = transformer.preprocess('data', i)
    net.blobs['data'].data[...] = i
    # print(net.blobs['data'].data[0][0, :10, :10])
    # net.blobs['data'].data[...] = transformer.preprocess('data', i)

    # note we can change the batch size on-the-fly
    # since we classify only one image, we change batch size from 10 to 1

    #compute

    out = net.forward()
    fc7 = net.blobs['fc7'].data[0].copy()
    final_data.append(fc7)

print(len(final_data))
print(final_data)

scipy.io.savemat('extracted_features_nonmask.mat',{'features':final_data})
clf = svm.SVC(gamma=0.001)
# clf = joblib.load('saved_smv.joblib') 
# clf.fit(final_data,)
# joblib.dump(clf,'saved_smv.joblib')
# print ("Got the SVM")
# res = clf.predict(np.array(outlier_data))

# print(res)
# print('AAAAAAAAAAAAAAAAAAAAAA')




