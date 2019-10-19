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




caffe.set_mode_gpu()
imgfilepath = '/home/tb/Desktop/save_test/'

net = caffe.Net( '/home/tb/caffe/models/bvlc_alexnet/deploy.prototxt',
                 '/home/tb/caffe/models/bvlc_alexnet/BVLC_AlexNet.caffemodel',
                 caffe.TEST)
testingfilepath = glob.glob('/home/tb/Desktop/save_test/*.jpg')

final_data_mask = np.array(scipy.io.loadmat('//home/tb/Desktop/mtcnn/extracted_features.mat').get("features"))
final_data_nonmask = np.array(scipy.io.loadmat('/home/tb/Desktop/mtcnn/extracted_features_nonmask.mat').get("features"))


net.blobs['data'].reshape(1,3,227,227)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('/home/tb/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
# transformer.set_channel_swap('data', (2,1,0))
# transformer.set_raw_scale('data', 255.0)

testing_data = np.concatenate((np.array(final_data_mask),np.array(final_data_nonmask)))
print(testing_data)

# load input and configure preprocessing
# for t in testingfilepath:
#     print(t)
#     i = cv2.imread(t)
#     i = transformer.preprocess('data', i)
#     net.blobs['data'].data[...] = i
#     # print(net.blobs['data'].data[0][0, :10, :10])
#     # net.blobs['data'].data[...] = transformer.preprocess('data', i)

#     # note we can change the batch size on-the-fly
#     # since we classify only one image, we change batch size from 10 to 1

#     # compute

#     out = net.forward()
#     fc7 = net.blobs['fc7'].data[0].copy()
#     testing_data.append(fc7)

# print(len(final_data_mask))
# print(len(final_data_nonmask))
X_train = np.concatenate((np.array(final_data_mask),np.array(final_data_nonmask)),axis = 0)
Y_train = np.concatenate((np.ones((25876,1)),np.zeros((10034,1))),axis = None)
# print(len(X_train))
# print(len(Y_train)) 
    


clf = svm.LinearSVC(random_state=0, tol=1e-5, max_iter=1)   
# clf = joblib.load('saved_smv.joblib') 
clf.fit(X_train,Y_train)
# clf = joblib.load('/home/tb/Desktop/mtcnn/saved_smv.joblib')
joblib.dump(clf,'saved_smv.joblib')
print ("Loaded the SVM")
res = clf.predict(np.array(testing_data))
accuracy = accuracy_score(Y_train,res)
print(accuracy)





