import cv2
import numpy as np
import csv
import pickle


def load_cfar10_batch(path):
    with open(path, mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels


sift_extractor = cv2.xfeatures2d.SIFT_create()
features = []
labels=[]
des_list = []

images,class_labels=load_cfar10_batch("/home/iiitb/PycharmProjects/ML/_VisualRecognition/Assignment2/cifar_10/data_batch_1")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = sift_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        vectorized = np.float32(descriptor.reshape(-1,1))
        des_list.append(vectorized)
    else:
        index += [id]
    id += 1
for ele in sorted(index, reverse=True):
    del class_labels[ele]
labels=class_labels

images,class_labels=load_cfar10_batch("/home/iiitb/PycharmProjects/ML/_VisualRecognition/Assignment2/cifar_10/data_batch_2")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = sift_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        vectorized = np.float32(descriptor.reshape(-1,1))
        des_list.append(vectorized)
    else:
        index += [id]
    id += 1
for ele in sorted(index, reverse=True):
    del class_labels[ele]
labels=labels+class_labels

images,class_labels=load_cfar10_batch("/home/iiitb/PycharmProjects/ML/_VisualRecognition/Assignment2/cifar_10/data_batch_3")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = sift_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        vectorized = np.float32(descriptor.reshape(-1,1))
        des_list.append(vectorized)
    else:
        index += [id]
    id += 1
for ele in sorted(index, reverse=True):
    del class_labels[ele]
labels=labels+class_labels

images,class_labels=load_cfar10_batch("/home/iiitb/PycharmProjects/ML/_VisualRecognition/Assignment2/cifar_10/data_batch_4")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = sift_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        vectorized = np.float32(descriptor.reshape(-1,1))
        des_list.append(vectorized)
    else:
        index += [id]
    id += 1
for ele in sorted(index, reverse=True):
    del class_labels[ele]
labels=labels+class_labels

images,class_labels=load_cfar10_batch("/home/iiitb/PycharmProjects/ML/_VisualRecognition/Assignment2/cifar_10/data_batch_5")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = sift_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        vectorized = np.float32(descriptor.reshape(-1,1))
        des_list.append(vectorized)
    else:
        index += [id]
    id += 1
for ele in sorted(index, reverse=True):
    del class_labels[ele]
labels=labels+class_labels

images,class_labels=load_cfar10_batch("/home/iiitb/PycharmProjects/ML/_VisualRecognition/Assignment2/cifar_10/test_batch")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = sift_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        vectorized = np.float32(descriptor.reshape(-1,1))
        des_list.append(vectorized)
    else:
        index += [id]
    id += 1
for ele in sorted(index, reverse=True):
    del class_labels[ele]
labels=labels+class_labels

#stacking all the descriptors
descriptors=des_list[0]
for i in range(1,len(des_list)):
    descriptors=np.vstack([descriptors, des_list[i]])
#kmeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, centers = cv2.kmeans(descriptors, 150, None, criteria, 10, cv2.KMEANS_PP_CENTERS)#clusters and iterations
#feature vect, histogram of visual words
for i in range(len(des_list)):
    n=len(des_list[i])
    freq = np.histogram(label[:n],bins=range(0,151))[0].tolist()
    label=label[n:]
    features+=[freq]

features+=[labels]
with open('cifar_sift_features.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(features)

