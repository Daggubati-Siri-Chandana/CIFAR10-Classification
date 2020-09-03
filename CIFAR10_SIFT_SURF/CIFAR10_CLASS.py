from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
import csv
import numpy as np

features = []
for row in csv.reader(open('cifar_surf_features.csv', 'r'), delimiter=','):
     features.append([int(i) for i in row])
train_label=features[len(features)-1]
train_features=features[:len(features)-1]

# features = []
# for row in csv.reader(open('cifar_sift_test.csv', 'r'), delimiter=','):
#      features.append([int(i) for i in row])
# test_label=features[len(features)-1]
# test_features=features[:len(features)-1]
#
# train_features = train_features+test_features
# train_label = train_label+test_label

z = list(zip(train_features, train_label))
TRAIN, TEST = train_test_split(z, test_size=0.1, random_state=72)
train_features, train_label = zip(*TRAIN)
test_features, test_label = zip(*TEST)


test_features=np.array(test_features, dtype=np.uint32)
train_features=np.array(train_features, dtype=np.uint32)
test_label=np.array(test_label, dtype=np.uint32)
train_label=np.array(train_label, dtype=np.uint32)
#
# test_features=train_features[:5000]
# test_label=train_label[:5000]
# #

print("\nK nearest Neighbours")
knn=KNeighborsClassifier()
knn.fit(train_features,train_label)
knn_predict=knn.predict(test_features)
actual_output=list(test_label)
print("\nAccuracy Score ", metrics.accuracy_score(actual_output, knn_predict))
print('Confusion Matrix : ',confusion_matrix(actual_output,knn_predict))
# print('Report : ',classification_report(actual_output,knn_predict))
#

print("\nada boost")
ada=GradientBoostingClassifier()
ada.fit(train_features,train_label)
ada_predict=ada.predict(test_features)
actual_output=list(test_label)
print("\nAccuracy Score ", metrics.accuracy_score(actual_output, ada_predict))
print('Confusion Matrix : ',confusion_matrix(actual_output,ada_predict))
# print('Report : ',classification_report(actual_output,knn_predict))
#
