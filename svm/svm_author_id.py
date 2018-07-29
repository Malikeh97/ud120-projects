#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]


#########################################################
### your code goes here ###
clf = svm.SVC(kernel='linear')
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

# count = 0
# for i in range(0,len(pred))
#     if (pred[i] == labels_train[i]:
#         count = count+1
accuracy = accuracy_score(labels_test, pred)
print("accuracy: ", accuracy)

#########################################################
