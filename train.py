import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import svm,metrics
import matplotlib.pyplot as plt
#load dataset
X = np.loadtxt("UCI HAR Dataset/train/X_train.txt")
y = np.loadtxt("UCI HAR Dataset/train/y_train.txt")
X_test = np.loadtxt("UCI HAR Dataset/test/X_test.txt")
y_test = np.loadtxt("UCI HAR Dataset/test/y_test.txt")
print X.shape
print y.shape

#feature selection
clf = ExtraTreesClassifier()
clf = clf.fit(X,y)
print clf.feature_importances_
model = SelectFromModel(clf,prefit=True)
X_new = model.transform(X)
X_test_new = model.transform(X_test)
print X_new.shape
print X_test_new.shape

#classifier
classifier = svm.SVC(gamma=0.001)
classifier.fit(X_new,y)
predicted = classifier.predict(X_test_new)
print("Classification report for classifier %s:\n%s\n" % (classifier,metrics.classification_report(y_test,predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test,predicted))
