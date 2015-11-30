import numpy as np
from numpy import arange
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import time
import matplotlib.pyplot as plt

mnist = fetch_mldata("MNIST Original")
n_train = 60000
n_test = 10000
train_idx = arange(0,n_train)
test_idx = arange(n_train+1,n_train+n_test)

X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]
k = 3
precisions = []
ns = []
for n in range(1,30):
  print "Reducing dimensions using PCA with components :",n
  start_time = time.time()
  pca = PCA(n_components = n)
  pca.fit(X_train)
  print "Transforming training"
  X_train_pca = pca.transform(X_train)
  print "Transforming testing"
  X_test_pca = pca.transform(X_test)
  end_time = time.time()
  print "Running time for PCA:", end_time - start_time
  print X_train_pca.shape
  print X_test_pca.shape

  print "Applying KNN algorithm with neighbours:", k
  start_time = time.time()
  clf = KNeighborsClassifier(n_neighbors=n)
  clf.fit(X_train_pca, y_train)
  print "Making predictions..."
  y_pred = clf.predict(X_test_pca)
  print "Evaluating results..."
  precision=metrics.precision_score(y_test, y_pred)
  print "Precision=",precision
  end_time = time.time()
  print "Running time for KNN:", end_time - start_time
  ns.append(n)
  precisions.append(precision)

plt.plot(ns,precisions)
plt.ylabel('Precision')
plt.xlabel('# of reduced features with PCA')
plt.show()
