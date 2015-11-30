import numpy as np
from numpy import arange
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import time

mnist = fetch_mldata("MNIST Original")
n_train = 60000
n_test = 10000
train_idx = arange(0,n_train)
test_idx = arange(n_train+1,n_train+n_test)

X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]

for n in range(8,9):
  print "Applying KNN algorithm with neighbours:", n
  start_time = time.time()
  clf = KNeighborsClassifier(n_neighbors=n)
  clf.fit(X_train, y_train)
  print "Making predictions..."
  y_pred = clf.predict(X_test)
# Creating confusion matrix
  conf_matrix = confusion_matrix(y_test, y_pred)
  print conf_matrix
# Evaluate the prediction
  print "Evaluating results..."
  print "Precision: \t", metrics.precision_score(y_test, y_pred)
  end_time = time.time()
  print "Running time for KNN:", end_time - start_time

