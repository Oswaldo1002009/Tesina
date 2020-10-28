import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pickle

#Load dataset
df = pd.read_csv("UNSW_2018_IoT_Botnet_Dataset_100k.csv")

#Get the best categories and label of the dataset
features = ["seq","stddev","min","state_number","mean","drate","srate","max","attack"]
df = df.filter(items=features)

#Split into X (matrix) and y (array)
dataset = df.values
X = dataset[:,0:8]
#X = pd.DataFrame(X).fillna(0)
y = dataset[:,8]

#Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Start algorithm
n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors, weights='distance')
print("Starting algorithm")
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.5f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.5f}'
     .format(knn.score(X_test, y_test)))

pickle.dump(knn, open('KNN_Model', 'wb'))

def new_dataset(path):
        csv = pd.read_csv(path)
        csv = csv.filter(items=features)
        dataset = csv.values
        X = dataset[:,0:8]
        X = pd.DataFrame(X).fillna(0)
        X = scaler.fit_transform(X)
        y = dataset[:,8]
        pred = knn.predict(X)
        cf = confusion_matrix(y, pred)
        tn, fn, fp, tp = cf.ravel()
        print(cf)

#new_dataset("UNSW_2018_IoT_Botnet_Dataset_beta.csv")

'''#Confussion matrix and scores
def cf_calc(X, y):
     X = scaler.fit_transform(X)
     pred = knn.predict(X)
     tn, fn, tp, fp = confusion_matrix(y, pred).ravel()
     print(confusion_matrix(y, pred))
     #print(tn, fn, tp, fp)
     accuracy = (tp + tn) / (tp + fp + tn + fn)
     false_alarm = fp / (tn + fp)
     precision = tp / (tp + fp)
     recall = tp / (tp + fn)
     f1_score = (2 * precision * recall) / (precision + recall)
     print("Accuracy: {:.5f}\nFalse Alarm Rate: {:.5f}\nPrecision: {:.5f}\nRecall: {:.5f}\nF1-measure: {:.5f}"
        .format(accuracy, false_alarm, precision, recall, f1_score))'''