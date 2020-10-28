import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

#Load dataset
df = pd.read_csv("UNSW_2018_IoT_Botnet_Dataset_100k.csv")

#Extract features and label
features = ["seq","stddev","min","state_number","mean","drate","srate","max","attack"]
df = df.filter(items=features)

#Fill na's
dataset = df.values
X = dataset[:,0:8]
X = pd.DataFrame(X).fillna(0)
y = dataset[:,8]

#Normalize data
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

#Split training and test
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.3, random_state=0)

#Model creation
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2000,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2000,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1,activation=tf.nn.sigmoid))
model.compile(optimizer = "sgd", loss = 'binary_crossentropy', metrics = ['accuracy'] )
history = model.fit(X_train, y_train, epochs = 9, validation_data=(X_test, y_test))

model.save("CNN_Model.h5")

#Confussion matrix
[pred] = model.predict(X_scale).T
#print(len(pred), len(y))
#print(pred, y)
pred = np.where(pred>0.5, 1, pred)
pred = np.where(pred<0.5, 0, pred)
tn, fn, fp, tp = confusion_matrix(y, pred).ravel()
print(confusion_matrix(y, pred))
accuracy = (tp + tn) / (tp + fp + tn + fn)
false_alarm = fp / (tn + fp)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = (2 * precision * recall) / (precision + recall)
print("Accuracy: {:.5f}\nFalse Alarm Rate: {:.5f}\nPrecision: {:.5f}\nRecall: {:.5f}\nF1-measure: {:.5f}"
        .format(accuracy, false_alarm, precision, recall, f1_score))

def new_dataset(path):
        csv = pd.read_csv(path)
        csv = csv.filter(items=features)
        dataset = csv.values
        X = dataset[:,0:8]
        X = pd.DataFrame(X).fillna(0)
        X = min_max_scaler.fit_transform(X)
        y = dataset[:,8]
        pred = model.predict(X)
        pred = np.where(pred>0.5, 1, pred)
        pred = np.where(pred<0.5, 0, pred)
        cf = confusion_matrix(y, pred)
        tn, fn, fp, tp = cf.ravel()
        print(cf)

new_dataset("UNSW_2018_IoT_Botnet_Dataset_Equitative.csv")
new_dataset("UNSW_2018_IoT_Botnet_Dataset_beta.csv")