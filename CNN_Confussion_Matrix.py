from sklearn.metrics import confusion_matrix
import pickle
from os import listdir
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import tensorflow as tf
import numpy as np

features = ["seq","stddev","min","state_number","mean","drate","srate","max","attack"]

model = tf.keras.models.load_model("CNN_Model.h5")

def new_dataset(path):
    csv = pd.read_csv(path)
    csv = csv.filter(items=features)
    dataset = csv.values
    X = dataset[:,0:8]
    X = pd.DataFrame(X).fillna(0)
    X = MinMaxScaler().fit_transform(X)
    y = dataset[:,8]
    pred = model.predict(X)
    pred = np.where(pred>0.5, 1, pred)
    pred = np.where(pred<0.5, 0, pred)
    cf = confusion_matrix(y, pred)
    tn, fn, fp, tp = cf.ravel()
    print(cf)

new_dataset("UNSW_2018_IoT_Botnet_Dataset_Equitative.csv")
new_dataset("UNSW_2018_IoT_Botnet_Dataset_100k.csv")
new_dataset("UNSW_2018_IoT_Botnet_Dataset_beta.csv")