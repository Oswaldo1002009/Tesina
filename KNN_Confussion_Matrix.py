from sklearn.metrics import confusion_matrix
import pickle
from os import listdir
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

features = ["seq","stddev","min","state_number","mean","drate","srate","max","attack"]

model = pickle.load(open("KNN_Model", 'rb'))

#Confussion matrix
def new_dataset(path):
    csv = pd.read_csv(path)
    csv = csv.filter(items=features)
    dataset = csv.values
    X = dataset[:,0:8]
    X = pd.DataFrame(X).fillna(0)
    X = MinMaxScaler().fit_transform(X)
    y = dataset[:,8]
    pred = model.predict(X)
    cf = confusion_matrix(y, pred)
    tn, fn, fp, tp = cf.ravel()
    #Scores
    print(cf)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    false_alarm = fp / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    print("Accuracy: {:.5f}\nFalse Alarm Rate: {:.5f}\nPrecision: {:.5f}\nRecall: {:.5f}\nF1-measure: {:.5f}"
        .format(accuracy, false_alarm, precision, recall, f1_score))

#new_dataset("UNSW_2018_IoT_Botnet_Dataset_Equitative.csv")

new_dataset("UNSW_2018_IoT_Botnet_Dataset_100k.csv")
    
