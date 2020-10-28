import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

complete=["pkSeqID","stime","flgs","proto","saddr","sport","daddr","dport","pkts","bytes","state","ltime",
"seq","dur","mean","stddev","smac","dmac","sum","min","max","soui","doui","sco","dco","spkts","dpkts",
"sbytes","dbytes","rate","srate","drate","attack","category","subcategory"]

best_8 = ["seq","stddev","min","state","mean","drate","srate","max","attack"]

normal = pd.DataFrame(columns=best_8)

files = [f for f in listdir("EntireDataset/") if f.endswith(".csv")]
for f in files:
    temp = pd.read_csv("EntireDataset/"+f, names=complete)
    print("File: " + f)
    extracted = temp.loc[temp["category"] == 'Normal'].filter(items=best_8) #Get just 8 features
    print("Extracted " + str(len(extracted)) + " elements")
    normal = normal.append(extracted, sort=False) #Append results
    print("Actual number:", len(normal))
    print("Appended " + f)

print("done")

print(str(len(normal)))

normal["state"].replace({"RST":1, "CON":2, "REQ":3, "INT":4, "URP":5,
                        "FIN":6, "ACC":7, "NRS":8, "ECO":9, "TST":10, "MAS":11}, inplace=True)

normal.to_csv('UNSW_2018_IoT_Botnet_Dataset_Normal.csv')
print("finished")