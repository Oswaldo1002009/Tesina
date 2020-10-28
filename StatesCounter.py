import os
from os import listdir
from os.path import isfile, join
import pandas as pd

files = [f for f in listdir("AllCategories/") if f.endswith(".csv")]
for f in files:
    temp = pd.read_csv("AllCategories/"+f)
    print( temp.groupby(['state','state_number']).size().reset_index().rename(columns={0:'count'}) )