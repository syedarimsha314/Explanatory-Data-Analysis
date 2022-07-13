# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 22:01:02 2022

@author: HP
"""

import numpy as np
import math
import pandas as pd
# data visualization
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("sydney_house_prices.csv")
suburbs = pd.read_csv("sydney_suburbs.csv")
#Gathering some information regarding datasets
print(data.describe())
print(data.shape)
print(data.columns)
print(data.info())
print("\n Number of rows: {0:,d} \n Number of features: {1:10,d}".format(data.shape[0],data.shape[1]))
print(suburbs.describe())
print(suburbs.shape)
print(suburbs.columns)
print(suburbs.info())
print("\n Number of rows: {0:,d} \n Number of features: {1:10,d}".format(suburbs.shape[0],suburbs.shape[1]))

#Counting missing data for each feature
print(data.isnull().sum())
#Visualizing missing values
print("Visualizing missing data")
msno.bar(data)

# Counting unique values for each feature of the data set
count= pd.DataFrame()
for i in data.columns:
    l = len(data[i].value_counts())
    new_row = {'feature' : i, 'total_unique_values' : l}
    count = count.append(new_row, ignore_index= True)
print(count)
print(data['suburb'].describe())

print('Counting records for each unique value per feature')
for i in data.columns:
    print(data[i].value_counts().to_frame())

#*Findings
#199,504 data points (records), 9 features
#Incorrect datatypes existing [Date] [bed] [car]
#Missing values existing [bed] [car]
#Inconsistent feature name
#Outliers: [sellPrice] [bed] [bath] [car]
#Other things to consider when doing analysis:
#house is the most sold property type
#number of car spaces almost range between (1,10), most common is 1 or 2