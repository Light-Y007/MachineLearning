#Multiple Linear Regression

#Data Preprocessing

#importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical data
#encoding the independent variable
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)

#Avoiding the dummy variable trap

x = x[:,1:]

#Splitting the data into training set and test set

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

'''
#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

#Fitting Multiple Linear Regression to the traonong set
from sklearn.linear_model import LinearRegression























