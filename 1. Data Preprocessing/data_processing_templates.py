#Data Preprocessing

#importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Taking care of missing data
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import OneHotEncoder , LabelEncoder
#imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#imputer = imputer.fit(X[:, 1:3])
#X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorical data
#from sklearn.compose import ColumnTransformer
#ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
#X = ct.fit_transform(X)
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

#Splitting the data into training set and test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''
