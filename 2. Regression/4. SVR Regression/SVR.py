#Polynomial Regression

#importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
'''
#Splitting the data into training set and test set

no need for spliting because only 10 records are there

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting the SVR to the dataset

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf' )

regressor.fit(X , y)
#Create your regressor here


# Predicting a new result 
y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))


#Visualising the SVR Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show() 


