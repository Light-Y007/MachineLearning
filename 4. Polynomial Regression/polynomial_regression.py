#Polynomial Regression

#importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''#Taking care of missing data
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

no need for spliting because only 10 records are there

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting the Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
 

#Fitting the Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualising the Linear Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression 4th Degree more smooth using X_grid)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))


























