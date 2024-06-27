import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('d:/MACHINELEARNING/Dataset.csv', sep=';')

print(dataset.head())
print(dataset.columns)

numerical_columns = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
numerical_data = dataset[numerical_columns]

X = numerical_data.iloc[:, :-1].values
y = numerical_data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print( X[:5])

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

print(regressor.predict([[160000, 130000, 300000]]))
