#importing the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
#suppose my filename is Salary_Data.csv

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[ : , : -1].values
y=dataset.iloc[ : , -1].values

#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = test_train_split(x,y,test_size=0.2,random_state=0)

#Training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#if you want to visualise the training set results
plt.scatter(x_train,y_train,color='red)
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Training Set)')         #making the graph look beautiful
plt.xlabel('Years Of Experience');
plt.ylabel('Salary')
plt.show()

#if you want to visualise the test set results
plt.scatter(x_test,y_test,color='red)
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Test Set)')         #making the graph look beautiful
plt.xlabel('Years Of Experience');
plt.ylabel('Salary')
plt.show()
